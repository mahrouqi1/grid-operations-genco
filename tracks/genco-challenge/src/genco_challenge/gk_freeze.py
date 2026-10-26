"""``gridfm_graphkit`` CLI with two opt-in plugins, applied at runtime (the cloned package is untouched).

1. Partial freezing (few-shot fine-tuning). ``FEWSHOT_TRAINABLE`` holds comma-separated regexes
   matched (``re.match``) against the model's parameter names (prefix ``model.``); every other
   parameter gets ``requires_grad=False``. graphkit builds AdamW over all parameters, and AdamW
   skips parameters without gradients, so frozen weights stay exactly at their pretrained values.

2. Slack |V| as a known input (``FEWSHOT_SLACK_VM=1``). graphkit's power-flow mask
   (``AddPFHeteroMask``) marks the REF bus |V| as unknown, although a power-flow problem specifies
   it (the slack generator's voltage setpoint). With the plugin the REF |V| entry is unmasked, so
   (a) the setpoint is fed to the model as an input feature instead of the mask value 0, and
   (b) the GNS model's ``torch.where(bus_mask, pred, fixed)`` overwrites the predicted REF |V|
   with the setpoint at every layer, and the physics loss can no longer move it. The supervised
   ``MaskedBusMSE`` drops the REF |V| term (it is no longer an unknown). Everything else, including
   the masking of REF Qg and the REF generator's Pg, is graphkit's own.

3. Input-only normalizer fit (``FEWSHOT_INPUT_NORM=1``). graphkit's ``HeteroDataMVANormalizer.fit``
   sets baseMVA to the 95th percentile of the nonzero Pd, Qd, generator p_mw and bus Qg values of
   the training split. Bus Qg and the slack generator's p_mw are AC power-flow OUTPUTS (labels).
   With the plugin the fit uses Pd, Qd, the non-slack generators' p_mw (power-flow setpoints) and,
   for slack generators, the DC power-flow dispatch ``p_mw_dc`` (computed from inputs only). The
   saved stats have the same keys, so evaluation with ``--normalizer_stats`` is unchanged.

Everything else is graphkit's own CLI, unchanged.

    FEWSHOT_TRAINABLE='model\\.mlp_bus,model\\.mlp_gen' python -m genco_challenge.gk_freeze finetune ...
    FEWSHOT_SLACK_VM=1 python -m genco_challenge.gk_freeze evaluate ...
"""
from __future__ import annotations

import os
import re

import gridfm_graphkit.cli as gk_cli
from gridfm_graphkit.__main__ import main
from gridfm_graphkit.datasets import masking as gk_masking
from gridfm_graphkit.datasets.globals import VM_H

_orig_get_task = gk_cli.get_task
_orig_pf_mask_forward = gk_masking.AddPFHeteroMask.forward


def _get_task(*args, **kwargs):
    task = _orig_get_task(*args, **kwargs)
    pats = [re.compile(p) for p in os.environ.get("FEWSHOT_TRAINABLE", "").split(",") if p]
    if pats:
        n_tr = n_all = 0
        for name, p in task.named_parameters():
            p.requires_grad = any(r.match(name) for r in pats)
            n_all += p.numel()
            n_tr += p.numel() if p.requires_grad else 0
        print(f"[gk_freeze] trainable {n_tr}/{n_all} parameters ({os.environ['FEWSHOT_TRAINABLE']})", flush=True)
    return task


def _pf_mask_slack_vm_known(self, data):
    """graphkit's PF mask, then REF |V| marked known (runs before ``ApplyMasking``)."""
    data = _orig_pf_mask_forward(self, data)
    data.mask_dict["bus"][data.mask_dict["REF"], VM_H] = False
    return data


def _fit_input_only(self, data_path: str, scenario_ids) -> dict:
    """``HeteroDataMVANormalizer.fit`` without AC outputs: no bus Qg, slack p_mw replaced by p_mw_dc."""
    import numpy as np
    import pandas as pd
    import torch

    bus = pd.read_parquet(os.path.join(data_path, "bus_data.parquet"), columns=["scenario", "Pd", "Qd", "vn_kv"])
    gen = pd.read_parquet(os.path.join(data_path, "gen_data.parquet"),
                          columns=["scenario", "p_mw", "p_mw_dc", "is_slack_gen"])
    assert bus.scenario.min() == 0 and bus.scenario.max() == len(bus.scenario.unique()) - 1
    bus = bus[bus["scenario"].isin(scenario_ids)]
    gen = gen[gen["scenario"].isin(scenario_ids)]
    if self.baseMVA is None:
        pg = pd.Series(np.where(gen["is_slack_gen"].to_numpy() == 1, gen["p_mw_dc"], gen["p_mw"]))
        vals = pd.concat([s[s != 0] for s in (bus["Pd"], bus["Qd"], pg)])
        self.baseMVA = float(np.percentile(vals, 95))
        print(f"[gk_freeze] input-only normalizer: baseMVA {self.baseMVA:.4f}", flush=True)
    self.vn_kv_max = float(bus["vn_kv"].max())
    return {"baseMVA_orig": torch.tensor(self.baseMVA_orig, dtype=torch.float),
            "baseMVA": torch.tensor(self.baseMVA, dtype=torch.float),
            "vn_kv_max": torch.tensor(self.vn_kv_max, dtype=torch.float)}


gk_cli.get_task = _get_task
if os.environ.get("FEWSHOT_FIX_SANITY") == "1":
    # 5. graphkit's best-model saver also fires in Lightning's pre-training sanity validation (2 batches only), so a
    #    lucky 2-batch value can make a run that never beats it return the PRETRAINED weights. With this plugin the
    #    saver ignores sanity checking, and additionally keeps the best checkpoint by "Validation loss" (the training
    #    objective) as best_by_loss_state_dict.pt next to the residual-monitored one.
    import torch
    from gridfm_graphkit.training import callbacks as gk_cb
    _orig_val_end = gk_cb.SaveBestModelStateDict.on_validation_end

    def _val_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        _orig_val_end(self, trainer, pl_module)
        cur = trainer.callback_metrics.get("Validation loss")
        if cur is not None and float(cur) < getattr(self, "_best_loss", float("inf")):
            self._best_loss = float(cur)
            path = gk_cb.best_model_artifact_path(trainer.logger, "best_by_loss_state_dict.pt")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self._canonical_state_dict(pl_module), path)

    gk_cb.SaveBestModelStateDict.on_validation_end = _val_end
    print("[gk_freeze] sanity-check saves OFF; best-by-loss checkpoint also kept", flush=True)
if os.environ.get("FEWSHOT_MONITOR"):   # 4. checkpoint/early-stop/LR monitor override (default: graphkit's "Validation layer_11_residual")
    gk_cli.BEST_CHECKPOINT_MONITOR = os.environ["FEWSHOT_MONITOR"]
    print(f"[gk_freeze] best-checkpoint monitor: {gk_cli.BEST_CHECKPOINT_MONITOR}", flush=True)
if os.environ.get("FEWSHOT_SLACK_VM") == "1":
    gk_masking.AddPFHeteroMask.forward = _pf_mask_slack_vm_known
    print("[gk_freeze] slack |V| plugin ON: REF bus Vm unmasked (fed as input, held at setpoint)", flush=True)
if os.environ.get("FEWSHOT_INPUT_NORM") == "1":
    from gridfm_graphkit.datasets.normalizers import HeteroDataMVANormalizer
    HeteroDataMVANormalizer.fit = _fit_input_only
    print("[gk_freeze] input-only normalizer ON (no AC Qg / slack Pg in the baseMVA fit)", flush=True)

if os.environ.get("FEWSHOT_PRIOR_DIR"):
    # 6. State-in-the-input "corrector" (opt-in, 2026-10-09). A per-scenario prior state (here the DC + two fast-decoupled
    #    iterations state "fd2", computed from inputs only) is read from FEWSHOT_PRIOR_DIR/<tree>/<grid>.npy, an array
    #    [scenarios, buses, 2] = (|V| pu, angle rad), where <tree> is the dataset root's parent folder name (train, eval2)
    #    and the scenario is the processed index. (a) The dataset attaches it as data["bus"].prior. (b) ApplyMasking, after
    #    graphkit's own masking, writes the prior into the entries the mask zeroed (Vm, Va at PQ buses; Va at PV buses; and
    #    nothing else: known entries are untouched), so the model receives the prior through its ordinary input channels.
    #    (c) FEWSHOT_PRIOR_MODE=corr: the model's bus head output is ADDED to the prior (the network predicts a correction;
    #    the where(mask, pred, fixed) step then keeps known entries). FEWSHOT_PRIOR_MODE=direct: output used as is.
    #    (d) FEWSHOT_PRIOR_ZERO_HEAD=1: after the first load_state_dict (the pretrained weights) the last Linear of
    #    mlp_bus is zeroed, so the initial correction is exactly 0.
    import numpy as np
    import torch
    from gridfm_graphkit.datasets import transforms as gk_tf
    from gridfm_graphkit.datasets.powergrid_hetero_dataset import HeteroGridDatasetDisk
    from gridfm_graphkit.datasets.globals import VA_H as _VA_H

    _PRIOR_DIR = os.environ["FEWSHOT_PRIOR_DIR"]
    _PRIOR_MODE = os.environ.get("FEWSHOT_PRIOR_MODE", "corr")
    _prior_tabs: dict = {}
    _orig_ds_get = HeteroGridDatasetDisk.get
    _orig_apply_masking = gk_tf.ApplyMasking.forward

    def _ds_get(self, idx):
        data = _orig_ds_get(self, idx)
        root = os.path.abspath(str(self.root))
        key = (os.path.basename(os.path.dirname(root)), os.path.basename(root))
        if key not in _prior_tabs:
            _prior_tabs[key] = torch.from_numpy(np.load(os.path.join(_PRIOR_DIR, key[0], key[1] + ".npy"))).float()
        sid = int(data["scenario_id"].reshape(-1)[0])
        tab = _prior_tabs[key][sid]
        assert tab.shape[0] == data["bus"].x.shape[0], (key, sid, tab.shape, data["bus"].x.shape)
        data["bus"].prior = tab.clone()
        return data

    def _apply_masking_prior(self, data):
        data = _orig_apply_masking(self, data)
        x, m, pr = data.x_dict["bus"], data.mask_dict["bus"], data["bus"].prior
        for col, k in ((VM_H, 0), (_VA_H, 1)):
            sel = m[:, col]
            x[sel, col] = pr[sel, k]
        return data

    HeteroGridDatasetDisk.get = _ds_get
    gk_tf.ApplyMasking.forward = _apply_masking_prior
    _prev_get_task = gk_cli.get_task

    def _get_task_prior(*args, **kwargs):
        task = _prev_get_task(*args, **kwargs)
        if _PRIOR_MODE == "corr":
            stash = {}

            def _pre(mod, a, kw):
                xd = a[0] if a else kw["x_dict"]
                stash["fixed"] = xd["bus"][:, VM_H:_VA_H + 1]

            task.model.register_forward_pre_hook(_pre, with_kwargs=True)
            task.model.mlp_bus.register_forward_hook(lambda mod, i, out: out + stash["fixed"])
        if os.environ.get("FEWSHOT_PRIOR_ZERO_HEAD") == "1":
            orig, done = task.load_state_dict, [False]

            def _lsd(*a, **k):
                r = orig(*a, **k)
                if not done[0]:
                    done[0] = True
                    last = [m for m in task.model.mlp_bus if isinstance(m, torch.nn.Linear)][-1]
                    torch.nn.init.zeros_(last.weight)
                    torch.nn.init.zeros_(last.bias)
                    print("[gk_freeze] prior plugin: last mlp_bus Linear zeroed (initial correction = 0)", flush=True)
                return r

            task.load_state_dict = _lsd
        print(f"[gk_freeze] prior plugin ON: mode {_PRIOR_MODE}, dir {_PRIOR_DIR}", flush=True)
        return task

    gk_cli.get_task = _get_task_prior

if os.environ.get("FEWSHOT_VAL_DVA") == "1":
    # 7. Labeled validation branch-angle-difference metric (opt-in, 2026-10-09, corrector v2). ReconstructionTask's
    #    validation_step is replaced by an identical one that also logs, on the labeled validation split,
    #    "Validation branch_dva" = mean over bus-bus edges of |wrap(dVa_pred - dVa_label)| in degrees (the lead metric of
    #    the fd2 studies) and, when the prior plugin is on, "Validation branch_dva_prior" = the same for the prior state
    #    itself (the fd2 baseline on the same split). Select/early-stop on it with FEWSHOT_MONITOR="Validation branch_dva".
    import math
    import torch
    from gridfm_graphkit.tasks.reconstruction_tasks import ReconstructionTask
    from gridfm_graphkit.datasets.globals import VA_H as _VAH, VA_OUT as _VAOUT

    def _dva(va_a, va_b, ei):
        d = (va_a[ei[0]] - va_a[ei[1]]) - (va_b[ei[0]] - va_b[ei[1]])
        d = torch.remainder(d + math.pi, 2 * math.pi) - math.pi
        return d.abs().mean() * (180.0 / math.pi)

    def _validation_step_dva(self, batch, batch_idx):
        output, loss_dict = self.shared_step(batch)
        loss_dict["loss"] = loss_dict["loss"].detach()
        ei = batch.edge_index_dict[("bus", "connects", "bus")]
        yv = batch.y_dict["bus"][:, _VAH]
        loss_dict["branch_dva"] = _dva(output["bus"][:, _VAOUT], yv, ei).detach()
        if "prior" in batch["bus"]:
            loss_dict["branch_dva_prior"] = _dva(batch["bus"].prior[:, 1], yv, ei).detach()
        for metric, value in loss_dict.items():
            self.log(f"Validation {metric}", value, batch_size=batch.num_graphs, sync_dist=True, on_epoch=True,
                     logger=True, on_step=False)
        return loss_dict["loss"]

    ReconstructionTask.validation_step = _validation_step_dva
    print("[gk_freeze] labeled validation branch-dVa metric ON", flush=True)

if os.environ.get("FEWSHOT_GRAD_CLIP") or os.environ.get("FEWSHOT_WARMUP_STEPS") or os.environ.get("FEWSHOT_STEPLOG") or os.environ.get("FEWSHOT_PROBE") == "1":
    # 8. Training-stability probes (opt-in, 2026-10-10, case118 retrain-isolation study). All three are independent env switches:
    #    FEWSHOT_GRAD_CLIP=x       Lightning Trainer(gradient_clip_val=x, algorithm "norm"): clip the global grad norm to x.
    #    FEWSHOT_WARMUP_STEPS=N    linear LR warm-up: for global_step < N the LR of every param group is base*(step+1)/N;
    #                              at step N it is set back to base (the plateau scheduler is not expected to fire before N).
    #    FEWSHOT_STEPLOG=path[:K]  write the first K (default 50) optimizer steps as jsonl: global_step, epoch, training loss,
    #                              global grad norm BEFORE clipping, lr. Nothing else changes.
    import json as _json
    import torch
    import lightning as _L
    from lightning.pytorch.callbacks import Callback as _Callback

    def _cfg():
        """Settings are re-read from the environment each time a Trainer is built, so one process can run many specs."""
        clip = float(os.environ["FEWSHOT_GRAD_CLIP"]) if os.environ.get("FEWSHOT_GRAD_CLIP") else None
        warm = int(os.environ["FEWSHOT_WARMUP_STEPS"]) if os.environ.get("FEWSHOT_WARMUP_STEPS") else 0
        sl = os.environ.get("FEWSHOT_STEPLOG")
        sl_path, sl_k = (sl.rsplit(":", 1)[0], int(sl.rsplit(":", 1)[1])) if sl and sl.rsplit(":", 1)[-1].isdigit() else (sl, 50)
        return clip, warm, sl_path, sl_k

    class _StabilityProbe(_Callback):
        def __init__(self):
            _, self.warm, self.sl_path, self.sl_k = _cfg()
            self.base = None
            self.rows = []
            self.pending = {}

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            opt = trainer.optimizers[0]
            if self.base is None:
                self.base = [g["lr"] for g in opt.param_groups]
            if self.warm and trainer.global_step < self.warm:
                for g, b in zip(opt.param_groups, self.base):
                    g["lr"] = b * (trainer.global_step + 1) / self.warm
            elif self.warm and trainer.global_step == self.warm:
                for g, b in zip(opt.param_groups, self.base):
                    g["lr"] = b

        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            if self.sl_path and len(self.rows) < self.sl_k:
                gs = [p.grad.detach().float().norm() for p in pl_module.parameters() if p.grad is not None]
                self.pending["gn"] = float(torch.stack(gs).norm()) if gs else float("nan")
                self.pending["lr"] = optimizer.param_groups[0]["lr"]

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if self.sl_path and len(self.rows) < self.sl_k and "gn" in self.pending:
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs
                self.rows.append({"step": trainer.global_step, "epoch": trainer.current_epoch, "loss": float(loss),
                                  "grad_norm_preclip": self.pending["gn"], "lr": self.pending["lr"]})
                os.makedirs(os.path.dirname(os.path.abspath(self.sl_path)), exist_ok=True)   # rewritten every step: runs shorter than K steps still leave their log
                with open(self.sl_path, "w") as fh:
                    fh.write("\n".join(_json.dumps(r) for r in self.rows) + "\n")

    _OrigTrainer = _L.Trainer

    class _ProbeTrainer(_OrigTrainer):
        def __init__(self, *a, **k):
            clip = _cfg()[0]
            if clip is not None:
                k["gradient_clip_val"] = clip
                k["gradient_clip_algorithm"] = "norm"
            k["callbacks"] = list(k.get("callbacks") or []) + [_StabilityProbe()]
            super().__init__(*a, **k)

    gk_cli.L.Trainer = _ProbeTrainer
    print(f"[gk_freeze] stability probes ON (settings read per Trainer): {_cfg()}", flush=True)

if os.environ.get("FEWSHOT_EXTRA_PLUGIN"):
    # 9. Extra opt-in plugins from another module (2026-10-10, corrector-above-162 study; additive, off unless the env var is set).
    import importlib
    for _mod in os.environ["FEWSHOT_EXTRA_PLUGIN"].split(","):   # comma-separated since 2026-10-19 (transfer runs)
        if _mod.strip():
            importlib.import_module(_mod.strip())

if __name__ == "__main__":
    main()
