"""Opt-in graphkit plugins for the corrector-above-162 study (2026-10-10). Loaded by gk_freeze only when
``FEWSHOT_EXTRA_PLUGIN=genco_challenge.corr3_plugins`` is set (a 3-line additive hook at the end of gk_freeze).
Nothing here runs otherwise. Each switch is independent:

``FEWSHOT_TRAIN_DVA=1``   log "Training branch_dva" (and "Training branch_dva_prior" when the prior plugin is on) per epoch,
                          the same quantity as the validation branch_dva, on the TRAINING batches (model in train mode, so a
                          slight under-estimate of the eval-mode fit; there is no dropout in this model, so it is the same).
``FEWSHOT_Q_LOSS=lam``    add ``lam * mean|reactive residual|`` (layer-weighted like the physics loss, same normalised
                          units) to the physics term. The Q residual is the one graphkit already computes inside the model
                          (``ComputeNodeResiduals``); only its weight changes, the network is untouched.
``FEWSHOT_DVA_LOSS=mu``   (post hoc diagnostic) add ``mu * mean|wrapped branch angle-difference error|`` in degrees to the loss: the
                          metric itself as a training objective, to test whether the label MSE on absolute angles is the problem.
``FEWSHOT_MM_DIR=dir``    extra INPUT features: the fd2 power mismatch (P and Q, scaled, from ``dir/<tree>/<grid>.npy``
                          [scenarios, buses, 2]) appended as bus input columns 15 and 16. ``input_proj_bus``'s first Linear is
                          widened (``model.input_bus_dim`` + 2); pretrained weights are padded with zero columns so the start
                          is unchanged (with the zeroed head the start is still exactly fd2).
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch

import gridfm_graphkit.cli as gk_cli
from gridfm_graphkit.datasets import transforms as gk_tf
from gridfm_graphkit.datasets.globals import VA_H, VA_OUT
from gridfm_graphkit.datasets.powergrid_hetero_dataset import HeteroGridDatasetDisk
from gridfm_graphkit.tasks.reconstruction_tasks import ReconstructionTask

if os.environ.get("FEWSHOT_TRAIN_DVA") == "1":
    def _dva(va_a, va_b, ei):
        d = (va_a[ei[0]] - va_a[ei[1]]) - (va_b[ei[0]] - va_b[ei[1]])
        d = torch.remainder(d + math.pi, 2 * math.pi) - math.pi
        return d.abs().mean() * (180.0 / math.pi)

    def _training_step(self, batch):
        output, loss_dict = self.shared_step(batch)
        ei = batch.edge_index_dict[("bus", "connects", "bus")]
        extra = {"Training Loss": loss_dict["loss"].detach(), "Learning Rate": self.optimizer.param_groups[0]["lr"]}
        with torch.no_grad():
            extra["Training branch_dva"] = _dva(output["bus"][:, VA_OUT], batch.y_dict["bus"][:, VA_H], ei)
            if "prior" in batch["bus"]:
                extra["Training branch_dva_prior"] = _dva(batch["bus"].prior[:, 1], batch.y_dict["bus"][:, VA_H], ei)
        for k, v in extra.items():
            self.log(k, v, batch_size=batch.num_graphs, sync_dist=False, on_epoch=(k != "Learning Rate"), prog_bar=False,
                     logger=True, on_step=(k in ("Training Loss", "Learning Rate")))
        return loss_dict["loss"]

    ReconstructionTask.training_step = _training_step
    print("[corr3] training branch-dVa logging ON", flush=True)

if os.environ.get("FEWSHOT_Q_LOSS"):
    from gridfm_graphkit.models import utils as gk_mu
    from gridfm_graphkit.training import loss as gk_loss

    _LAM = float(os.environ["FEWSHOT_Q_LOSS"])
    _STASH: list = []
    _orig_res = gk_mu.ComputeNodeResiduals.forward

    def _res(self, P_in, Q_in, bus_data_pred, bus_data_orig):
        rp, rq = _orig_res(self, P_in, Q_in, bus_data_pred, bus_data_orig)
        _STASH.append(rq.abs().mean())
        return rp, rq

    gk_mu.ComputeNodeResiduals.forward = _res
    _P = gk_loss.LayeredWeightedPhysicsLoss
    _orig_loss = _P.forward

    def _loss(self, pred, target, edge_index=None, edge_attr=None, mask=None, model=None, x_dict=None):
        out = _orig_loss(self, pred, target, edge_index, edge_attr, mask, model, x_dict)
        qs = list(_STASH)
        _STASH.clear()
        # audit corrector-timing R2-F4: a count mismatch used to drop the Q term silently; with lambda > 0 it is an error
        if _LAM > 0 and len(qs) != len(model.layer_residuals):
            raise RuntimeError(f"reactive-residual stash has {len(qs)} entries, model has {len(model.layer_residuals)} "
                               "layer residuals: the Q term cannot be weighted per layer")
        if len(qs) == len(model.layer_residuals) and qs:
            L = len(qs)
            raw = [self.base_weight ** (L - i - 1) for i in range(L)]
            s = sum(raw)
            qterm = sum((w / s) * q for w, q in zip(raw, qs))
            out["loss"] = out["loss"] + _LAM * qterm
            out["Reactive residual (layer-weighted)"] = float(qterm.detach())
        return out

    _P.forward = _loss
    # the stash must not grow when the loss is not called (evaluation / prediction)
    _orig_get_task_q = gk_cli.get_task

    def _get_task_q(*a, **k):
        task = _orig_get_task_q(*a, **k)
        task.model.register_forward_pre_hook(lambda m, i: _STASH.clear())
        return task

    gk_cli.get_task = _get_task_q
    print(f"[corr3] reactive-residual loss term ON, lambda {_LAM}", flush=True)

if os.environ.get("FEWSHOT_MM_DIR"):
    _MM_DIR = os.environ["FEWSHOT_MM_DIR"]
    _mm_tabs: dict = {}
    _prev_get = HeteroGridDatasetDisk.get
    _prev_mask = gk_tf.ApplyMasking.forward

    def _get(self, idx):
        data = _prev_get(self, idx)
        root = os.path.abspath(str(self.root))
        key = (os.path.basename(os.path.dirname(root)), os.path.basename(root))
        if key not in _mm_tabs:
            _mm_tabs[key] = torch.from_numpy(np.load(os.path.join(_MM_DIR, key[0], key[1] + ".npy"))).float()
        sid = int(data["scenario_id"].reshape(-1)[0])
        tab = _mm_tabs[key][sid]
        assert tab.shape[0] == data["bus"].x.shape[0]
        data["bus"].mm = tab.clone()
        return data

    def _mask(self, data):
        data = _prev_mask(self, data)
        x = data["bus"].x
        if x.shape[1] == 15 and "mm" in data["bus"]:
            data["bus"].x = torch.cat([x, data["bus"].mm.to(x.dtype)], dim=1)
        return data

    HeteroGridDatasetDisk.get = _get
    gk_tf.ApplyMasking.forward = _mask
    _orig_get_task_mm = gk_cli.get_task

    def _get_task_mm(args, *a, **k):
        args.model.input_bus_dim = 17
        task = _orig_get_task_mm(args, *a, **k)
        orig = task.load_state_dict

        def _lsd(sd, *aa, **kk):
            sd = dict(sd)
            for key, w in list(sd.items()):
                if key.endswith("input_proj_bus.0.weight") and w.shape[1] == 15:
                    sd[key] = torch.cat([w, torch.zeros(w.shape[0], 2, dtype=w.dtype)], dim=1)
            return orig(sd, *aa, **kk)

        task.load_state_dict = _lsd
        return task

    gk_cli.get_task = _get_task_mm
    print(f"[corr3] fd2-mismatch input features ON (cols 15, 16), dir {_MM_DIR}", flush=True)

if os.environ.get("FEWSHOT_DVA_LOSS"):
    _MU = float(os.environ["FEWSHOT_DVA_LOSS"])
    _orig_shared = ReconstructionTask.shared_step

    def _shared(self, batch):
        output, loss_dict = _orig_shared(self, batch)
        ei = batch.edge_index_dict[("bus", "connects", "bus")]
        va, yv = output["bus"][:, VA_OUT], batch.y_dict["bus"][:, VA_H]
        d = (va[ei[0]] - va[ei[1]]) - (yv[ei[0]] - yv[ei[1]])
        l1 = (torch.remainder(d + math.pi, 2 * math.pi) - math.pi).abs().mean() * (180.0 / math.pi)
        loss_dict["loss"] = loss_dict["loss"] + _MU * l1
        loss_dict["dVa L1 term (deg)"] = l1.detach()
        return output, loss_dict

    ReconstructionTask.shared_step = _shared
    print(f"[corr3] branch-dVa L1 loss term ON, mu {_MU}", flush=True)
