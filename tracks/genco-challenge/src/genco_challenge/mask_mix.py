"""gk_freeze extra plugin: choose, per sample, between graphkit's pretraining mask and its power-flow mask (transfer
experiment, the study plan section 3, 2026-10-19).

Enable with ``FEWSHOT_EXTRA_PLUGIN=genco_challenge.mask_mix`` (comma-separate to combine with ``save_at_steps``).
The task stays graphkit's ``PreTraining`` (same losses); only ``AddPretrainMask.forward`` is wrapped. Settings:
  MASK_POLICY  pretrain  -> pretraining mask always (arm D; the plugin is then a no-op)
               pf        -> power-flow mask (``AddPFHeteroMask``) always (arm a)
               halfmix   -> pretraining mask for epochs < MASK_EPOCHS/2, then a 50/50 per-sample mix (arm b)
               curriculum-> power-flow fraction e / (MASK_EPOCHS - 1) at epoch e, 0 -> 1 (arm c)
               mix50     -> a constant 50/50 per-sample mix from the first step (an external reviewer's C01 / C11, 2026-10-19 22:00)
  MASK_EPOCHS  the run's epoch count (required for halfmix / curriculum)

The epoch reaches the dataloader workers through a module global set by a Trainer callback at the end of each epoch
(for the next one) and at train start. graphkit's DataLoaders are not persistent, so workers fork afresh each epoch
and inherit it. The per-sample draw uses torch's RNG inside the worker (seeded by Lightning / torch per worker).
"""
from __future__ import annotations

import os

import torch
from gridfm_graphkit import cli as gk_cli
from gridfm_graphkit.datasets import masking as gk_masking
from lightning.pytorch.callbacks import Callback

POLICY = os.environ.get("MASK_POLICY", "pretrain")
EPOCHS = int(os.environ.get("MASK_EPOCHS", "0") or 0)
if POLICY not in ("pretrain", "pf", "halfmix", "curriculum", "mix50"):
    raise ValueError(f"MASK_POLICY={POLICY!r}")
if POLICY in ("halfmix", "curriculum") and EPOCHS < 2:
    raise ValueError("MASK_EPOCHS >= 2 required for halfmix / curriculum")

_EPOCH = 0                      # set in the main process before each epoch's workers fork


def pf_fraction(epoch: int, policy: str = POLICY, epochs: int = EPOCHS) -> float:
    if policy == "pretrain":
        return 0.0
    if policy == "pf":
        return 1.0
    if policy == "mix50":
        return 0.5
    if policy == "halfmix":
        return 0.0 if epoch < epochs / 2 else 0.5
    return min(1.0, max(0.0, epoch / (epochs - 1)))       # curriculum


_orig_pretrain_forward = gk_masking.AddPretrainMask.forward
_pf_mask = gk_masking.AddPFHeteroMask()


_SEEN: set = set()


def _mixed_forward(self, data):
    q = pf_fraction(_EPOCH)
    if os.environ.get("MASK_MIX_DEBUG") and (os.getpid(), _EPOCH) not in _SEEN:   # once per worker per epoch
        _SEEN.add((os.getpid(), _EPOCH))
        print(f"[mask_mix] worker {os.getpid()} sees epoch {_EPOCH}, fraction {q:.3f}", flush=True)
    if q > 0 and (q >= 1 or torch.rand(()).item() < q):
        return _pf_mask(data)
    return _orig_pretrain_forward(self, data)


gk_masking.AddPretrainMask.forward = _mixed_forward


class _EpochSetter(Callback):
    def on_train_start(self, trainer, pl_module):
        global _EPOCH
        _EPOCH = trainer.current_epoch

    def on_train_epoch_start(self, trainer, pl_module):
        global _EPOCH
        _EPOCH = trainer.current_epoch
        if trainer.global_rank == 0:
            print(f"[mask_mix] epoch {_EPOCH}: policy {POLICY}, power-flow mask fraction {pf_fraction(_EPOCH):.3f}",
                  flush=True)

    def on_train_epoch_end(self, trainer, pl_module):
        global _EPOCH
        _EPOCH = trainer.current_epoch + 1


_Base = gk_cli.L.Trainer


class _MixTrainer(_Base):
    def __init__(self, *a, **k):
        k["callbacks"] = list(k.get("callbacks") or []) + [_EpochSetter()]
        super().__init__(*a, **k)


gk_cli.L.Trainer = _MixTrainer
print(f"[mask_mix] policy {POLICY}, epochs {EPOCHS}", flush=True)
