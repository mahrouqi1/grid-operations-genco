"""gk_freeze extra plugin: save the task state dict at chosen optimizer steps (unlearning curves, 2026-10-19).

Enable with ``FEWSHOT_EXTRA_PLUGIN=genco_challenge.save_at_steps``. Settings:
  SAVE_AT_STEPS  comma-separated global steps, e.g. "0,25,50,100,250,500,1000"; step 0 is saved at train start
                 (the initial weights, before any update); the final weights are saved at train end as ``final``.
  SAVE_DIR       directory for ``state_step<N>.pt`` and ``state_final.pt``.
The files hold ``pl_module.state_dict()``, the same object graphkit's best-model callback saves and ``--model_path``
loads, so every scorer that takes a checkpoint path can read them. Rank 0 only.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from gridfm_graphkit import cli as gk_cli
from lightning.pytorch.callbacks import Callback

_STEPS = sorted({int(s) for s in os.environ.get("SAVE_AT_STEPS", "").split(",") if s.strip()})
_DIR = Path(os.environ.get("SAVE_DIR", "saved_steps"))


class _SaveAtSteps(Callback):
    def __init__(self):
        self.done: set[int] = set()

    def _save(self, trainer, pl_module, tag: str):
        if trainer.global_rank != 0:
            return
        _DIR.mkdir(parents=True, exist_ok=True)
        torch.save(pl_module.state_dict(), _DIR / f"state_{tag}.pt")

    def on_train_start(self, trainer, pl_module):
        if 0 in _STEPS:
            self._save(trainer, pl_module, "step0")
            self.done.add(0)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        s = trainer.global_step          # optimizer steps completed so far
        if s in _STEPS and s not in self.done:
            self._save(trainer, pl_module, f"step{s}")
            self.done.add(s)

    def on_train_end(self, trainer, pl_module):
        self._save(trainer, pl_module, "final")


_Base = gk_cli.L.Trainer


class _SaveTrainer(_Base):
    def __init__(self, *a, **k):
        k["callbacks"] = list(k.get("callbacks") or []) + [_SaveAtSteps()]
        super().__init__(*a, **k)


gk_cli.L.Trainer = _SaveTrainer
print(f"[save_at_steps] saving at steps {_STEPS} and at train end into {_DIR}", flush=True)
