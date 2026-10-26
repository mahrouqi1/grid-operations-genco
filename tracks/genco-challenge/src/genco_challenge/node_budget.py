"""gk_freeze extra plugin: node-budgeted training batches (transfer experiment, an external reviewer point 4, 2026-10-19).

graphkit batches a fixed number of graphs (``training.batch_size``). With ladder grids from 64 to 9,591 buses, a
batch of 32 large graphs does not fit in GPU memory (local smoke 2026-10-19: OOM at batch 8 on a 24 GB RTX 3090).
This plugin replaces only the TRAINING loader's batching: each epoch the training indices are shuffled (seeded by
``NODE_BUDGET_SEED`` + epoch) and packed greedily into batches of at most ``NODE_BUDGET`` buses and at most
``training.batch_size`` graphs. Validation, test and predict loaders are untouched.

Enable with ``FEWSHOT_EXTRA_PLUGIN=...,genco_challenge.node_budget`` and ``NODE_BUDGET=<buses>``. Unset or 0 = off.
"""
from __future__ import annotations

import os
import random

from torch.utils.data import Sampler
from torch_geometric.loader import DataLoader

from gridfm_graphkit.datasets import hetero_powergrid_datamodule as gk_dm

BUDGET = int(os.environ.get("NODE_BUDGET", "0") or 0)
SEED = int(os.environ.get("NODE_BUDGET_SEED", "0") or 0)


def pack(sizes: list[int], budget: int, max_graphs: int, rng: random.Random) -> list[list[int]]:
    """Shuffle indices, then pack greedily in that order: a batch closes when the next graph would exceed the bus
    budget or the graph cap. A single graph larger than the budget gets a batch of its own."""
    order = list(range(len(sizes)))
    rng.shuffle(order)
    batches, cur, tot = [], [], 0
    for i in order:
        n = sizes[i]
        if cur and (tot + n > budget or len(cur) >= max_graphs):
            batches.append(cur); cur, tot = [], 0
        cur.append(i); tot += n
    if cur:
        batches.append(cur)
    return batches


class NodeBudgetBatchSampler(Sampler):
    def __init__(self, sizes, budget, max_graphs, seed):
        self.sizes, self.budget, self.max_graphs, self.seed = sizes, budget, max_graphs, seed
        self.epoch = 0
        self._n = len(pack(sizes, budget, max_graphs, random.Random(seed)))     # length estimate for Lightning

    def __iter__(self):
        b = pack(self.sizes, self.budget, self.max_graphs, random.Random(self.seed + self.epoch))
        self.epoch += 1          # graphkit's loaders are rebuilt per fit only; advance the shuffle every pass
        return iter(b)

    def __len__(self):
        return self._n


def _sizes(concat) -> list[int]:
    """Buses per sample for a ConcatDataset of per-network datasets (constant within a network: one read each)."""
    out = []
    for ds in concat.datasets:
        n = int(ds[0]["bus"].x.size(0)) if len(ds) else 0     # bus feature rows (NodeStorage has no num_nodes here)
        out += [n] * len(ds)
    return out


_orig_train_dataloader = gk_dm.LitGridHeteroDataModule.train_dataloader


def _train_dataloader(self):
    if BUDGET <= 0:
        return _orig_train_dataloader(self)
    sizes = _sizes(self.train_dataset_multi)
    bs = NodeBudgetBatchSampler(sizes, BUDGET, self.batch_size, SEED)
    print(f"[node_budget] {len(sizes)} training graphs, {len(bs)} batches per epoch at <= {BUDGET} buses and <= "
          f"{self.batch_size} graphs (largest graph {max(sizes)} buses)", flush=True)
    return DataLoader(self.train_dataset_multi, batch_sampler=bs, num_workers=self.args.data.workers, pin_memory=True)


gk_dm.LitGridHeteroDataModule.train_dataloader = _train_dataloader
print(f"[node_budget] budget {BUDGET} buses per training batch (0 = off, graphkit's fixed batch size)", flush=True)
