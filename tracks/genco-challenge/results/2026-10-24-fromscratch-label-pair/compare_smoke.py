"""Smoke comparison of the two arms (original vs corrected): data order, init, loss, throughput, checkpoints.
Run: python -I compare_smoke.py <smoke_root> > smoke_compare.txt   (smoke_root holds original/ and corrected/)"""
import json, sys
from pathlib import Path
import torch

root = Path(sys.argv[1])
A, B = root / "original", root / "corrected"
rd = lambda p: [json.loads(l) for l in open(p) if l.strip()]
oa, ob = rd(A / "orderlog.jsonl"), rd(B / "orderlog.jsonl")
n = min(len(oa), len(ob))
same = [oa[i]["sha"] == ob[i]["sha"] for i in range(n)]
print(f"order log lines: original {len(oa)}, corrected {len(ob)}; compared {n}; identical sha on {sum(same)} of {n}")
if not all(same):
    print("first mismatch at update", oa[same.index(False)]["step"])
for i in range(3):
    print(f"update {oa[i]['step']}: ids original {oa[i].get('scenario_id', '')[:8]}... corrected {ob[i].get('scenario_id', '')[:8]}...  sha {oa[i]['sha']} / {ob[i]['sha']}")
sa = torch.load(A / "steps/state_step0.pt", map_location="cpu"); sb = torch.load(B / "steps/state_step0.pt", map_location="cpu")
ident = all(torch.equal(sa[k], sb[k]) for k in sa)
print(f"step-0 (random init) weights bitwise identical across arms: {ident} ({len(sa)} tensors)")
for tag in ("original", "corrected"):
    p = rd(root / tag / "progress.jsonl")
    print(tag, "steps", p[0]["step"], "->", p[-1]["step"], "loss", round(p[0]["loss"], 3), "->", round(p[-1]["loss"], 3),
          "lr", p[-1]["lr"], "| updates/s over the last 150:", round((p[-1]["step"] - p[-7]["step"]) / (p[-1]["wall_s"] - p[-7]["wall_s"]), 2))
    print("  files:", sorted(x.name for x in (root / tag / "steps").glob("*.pt")), "| resume:", sorted(x.name for x in (root / tag / "resume").glob("*")))
for tag in ("original", "corrected"):
    c = torch.load(root / tag / "resume/full_latest.ckpt", map_location="cpu", weights_only=False)
    print(tag, "full ckpt: global_step", c["global_step"], "epoch", c["epoch"], "optimizer_states", len(c["optimizer_states"]),
          "Adam state tensors", len(c["optimizer_states"][0]["state"]), "lr_scheduler", bool(c["lr_schedulers"]))
fa = torch.load(A / "steps/state_final.pt", map_location="cpu"); fb = torch.load(B / "steps/state_final.pt", map_location="cpu")
d = max(float((fa[k].float() - fb[k].float()).abs().max()) for k in fa)
print("max |weight difference| between arms after 300 updates:", d)
