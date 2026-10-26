"""Zero-shot released-GENCO predictions for the 11 grids (graphkit evaluate via ladder_eval.evaluate; model 'zeroshot').
Writes results.jsonl (ladder_eval rows incl. error summaries) and mlruns/ under this dir. GPU: CUDA_VISIBLE_DEVICES=0."""
import json, sys
from pathlib import Path
from genco_challenge import ladder_eval as LE
HERE = Path(__file__).resolve().parent
import os, yaml
_BS = int(os.environ.get("EVAL_BATCH", "0"))
if _BS:  # batch 64 does not fit in 24 GB for the 20M-parameter model at 8-9k buses (OOM): predictions are batch-size independent in eval mode
    _sl = yaml.safe_load
    yaml.safe_load = lambda *a, **k: (lambda c: (c.setdefault("training", {}).__setitem__("batch_size", _BS), c)[1])(_sl(*a, **k))
ROOT = LE.REPO / "data/external/genco-robust-start"
GRIDS = json.loads((HERE / "protocol.json").read_text())["grids"]
grids = [g for fam in GRIDS.values() for g in fam]
res = HERE / "results.jsonl"
done = {json.loads(l)["grid"] for l in res.read_text().splitlines()} if res.exists() else set()
for g in grids:
    if g in done:
        continue
    try:
        row = LE.evaluate(g, "zeroshot", HERE / "mlruns", data_root=ROOT)
    except Exception as e:
        row = {"grid": g, "model": "zeroshot", "error": repr(e)}
    with res.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(g, "error" in row and row["error"] or "ok", flush=True)
(HERE / "SCORE_DONE").write_text("done\n")
