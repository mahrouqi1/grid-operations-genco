"""Amendment 1 (written before any AC-labelled run and before any as-run run on case5/24/240). UNAUDITED."""
import json, hashlib, time
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
P = json.loads((HERE / "protocol.json").read_text())
G12 = [g for g in P["grids"] if g not in P["grid_data_status"]["skipped_no_data"]]
q = [{"id": f"wu100L_{g}_n100_s{s}", "prio": 0, "grid": g, "n": 100, "seed": s, "split_seed": s, "tag": "wu100L", "labels": "ac",
      "env": {"FEWSHOT_WARMUP_STEPS": "100"}, "epochs": 200, "note": "labelled recipe (AC targets, published normalizer)"} for s in (0, 1, 2) for g in G12]
am = {
 "written": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
 "finding": "DISCOVERED AFTER 15 AS-RUN ROWS (5 grids: case39/73/89/162/300 x 3 draws, results.jsonl, tag wu100) AND BEFORE ANY AC-LABELLED RUN: the B4 study's run.py imports genco_on_fd2/scripts/common.py, which imports genco_challenge.labelfree, and labelfree.py patches fewshot at import time: fs.TRAIN_DATA = train_dcprior and fs.ENV += FEWSHOT_INPUT_NORM=1. So every B4 fine-tune (and every row of this study so far, which copied B4's run.py) trained with bus-voltage targets Va := Va_dc (DC power-flow angle) and Vm := 1.0 at PQ buses, and an INPUT-ONLY normalizer; the AC voltage solution was never a training target. The B4 report/protocol call this 'n labelled scenarios' and 'the published recipe unchanged'; the former is not true as run (the training signal is the physics residual loss plus a DC-prior MSE; the AC label is never read), the latter differs in targets and normalizer. Evidence: B4 train log line 'input-only normalizer ON'; fewshot._train reads the module global TRAIN_DATA; dcprior.build.",
 "consequence": "Two variants. 'dcprior' (tag wu100) = B4 as it was actually run: label-free few-shot (n unlabelled load scenarios of the new grid). 'ac' (tag wu100L) = the intended labelled recipe: fs.TRAIN_DATA = data/external/genco-ladder/train (AC Va/Vm labels), no FEWSHOT_INPUT_NORM, separate split dir fewshot/splits_L (the dcprior pool does not filter collapsed AC labels, the AC pool does), everything else identical (released checkpoint, warm-up 100, AdamW 5e-4, 200 epochs, same draws 0-2, same eval2 scoring).",
 "primary_variant": "ac (wu100L): the goal is a claim about N LABELLED scenarios. The same success test (draw-mean branch dVa < DC on >= 10 of the 15 grids; the 3 skipped grids count as failures) is applied to BOTH variants and both are reported; no variant is chosen after seeing AC-labelled results. The as-run variant is reported as what it is: a label-free result.",
 "dcprior_variant_scope": "grids 14/30/57/118 reused from B4 (as-run, hence also dcprior), 39/73/89/162/300 already run, 5/24/240 queued after generation (train_dcprior copies built with genco_challenge.dcprior). case300 draws are rerun here (B4's OOM'd).",
 "ac_variant_scope": "all 12 grids with data x 3 draws = 36 runs, none reused (B4 has no AC-labelled rows).",
 "n_new_specs": len(q)}
(HERE / "protocol_amendment1.json").write_text(json.dumps(am, indent=1))
(HERE / "protocol_amendment1.sha256").write_text(hashlib.sha256((HERE / "protocol_amendment1.json").read_bytes()).hexdigest() + "  protocol_amendment1.json\n")
(HERE / "queue_amendment1.json").write_text(json.dumps(q, indent=1))
print(len(q))
