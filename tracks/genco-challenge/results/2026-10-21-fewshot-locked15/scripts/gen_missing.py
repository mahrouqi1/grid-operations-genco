"""Generate the few-shot training pool + fresh eval2 set for HELDOUT15 grids that lack them (case5, case24, case240),
by the SAME procedures as fewshot.gen / fresh_eval_data.gen (same configs, seeds 7007 / 41113, profile window 1600..1799).
Only difference: training pool requested = 700 scenarios (not 1200) to keep generation to ~1 h. UNAUDITED."""
import json, sys
from genco_challenge import fewshot as fs, fresh_eval_data as fe
fs.N_TRAIN_GEN = 700
for g in sys.argv[1:]:
    print(json.dumps({"eval_copy": fs.eval_copy(g)}), flush=True)
    print(json.dumps({"train": fs.gen(g, 6)}), flush=True)
    print(json.dumps({"eval2": fe.gen(g, 6)}), flush=True)
    print(json.dumps({"pool": len(fs.pool(g))}), flush=True)
