"""Cost-matched classical states on the eval2 scenarios graphkit scored: fd1..fd4, nr1..nr3 (and dc, fd2) with the
same metrics. CPU only. Rows split='eval2_classical'. Needs the zero-shot prediction file only for the scenario list."""
import numpy as np, pandas as pd
from common import *
from s1_score import rows_for, write
from genco_challenge.fresh_eval_data import EVAL2

for g in fs.GRIDS:
    pred = pd.read_parquet(pred_file(W / "fs_eval2", f"zeroshot_eval2_{g}", g))
    acc = {}
    for o in bundles(EVAL2, g, pred):
        sc = o.sc
        st = {"dc": (sc.vm_set, o.va_dc)}
        for k in ("fd1", "fd2", "nr1", "nr2", "nr3"):
            st[k] = fd_state(o, k)
        v = o.pr["fd2"]["v"].copy()
        for k in (3, 4):                               # extra XB iterations from fd2 (same guard as priors_for: keep if finite)
            vn = sc.fdxb_step(v)
            if hy._finite(vn):
                v = vn
            st[f"fd{k}"] = (np.abs(v), np.radians(lp2._unwrapped(sc, v)))
        for k, (vm, va) in st.items():
            acc.setdefault(k, []).append(score(o, vm, va))
    acc = {k: np.array(v) for k, v in acc.items()}
    write(rows_for("eval2_classical", g, acc))
    print(g, flush=True)
