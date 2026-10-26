"""Success test of protocol.json, computed from the saved files only (UNAUDITED).
inputs:  per_scenario/<case>_n<N>.npz  (ids + one (S,7) score array per state: dc, fd2..fd30, corrector; columns common.COLS)
         timing.jsonl                  (kind=summary rows from scripts/timing.py; the last non-smoke row per case is used)
outputs: verdict.json, summary_table.md
usage:   python verdict.py [--n 100]   (env grid-operations-genco, PYTHONPATH=tracks/genco-challenge/src)"""
import argparse, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
OUT = HERE.parent
sys.path.insert(0, str(HERE.parents[1] / "2026-10-09-genco-on-fd2/scripts"))
from common import agg, boot_ratio          # noqa: E402

CASES = ["case2869pegase", "case9241pegase", "case6470rte"]
KMAX = 30


def timing(case):
    rows = [json.loads(l) for l in (OUT / "timing.jsonl").read_text().splitlines()] if (OUT / "timing.jsonl").exists() else []
    rows = [r for r in rows if r.get("kind") == "summary" and r.get("case") == case and not r.get("smoke")]
    if not rows:
        return None
    reps = [json.loads(l) for l in (OUT / "timing.jsonl").read_text().splitlines()]
    reps = [r for r in reps if r.get("kind") == "repeat" and r.get("case") == case and not r.get("smoke")]
    last = rows[-1]
    n = last["reps"]
    reps = reps[-n:]
    last["dc_ms"] = float(np.median([r["asm_ms"] + r["dc_ms"] for r in reps])) if reps else None
    last["gpu_states"] = sorted({r["gpu_before"] for r in reps} | {r["gpu_after"] for r in reps})
    return last


def judge(acc, t_fd, t_cor, label):
    """t_fd[k-1] = fd_k total ms (k = 1..KMAX). Comparator = cheapest fd_k (k >= 2) with time >= t_cor; fd30 if none."""
    ks = [k for k in range(2, KMAX + 1) if t_fd[k - 1] >= t_cor]
    k_star = ks[0] if ks else KMAX
    pt, lo, hi = boot_ratio(acc["corrector"], acc[f"fd{k_star}"], "dva")
    strict = {}
    for k in (ks or [KMAX]):
        p, l, h = boot_ratio(acc["corrector"], acc[f"fd{k}"], "dva")
        strict[k] = [p, l, h]
    return {"time_basis": label, "t_corrector_ms": t_cor, "k_matched": k_star, "any_fd_k_as_expensive": bool(ks), "t_fd_k_matched_ms": t_fd[k_star - 1],
            "ratio_vs_matched": [pt, lo, hi], "beats_matched": bool(hi < 1.0),
            "strict_all_costlier": {str(k): v for k, v in strict.items()}, "beats_all_costlier": bool(all(v[2] < 1.0 for v in strict.values()))}


def run(n):
    res = {"n_train": n, "grids": {}}
    md = [f"Branch dVa (deg), |V| MAE (pu), active residual (MW/bus) and time (ms/scenario) per grid and method, corrector n = {n}.\n"]
    for c in CASES:
        f = OUT / "per_scenario" / f"{c}_n{n}.npz"
        if not f.exists():
            res["grids"][c] = {"status": "missing per-scenario file"}
            continue
        z = np.load(f); acc = {k: z[k] for k in z.files if k != "ids"}
        t = timing(c)
        g = {"n_scored": int(len(z["ids"]))}
        vs = {}
        for k in ("dc", "fd2", "fd3", "fd4", "fd5"):
            vs[k] = boot_ratio(acc["corrector"], acc[k], "dva")
        g["corrector_vs"] = vs
        g["fd2_input_is_dc_frac"] = float(np.mean(z["fd2_input_is_dc"][:, 0])) if "fd2_input_is_dc" in z.files else None
        g["corrector_beats_fd_k_for_k"] = [k for k in range(2, KMAX + 1) if boot_ratio(acc["corrector"], acc[f"fd{k}"], "dva")[2] < 1.0]
        g["corrector_beats_dc"] = bool(boot_ratio(acc["corrector"], acc["dc"], "dva")[2] < 1.0)
        if t is None:
            g["status"] = "no timing row"
        else:
            g["timing"] = {"gpu_states_seen": t["gpu_states"], "batch_used": t["batch_used"], "corrector_ms": t["corrector_ms"], "corrector_range": t["corrector_range"],
                           "corrector_fwd_only_ms": t["corrector_fwd_only_ms"], "fd2_ms": t["fd_total_ms_median"][1], "reps": t["reps"]}
            g["primary"] = judge(acc, t["fd_total_ms_median"], t["corrector_ms"], "full corrector pipeline")
            g["generous"] = judge(acc, t["fd_total_ms_median"], t["corrector_fwd_only_ms"], "corrector forward-only lower bound")
            ks = g["primary"]["k_matched"]
            rows = [("dc", None), ("fd2_input (corrector's start; guarded fd2)", "in"), ("fd2", 2), ("fd3", 3), ("fd4", 4), ("fd5", 5), (f"fd{ks} (time-matched)", ks), ("corrector", None)]
            md.append(f"\n**{c}** ({g['n_scored']} scored test scenarios, GPU batch {t['batch_used']})\n\n| method | branch dVa deg | |V| MAE pu | active res MW/bus | ms/scenario |\n|---|---:|---:|---:|---:|")
            for name, k in rows:
                key = "corrector" if name == "corrector" else "dc" if name == "dc" else "fd2_input" if k == "in" else f"fd{k}"
                a = agg(acc[key])
                tm = t["corrector_ms"] if name == "corrector" else (t["fd_total_ms_median"][0] * 0 + 0) if False else None
                if name == "corrector":
                    tm = t["corrector_ms"]
                elif name == "dc":
                    tm = t["dc_ms"]
                elif k == "in":
                    tm = t["fd_total_ms_median"][1]
                else:
                    tm = t["fd_total_ms_median"][k - 1]
                md.append(f"| {name} | {a['branch_dva_deg']:.5f} | {a['vm_pu']:.2e} | {a['p_res_mw']:.3g} | {'' if tm is None else f'{tm:.1f}'} |")
        res["grids"][c] = g
    done = [c for c in CASES if "primary" in res["grids"].get(c, {})]
    res["n_grids_with_result"] = len(done)
    res["primary_wins"] = [c for c in done if res["grids"][c]["primary"]["beats_matched"]]
    res["generous_wins"] = [c for c in done if res["grids"][c]["generous"]["beats_matched"]]
    res["strict_wins"] = [c for c in done if res["grids"][c]["primary"]["beats_all_costlier"]]
    res["SUCCESS_primary"] = bool(len(done) == 3 and len(res["primary_wins"]) >= 2)
    res["SUCCESS_generous_time"] = bool(len(done) == 3 and len(res["generous_wins"]) >= 2)
    (OUT / f"verdict_n{n}.json").write_text(json.dumps(res, indent=1))
    (OUT / f"summary_table_n{n}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md)); print(json.dumps({k: v for k, v in res.items() if k != "grids"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=100)
    run(ap.parse_args().n)
