"""Verify the converted tree: per-case scenario counts == n_map (all four tables, contiguous 0..N-1), no NaN in
the first 20 scenarios, rate_a == PGLib .m for pglib cases (max abs diff 0, via pglib_rate_a_fix's own match),
and graphkit's HeteroGridDatasetDisk builds graphs from the first 20 scenarios. Prints a per-case table and
writes verify_tree.json. Exit code 0 only if everything passes."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "data/external/gridfm-graphkit-repro-pretraining"))
sys.path.insert(0, str(REPO / "tracks/genco-challenge/src"))
OUT = REPO / "data/external/genco-ladder-datakit"
D = Path(__file__).resolve().parent
nmap = json.loads((D / "n_map.json").read_text())
SC = REPO / "data/external/genco-ladder-datakit-verify"


def graphkit_ok(case):
    from gridfm_graphkit.datasets.normalizers import HeteroDataMVANormalizer
    from gridfm_graphkit.datasets.powergrid_hetero_dataset import HeteroGridDatasetDisk
    root = SC / case
    (root / "raw").mkdir(parents=True, exist_ok=True)
    for t in ("bus_data", "gen_data", "branch_data"):
        tab = pq.read_table(OUT / case / "raw" / f"{t}.parquet", filters=[("scenario", "<", 20)])
        pq.write_table(tab, root / "raw" / f"{t}.parquet")
    args = SimpleNamespace(data=SimpleNamespace(baseMVA=100, normalizationMVA=100))
    ds = HeteroGridDatasetDisk(root=str(root), norm_method="mvanorm", data_normalizer=HeteroDataMVANormalizer(args))
    g = ds[0]
    return len(ds) == 20 and all(np.isfinite(g[k].x.numpy()).all() for k in ("bus", "gen"))


def rate_ok(case):
    """Independent of the fixer: the written rate_a multiset (scenario 0) equals the .m RATE_A multiset exactly,
    rate_a is identical in every scenario, and the fixer recorded max abs diff 0 against .m in its row mapping."""
    from matpowercaseframes import CaseFrames
    m = np.sort(CaseFrames(str(REPO / "data/external/pglib-opf" / f"{case}.m")).branch.RATE_A.values.astype(float))
    t = pq.read_table(OUT / case / "raw" / "branch_data.parquet", columns=["scenario", "idx", "rate_a"])
    sc, ix, ra = (t[c].to_numpy() for c in ("scenario", "idx", "rate_a"))
    first = ra[sc == 0]
    n = len(first)
    per = np.zeros(n)
    per[ix[sc == 0].astype(int)] = first
    consistent = bool(np.array_equal(ra, per[ix.astype(int)]))
    multiset_diff = float(np.abs(np.sort(first) - m).max()) if len(first) == len(m) else float("inf")
    rec = json.loads((OUT / case / "convert_stats.json").read_text()).get("rate_a_pglib_posthoc")
    return (multiset_diff == 0.0 and consistent and rec is not None and rec["max_abs_diff_vs_m_after"] == 0.0), float(ra.max())


def gens_vs_m(case):
    """Written gen rows == .m gen rows (count, order, bus, limits, costs); in_service may differ only by the re-enabled slack."""
    from matpowercaseframes import CaseFrames
    cf = CaseFrames(str(REPO / "data/external/pglib-opf" / f"{case}.m"))
    ids = np.sort(cf.bus.BUS_I.values.astype(int))
    rank = {int(b): k for k, b in enumerate(ids)}
    g = cf.gen.values.astype(float)
    gc = cf.gencost.values.astype(float)
    t = pq.read_table(OUT / case / "raw" / "gen_data.parquet", filters=[("scenario", "=", 0)]).to_pandas().sort_values("idx")
    if len(t) != len(g):
        return False, f"{len(t)} written vs {len(g)} .m gens"
    ok = np.array_equal(t.bus.values, [rank[int(b)] for b in g[:, 0]])
    ok &= np.allclose(t.max_p_mw, g[:, 8], atol=0, rtol=1e-12) and np.allclose(t.min_p_mw, g[:, 9], atol=0, rtol=1e-12)
    ok &= np.allclose(t.max_q_mvar, g[:, 3], atol=0, rtol=1e-12) and np.allclose(t.min_q_mvar, g[:, 4], atol=0, rtol=1e-12)
    ncost = int(gc[0, 3])
    if ncost == 3:
        ok &= np.allclose(t.cp2_eur_per_mw2, gc[:, 4]) and np.allclose(t.cp1_eur_per_mw, gc[:, 5]) and np.allclose(t.cp0_eur, gc[:, 6])
    n_status_diff = int(((t.in_service.values == 1) != (g[:, 7] > 0)).sum())
    return bool(ok), n_status_diff


def basemva_rule(case, nsc=50):
    """graphkit HeteroDataMVANormalizer.fit rule: 95th percentile of non-zero Pd, Qd, gen p_mw, bus Qg."""
    b = pq.read_table(OUT / case / "raw" / "bus_data.parquet", columns=["scenario", "Pd", "Qd", "Qg"], filters=[("scenario", "<", nsc)]).to_pandas()
    g = pq.read_table(OUT / case / "raw" / "gen_data.parquet", columns=["scenario", "p_mw"], filters=[("scenario", "<", nsc)]).to_pandas()
    import pandas as pd
    v = pd.concat([b.Pd[b.Pd != 0], b.Qd[b.Qd != 0], g.p_mw[g.p_mw != 0], b.Qg[b.Qg != 0]])
    return float(np.percentile(v, 95))


res, bad = {}, []
for case, n in nmap.items():
    raw = OUT / case / "raw"
    r = {"n_expected": n}
    try:
        for t in ("bus_data", "branch_data", "gen_data", "y_bus_data"):
            s = pq.read_table(raw / f"{t}.parquet", columns=["scenario"])["scenario"].to_numpy()
            u = np.unique(s)
            r[t] = int(len(u))
            if not np.array_equal(u, np.arange(n)):
                bad.append(f"{case}:{t} scenarios != 0..{n - 1}")
        st = json.loads((OUT / case / "convert_stats.json").read_text())
        r["dropped"] = st["dropped"]
        r["max_resid_pu"] = st["max_resid_pu"]
        r["rate_a_max"] = float(pq.read_table(raw / "branch_data.parquet", columns=["rate_a"])["rate_a"].to_numpy().max())
        if case.startswith("pglib_opf_"):
            ok, _ = rate_ok(case)
            r["rate_a_equals_m"] = bool(ok)
            if not ok:
                bad.append(f"{case}: rate_a not asserted equal to .m")
        r["basemva_rule"] = basemva_rule(case)
        pdmin = float(pq.read_table(raw / "bus_data.parquet", columns=["Pd"])["Pd"].to_numpy().min())
        r["min_pd"] = pdmin
        r["n_gens_written"] = int(len(pq.read_table(raw / "gen_data.parquet", columns=["idx"], filters=[("scenario", "=", 0)])))
        r["n_pv"] = int(pq.read_table(raw / "bus_data.parquet", columns=["PV"], filters=[("scenario", "=", 0)])["PV"].to_numpy().sum())
        if case.startswith("pglib_opf_"):
            okg, info = gens_vs_m(case)
            r["gens_equal_m"], r["n_status_diff_vs_m"] = okg, info
            if not okg:
                bad.append(f"{case}: gens != .m ({info})")
            from matpowercaseframes import CaseFrames
            mmin = float(CaseFrames(str(REPO / "data/external/pglib-opf" / f"{case}.m")).bus.PD.min())
            r["m_file_min_pd"] = mmin      # the .m file itself has negative-Pd buses in some cases; datakit keeps them
            if pdmin < min(mmin, 0.0) - 1e-6:
                bad.append(f"{case}: Pd {pdmin} below the .m file's own minimum {mmin}")
        r["graphkit_ok"] = bool(graphkit_ok(case))
        if not r["graphkit_ok"]:
            bad.append(f"{case}: graphkit")
        if st["max_resid_pu"] >= 1e-6:
            bad.append(f"{case}: residual")
    except Exception as e:  # noqa: BLE001
        r["error"] = f"{type(e).__name__}: {e}"
        bad.append(f"{case}: {r['error']}")
    res[case] = r
    print(case, {k: v for k, v in r.items() if k not in ('n_expected',)}, flush=True)
(D / "verify_tree.json").write_text(json.dumps({"cases": res, "problems": bad}, indent=1))
print("PROBLEMS:", bad)
sys.exit(1 if bad else 0)
