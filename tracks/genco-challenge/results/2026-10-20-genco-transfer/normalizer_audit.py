"""Normaliser audit before the transfer runs (plan section 9 point 5, an external reviewer). UNAUDITED.

Per tree (21 training ladder grids, 5 RTE validation grids, 8 corrected replay subgrids):
  * graphkit's own baseMVA rule (HeteroDataMVANormalizer.fit: 95th percentile of the NON-ZERO Pd, Qd, gen p_mw and
    bus Qg values of that network; fit per network on its own tables) and vn_kv_max;
  * unit sanity: |Va| range (graphkit multiplies by pi/180, so the raw tables must be in DEGREES), Vm range (pu),
    Pd range (MW), rate_a / baseMVA (the normalised edge input), GS/BS range;
  * inverse round trip on processed graphs where graphkit has already built them (smoke runs): load the graph,
    apply the saved normaliser's inverse to bus Pd/Qd/Vm/Va, compare with the raw table row.

    python normalizer_audit.py [--processed DIR ...]     -> normalizer_audit.md / .json next to this file
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
LADDER = REPO / "data/external/genco-ladder-datakit"
REPLAY = REPO / "data/external/genco-pf-pretraining-corrected"
SRC = REPO / "data/external/genco-pf-pretraining"
REPLAY_SUBS = ["subnet_case4837_goc_idx_35", "subnet_case20758_epigrids_idx_49", "subnet_case5658_epigrids_idx_8",
               "subnet_case4837_goc_idx_28", "subnet_case10480_goc_idx_40", "subnet_case4837_goc_idx_37",
               "subnet_case10480_goc_idx_26", "subnet_case19402_goc_idx_47"]


def table(d: Path, name: str, cols=None) -> pd.DataFrame:
    return pd.read_parquet(d / "raw" / f"{name}.parquet", columns=cols)


def audit_tree(d: Path, role: str) -> dict:
    bus = table(d, "bus_data", ["scenario", "Pd", "Qd", "Qg", "Vm", "Va", "vn_kv", "GS", "BS"])
    gen = table(d, "gen_data", ["p_mw"])
    br = table(d, "branch_data", ["scenario", "rate_a"])
    nz = pd.concat([bus.Pd[bus.Pd != 0], bus.Qd[bus.Qd != 0], gen.p_mw[gen.p_mw != 0], bus.Qg[bus.Qg != 0]])
    base = float(np.percentile(nz, 95))
    return {"tree": d.name, "role": role, "scenarios": int(bus.scenario.nunique()),
            "buses": int((bus.scenario == bus.scenario.min()).sum()), "baseMVA": base,
            "vn_kv_max": float(bus.vn_kv.max()), "Va_absmax_deg": float(bus.Va.abs().max()),
            "Vm_min": float(bus.Vm.min()), "Vm_max": float(bus.Vm.max()),
            "Pd_min": float(bus.Pd.min()), "Pd_max": float(bus.Pd.max()),
            "rate_over_base_max": float(br.rate_a.max() / base) if base else None,
            "GS_absmax": float(bus.GS.abs().max()), "BS_absmax": float(bus.BS.abs().max())}


def round_trip(proc_root: Path) -> list[dict]:
    """Processed graphs are stored in RAW units (graphkit normalises in its transform at load time). For graph 0 of
    each network: (1) stored bus targets vs the raw table row (identity); (2) graphkit's HeteroDataMVANormalizer
    fit_from_dict(saved stats) -> transform -> inverse_transform must return the stored values."""
    import copy
    import types
    import torch
    from gridfm_graphkit.datasets.globals import PD_H, QD_H, VA_H, VM_H
    from gridfm_graphkit.datasets.normalizers import HeteroDataMVANormalizer
    out = []
    for stats in sorted(set(glob.glob(str(proc_root / "*/processed/data_stats_*.pt")))):
        net = Path(stats).parents[1]
        g0 = net / "processed/data_index_0.pt"
        if not g0.exists():
            continue
        st = torch.load(stats, weights_only=False)
        g = torch.load(g0, weights_only=False)
        y = g.y_dict["bus"]
        bus = table(net, "bus_data", ["scenario", "bus", "Pd", "Qd", "Vm", "Va"])
        r = bus[bus.scenario == 0].sort_values("bus")
        ident = {"Pd": float(abs(y[:, PD_H].numpy() - r.Pd.to_numpy()).max()),
                 "Qd": float(abs(y[:, QD_H].numpy() - r.Qd.to_numpy()).max()),
                 "Vm": float(abs(y[:, VM_H].numpy() - r.Vm.to_numpy()).max()),
                 "Va": float(abs(y[:, VA_H].numpy() - r.Va.to_numpy()).max())}
        args = types.SimpleNamespace(data=types.SimpleNamespace(baseMVA=100))
        nz = HeteroDataMVANormalizer(args); nz.fit_from_dict(st)
        h = copy.deepcopy(g); nz.transform(h); h2 = copy.deepcopy(h)
        h2.is_normalized = torch.tensor([True]); h2.baseMVA = torch.tensor([float(h.baseMVA)])   # batched form expected
        nz.inverse_transform(h2)
        # graphkit's inverse_transform does NOT convert Va back to degrees (it stays in radians; ladder_eval applies
        # np.degrees to predictions), so Va is compared in radians
        ref = y.clone(); ref[:, VA_H] = torch.deg2rad(ref[:, VA_H])
        rt = float(max(abs(h2.y_dict["bus"][:, c] - ref[:, c]).max() for c in (PD_H, QD_H, VM_H, VA_H)))
        out.append({"tree": net.name, "baseMVA": float(st["baseMVA"]), "identity": ident, "transform_inverse_maxdiff": rt})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", nargs="*", type=Path, default=[], help="run data dirs with graphkit processed/")
    a = ap.parse_args()
    cases = (HERE.parents[0] / "2026-10-20-ladder-to-datakit/cases.txt").read_text().split()
    rows = [audit_tree(LADDER / c, "validation" if c.endswith("rte") else "train") for c in cases]
    for s in REPLAY_SUBS:
        d = REPLAY / s if (REPLAY / s / "raw").is_dir() else SRC / s
        rows.append(audit_tree(d, "replay"))
    rt = [x for p in a.processed for x in round_trip(p)]
    L = ["# Normaliser audit (generated by normalizer_audit.py; UNAUDITED)", "",
         "| tree | role | buses | baseMVA (graphkit rule) | vn_kv max | max abs Va (deg) | Vm range | Pd range (MW) | max rate_a / baseMVA |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['tree']} | {r['role']} | {r['buses']} | {r['baseMVA']:.1f} | {r['vn_kv_max']:.0f} | {r['Va_absmax_deg']:.1f} | "
                 f"{r['Vm_min']:.3f} to {r['Vm_max']:.3f} | {r['Pd_min']:.1f} to {r['Pd_max']:.1f} | {r['rate_over_base_max']:.0f} |")
    if rt:
        L += ["", "## Processed graphs (scenario 0, bus targets): stored vs raw table, and graphkit transform -> inverse", "",
              "| tree | baseMVA | stored-raw Pd (MW) | Qd | Vm (pu) | Va (deg) | transform->inverse max diff |", "|---|---|---|---|---|---|---|"]
        for d in rt:
            i = d["identity"]
            L.append(f"| {d['tree']} | {d['baseMVA']:.1f} | {i['Pd']:.1e} | {i['Qd']:.1e} | {i['Vm']:.1e} | {i['Va']:.1e} | "
                     f"{d['transform_inverse_maxdiff']:.1e} |")
    (HERE / "normalizer_audit.md").write_text("\n".join(L) + "\n")
    json.dump({"trees": rows, "round_trip": rt}, open(HERE / "normalizer_audit.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
