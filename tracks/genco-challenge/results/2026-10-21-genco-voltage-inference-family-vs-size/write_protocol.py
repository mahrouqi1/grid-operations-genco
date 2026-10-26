import json, hashlib, sys, glob
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import grids as G
ids = json.loads((HERE / "test_ids.json").read_text()); ver = json.loads((HERE / "verified.json").read_text())
man = {r["name"]: r for f in glob.glob(str(G.CARVED / "manifest_*.json")) for r in json.loads(Path(f).read_text())}
gen = {}
for f in HERE.glob("gen_?.log"):
    for l in f.read_text().splitlines():
        if l.startswith("{"): j = json.loads(l); gen[j["name"]] = j
sub, notrun = {}, {}
for n in G.SEL:
    nt = len(ids.get(n, {}).get("test", [])); r = man[n]
    if nt >= 20:
        sub[n] = {"family": G.FAMILY[n], "parent": G.PARENT[n], "tier": G.TIER[n], "n_bus": r["n_bus"], "n_gen": r["n_gen"], "n_cut_branches": r["n_cut_branches"], "tie_mw": r["tie_mw"], "internal_scale_mw": r["internal_scale_mw"],
                  "parent_load_scale": r.get("parent_load_scale", 1.0), "n_generated_valid": gen[n]["scenarios_written"], "n_verified": ver.get(n, {}).get("n_pass"), "n_test": nt}
    else: notrun[n] = {"n_test": nt, "datakit_status": gen.get(n, {}).get("status"), "scenarios_written": gen.get(n, {}).get("scenarios_written"), "n_bus": r["n_bus"], "family": G.FAMILY[n]}
sel = json.loads((HERE / "selected.json").read_text())
p = {
"written": "2026-10-21, committed BEFORE any GENCO-vs-interpolator evaluation on the carved UNSEEN-family subgrids. Before this: carving, OPF screening, datakit generation, label verification, test ids. No GENCO or interpolator numbers on these subgrids exist.",
"question": "Candidate 10: freshly carved subgrids of the SEEN families (GOC/EPIGRIDS) all win (19/19 at 40%). Candidate 9: unseen-family grids win 8/17. Separate FAMILY from SIZE: carve the UNSEEN families (RTE, PEGASE, Polish) into the same 150-950 bus pieces with the same procedure. If unseen pieces still lose -> family matters; if they win like the seen pieces -> size matters.",
"procedure": "carve.py is candidate 10's carve.py unchanged in logic (seeded BFS, tie-flow folding, REF = largest gen, AC-OPF screen, datakit generation via ladder_data, verify.py residual < 1e-6 pu on every scenario) with these additions needed because the unseen parents are harder, each used ONLY when the unchanged procedure fails: (1) parent base point: if plain Newton fails, damped Newton (backtracking), then load scaled down 0.9..0.5, then the case file's own dispatch; generators at type-1 buses now inject their file Q (a correction also applied from now on; it did not change which seen-family parents converge but the seen-family carved cases were made with the older code); no parent used a scaled load (all parent_load_scale = 1.0). (2) Parents case1888_rte, case6470_rte, case6495_rte, case6515_rte could NOT be given a base point by ANY of these fallbacks (Newton diverges or pandapower finds no reference bus), so RTE is represented by case2848_rte only. (3) Tier-b relaxation (tie_max 4.0) tried for case3012wp_k and case3375wp_k that had no/few tier-a candidates; extra candidate batches (offsets 20, 40, 50) were drawn for parents with few OPF-feasible candidates (case2848_rte, case2383wp_k, case3375wp_k, case3012wp_k). The PGLib RTE/PEGASE/Polish files are used as parents (they correspond to the grids of candidate 9 but are the PGLib versions, not the pandapower built-ins).",
"selection_rule": sel["rule"] + ". Of 30+ OPF-screened candidates, case3012wp_k produced no feasible candidate and case3375wp_k only one (tier b); so Polish is effectively one parent (case2383wp_k) plus one piece of case3375wp_k.",
"subgrids": sub, "not_run": notrun,
"run_rule": ">= 20 verified test scenarios, finite output for G and I1-I5 in all 15 configs",
"conditions_and_methods": "identical to candidates 9 and 10 (ocommon.py copy, run_interp.py, run_genco_o.py): 10/20/40% x 3 seeds, 0.5% noise at 20/40%; ratio = median_G / min median_I; scenario-cluster bootstrap 2000 seed 7; win = ratio<1 and CI upper<1.",
"interpolators": ["I1","I2","I3","I4","I5"], "bootstrap": {"n": 2000, "seed": 7}, "alpha": 0.05,
"primary_test": "at 40% noise-free (3 seeds pooled): proportion of carved UNSEEN-family subgrids where GENCO wins (CI rule) is LOWER than the carved SEEN-family 19/19 (reused from ../2026-10-21-genco-voltage-inference-control/verdict.json, not rerun) by a one-sided Fisher exact test (alternative: unseen proportion less) at p < 0.05 => 'FAMILY MATTERS'; otherwise 'NOT SHOWN'.",
"secondary_tests": ["same one-sided Fisher at 20% (seen 16/19)", "per family (RTE, PEGASE, Polish) at 40% and 20% against the same seen set (descriptive; RTE = case2848 only, Polish = case2383 + one piece)",
  "size distributions of the two carved sets (bus counts); they 'differ' if a two-sample KS p < 0.10 or the medians differ by more than 25%; the size-matched subset is ALWAYS reported: greedy nearest-neighbour matching on log(bus count) without replacement, caliper |log ratio| <= 0.25, Fisher one-sided on the matched pairs at 40% and 20%",
  "descriptive: 10%, 40% with noise, median ratios"],
"size_differ_ks_p": 0.10, "size_differ_median_frac": 0.25, "caliper_log": 0.25,
"caveats_preregistered": "RTE represented by ONE parent (case2848_rte) because the other four parents' base points are unobtainable by this procedure; Polish effectively one parent; with 18 pieces from ~6 parents the pieces are not independent draws of a family; the unseen parents are the PGLib versions; our carving differs from GENCO's; OPF screening favours easy pieces; unaudited.",
"post_sentences": {
 "FAMILY MATTERS (p < 0.05, unseen pieces lose)": "Carved pieces of the UNSEEN families (RTE, PEGASE, Polish), the same size and cut the same way as the seen-family pieces, do NOT show GENCO's advantage (X of N win at 40% against 19/19 for the seen families; one-sided Fisher p = P). So candidate 9's seen/unseen gap is a family effect, not a size effect: the pretraining families matter. Caveats: RTE is one parent, Polish about one, carving may differ from GENCO's, unaudited.",
 "NOT SHOWN (unseen pieces win like the seen ones)": "Carved pieces of the unseen families win at 40% about as often as the seen-family pieces (X of N vs 19/19, one-sided Fisher p = P). So the unseen-family failures of candidate 9 are better explained by SIZE (or whole-grid structure) than by the family: pieces of 150-950 buses of unseen families are handled as well as seen ones. Caveats as above; unaudited."},
"code_sha256_16": {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest()[:16] for f in ["grids.py", "carve.py", "gen.py", "verify.py", "make_ids.py", "ocommon.py", "run_interp.py", "run_genco_o.py", "verdict.py", "screen.jl"]}}
(HERE / "protocol.json").write_text(json.dumps(p, indent=1)); print(len(sub), "subgrids;", len(notrun), "not run:", list(notrun))
