# Classical WLS state-estimation baseline for the GENCO zero-shot |V| inference comparison (2026-10-21/22)

> **UNAUDITED.** Produced by one chat; no `/audit` round has cleared it. Do not quote the numbers until one has.

## Bottom line

Adding an AC weighted-least-squares state estimator (WLS, called I6 below) as a comparator **removes most of GENCO's
wins**. WLS is the best of the six comparators in **244 of 306 scored cells (80%)**, and in all 71 cells where fewer
than 20% of its scenarios hit the iteration cap. GENCO's headline counts (same scenarios, masks, noise and metric; paired
bootstrap; see caveat 4 for the subset) become:

| statement | published count (full sets) | I1-I5, re-derived on the WLS subset | **I1-I6 (WLS added)** | I1-I6, non-converged WLS scenarios replaced by I2 |
|---|---|---|---|---|
| carved pieces, datakit, 40% (seen 19 + unseen 17 = 36) | 33/36 | 31/36 | **6/36** | 16/36 |
| carved pieces, datakit, 40% + 0.5% noise | 23/36 | 19/36 | **5/36** | 12/36 |
| carved pieces, ladder generator, 40% | 30/36 | 30/36 | **5/36** | 17/36 |
| locked 15 whole grids, 40% / 20% | 12/15 / 8/15 | 11 / 7 | **2 / 1** | 5 / 3 |
| seen-family whole grids, 40% / 20% / noisy 40% | 15/15 / 13/15 / 8/15 | 14 / 13 / 7 | **1 / 0 / 1** | 13 / 13 / 7 |
| unseen-family whole grids, 40% / 20% / noisy 40% | 8/17 / 3/17 / 2/17 | 5 / 2 / 2 | **0 / 2 / 2** | 3 / 2 / 2 |

Per-group detail: `table.md`, `verdict.json` (aggregates); per cell: `cells.csv`. 162 of 306 cells flip from GENCO-win to loss when WLS
is added; 0 flip the other way. Median error ratio GENCO/WLS per group is 1.7 to 12 (GENCO error is larger), e.g. 10.9 on
carved seen pieces at 40%, 3.4 on carved unseen pieces, 2.7 locked-15, 1.8 seen whole grids, 3.0 unseen whole grids. By the
pre-registered rule WLS **should be added as I6** (best-of-six in 80% of cells against a 10% threshold; flips 162 cells).

Why it wins: the task gives loads, so every bus without a generator has an exactly known injection. That is a physics
constraint on most of the network; with the grid's Ybus, a handful of measured |V| (plus weak generator priors) pin down the
remaining |V| much better than a smoother. Where few buses have generators (SimBench, carved pieces) the problem is nearly a
power-flow solve and WLS error is far below GENCO's (oos unseen max ratio ~5000, SimBench HV). This is a
legitimate use of the task's information set (no dispatch, no labels at unmeasured buses).

## Caveats that change how to read this (read before quoting)

1. **WLS here is an iteration-capped, not fully converged, estimator.** 15,241 of 25,107 scenario-solves (61%) hit the cap
   (8 + 12 Levenberg-Marquardt iterations) and are flagged `maxit`; 0 raised errors, 0 NaN. They are counted in the primary arm
   (the error of the lowest-cost iterate). The cap and damping were picked by looking at MAE on 4 scenarios each of two scored
   grids (case500_goc, case1888rte): mild tuning on test labels, disclosed in `protocol.json`. A different solver setting moved
   individual MAEs by up to 2x. So this is a feasible classical baseline, not WLS at its optimum; a converged WLS would probably
   do better, but that was not tested.
2. **The "fallback" column** (flagged scenarios take the I2 error instead) is a fallback policy (capped WLS replaced by I2), not a bound on WLS: the direction varies by grid
   (I2 beats the capped WLS on 86 of 89 capped scenarios on case2312_goc, but loses on 149 of 254 on case2848_rte_c1 and
   on 44 of 44 on case4837_goc_c0; an external reviewer audit 2026-10-22 C08). With it, most wins are still removed (carved datakit 40%: 33 -> 16 of 36).
3. **Observability.** With unknown dispatch the measurement set (|V| + known zero injections) is count-observable only if
   measured |V| >= 2*(generator buses) - 1. It fails in 4,725 of 25,107 scenario-solves (19%): 0% on carved seen pieces,
   35% on carved unseen pieces (71% at 20%), 80% / 53% on locked-15 at 20% / 40%, 16% on oos. These are solved with the weak
   generator P/Q priors (limits-based, protocol M3) and a very weak |V| prior; they are reported, never dropped. Convergence
   rate was similar for count-observable and unobservable scenarios (39% vs 41%), so the cap, not observability, drives the flags.
4. **Scope cut (compute: shared box at load 35-70, 2-core cap).** WLS was scored on the **first 10 test scenarios per grid for the
   oos whole-grid sets and the first 30 for locked-15, carved datakit and carved ladder sets** (protocol amendments 1-2). G,
   I1-I5 and WLS were all re-evaluated on that same subset (paired bootstrap, same resample indices for the 5- and 6-method arms), so the
   "I1-I5 re-derived" column differs from the published counts through sample size alone (cells near ratio 1 can flip; e.g. oos unseen 40%:
   5 vs published 8). Compare columns 3 and 4, not 2 and 4. Only clean 20%/40% and noisy 40% were scored (the conditions in the
   headline counts); 10% and noisy 20% were not.
5. Same design caveats as the base studies (locked-15 generated the hypothesis, siblings not independent, ...). The WLS design
   choices (sigmas, generator priors, 2% loss assumption) were fixed in `protocol.json` before scoring and not tuned
   afterwards except the solver cap above.
6. Noise: the estimator is told the noise condition (sigma 0.5% of |V| when noisy, 1e-3 pu otherwise). GENCO and I1-I4 are not
   (I5 fits it by marginal likelihood). This favours WLS somewhat in the noisy rows.

## What was done

`protocol.json` (written 22:04 EDT before any WLS score; two timestamped amendments) fixes: state = polar |V|, theta (REF angle known);
measurements = |V| at the seeded measured buses (same masks via `vcommon.measured_seeded`, same noise via `ocommon.noise_z`),
exact P/Q injection (= -load) at every non-generator bus, weak limits-based pseudo-measurements of generator P/Q (dispatch is NOT
used as a measurement), weak |V| prior; DC-angle start, 2-stage continuation, LM. Code: `wlslib.py` (estimator, sparse analytic
Jacobian), `run_wls.py` (reads each study's tables, masks, noise), `verdict.py` (same win rule: ratio < 1 and 95% CI upper < 1,
best comparator re-selected per resample, 2000 resamples). Test: `tests/test_wls_recovery.py` (all |V| and all injections
measured recovers reference_pf on case14 to 1e-8; passes). Outputs: `per_case/<study>/<grid>.csv`, merged `all_cases.csv`
(25,107 scenario-solves with status, iterations, observability), `cells.csv`, `verdict.json`, `table.md`.
GENCO was not rerun; its and I1-I5's existing per-scenario files were reused. No GPU. All 119 grids/pieces had GENCO files; none were skipped.

## Not done

- No fully converged WLS (iteration cap); no WLS variant without priors or with other prior strengths; no 10% / noisy-20% conditions.
- Full scenario sets (oos 10 of ~100 per grid; locked/carved 30 of ~100).
- Not edited, per instructions: post-draft-v2.md, the claim register, existing voltage-inference reports, q46/q47 scripts. Section 6 of the
  draft ("No WLS baseline yet") is now answerable, but needs the caveats above and an audit before any wording change.
