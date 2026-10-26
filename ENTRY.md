# What GENCO does without fine-tuning: infer missing voltages, not solve power flow

Mazin Al-Mahrouqi (The Ohio State University)

**Contact:** mahrouqi.1@osu.edu
**Code and result files:** <https://github.com/mahrouqi1/grid-operations-genco>. One command rebuilds the figures and runs the checks: `bash reproduce.sh`. Details are in section 8.

## The short version

GENCO comes with its code, pretraining data and checkpoints. That let us look closely, and we thank the authors for it. We asked three questions. Are GENCO's own training labels physically valid, and what does the model learn from the ones that are not? The paper reports zero-shot transfer as unsolved; is there a task where the released checkpoint already works without fine-tuning? Does more pretraining help? This post is what we found, for people who use GENCO or plan to build on it.

1. **GENCO's response to collapsed training labels depends on the task.** About 3% of the sampled pretraining labels are collapsed roots of the power-flow equations. Under its masked pretraining task the model reproduces them. On the power-flow task, its voltage magnitudes at the collapsed buses are closer to a recovered power-flow solution than to the stored label in all 1,083 scored scenarios, with a median difference of 0.008 pu. This comparison does not establish feasibility of GENCO's full output.
2. **Without fine-tuning, GENCO can fill in missing voltages.** The weights are unchanged, but the input scale uses labelled target-grid scenarios, following GENCO's preprocessing; we have not measured the effect of that scaling. The paper's zero-shot limitation is about power-flow transfer, and our own power-flow tests agree with it (finding 4). Masked voltage-magnitude inference is a different task. There, give it 40% of the bus voltages and it infers the rest better than the best of five learning-free graph interpolators. It wins on 33 of 36 carved subgrids, 12 of the 15 locked-benchmark grids and 15 of 15 whole grids from its pretraining families. On whole grids from unseen families it wins 8 of 17. With 20% of the voltages measured, the locked-benchmark count was 8 of 15, below the bar we set in advance, and with 0.5% measurement noise the wins shrink (section 3). A state estimator that knows the generator limits is a harder comparator. On the subset we ran it on, GENCO's wins against the interpolators plus this estimator fall from 31 to 5 of 36 carved subgrids and from 11 to 2 of 15 locked grids (mask seed 0).
3. **Our continued-pretraining recipes did not improve the locked-benchmark result.** We ran 21 continued-pretraining runs (7 training recipes x 3 seeds), then 9 more runs with three fixes: a lower learning rate, a frozen backbone, and more of GENCO's original training data. None beat the released model on a locked 15-grid benchmark. About three quarters of our training graphs (73%, counting graphs from GENCO's original training data) came from GENCO's own pretraining families, so most of these runs do not isolate new families. A further control that used SimBench high-voltage grids, a family GENCO never saw in pretraining, together with GENCO's corrected original training data, was also no better than the released model, so these runs cannot tell whether the training recipe or unfamiliar grid families limit transfer.
4. **When every input is known, a classical method wins.** Two fast-decoupled iterations from a DC start beat every GENCO variant we tried on branch angle and voltage magnitude.

We also release what we built along the way: the RepairKit, which checks datakit-format datasets, attempts label repair and verifies the repaired tables, the checksums of the corrected tables with the command that regenerates them, and a frozen, leakage-checked protocol for testing any new training recipe.

## 1. Setup

GENCO (arXiv 2608.09921v2) is a graph foundation model for power systems. It is released with its code (gridfm-graphkit, gridfm-datakit), pretraining data and checkpoints.

We first reproduced the paper's case118 transfer numbers with the released checkpoints, to within 1e-5 MW. The largest difference is under 1e-5 MW: without fine-tuning, 11.078244 against 11.078253 published.

We registered the transfer, voltage-inference, continued-pretraining and classical-reference tests before their results existed (each follow-up arm before its own run). The label diagnostics in section 2 and the case118 replication had no written protocol, so treat them as exploratory.

## 2. GENCO's response to collapsed training labels depends on the task

**The defect.** 1,085 of 34,274 sampled scenarios are flagged (minimum |V| below 0.5 pu, or negative). Of these, 1,069 (3.1% of 34,274) have a collapsed |V| below 0.5 pu, and 16 are sign flips only. The cause is in the data generator. Its fast power-flow call (PowerModels, trust-region solver from a flat start) lands on the collapsed root. It reproduces the published label to below 1e-6 pu on 51 of 51 sampled collapsed scenarios. We have not established which property of the grids pushes it there.

**Under the pretraining mask,** the released model predicts the collapsed voltage at 96% of hidden collapsed buses, and at least 75% on each of the 9 subgrids. It reproduces the defect.

**Under the power-flow task, on the same systems,** the picture changes:
- It predicts a collapsed voltage at 0 of 1,143 hidden collapsed buses.
- At the collapsed buses it is closer to a physical solution than to the label in all 1,083 scored scenarios, with a median gap of 0.008 pu (the label itself is a median 1.135 pu away). In case4837_idx_35, where most collapsed labels are also negative, it sits 0.009 pu from the solution and 1.15 pu from the label.

**How we found the physical solutions, and what this does not show.** The 1,085 flagged labels contain a bus voltage that is negative or below 0.5 pu; in 941 of them the lowest bus is within 0.1 pu of zero, and at the 1,156 flagged buses, 1,148 of our solutions sit at 0.9 pu or above. The 0.5 pu cut is a convention, not a gap in the data. We searched for a power-flow solution with every voltage at least 0.5 pu and a power mismatch below 1e-9 pu. Newton's method started from flat and DC guesses, then from the solutions of neighbouring scenarios of the same subgrid, then by gradually morphing from healthy scenarios. This found a solution for all 1,083 scored collapsed scenarios. For 1,070 of them two different starts reached the same solution; 643 of those came from the neighbour or morphing starts. Across all 1,083, 40 solutions have a voltage below 0.70 pu. For the other 13, the stored label already satisfies the power-flow equations once its sign error is removed. Six scenarios are more than 0.2 pu from their solution; they have strained low-voltage solutions. A scenario can have more than one valid solution, so this shows GENCO lands near a physical solution, not that it finds a unique one. The comparison uses voltage magnitudes only.

We do not know whether this difference comes from pretraining or from the extra information supplied in the power-flow task. We evaluated the power-flow task only, not OPF.

**Cleaning the labels.** We continued pretraining on corrected labels, and separately on the original labels for comparison. Over three training seeds per label set, continued pretraining made held-out error worse than the released model's in both arms (median over the 15 held-out grids of the ratio to released; the corrected-label models still beat released on 2 to 4 of 15 grids on branch angle): median branch-angle error 1.16 times released with corrected labels (range 1.09 to 1.17) and 1.33 times with original labels (1.30 to 1.44); median |V| error 1.38 times (1.35 to 2.21) and 2.31 times (1.93 to 2.56), Across the three seeds, the branch-angle ranges do not overlap; the voltage-magnitude ranges do. For the added seeds, the original weight-identity check failed; a replacement check was chosen after the results were seen. The seed report documents this limitation. The share of hidden collapsed buses still predicted below 0.5 pu was 0.80 (0.70 to 0.83) with corrected labels and 0.94 (0.92 to 0.97) with original labels (mask seed 1000), lower with corrected labels in all three seeds. We also trained two GENCO-architecture models from scratch for about 115k updates (1.3% of the paper's schedule) on 28 of GENCO's 100 pretraining subgrids, one on the original labels and one on the corrected labels, with the same seed, initialisation and data order. Both failed the underfit check we set in advance at 12k updates (hidden-voltage error on normal scenarios 1.4 to 1.6 times that of a flat 1.0 pu guess) and were far from GENCO's schedule at the end (by 114k updates the error had fallen to 0.5 to 0.6 times the flat guess), so this run cannot say whether the label repair changes what GENCO learns. Both label sets worsened held-out accuracy under this training recipe.

![Top: the fraction of hidden collapsed buses that the model still predicts below 0.5 pu falls from 0.97 to 0.84 (mean over three mask seeds) when continued pretraining uses corrected labels and to 0.94 with the original labels (training seed 200; the axis starts at 0). Bottom: held-out error relative to the released model is worse with both label sets (training seed 200; the three-seed medians are in the text). The dashed line marks the 1.10 preservation tolerance.](tracks/genco-challenge/results/2026-10-18-genco-challenge-post/figures-v3/fig_c_unlearning.png)

## 3. Voltage inference from partial measurements, without fine-tuning

**The task.** Loads are known. Generator dispatch is unknown. The voltage magnitude |V| is measured at a seeded fraction of buses, which is GENCO's own masked regime. The GENCO paper already studies state estimation (its section 5.3); we test the released checkpoint's masked voltage inference across these grid sets without fine-tuning. The released checkpoint, without fine-tuning, predicts |V| at the other buses. We compare it with the best of five learning-free graph interpolators: harmonic, admittance-weighted, nearest electrical neighbour, load-aware, and graph kriging. None of them uses the power-flow equations. A model-based comparator follows below.

**Terms.** Carved subgrids are smaller networks extracted from parent grids. "Seen" means the grid family appeared in GENCO's pretraining, not necessarily that this particular grid did. The locked benchmark is 15 fixed grids that shaped earlier work; it is not a blind test. A mask seed fixes which buses are measured. Branch angle is the voltage-angle difference across a branch, and pu is per unit.

**Results at 40% measured, noise-free.** A win means GENCO's error is lower than the best interpolator's, and the upper end of its 95% bootstrap interval is also below 1.

| Test set | GENCO wins |
|---|---|
| Carved subgrids of 159 to 903 buses, all | 33 of 36 |
| of which from pretraining-family parents | 19 of 19 |
| of which from unseen families | 14 of 17 |
| Same carved subgrids, a second scenario generator (all pieces from one generator) | 30 of 36 (19 of 19 pretraining-family, 11 of 17 unseen) |
| Whole grids, locked benchmark | 12 of 15 |
| Whole grids, pretraining families | 15 of 15 |
| Whole grids, unseen families | 8 of 17 |

With 0.5% measurement noise at 40% measured, GENCO still wins 23 of 36 carved subgrids (12 of 19 seen, 11 of 17 unseen), 8 of 15 seen-family whole grids and 2 of 17 unseen-family whole grids. The margin collapses on seen-family pieces (median error ratio 0.47 to 0.96) and shrinks on unseen pieces (0.78 to 0.89).

**Family familiarity.** The evidence points that way, but it is weak. The primary tests at 20% measured (locked 15) and 40% (unseen whole grids) failed. The carved-subgrid control passed (19 of 19). Under one scenario generator, the seen pieces win 19 of 19 and the unseen pieces 11 of 17 (p = 0.006, Fisher's exact test on pieces, a secondary and unadjusted test). The pieces are not independent: the 19 seen and 17 unseen pieces come from 7 and 6 parent grids. A one-sided permutation test over parents gives p = 0.021, which we show only as a sensitivity check. Two registered tests did not show an effect: the family test on datakit pieces (14 of 17 against 19 of 19, p = 0.095) and the pieces-versus-whole test within the second scenario generator (11 of 17 pieces against 8 of 17 whole grids, p = 0.245). Seen and unseen pieces also differ in parent and composition.

**Against a state estimator.** On the same scenarios, masks and seeds, we added a weighted-least-squares (WLS) state estimator. It uses the known loads, the grid model and the true generator active-power limits, so it is a stronger comparator than the interpolators. At 40% measured, GENCO's wins against the five interpolators plus this WLS fall from 31 to 5 of 36 carved pieces, and from 11 to 2 of the 15 locked-benchmark grids. With the solver allowed 40 iterations in its first stage and 60 in its second (40/60) they fall to 2 of 36 and 1 of 15. That solver still fails to converge on some scenarios; replacing each non-converged solve with the fixed I2 interpolator (harmonic interpolation weighted by branch admittance) gives 4 of 36 and 1 of 15 instead. A win here means the error ratio to the best comparator is below 1, with the upper end of its 95% bootstrap interval also below 1. The 31 and 11 are interpolator-only counts on the subset WLS was run on (the first 30 scenarios per grid, mask seed 0). On the full test sets above they are 33 and 12.

On magnitudes: the median ratio of GENCO's error to WLS's error, with the true generator limits, is 7.3 on carved pieces, 2.7 on the locked 15, 1.7 on seen-family whole grids and 2.4 on unseen-family whole grids, so GENCO is clearly less accurate than the informed WLS; against the active-limit-blind WLS the medians are 2.2, 1.3, 0.58 and 0.96 (mask seed 0). Most of these WLS runs were stopped by the iteration cap. With the limit raised to 40/60 the medians against the informed WLS are 10.3, 2.8, 8.8 and 6.2, and against the active-limit-blind WLS 3.6, 1.4, 1.16 and 1.39. So GENCO is close to the active-limit-blind WLS (1.16) only on seen-family whole grids, and clearly behind on unseen-family whole grids, and even that is against a solver that fails to converge on 81% (seen) and 44% (unseen) of whole-grid scenarios (details below).

The WLS result comes with caveats:
- WLS ran on 10 to 30 scenarios per grid for the win counts (up to 100 per grid, 30 on whole grids, for the convergence figures). At mask seed 0, 60% of its solves with the true generator limits hit the iteration cap. With the limit raised to 40/60 it still failed to converge on 17% of locked-benchmark scenarios, 11% and 30% of seen and unseen pieces and 49% of whole-grid scenarios. Its solver was tuned on two scored grids.
- WLS leans on the generator limits. We removed generator-specific active-power limits: every in-service generator's Pmax was set to the median and Pmin to 0. Reactive-power limits were unchanged. We call this the active-limit-blind WLS. GENCO still sees the true limits. GENCO then beats the interpolators and this active-limit-blind WLS on 8 of 36 pieces, 3 of 15 locked-benchmark grids, 6 of 15 seen-family whole grids and 3 of 17 unseen-family whole grids (mask seed 0). With the iteration limit raised to 40/60 these become 4, 2, 1 and 2.
- The active-limit-blind WLS hit the iteration cap on 88% of piece scenarios and 99% of seen-family whole-grid scenarios (mask seed 0). If every capped solve is replaced by the fixed I2 interpolator, the counts become 28, 5, 13 and 3 (mask seed 0). With the iteration limit raised to 40/60 on all 36 pieces, GENCO wins 4 of 36 (8 at the original limit). That solver still fails to converge on 35% of piece scenarios, 25% of locked-benchmark scenarios, 81% of seen-family whole-grid scenarios and 44% of unseen-family whole-grid scenarios; replacing each non-converged solve with the fixed I2 interpolator gives 9, 4, 10 and 3 for pieces, locked, seen whole and unseen whole.

![Top: each dot is one grid: GENCO's error divided by the best interpolator's error at 40% measured (error = median over scenarios of the mean absolute |V| error on unmeasured buses); filled circles mark wins under the bootstrap rule, with win counts above and the 0.5% noise counts below. Bottom: GENCO's error divided by the WLS error, with the true generator limits (squares) and the active-limit-blind WLS (triangles); hollow markers use the original iteration budget and filled markers use 40/60 iterations. Bars and printed values show group medians for each budget (mask seed 0, log axis). Values above 120 are clipped: 5 grids with true limits (largest 6,146) and 4 with the active-limit-blind WLS (largest 1,606), all SimBench high-voltage grids.](tracks/genco-challenge/results/2026-10-18-genco-challenge-post/figures-v3/fig_a_voltage_inference_v35.png)

## 4. Continued pretraining did not help in our study

Continued pretraining means taking the released checkpoint and training it further on more data. A masking policy decides which values are hidden during that training. We ran two sets of runs: a first set of 21 runs, and a second set of 9 runs that tried three fixes after the first set failed. A third control, 3 runs on the SimBench high-voltage grids, is described under Reading.

**Design.** We froze the protocol before any results and logged amendments with times.
- We continued pretraining the released checkpoint on 37,000 verified operating points from 21 grids that we generated (our generated data), plus GENCO's original training data: 7,929 graphs from its 7 pretraining subgrids, with corrected labels. None of the 21 grids is in the benchmark.
- 6 SimBench HV grids (12,000 points) are a family GENCO's pretraining did not include.
- 15 grids are from GENCO's own pretraining families: 13 GOC grids (13,000 points) and 2 EPIGRIDS grids (12,000 points). That is 25,000 points, two thirds of the draw. Six of these 15 are whole-grid versions of the parents of its pretraining subgrids (GOC 4619, 4837, 4917, 9591; EPIGRIDS 5658, 7336). The other 9 are GOC grids with other parents.
- Counting the 7,929 original-data graphs, which also come from GENCO's families, the first set trained on 44,929 graphs: 73% from GENCO's own families and 27% SimBench HV. In the second set, the run with 9 times as much original data trained on 108,361 graphs: 89% and 11%. These are graph counts, not weighted by grid size.
- First set: 7 training setups, 3 seeds each. Five used our generated data together with the original data: four masking policies (random, PF-only, half-then-mix, curriculum) and a constant 50/50 mix of the two masks. Two were controls using only the original data (random mask, constant 50/50 mask).
- Validation on an RTE family (2 grids in the first set, 5 in the second). Scoring on the locked 15-grid benchmark.
- Leakage check: maximum branch-impedance Jaccard 0.16 on the grids kept (one secondary grid at 0.43 was dropped).
- Cost: 73.4 OSC GPU-hours for the first set and a further 15.0 for the second.

**First set (21 runs): no-go.** Here "no-go" means the continued-pretraining runs did not beat the released model.
- The arm that validation selected is 50% worse than released on branch angle (median 5.52 against 3.69 times DC). It is better on 1 of 15 grids.
- At a matched 12,000 updates, every training setup is worse than released:

| Training data | Branch angle vs released |
|---|---|
| GENCO's original data only | +21% |
| Our generated data (with the original data) | +54.5% |
| Our generated data (with the original data), with power-flow masking | +87% to +106% |

- Without our generated data, masking showed no measurable effect over three seeds.
- |V| error is also worse than released on 12 to 14 of 15 grids.
- The generated-data contrast compares training setups. It does not isolate the effect of the data. The runs match optimizer updates but not batches. Batches with only the original data held 32 graphs, about 20,138 nodes per step (the 7,929 original-data graphs average 629.3 buses). Batches with generated plus original data averaged about 47,118 nodes and 16.03 graphs per step: 2.34 times as many nodes per step as the original-data-only batches. The 15 runs with generated data trained on OSC and the 6 original-data-only runs locally. So +21% against +54.5% compares data, batch size and composition together.

![Bars show how much worse than the released model each training setup is on branch angle at matched 12,000 updates, grouped as original data only, generated data, and generated data with power-flow masking; the squares count, out of the 15 locked grids, those where |V| is worse and those where branch angle is better.](tracks/genco-challenge/results/2026-10-18-genco-challenge-post/figures-v3/fig_b_transfer_12k.png)

**Second set (9 runs): no-go again.** The three fixes were a lower learning rate, a frozen backbone (the model's 12 message-passing layers kept fixed), and 9 times as much of GENCO's original training data. We froze this protocol before the runs and made one amendment, about how often validation was run, after they finished; it changed which checkpoint one run selected, and the report in the repo gives the sensitivity.
- The arm that validation selected (9 times the original data) is 31% worse (4.83 against 3.69). The lower learning rate is 12% worse.
- The frozen backbone never beats the released weights on validation.
- A read of each run's final state, with no selection (a pre-registered amendment, not an input to the go/no-go), gives branch angle +48.1% for the lower learning rate, +31.4% for the frozen backbone and +47.6% for 9 times the original data. Each is better on 2 of 15 grids. The frozen backbone is also worse at its final state, although its 12 message-passing layers are bitwise unchanged.
- At the selected checkpoints, |V| error is worse than released on 12 of 15 grids for 9 times the original data (median 2.1 times) and on 10 of 15 for the lower learning rate (1.4 times).

**Validation did not predict the benchmark.** In both sets of runs, gains on the RTE validation family (up to 19%, and 14%) did not transfer. Two arms with the same settings through 14k updates differ by up to 2.8 times on the validation metric (median 1.6). In the first set, validation selected the released weights themselves in 6 of the 21 runs, and in the second in 4 of the 9 runs. If you test a training recipe on GENCO, we suggest a validation family of more than a couple of grids and the released weights as a candidate.

**Reading.** Together with section 3, GENCO looks strongest where the grid family resembles its pretraining. This is a hypothesis. Section 3 also shows it winning 12 of the 15 locked-benchmark grids, none of them from its pretraining families. The training mix was 73% GENCO's own families (89% in the run with 9 times the original data), with the rest SimBench HV, and it lowered accuracy on the locked 15. That is not a test of training on new families alone. We also continued pretraining on the SimBench high-voltage grids, the only unseen-family grids in our training mix, together with GENCO's own corrected training subgrids (3 seeds). Judged by the rules we fixed in advance, the result is partial: at 12,000 updates it beat the run on our mostly-familiar generated data, but at the validation-selected checkpoint it did not, and on the 15 locked grids it was no better than the released model (median branch-angle error 4.4 times DC at 12,000 updates, against 3.7). So this control cannot tell whether the training recipe or unfamiliar grid families limit transfer.

**What we release for others:** a frozen, leakage-checked protocol that any training recipe can be run through, matched-update contrasts of data, masking and original data, and a list of things we did not try (other learning rates and schedules, longer runs, other draws of generated data).

## 5. A classical reference when all inputs are known

With every power-flow input known, two fast-decoupled iterations from DC (we call it fd2) were better than every GENCO variant we tried on branch angle and voltage magnitude (adapted GENCO had a lower active-power residual on two or three small grids). fd2 takes roughly 10 to 65 ms per scenario on grids of 1,900 to 9,300 buses, with no training. We measured this on a shared, loaded workstation. The table gives the gap, which is the bar a learned solver has to clear in this regime.

| GENCO variant | Result |
|---|---|
| Without fine-tuning (released checkpoint, as is) | 0 of 9 grids beat fd2. Branch-angle error is 25 to 129 times fd2's (on the 9 of the 15 grids we evaluated). |
| Few-shot with 100 labels | Beats DC on 7 of the 12 grids run (7 of 15 counting the 3 largest as not run). Never beats guarded fd2 (0 of 12). Beats raw fd2 on 1 of 12 (case240). |
| Learned corrector on large grids | 1.5 and 3.7 times fd2's error on case2869 and case9241. On case6470rte the fd2 guard replaced fd2 by DC in every scenario, so the corrector there was fed DC (1.2 times raw fd2's error, 0.64 times DC's). A corrector on raw fd2 was not run. |
| GENCO as a Newton start | On the 11 large grids we tried, never faster or more robust than fd2. |

One narrow exception: few-shot GENCO on case240 beats the raw fd2 iterate (3.16 deg), which is itself 3.3 times worse than DC (0.97 deg). Against the guarded fd2, which falls back to DC, fd2 wins there too, and GENCO (1.25 deg) is worse than DC. On case240, few-shot GENCO beats raw fd2 but remains worse than DC and guarded fd2.

When the dispatch is unknown, so not every input is known, GENCO's state inference for branch angle did not beat the best of three baselines built on an assumed dispatch on any of the 15 grids (error ratios 1.37 to 8.24). This is a different quantity from the |V| task in section 3.

## 6. Limits

- The 15 benchmark grids shaped earlier work. It is a locked benchmark, not a blind test.
- The carved subgrids are our own carving. All 5 RTE pieces come from one parent (case2848rte), 5 of the 6 Polish pieces from one parent (case2383wp), and the PEGASE pieces are 2 each from three parents.
- The state-estimator comparison covers only 10 to 30 scenarios per grid (one mask seed) and used an iteration-limited solver; raising the limit to 40/60 still leaves it unconverged on 17% of locked-benchmark scenarios, 11% and 30% of seen and unseen pieces and 49% of whole-grid scenarios.
- Our voltage-inference results are errors in pu, not counts of limit violations. We did not save per-bus predictions, so we cannot say how many violations they miss or falsely flag.
- GENCO's input scale is fitted on the target grid's solved outputs. Its normalizer sets the base power from the 95th percentile of the nonzero loads (Pd, Qd), generator active power and bus reactive generation on the target grid's training split. The generator quantities are solved outputs that the inference tasks hide. On the four grids we recomputed (case14, case118, case1354, case3120sp) the fitted scale is 1.49 to 2.26 times a loads-only scale. The weights are used without fine-tuning, but the input preprocessing is not label-free: a few labelled target-grid scenarios set the scale. We have not measured whether this changes any accuracy number.
- Two thirds of the continued-pretraining study's 37,000 drawn points, and about three quarters of its training graphs once the original-data graphs are counted (73%; 89% in the second set's run with 9 times the original data), came from GENCO's own pretraining families. The control trained on the SimBench HV grids with GENCO's corrected subgrids did not settle whether the recipe or unfamiliar families limit transfer (section 4).
- The continued-pretraining no-go tested one family of training setups: learning rate 3e-5, up to about 28,000 updates per run, one draw of 37,000 points, 3 seeds per setup. The second set varied the learning rate, froze the backbone, or added more original data. We did not try learning rates other than 3e-5 and 1e-5, longer runs, other optimisers or schedules, or other draws of generated data.
- Few-shot GENCO was not run on the 3 largest benchmark grids (case1354, case2736sp, case3120sp).
- Section 5 timings come from a shared workstation under load (load average 11 to 46), so cross-grid timing is only indicative.
- An earlier warm-up study trained on DC-prior targets through an import side effect. We re-cite its claim from an AC-labelled rerun (100 labels: 0.32, 0.41, 0.71, 1.05 of DC on case14, 30, 57, 118). The conclusion stands: a warm-up fine-tune beats DC on small grids.
- OPF was not evaluated for the invalid-label finding.

## 7. How we checked

We used separate AI review sessions, which had not produced the results, to check result files and claims in repeated rounds. Those checks led to the changes described below. The three newest training results (the three-seed continued pretraining, the SimBench control and the from-scratch pair) had two review rounds with minor items still open, and their reports remain marked unaudited. The audits changed several things, and the changes are in the text above: continued-pretraining runs now treat the released weights as a selection candidate and validate on 5 RTE grids; voltage inference has a generator-controlled rerun and a family-matched whole-versus-piece comparison; the classical-reference verdicts were reproduced; and the training-set size and the share from GENCO's own families were corrected. If you find a number that does not match its result file, please tell us.

## 8. Reproduce it

Code and result files: <https://github.com/mahrouqi1/grid-operations-genco>. Code is under Apache-2.0 and result files and figures under CC BY 4.0. Third-party code, data and checkpoints keep their own licences and are not redistributed.

Run `bash reproduce.sh` in the pinned environment (`environment-genco.lock.yml`; `REPRODUCE.md` has the setup) to check files against their checksums, check the environment, rebuild the figures and run the tests. It prints PASS or FAIL per step. This does not rerun the scoring; the levels below say what that needs.

**In the repo:** the experiment code, the RepairKit and a minimal `core/grid_ops`; the unit tests; the small result files (reports, tables, json, csv) that each number in this post rests on; the figure scripts and committed figures; the lock files and `CHECKSUMS.sha256`.

**Not in the repo, and how to regenerate it.** `OMITTED.md` lists each item with its command.
- Generated ladder data, checkpoints, training logs and per-scenario tables. Regenerate with the modules named in `OMITTED.md`, for example `python -m genco_challenge.ladder_data` (the full command line is in that file).
- The corrected pretraining tables. `EXTERNAL-DATA.md` gives their size and sha256. Regenerate and check them with:

```
python -m genco_challenge.repairkit repair <raw_dir> <out_dir>
python -m genco_challenge.repairkit verify <raw_dir> <out_dir>
python tools/check_external.py corrected-pretraining <path to the regenerated tables>
```

- The released GENCO checkpoints and the authors' pretraining data. These come from the authors, not from us.

**Three levels of reproduction.**

| Level | Needs | What it shows |
|---|---|---|
| 1 | the pinned environment, no GPU, no external data | `bash reproduce.sh`: files match checksums, the environment matches the lock, figures rebuild, tests pass |
| 2 | level 1, plus Julia with PowerModels, PGLib-OPF, and the authors' datakit and graphkit branches | the large data sets regenerate from public inputs, and `tools/check_external.py` compares them |
| 3 | level 2, plus the released GENCO checkpoints and a GPU | the scored numbers themselves; the committed result files are the record |

Regeneration is not seed-free from end to end. Datakit draws the scenarios and Julia solves them on the host, so a byte difference in a regenerated parquet file is possible. Compare tables before calling it a failure. The level 1 command does not rerun the scoring; that needs level 3.

## What we'd like to explore with the community

We would be glad to work on any of these with GENCO's authors and other users. GENCO as a learned prior inside WLS state estimation is one: GENCO alone lost ground once the estimator was added as a comparator. A hybrid of physics and GENCO is another, since a classical method won when every input was known. A third is solvability-aware training: about 3% of the sampled pretraining labels are collapsed roots, and some recovered solutions remain at low voltage. A fourth is a cleaner test of whether unfamiliar grid families limit transfer, since our SimBench-HV control could not separate that from the training recipe. If any of these overlaps with your work, or if you have a number that disagrees with ours, write to mahrouqi.1@osu.edu.
