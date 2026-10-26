# GENCO infers missing bus voltages zero-shot, and 3% of its pretraining labels are collapsed power-flow roots

**Participant:** Mazin Al-Mahrouqi, PhD student, The Ohio State University · **Contact:** mahrouqi.1@osu.edu · **Code and result files:** <https://github.com/mahrouqi1/grid-operations-genco>

## Summary

The released GENCO checkpoint infers missing voltage magnitudes on new grids without fine-tuning. Once 30% or more of bus voltage magnitudes are measured, it beats the best of five graph interpolators on 11 to 12 of the 15 benchmark grids. At about 20%, the coverage it sees in pretraining, it wins 8/15, short of the bar we set in advance. The interpolators test whether GENCO adds more than smoothing the measurements over the network. A weighted-least-squares state estimator remains more accurate.

The data audit also found a separate problem: about 3% of the sampled pretraining labels are collapsed power-flow roots. GENCO reproduces these voltages under the pretraining mask but avoids them under the power-flow task. At the collapsed buses, its voltage magnitudes are closer to a recovered power-flow solution than to the stored label in all 1,083 scored scenarios. This is a comparison of voltage magnitudes, not a feasibility test of GENCO's full output.

Continued pretraining of the released checkpoint did not improve the benchmark result in our runs. For power flow on new topologies, our findings support the paper's route of grid-specific fine-tuning. We release the diagnostic results and RepairKit, which checks datakit-format datasets, attempts label repair and verifies the repaired tables, together with corrected-table checksums and the command that regenerates them.

## Setup

We use the released checkpoint (20,069,187 parameters) and GENCO's data tools, gridfm-datakit and gridfm-graphkit. We reproduced the paper's case118 zero-shot transfer number to within 1e-5 MW (11.078244 against 11.078253 published). Protocols for the voltage-inference, continued-pretraining and power-flow tests were committed before the reported comparisons; earlier experiments informed their design. The label diagnostics are exploratory.

The test sets are:

- **Locked benchmark:** 15 PGLib grids from 5 to 3,120 buses, fixed before the later experiments. It shaped earlier work, so it is a benchmark, not a blind test.
- **Carved subgrids:** 36 networks of 159 to 903 buses cut from larger grids, 19 from families in GENCO's pretraining ("seen") and 17 from families it never saw (RTE, PEGASE, Polish).
- **Whole grids:** 15 from seen families and 17 from unseen families (RTE, PEGASE, Polish, SimBench HV).

Here, "seen" refers to the grid family, not necessarily the particular grid. Voltage magnitudes are reported in per unit (pu); branch angle means the voltage-angle difference across a branch.

## 1. Zero-shot voltage inference

**Task.** Loads are known, generator dispatch is unknown, and voltage magnitudes are measured at a seeded fraction of buses. The released checkpoint predicts voltage magnitudes at the remaining buses with its weights unchanged. Its input normalizer is fitted on solved target-grid scenarios, following GENCO's preprocessing; the effect of that scaling has not been measured.

The paper's state-estimation study (section 5.3) trains a separate model for each grid from voltage-magnitude, bus injection and branch flow measurements. Here the unchanged pretrained checkpoint gets known loads and voltage-magnitude measurements only, the quantity its pretraining hides most; other measurement combinations are a useful extension.

**Baseline.** Graph interpolation estimates missing voltages from measurements and network structure without training across scenarios or grids. It tests whether GENCO improves on smoothing over the grid. The comparison uses the best per grid of five standard estimators: graph smoothing (harmonic interpolation, unweighted and admittance-weighted), nearest electrical neighbour, a load-aware correction, and Gaussian-process kriging on the network graph. All receive the same available topology, admittances, loads and measured voltages; the methods use different subsets of that information.

**Measurement coverage.** GENCO's pretraining hides each bus voltage with probability 0.8, leaving about 20% visible. We fixed three coverage levels in advance, with 20% as the primary test: on the locked 15 GENCO wins 6/15, 8/15 and 12/15 at 10%, 20% and 40%, against a pre-set bar of 10, so the primary test falls short. A descriptive sweep at mask seed 0, run afterwards under its own committed protocol, gives the shape: on the locked 15, 6/15 at 10%, 7/15 at 20%, 11/15 at 30% and 12/15 at every level from 40% to 90%; on the 36 carved subgrids, 27/36 at 20%, a peak of 33/36 at 50% and 60%, and 27/36 at 90%. GENCO never beats the interpolators on case240, case300 or case3120 at any level. Several grids sit near the significance boundary, so a one-grid difference between levels is within noise. The rest of this section is at 40%.

![Share of grids where GENCO beats the best interpolator against the measured fraction, for the locked 15 and the 36 carved subgrids (mask seed 0, same win rule; a descriptive sweep run after the pre-set tests). The dashed segment is the bar set in advance for the locked-15 test at 20% (10/15); the pre-set test pools three mask seeds and gave 8/15, while the seed-0 sweep shown here gives 7/15; the dotted line marks the coverage GENCO sees in pretraining. Counts at 40% can differ by one grid from the table below, which pools mask seeds for the carved subgrids.](tracks/genco-challenge/results/2026-10-18-genco-challenge-post/figures-v5/fig_sweep_v5.png)

Error is the median across scenarios of the mean absolute voltage-magnitude error on unmeasured buses. A win requires GENCO's error to be lower than the best interpolator's, with the upper end of the 95% bootstrap interval of their error ratio below 1.

| Test set (40% measured) | GENCO wins |
|---|---|
| Carved subgrids, all | 33/36 |
| Seen families | 19/19 |
| Unseen families | 14/17 |
| Same subgrids, a second scenario generator | 30/36 |
| Locked benchmark | 12/15 |
| Whole grids, seen families | 15/15 |
| Whole grids, unseen families | 8/17 |

With 0.5% measurement noise, GENCO still wins 23/36 carved subgrids, 8/15 seen-family whole grids and 2/17 unseen-family whole grids.

Performance is stronger on families represented in pretraining. With a common scenario generator, GENCO wins 19/19 seen pieces and 11/17 unseen pieces (p = 0.006, secondary Fisher's exact test). The pieces come from 7 and 6 parent grids, so they are not independent; a permutation test over parents gives p = 0.021. Two pre-specified tests did not reach significance (p = 0.095 and 0.245). GENCO also wins 12/15 locked grids from families absent from pretraining. The pattern may reflect family familiarity, but we did not test this directly.

![GENCO's voltage-magnitude error divided by the best interpolator's error at 40% measured. Each dot is a grid; each bar is a group median. Error is the median across scenarios of mean absolute error on unmeasured buses. Ratios below 1 favour GENCO. Filled dots are significant GENCO wins: the ratio and the upper end of its 95% bootstrap interval are below 1. Hollow dots are not significant wins; hollow dots above 1 are grids where GENCO's error is higher. Counts above show wins without noise; counts below show wins with 0.5% measurement noise.](tracks/genco-challenge/results/2026-10-18-genco-challenge-post/figures-v5/fig_voltage_v5.png)

**Comparison with a state estimator.** We also ran a weighted-least-squares (WLS) estimator using known loads, the grid model and generator limits. Accuracy comparisons use the same scenarios and masks: mask seed 0, the first 30 scenarios per grid and 10 on whole grids. A mask seed fixes which buses are measured.

WLS builds a dispatch prior from generator limits. We tested true limits and an active-limit-blind variant that sets every generator's Pmax to the median and Pmin to 0, retaining reactive limits. GENCO receives true limits in both comparisons. WLS was allowed 40 iterations in its first stage and 60 in its second (40/60). The table reports the median across grids of the ratio of GENCO's error to WLS error; values above 1 favour WLS. These ratios use the estimator outputs, including outputs from solves that did not converge.

| Set | WLS, true limits | WLS, active-limit-blind |
|---|---|---|
| Carved subgrids | 10.3 | 3.6 |
| Locked 15 | 2.8 | 1.4 |
| Whole grids, seen | 8.8 | 1.16 |
| Whole grids, unseen | 6.2 | 1.39 |

When WLS with true limits joins the interpolators, GENCO wins 2/36 carved subgrids and 1/15 locked grids. With active-limit-blind WLS, the respective win counts are 4 carved, 2 locked, 1 seen-family whole and 2 unseen-family whole grids.

Convergence was evaluated on the full WLS run, which includes more scenarios than the accuracy-comparison subset. With true limits, WLS fails on 17% of locked-benchmark scenarios, 11% and 30% of seen and unseen pieces, and 49% of whole-grid scenarios. The active-limit-blind failure rates are unweighted means across grids over the accuracy-comparison scenarios: 35% on pieces, 25% on the locked benchmark, 81% on seen-family whole grids and 44% on unseen-family whole grids. Replacing each failed true-limit solve with the fixed I2 interpolator, harmonic interpolation weighted by branch admittance, gives GENCO 4/36 carved-subgrid wins and 1/15 locked-grid wins.

The iteration budget matters. At the initial 8/12 limit, 60% of solves with true limits hit the cap; the active-limit-blind version hit it on 88% of piece scenarios and 99% of seen-family whole-grid scenarios. WLS is more accurate than interpolation, but it needs generator information and fails to converge on many scenarios.

## 2. Collapsed labels in the pretraining data

**Finding.** Of 34,274 sampled pretraining scenarios, 1,085 contain a voltage magnitude below 0.5 pu or a negative stored magnitude. Of these, 1,069 (3.1%) have collapsed magnitudes and 16 have sign flips only. In 941 of the 1,085, the lowest bus voltage is within 0.1 pu of zero. These roots can satisfy the power-flow equations while being unsuitable labels for the intended operating state.

**Origin.** The data generator uses PowerModels' fast power-flow call, a trust-region solver from a flat start, consistent with GENCO's published data log. Rerunning that call on GENCO's raw tables reproduces the stored label to below 1e-6 pu on 51/51 sampled collapsed scenarios across all 9 affected subgrids. This locates the problem in label generation.

On PGLib grids we generated with the same call, we traced one mechanism: on case1354, the solver reaches a Vm = 0 root at a zero-injection leaf bus with no load or generator. That bus accounts for 93% of collapsed-bus observations there; on case240, case2736sp and case2746wop every collapsed bus is zero-injection. The RTE grids and case2869 show a broader family of low or negative voltages. This mechanism has not been traced on GENCO's published subgrids.

Switching to PowerModels' Newton method repaired 106/110 collapsed scenarios on our generated grids, but only 15/51 on GENCO's published subgrids, where it mostly fails to converge. A solver-method change alone is therefore insufficient on the published examples. Our relabeling procedure recovered acceptable labels for all flagged scenarios in the local sample, with sign-only cases checked separately.

**Task-dependent model response.** Under the pretraining mask, the released checkpoint predicts collapsed voltages at 96% of hidden collapsed buses, and at least 75% on each affected subgrid. Under the power-flow task, it predicts a collapsed voltage at 0/1,143 hidden collapsed buses. At those buses, its magnitude is a median 0.008 pu from a recovered solution; the stored label is a median 1.135 pu away.

We recovered solutions using Newton's method from flat and DC starts, neighbouring scenarios, and continuation from healthy scenarios. Every recovered voltage is at least 0.5 pu and the power mismatch is below 1e-9 pu in all 1,083 scored scenarios. Two starts agree for 1,070; for the remaining 13, the stored label satisfies the equations after its sign error is removed. At the 1,156 flagged buses, 1,148 recovered magnitudes are at least 0.9 pu, so the 0.5 pu threshold separates the groups cleanly.

Still, 40 recovered solutions have a voltage below 0.70 pu, and solutions need not be unique. GENCO's magnitudes at these buses are close to a recovered solution. The difference between tasks could come from pretraining or from the extra inputs in the power-flow task; we did not separate them.

**Continued pretraining on corrected labels.** Over three training seeds, the share of hidden collapsed buses still predicted below 0.5 pu falls to 0.80 (0.70 to 0.83), compared with 0.94 using original labels and 0.97 for the released checkpoint. Held-out branch-angle error nevertheless worsens with both label sets: a median 1.16 times the released model's error with corrected labels and 1.33 times with original labels. None of these continuations is an improved checkpoint.

The run lasted 4,960 updates, about 0.06% of the paper's 8.73M. For training seed 200, the collapsed-prediction share falls from 0.97 to 0.91 at 1,000 updates and 0.84 at the end. No checkpoint between those last two was scored, so the curve does not establish when that later reduction occurred. Most hidden collapsed buses are still predicted below the threshold at the end; longer training was not tested.

![Share of hidden collapsed buses predicted below 0.5 pu during continued pretraining on corrected or original labels. Training seed 200, mean of three mask seeds; the individual mask seeds (thin lines and band) nearly coincide. Dashed segments span 1,000 to 4,960 updates, where no checkpoint was scored.](tracks/genco-challenge/results/2026-10-18-genco-challenge-post/figures-v5/fig_corrected_v51.png)

## 3. Continued pretraining for power-flow transfer

We continued pretraining the released checkpoint on 37,000 generated operating points from 21 grids, together with GENCO's corrected original data (7,929 graphs), mostly on the Ohio Supercomputer Center cluster (73.4 GPU-hours for the first set). The first set tested 7 recipes, varying masks and data mixes, with 3 seeds each. A second set tested a lower learning rate, a frozen backbone and 9 times as much original data. None of these 30 runs improved on the released model on the locked 15.

Protocols, including a leakage check of the training grids against the benchmark grids, were frozen before the runs. At matched 12,000 updates, branch-angle error was 21% worse with original data only and 54.5% worse with generated data. This compares training recipes, including their data and batching, rather than isolating a data effect. The run that validation selected was 50% worse than the released model on branch angle.

About three quarters of the training graphs came from GENCO's pretraining families. A further control used SimBench HV grids, which GENCO never saw, together with corrected original data. It also remained worse than the released checkpoint (4.4 times DC at 12,000 updates, against 3.7). None of the continuations improved transfer. Because the recipes also changed the data, we cannot say whether grid-family novelty or the recipe is responsible.

## 4. Power flow on new topologies

In the power-flow task the model is given the loads, the active-power output of the non-reference generators, the PV-bus voltage setpoints and the reference angle, and predicts the remaining voltages and angles. The released checkpoint's branch-angle error is a median 3.69 times DC power flow on the locked 15. This extends the paper's stated zero-shot transfer limitation across the benchmark; the paper shows that about 1,000 grid-specific fine-tuning samples are needed to beat DC on IEEE 118. Our continued-pretraining recipes did not close that gap.

Two fast-decoupled iterations from a DC start (fd2) provide a stronger classical reference than DC alone. No GENCO variant beat guarded fd2, which falls back to DC when its mismatch is larger, on branch angle or voltage magnitude; few-shot GENCO beat raw fd2 on one grid (case240), where raw fd2 was itself worse than DC. It takes roughly 10 to 65 ms per scenario on grids of 1,900 to 9,300 buses on a shared workstation. With 100 labels, fine-tuned GENCO beats DC on 7/12 tested grids. The results support grid-specific fine-tuning, with guarded fd2 as the reference to beat alongside DC.

## Limitations

- The locked benchmark shaped earlier work, and carved subgrids share relatively few parent grids.
- The unchanged checkpoint uses preprocessing fitted on solved target-grid scenarios; its accuracy without that information has not been measured.
- WLS was tested at 40% measured, without noise, with one mask seed; its parameters were tuned on two scored grids.
- The zero-injection mechanism was traced on our generated PGLib grids, not on GENCO's published subgrids or the RTE grids.
- The solver-cause reports, the additional label-training seeds and the SimBench control are still marked unaudited in their reports. The identity check for the added training seeds was changed after seeing results; the seed report in the repo documents it.

AI-assisted review sessions checked the result files and claims in repeated rounds; independent replication is welcome.

## Code and data

Code, result files, figure scripts and reproduction instructions: <https://github.com/mahrouqi1/grid-operations-genco> (Apache-2.0 for code, CC BY 4.0 for results and figures). `bash reproduce.sh` checks the released files, rebuilds figures and runs tests; the README documents the additional requirements for rerunning scoring.

## Next steps

Natural extensions are GENCO as a learned prior inside WLS state estimation, a datakit power-flow procedure that avoids collapsed roots on the published subgrids, and a transfer experiment that separates grid-family novelty from training recipe. We welcome collaboration with GENCO's authors and other users on these extensions: mahrouqi.1@osu.edu.
