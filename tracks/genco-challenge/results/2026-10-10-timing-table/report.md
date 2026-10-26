# Equal-hardware, setup-included timing table (gate G1-4)

> **CORRECTION 2026-10-10 14:30 (audit genco-corrector-timing R1-F1, needs-rerun):** items 5 and 6 of the
> bottom line explain the gap to the paper wrongly. The paper's 28 to 29x RECOMPUTES EXACTLY from the timing
> files shipped in the cloned graphkit and datakit repos: Tiny, goc2000 best is batch 128 at 0.4734 ms
> per scenario against PowerModels best (p = 152 workers, measured parallel amortised time) 13.329 ms = 28.2x;
> goc10000 best is batch 64 at 2.0933 ms against 60.847 ms = 29.1x. Our 10.7x and 10.1x differ by two
> factors: the paper's GENCO time is 6.7 to 7.1x faster than our batch-64 forward (H100, their batch sweep to
> 16384; ours stopped at 64), and the paper's PowerModels reference is 2.5 to 2.6x FASTER than our
> single-process time divided by 64. The statement 'a further 2.7 to 2.8x on a faster GPU with compile'
> is wrong: with the paper's GENCO time and OUR PowerModels/64 the ratio would be 72x. Correct wording: the
> paper's number is internally consistent; on our hardware and protocol the same comparison gives 10.1x to
> 10.7x (forward only, batch 64, resident data), and the difference is GPU speed and the PowerModels
> reference, not an error in the paper. Also note (F2) the ladder-grid GENCO costs use the 20.07M-parameter
> pretrained model while goc2000 and goc10000 use the 1.28M-parameter Tiny model, so cross-grid statements about
> GENCO cost growth are not like for like; and (F3) at case14 PowerModels `compute_ac_pf` solve only (1.29 ms)
> is cheaper than fd2 total (1.70 ms, setup included), so 'fd2 is the cheapest on every grid' holds only with
> setup counted on both sides. The remaining findings (F4 to F8) are in the audit file; a rerun with a larger
> batch sweep is not yet done.
>
> **Added 2026-10-18 (R1-F4):** 10.7x and 10.1x use the median of n = 3 Ipopt solves from a flat PQ start; see the
> note under bottom line 5.

2026-10-10. `scripts/`, `timing.jsonl`, `tables.md` (generated from `timing.jsonl`, T0 has per-block load),
`summary.json`, `logs/`. Built by a Sonnet subagent (commit 9b60d97; it could not write this file, so the lead
chat saved its report text). UNAUDITED.

Machine: Intel i9-10920X, RTX 3090 (GPU 0 only), torch 2.8.0+cu128, Julia 1.13.1 with one thread. CPU work
pinned with `taskset -c 16-19`. Every block ran at a 1-minute load average between 9.3 and 14.3 (below the
15 threshold), but it was still a shared machine: another user's job ran on GPU 1 (72 to 88 percent
utilisation) and NFS was in use.

## Bottom line

1. One scenario at a time, setup included, **fd2 is the cheapest pipeline on every grid**: fd2 total
   (assembly, DC, two fast-decoupled iterations) is 1.70 ms at 14 buses, 7.05 at 1,354, 11.5 at 2,000 and
   49.5 at 10,000.
2. **GENCO from raw tables to prediction costs 47 to 57 ms at batch 1 on every grid.** The batch-1 forward is
   launch-bound at about 36 to 39 ms regardless of grid size.
3. **The corrector pipeline (fd2 plus GENCO) costs 49 to 107 ms**, always more than twice fd2.
4. **Against our converged fd2nr solve, GENCO (Tiny at 2,000 and 10,000 buses) wins from 300 buses up when
   batched. At 10,000 buses it also wins at batch 1 (56.5 against 239 ms). At 2,000 buses batch 1 loses
   narrowly (52.5 against 49.2 ms).** At 10,000 buses plain Newton from flat or DC converges in 0 of 30 scenarios, so fd2nr is the converged
   classical reference there.
5. **The paper's 28 to 29x is not reproduced under any protocol measurable here.** Forward only, best GPU
   batch (64), data resident, against single-process PowerModels divided by 64, GENCO Tiny gives 10.7x at
   2,000 buses and 10.1x at 10,000 (the earlier noisy-box figures were 6.0x and 3.5x). Without the "divided
   by 64" the same comparison is 687x and 644x. With conversion included at the best batch, divided by 64,
   it is 2.8x and 5.4x; at batch 1 with conversion it is 0.65x and 2.7x.
   The 10.7x and 10.1x rest on the median of only n = 3 Ipopt solves (the first sorted scenarios), started with PQ
   buses flat, while the datakit started every bus at the OPF |V| (`scripts/t_pm.py` header). The solves recorded
   at generation (264 on goc2000, 132 on goc10000) give, with the same GENCO forward time and the same division
   by 64, a median of 6.5x and 8.0x and a mean of 10.1x and 8.2x. These recorded times are Ipopt `solve_time` only (no
   parse or model build) and were taken under load 50 to 70. Our Ipopt path is wall time including the .m parse
   (298 ms at 2,000 buses, 1,341 ms at 10,000), so the two cannot be compared to attribute the difference to the
   start point. Parse alone covers about 70 percent of the 1,929 ms gap at 10,000 buses (9,598 against 7,669 ms).
   The PowerModels side here also includes reading the case from a .m file, while the GENCO side is forward only
   on resident data. The lower recorded ratios of 6.5x and 8.0x still leave "not reproduced" unthreatened (added
   2026-10-18, audit R1-F4, reworded after R2-F2).
6. The paper's number would be reproduced only in the fixed-base-case, resident-data, batched protocol on a
   faster GPU with `torch.compile`. A further 2.7 to 2.8x on top of our 10.1 to 10.7x would reach 28 to 29x,
   but that is arithmetic, not a measurement: no H100, no compile was tried. It is not reproduced with
   distinct scenarios converted from tables on this RTX 3090.

## Table (ms per scenario, median)

GOC columns use the Tiny model; Small is in `tables.md`. Ladder grids use the pretrained GENCO weights
zero-shot, so this table is cost only.

| method | case14 | case57 | case118 | case300 | case1354 | goc2000 | goc10000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| (a) GENCO forward, GPU b1 | 36.7 | 37.0 | 39.1 | 37.6 | 38.0 | 36.3 | 36.2 |
| (a) GENCO forward, GPU b16 | 2.38 | 2.36 | 2.40 | 2.30 | 7.59 | 3.40 | 15.08 |
| (a) GENCO forward, GPU b64 | 0.63 | 0.59 | 0.83 | 1.64 | 6.92 | 3.19 | 14.91 |
| (a) GENCO forward, CPU 4 threads, b1 | 37.7 | 43.4 | 55.1 | 73.6 | 232 | 98.0 | 324 |
| (b) conversion + transfer + forward + readback, b1 | 47.5 | 48.3 | 49.6 | 50.4 | 50.6 | 52.5 | 56.5 |
| (b) same, b16 | 10.2 | 9.86 | 10.2 | 10.4 | 16.4 | 13.2 | 28.8 |
| (b) same, b64 | 8.01 | 8.04 | 8.41 | 9.45 | 15.5 | 12.4 | 28.0 |
| (c) Newton from flat, total | 8.84 | 9.91 | 10.5 | 17.9 | 51.2 | 76.8 | 2139 (0/30 converge) |
| (c) Newton from DC, total | 7.05 | 10.0 | 10.8 | 18.4 | 52.8 | 79.9 | 2152 (0/30 converge) |
| (c) Newton from GENCO \|V\| + DC angle, total incl. GENCO | 55.2 | 57.8 | 58.7 | 69.6 | 87.3 | 98.3 | 315 |
| (d) fd2 total | 1.70 | 1.90 | 2.04 | 2.88 | 7.05 | 11.5 | 49.5 |
| (d) fd3 total | 1.74 | 1.94 | 2.10 | 2.98 | 7.38 | 12.0 | 52.0 |
| (e) fd2nr total | 5.45 | 6.25 | 6.67 | 12.7 | 32.2 | 49.2 | 239 |
| (f) corrector = fd2 + GENCO (b) b1 + add | 49.3 | 50.6 | 51.8 | 53.6 | 57.9 | 65.0 | 107 |
| (g) PowerModels `compute_ac_pf`, solve only | 1.29 | 2.99 | 6.18 | 15.5 | 97.4 | not label solver* | not converged* |
| (g) PowerModels fast path (parse + solve + flows) | 6.33 | 11.7 | 25.5 | 54.9 | 303 | not label solver* | not converged* |
| (g) PowerModels Ipopt path | 14.5 | 35.3 | 81.5 | 180 | 1243 | 2195 | 9598 |
| (g) Ipopt path divided by 64 (graphkit convention) | 0.23 | 0.55 | 1.27 | 2.82 | 19.4 | 34.3 | 150 |
| PowerModels recorded at generation (load 50 to 70), mean | 2.89 | 3.67 | 10.6 | 29.2 | 531 | 2065 | 7803 |

*The GOC labels came from the Ipopt path (`pf_fast: false`). On the rebuilt scenario `compute_ac_pf`
converges to a different root at 2,000 buses and fails at 10,000 (12 of 12 runs). The Ipopt path reproduces
the labels to about 5e-12 pu at both sizes. On the ladder grids `compute_ac_pf` reproduces the labels to
about 5e-12 pu.

## Speedups (reference time over method time)

The reference is the fast path on the ladder grids and the Ipopt path on the GOC grids.

| grid | PM / forward best | PM/64 / forward best | PM / (b) b1 | PM / (b) best batch | PM / fd2 | PM / fd2nr |
|---|---:|---:|---:|---:|---:|---:|
| case14 | 9.97 | 0.16 | 0.13 | 0.79 | 3.71 | 1.16 |
| case57 | 19.7 | 0.31 | 0.24 | 1.45 | 6.14 | 1.87 |
| case118 | 30.8 | 0.48 | 0.52 | 3.04 | 12.5 | 3.83 |
| case300 | 33.5 | 0.52 | 1.09 | 5.81 | 19.1 | 4.33 |
| case1354 | 43.8 | 0.68 | 6.00 | 19.6 | 43.1 | 9.44 |
| goc2000 Tiny | 687 | 10.7 | 41.8 | 177 | 191 | 44.6 |
| goc10000 Tiny | 644 | 10.1 | 170 | 342 | 194 | 40.2 |

Against our own converged Newton, fd2nr over GENCO (b) best batch is 0.68, 0.78, 0.79, 1.34, 2.08, 3.98 and
8.53 across the seven grids. Per-method and Small-model speedups are in `tables.md` (T2).

## Setup costs, separately

- Ybus csr build: 0.23 ms at 14 buses to 0.95 ms at 10,000. Full assembly (Ybus, Sbus, B', B''): 1.45 to
  5.89 ms. DC solve: 0.08 to 14.0 ms.
- GENCO graph build in memory (merge, build, normalise, collate): about 9.6 to 16.7 ms. Host-to-device
  transfer: 0.42 to 1.19 ms. Model load, once: 0.56 to 0.98 s, plus 0.14 to 0.19 s for the first CUDA call.
- PowerModels parse: 5.2 ms at 14 buses to 1,341 ms at 10,000. Julia start: about 8 s.
- graphkit's shipped one-scenario conversion (parquet read plus `.pt` write): 58 ms at 14 buses to 138 ms at
  10,000. The in-memory build is faster and reproduces the dataset item exactly.

## Caveats

- 30 timed repeats after 5 warm-ups, except: Newton at 10,000 buses used 30 instead of 40 repeats (and never
  converges); GENCO CPU batch 16 at case1354: 12; goc10000 Small GPU batch 64 forward: 12; PowerModels
  `compute_ac_pf` at goc10000: 12 (and it did not converge); the graphkit shipped conversion row: 12. Ipopt:
  10 scenarios at 14 to 300 buses, 5 at 1,354, 3 at 2,000 and 10,000; these IQRs are indicative only.
- Corrector cost only: pipeline (f) uses the pretrained GENCO weights (same architecture as the corrector);
  the trained head was not loaded, so it measures cost, not accuracy.
- The first goc10000 PowerModels run errored (`compute_ac_pf` did not converge and a fidelity read hit a
  KeyError); the script was patched and the Ipopt part rerun separately. The error row stays in
  `timing.jsonl`; the tables use the latest row per kind.
- Not tried: `torch.compile`, CUDA graphs, an H100. The batch-1 forward is launch-bound, so these would
  change the batch-1 numbers a lot.
- Unaudited: the in-memory PowerModels input rebuild is checked only against scenario-0 residuals. Whether
  the paper divides PowerModels by 64 is unverified.
