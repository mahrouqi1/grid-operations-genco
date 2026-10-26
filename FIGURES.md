# Figure and table manifest

Every figure and table, the script that makes it, the result files the script reads (all in this repo)
and the files it writes (`figures/` next to the script). `bash reproduce.sh figures` rebuilds all of them;
`CHECKSUMS.sha256` pins the shipped outputs.

## Fig 1: coverage sweep: share of grids where GENCO beats the best interpolator against the measured fraction, locked 15 and 36 carved subgrids, mask seed 0 (post figure 1)

- script: `tracks/genco-challenge/results/2026-10-26-figures-v5/make_fig_sweep_v5.py`
- reads:
  - `tracks/genco-challenge/results/2026-10-26-mask-sweep/sweep_summary.json`
- writes (.png):
  - `tracks/genco-challenge/results/2026-10-18-genco-challenge-post/figures-v5/fig_sweep_v5`

## Fig 2: voltage inference: GENCO error over best interpolator error per grid at 40% measured, win counts, and the 0.5% noise row (post figure 2)

- script: `tracks/genco-challenge/results/2026-10-26-figures-v5/make_fig_voltage_v5.py`
- reads:
  - `tracks/genco-challenge/results/2026-10-21-genco-voltage-inference/verdict.json`
  - `tracks/genco-challenge/results/2026-10-21-genco-voltage-inference-oos/verdict.json`
  - `tracks/genco-challenge/results/2026-10-21-genco-voltage-inference-control/verdict.json`
  - `tracks/genco-challenge/results/2026-10-21-genco-voltage-inference-family-vs-size/verdict.json`
  - `tracks/genco-challenge/results/2026-10-21-genco-voltage-inference-generator-control/verdict.json`
  - `tracks/genco-challenge/results/2026-10-21-genco-voltage-inference-oos/protocol.json`
  - `tracks/genco-challenge/results/2026-10-21-genco-voltage-inference-generator-control/protocol.json`
- writes (.png):
  - `tracks/genco-challenge/results/2026-10-18-genco-challenge-post/figures-v5/fig_voltage_v5`

## Fig 3: collapse reproduction versus updates, corrected versus original labels, three mask seeds (post figure 3)

- script: `tracks/genco-challenge/results/2026-10-26-figures-v5/make_fig_corrected_v51.py`
- reads:
  - `tracks/genco-challenge/results/2026-10-20-genco-continued-pretraining/pooled.json`
  - `tracks/genco-challenge/results/2026-10-20-genco-continued-pretraining/seeds/pooled_seeds.json`
- writes (.png):
  - `tracks/genco-challenge/results/2026-10-18-genco-challenge-post/figures-v5/fig_corrected_v51`
