# Figure and table manifest

Every figure and table, the script that makes it, the result files the script reads (all in this repo)
and the files it writes (`figures/` next to the script). `bash reproduce.sh figures` rebuilds all of them;
`CHECKSUMS.sha256` pins the shipped outputs.

## Fig a: voltage inference: GENCO error over best interpolator error per grid, and over WLS error (mask seed 0)

- script: `tracks/genco-challenge/results/2026-10-25-figure-a-v35/make_fig_a_v35.py`
- reads:
  - `tracks/genco-challenge/results/2026-10-21-genco-voltage-inference/verdict.json`
  - `tracks/genco-challenge/results/2026-10-21-genco-voltage-inference-oos/verdict.json`
  - `tracks/genco-challenge/results/2026-10-21-genco-voltage-inference-control/verdict.json`
  - `tracks/genco-challenge/results/2026-10-21-genco-voltage-inference-family-vs-size/verdict.json`
  - `tracks/genco-challenge/results/2026-10-21-genco-voltage-inference-generator-control/verdict.json`
  - `tracks/genco-challenge/results/2026-10-21-genco-voltage-inference-oos/protocol.json`
  - `tracks/genco-challenge/results/2026-10-21-genco-voltage-inference-generator-control/protocol.json`
  - `tracks/genco-challenge/results/2026-10-21-genco-voltage-wls/cells_v2.csv`
  - `tracks/genco-challenge/results/2026-10-21-genco-voltage-wls/cells_v3.csv`
  - `tracks/genco-challenge/results/2026-10-21-genco-voltage-wls/wls_only_v3.csv`
  - `tracks/genco-challenge/results/2026-10-25-figure-a-v35/q_g_ratios_per_grid_v35.csv`
- writes (.png):
  - `tracks/genco-challenge/results/2026-10-18-genco-challenge-post/figures-v3/fig_a_voltage_inference_v35`

## Fig b: 12k-update contrast of the continued-pretraining arms (transfer study)

- script: `tracks/genco-challenge/results/2026-10-24-figure-redesign/make_figs_v3.py`
- reads:
  - `tracks/genco-challenge/results/2026-10-20-genco-transfer/contrast_12k.json`
- writes (.png):
  - `tracks/genco-challenge/results/2026-10-18-genco-challenge-post/figures-v3/fig_b_transfer_12k`

## Fig c: collapse reproduction versus updates, corrected versus original labels, with held-out error

- script: `tracks/genco-challenge/results/2026-10-24-figure-redesign/make_figs_v3.py`
- reads:
  - `tracks/genco-challenge/results/2026-10-20-genco-continued-pretraining/pooled.json`
  - `tracks/genco-challenge/results/2026-10-20-genco-continued-pretraining/preservation.json`
- writes (.png):
  - `tracks/genco-challenge/results/2026-10-18-genco-challenge-post/figures-v3/fig_c_unlearning`
