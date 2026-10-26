# grid-operations-genco

Code, small result files and figure scripts behind the post "GENCO infers missing bus voltages zero-shot, and 3% of its pretraining labels are collapsed power-flow roots", written for the GridFM GENCO Challenge.

- Post on Hugging Face: <https://huggingface.co/spaces/gridfm/genco-challenge/discussions/35>
- The same text is in this repo as [`ENTRY.md`](ENTRY.md); where this summary and ENTRY.md differ, ENTRY.md is the reference.
- Author: Mazin Al-Mahrouqi, The Ohio State University (mahrouqi.1@osu.edu).

## What the study found

1. About 3% of the sampled pretraining labels are collapsed power-flow roots, produced by the data generator's fast power-flow call. Under its masked pretraining task GENCO reproduces them; on the power-flow task its voltage magnitudes at those buses are closer to a recovered power-flow solution than to the stored label.
2. Without fine-tuning, GENCO infers missing bus voltage magnitudes better than the best of five graph interpolators once 30% or more of bus voltages are measured (11 to 12 of 15 benchmark grids); at about 20%, the coverage it sees in pretraining, it falls short of the bar set in advance (8/15). A weighted-least-squares state estimator remains more accurate.
3. Continued pretraining of the released checkpoint did not improve it on a 15-grid benchmark in my runs.
4. Zero-shot power flow with every input known matches the limitation the GENCO paper states: the released checkpoint is less accurate than DC power flow without grid-specific fine-tuning.

Also here: the RepairKit (`tracks/genco-challenge/src/genco_challenge/repairkit/`), which checks gridfm-datakit-format datasets, attempts label repair and verifies the repaired tables.

## Reproduce

```
conda env create -f environment-genco.lock.yml     # env name: grid-operations-genco
conda activate grid-operations-genco
export PYTHONNOUSERSITE=1
bash reproduce.sh
```

`reproduce.sh` needs no GPU and no private data. It runs four steps, prints PASS or FAIL for each, then `RESULT: PASS` or `RESULT: FAIL` (exit code 0 only if all passed):

1. `checksums`: every file in the repo matches `CHECKSUMS.sha256`.
2. `env`: `tools/check_env.py` compares every pinned package with what is installed.
3. `figures`: rebuilds the three figures of the post (`tracks/genco-challenge/results/2026-10-18-genco-challenge-post/figures-v5/`) from the committed result files.
4. `tests`: runs the unit tests (`pytest -m "not slow"`) on synthetic fixtures.

One step alone: `bash reproduce.sh checksums` (or `env`, `figures`, `tests`). Run the checksum step before the figures step, because the figures are rebuilt in place. [`FIGURES.md`](FIGURES.md) lists each figure with its script, the result files it reads and the files it writes. [`REPRODUCE.md`](REPRODUCE.md) has the environment notes and the two further levels (regenerating the large data sets, and re-scoring with the released checkpoints on a GPU).

The three PNGs in the repo are rendered in the pinned environment (matplotlib 3.11.2), so `reproduce.sh` rebuilds them byte for byte. They are the same files as the ones uploaded with the Hugging Face post.

The step-by-step scoring runs themselves are not rerun by `reproduce.sh`; the committed result files are the record of those numbers.

## Layout

- `core/grid_ops/`: the minimal shared substrate (test cases and the transparent Newton-Raphson reference solver used as the exact oracle).
- `tracks/genco-challenge/src/genco_challenge/`: the experiment code (data generation with the authors' datakit, scoring, continued-pretraining plugins, RepairKit, ladder-to-datakit conversion).
- `tracks/genco-challenge/results/<date>-<slug>/`: small result files (reports, tables, json, csv and the scripts that make them) that each number in the post rests on.
- `tests/`: unit tests.

## What is not in this repo

Large artifacts (per-scenario tables, generated grid data, checkpoints, logs) are left out. [`OMITTED.md`](OMITTED.md) lists each one with the command that regenerates it, and [`EXTERNAL-DATA.md`](EXTERNAL-DATA.md) gives the sizes and sha256 of the larger data sets I derived. GENCO's own code, data and checkpoints come from its authors and are not redistributed here. Regenerating the data needs the authors' released checkpoints and the `genco-paper-repro` branches of gridfm-graphkit and gridfm-datakit, PGLib-OPF v23.07, Julia with PowerModels, and a GPU.

## Licence

Code: Apache License 2.0 (`LICENSE`, `NOTICE`). Result files and figures: CC BY 4.0 (`LICENSE-DATA.md`). Third-party code, data and checkpoints keep their own licences. To cite, see `CITATION.cff`.
