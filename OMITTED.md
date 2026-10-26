# Omitted from this repo

Large or non-essential artifacts are left out. Each line says what and why; the regeneration
commands need the authors' released checkpoints and datakit/graphkit genco-paper-repro branches,
PGLib-OPF v23.07 and the other external inputs named in README.md.

## Not exported, with the command that regenerates it

- generated ladder data (the 21 transfer-study training grids, 1.7M verified points; the RTE validation trees) and the real-grid operating-point shards: shards: powerflow_surrogate.generate (shipped) driven by run_generate in the private powerflow-surrogate package (not exported); then python -m genco_challenge.ladder_to_datakit ; python -m genco_challenge.pglib_rate_a_fix ; python -m genco_challenge.renumber_raw (see each module docstring; needs the datakit/graphkit genco-paper-repro branches and PGLib-OPF v23.07)
- continued-pretraining checkpoints and the per-arm mlruns of 2026-10-20-genco-transfer (about 67 GB, 24 GB in val/ alone), 2026-10-20-genco-continued-pretraining and the retrain arms: the SLURM job scripts are not exported; the per-arm results.jsonl, run_meta.json and tally.md ARE shipped. Scoring a checkpoint: python -m genco_challenge.score_ckpts --out DIR --ckpt NAME=path/to/state.pt ; training plugins genco_challenge.mask_mix, node_budget, save_at_steps (FEWSHOT_EXTRA_PLUGIN)
- 2026-10-21-genco-voltage-inference/interp_per_scenario.jsonl (9.6 MB, over the size limit; verdict.py of that dir reads it): python run_interp.py in 2026-10-21-genco-voltage-inference (needs the generated trees); the verdict.json and table.md it produced are shipped
- the per_grid/ per-scenario outputs of the five voltage-inference dirs (interp_*.jsonl and genco_*.jsonl, about 110 MB; the verdict.json and table.md of each dir are shipped) and the corrector-at-scale per-scenario .npz files: run_interp.py and run_genco_o.py (stage_pieces.py / gen_ladder.py for the generator-control pieces) in each voltage-inference result dir; run_eval.sh in corrector-at-scale/scripts. verdict.py of each dir recomputes the verdict from them
- corrected pretraining tables (9 subgrids, about 2.1 GB): python -m genco_challenge.repairkit repair <raw_dir> <out_dir>, or tracks/genco-challenge/results/2026-10-10-collapsed-repair-data/repair.py; sizes and sha256 in EXTERNAL-DATA.md
- training logs, mlruns, checkpoints, job-claim files, *_DONE markers: `python -m genco_challenge.fewshot / labelfree / fresh_eval / score_ckpts (see each module docstring for the command line)`

## Left out of the result directories (listing of names, not contents)

- tracks/genco-challenge/results/2026-10-07-label-validation: meta-before/, per_grid/ (only the small files named in evidence_extra are shipped) (not in the evidence include list (logs, parquet, gz, scripts, subdirs))
- tracks/genco-challenge/results/2026-10-09-genco-on-fd2: scripts/ (only the small files named in evidence_extra are shipped) (not in the evidence include list (logs, parquet, gz, scripts, subdirs))
- tracks/genco-challenge/results/2026-10-10-collapsed-repair-data: logs/, per_grid/ (only the small files named in evidence_extra are shipped) (not in the evidence include list (logs, parquet, gz, scripts, subdirs))
- tracks/genco-challenge/results/2026-10-10-timing-table: logs/, scripts/ (only the small files named in evidence_extra are shipped) (not in the evidence include list (logs, parquet, gz, scripts, subdirs))
- tracks/genco-challenge/results/2026-10-19-genco-retrain-scoring: fewshot/, schedule-test/, zeroshot/ (only the small files named in evidence_extra are shipped) (not in the evidence include list (logs, parquet, gz, scripts, subdirs))
- tracks/genco-challenge/results/2026-10-20-genco-continued-pretraining: GPU_FREE, pmask/ (only the small files named in evidence_extra are shipped), seeds/ (only the small files named in evidence_extra are shipped), zeroshot-final/, zeroshot-step500/ (not in the evidence include list (logs, parquet, gz, scripts, subdirs))
- tracks/genco-challenge/results/2026-10-20-genco-transfer: heldout/, heldout-12k/, heldout-12k-simbench/, heldout-rescue/, heldout-rescue-final/, heldout-simbench/, identity-check/, smoke-H/, val/, val-released/, val-released-rescue/, val-rescue/, val-simbench/ (not in the evidence include list (logs, parquet, gz, scripts, subdirs))
- tracks/genco-challenge/results/2026-10-21-corrector-at-scale: logs/, per_scenario/, scripts/ (only the small files named in evidence_extra are shipped) (not in the evidence include list (logs, parquet, gz, scripts, subdirs))
- tracks/genco-challenge/results/2026-10-21-fewshot-locked15: claims/, logs/, scripts/ (only the small files named in evidence_extra are shipped) (not in the evidence include list (logs, parquet, gz, scripts, subdirs))
- tracks/genco-challenge/results/2026-10-21-fewshot-locked15-pool-fix: scripts/ (only the small files named in evidence_extra are shipped) (not in the evidence include list (logs, parquet, gz, scripts, subdirs))
- tracks/genco-challenge/results/2026-10-21-genco-robust-start: identity/ (only the small files named in evidence_extra are shipped), per_scenario/ (only the small files named in evidence_extra are shipped), timing/ (only the small files named in evidence_extra are shipped) (not in the evidence include list (logs, parquet, gz, scripts, subdirs))
- tracks/genco-challenge/results/2026-10-21-genco-voltage-inference/interp_per_scenario.jsonl (9.6 MB) (over the size limit)
- tracks/genco-challenge/results/2026-10-21-genco-voltage-inference-control: per_grid/ (not in the evidence include list (logs, parquet, gz, scripts, subdirs))
- tracks/genco-challenge/results/2026-10-21-genco-voltage-inference-family-vs-size: per_grid/ (not in the evidence include list (logs, parquet, gz, scripts, subdirs))
- tracks/genco-challenge/results/2026-10-21-genco-voltage-inference-generator-control: per_grid/ (not in the evidence include list (logs, parquet, gz, scripts, subdirs))
- tracks/genco-challenge/results/2026-10-21-genco-voltage-inference-oos: per_grid/ (not in the evidence include list (logs, parquet, gz, scripts, subdirs))
- tracks/genco-challenge/results/2026-10-21-genco-voltage-wls: locks_v2/, locks_v2q/, locks_v3/, logs/, logs_v2/, logs_v3/, ok_v3/, per_case/, per_case_v2/, per_case_v3/, tests/ (not in the evidence include list (logs, parquet, gz, scripts, subdirs))
- tracks/genco-challenge/results/2026-10-22-genco-voltage-limit-ablation: DONE_pieces, PARTIAL_seen_whole, PARTIAL_unseen_whole, locks/, per_grid/, repro/, timing/ (not in the evidence include list (logs, parquet, gz, scripts, subdirs))
- tracks/genco-challenge/results/2026-10-24-voltage-limit-counts: .gitignore (not in the evidence include list (logs, parquet, gz, scripts, subdirs))
- tracks/genco-challenge/results/2026-10-24-fromscratch-label-pair: control/ (only the small files named in evidence_extra are shipped), fig_collapse_and_error.png, l15/, pmask/ (only the small files named in evidence_extra are shipped), rte/, rte_rev/ (not in the evidence include list (logs, parquet, gz, scripts, subdirs))
