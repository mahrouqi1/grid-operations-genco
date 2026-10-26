# Commands in the order they were run (repo root; env grid-operations-genco; PYTHONPATH=tracks/genco-challenge/src:<this dir>/scripts;
# nice -n 10; PATH must contain the env bin so that `gridfm_graphkit` resolves).
python scripts/write_protocol.py                     # protocol.json + protocol.sha256, BEFORE anything else
python scripts/s1_predict.py val                     # builds val sets (case14, case30), zero-shot GENCO via gk_freeze (input-only normalizer)
python scripts/s1_predict.py eval2                   # zero-shot GENCO on the 9 eval2 sets
python scripts/s1_score.py val                       # blend selection + gate input (stage1_blend_selection.json)
python scripts/s1_score.py eval --tag eval2          # stage 1 rows on eval2
python scripts/s1b_classical.py                      # cost-matched classical states (fd1..fd4, nr1..nr3)
python scripts/s2_existing.py score-dc               # SUPPLEMENTARY: 27 existing DC-prior adapted checkpoints, no GPU
python scripts/s2_existing.py eval-fd2; python scripts/s2_existing.py score-fd2   # SUPPLEMENTARY: 2 existing fd2-prior checkpoints
FEWSHOT_INPUT_NORM=1 python scripts/s0_timing.py classical|gpu|cpu               # timing.jsonl
python scripts/make_tables.py                        # tables.md from results.jsonl / timing.jsonl
# The first zero-shot run (graphkit default normalizer fit, because fewshot.evaluate bypasses the plugin when slack_vm is off) was
# moved aside and kept as a sensitivity set: results_defaultnorm_zeroshot.jsonl, tables_old_defaultnorm.md, stage1_blend_selection_defaultnorm.json.
