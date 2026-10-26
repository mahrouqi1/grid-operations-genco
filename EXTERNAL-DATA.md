# Big files that are NOT shipped

Size and sha256 are those of the authors' working copy of each set, recorded by hashing every file of the
per-grid `raw/` directories. The tree hash is the sha256 of the sorted lines `<sha256>  <bytes>  <path>`.
Per-file hashes are in `EXTERNAL-DATA.json`; compare a regenerated copy with
`python tools/check_external.py <set> <root>`. Regeneration draws scenarios with datakit and solves them in
Julia, so a byte difference in a regenerated parquet file is possible; compare tables before calling it a
failure. Needs the env, the Julia depot and the external inputs of `REPRODUCE.md`.

## corrected-pretraining

GENCO pretraining subgrids with the 1,085 collapsed or sign-flipped scenarios repaired, 9 subgrids

- location in the working tree: `data/external/genco-pf-pretraining-corrected/<grid>/raw/` (subnet_*/raw)
- files: 248, bytes: 2149548023 (2.150 GB)
- tree sha256: `a68878b521934a146a60a6551ef2cc1ff46e4dc8b95d60ee2dc6b335c0e9f970`
- public inputs: the authors' released GENCO pf pretraining subnets (data/external/genco-pf-pretraining)
- regenerate:
  ```
  python tracks/genco-challenge/results/2026-10-10-collapsed-repair-data/repair.py
  python tracks/genco-challenge/results/2026-10-10-collapsed-repair-data/verify.py   # read-back check
  ```
- check: `python tools/check_external.py corrected-pretraining <path to genco-pf-pretraining-corrected>`
