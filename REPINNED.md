# Text cleaning of the result files

111 of the shipped files had text removed or replaced on the way into this repo: private file-system paths became
placeholders such as `<REPO>`, and names, internal file pointers and internal question ids were taken out of reports, protocols and
scripts. No number in a result file was changed.

Some protocols are pinned by a sha256 (a `*.sha256` file, or a hash quoted in a later amendment). Where such a file was cleaned, the
quoted hash was recomputed on the cleaned file; the pins that moved are listed here (file, files that quote its hash).

- `tracks/genco-challenge/results/2026-10-21-fewshot-locked15/protocol.json`: quoted in 1 file(s)
- `tracks/genco-challenge/results/2026-10-21-genco-voltage-wls/launch_v2.sh`: quoted in 3 file(s)
- `tracks/genco-challenge/results/2026-10-21-genco-voltage-wls/launch_v2q.sh`: quoted in 1 file(s)
- `tracks/genco-challenge/results/2026-10-21-genco-voltage-wls/protocol_amendment_3.json`: quoted in 3 file(s)
- `tracks/genco-challenge/results/2026-10-21-genco-voltage-wls/protocol_amendment_3b.json`: quoted in 3 file(s)
- `tracks/genco-challenge/results/2026-10-21-genco-voltage-wls/protocol_amendment_3c.json`: quoted in 2 file(s)
- `tracks/genco-challenge/results/2026-10-21-genco-voltage-wls/tuning_evidence_2026-10-22.md`: quoted in 1 file(s)
- `tracks/genco-challenge/results/2026-10-26-mask-sweep/protocol.json`: quoted in 1 file(s)
