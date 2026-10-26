# 12,000-update contrast on the locked 15 (contrast_12k.py; audit R1-F4 prepared result; UNAUDITED)

Released median branch-angle ratio 3.692.

| arm | median seed-mean angle ratio | change vs released | grids better than released (angle) | grids with \|V\| error worse than released |
|---|---|---|---|---|
| C00 | 4.477 | +21.3% | 1 | 14 |
| C01 | 4.451 | +20.6% | 3 | 12 |
| D | 5.703 | +54.5% | 0 | 14 |
| a | 6.909 | +87.1% | 1 | 12 |
| b | 5.852 | +58.5% | 0 | 12 |
| c | 7.131 | +93.2% | 0 | 13 |
| m | 7.598 | +105.8% | 0 | 12 |

Data effect at 12k: D vs C00 (random mask), m vs C01 (50/50). Mask effect at 12k: C01 vs C00 (no ladder), a/c/m vs D (with ladder); b at 12k is still pure pretraining mask (equivalent to D).
