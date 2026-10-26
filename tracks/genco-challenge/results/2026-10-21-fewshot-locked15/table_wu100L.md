| grid | buses | source | draws | dVa DC (deg) | dVa fd2 (deg) | dVa GENCO (deg) | ratio to DC (mean) | draws <DC | ratio to fd2 (mean) | |V| ratio DC | p-res ratio DC | verdict |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| case5_pjm | 5 | 2026-10-21 | 3 | 0.386 | 0.0146 | 0.027 | 0.07 | 3/3 | 1.8 | 0.10 | 0.01 | beats DC |
| case14_ieee | 14 | 2026-10-21 | 3 | 0.411 | 0.0189 | 0.133 | 0.32 | 3/3 | 7.1 | 0.05 | 0.07 | beats DC |
| case24_ieee_rts | 24 | 2026-10-21 | 3 | 0.272 | 0.0101 | 0.110 | 0.40 | 3/3 | 10.8 | 0.06 | 0.07 | beats DC |
| case30_ieee | 30 | 2026-10-21 | 3 | 0.288 | 0.0085 | 0.117 | 0.41 | 3/3 | 13.7 | 0.10 | 0.16 | beats DC |
| case39_epri | 39 | 2026-10-21 | 3 | 0.254 | 0.0234 | 0.337 | 1.33 | 0/3 | 14.4 | 0.19 | 1.09 | loses to DC |
| case57_ieee | 57 | 2026-10-21 | 3 | 0.389 | 0.0308 | 0.277 | 0.71 | 3/3 | 9.0 | 0.28 | 0.23 | beats DC |
| case73_ieee_rts | 73 | 2026-10-21 | 3 | 0.311 | 0.0101 | 0.256 | 0.82 | 3/3 | 25.4 | 0.03 | 0.11 | beats DC |
| case89_pegase | 89 | 2026-10-21 | 3 | 0.360 | 0.0168 | 0.307 | 0.85 | 3/3 | 18.3 | 0.05 | 0.05 | beats DC |
| case118_ieee | 118 | 2026-10-21 | 3 | 0.354 | 0.0209 | 0.370 | 1.05 | 1/3 | 17.7 | 0.04 | 0.10 | loses to DC |
| case162_ieee_dtc | 162 | 2026-10-21 | 3 | 0.240 | 0.0204 | 0.242 | 1.01 | 1/3 | 11.9 | 0.23 | 0.35 | loses to DC |
| case240_pserc | 240 | 2026-10-21 | 3 | 0.968 | 0.4503 | 1.167 | 1.21 | 0/3 | 2.6 | 0.15 | 0.98 | loses to DC |
| case300_ieee | 300 | 2026-10-21 | 3 | 0.567 | 0.0273 | 0.886 | 1.56 | 0/3 | 32.4 | 0.27 | 0.24 | loses to DC |
| case1354_pegase | 1354 | - | 0 | | | | | | | | | NOT RUN: datakit labels collapse (fewshot.py: 69/97 eval, ~76% train); fewshot study excluded it by design |
| case2736sp_k | 2736 | - | 0 | | | | | | | | | NOT RUN: training-pool generation (1200 scen) projected well over the 1 h budget (ladder: 2,735 s for 200 scenarios, only 74 survive) |
| case3120sp_k | 3120 | - | 0 | | | | | | | | | NOT RUN: same (1,070 s for 200 scenarios, 99 survive) |
