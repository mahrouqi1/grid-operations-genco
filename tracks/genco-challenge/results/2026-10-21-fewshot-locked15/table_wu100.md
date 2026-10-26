| grid | buses | source | draws | dVa DC (deg) | dVa fd2 (deg) | dVa GENCO (deg) | ratio to DC (mean) | draws <DC | ratio to fd2 (mean) | |V| ratio DC | p-res ratio DC | verdict |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| case5_pjm | 5 | 2026-10-21 | 3 | 0.386 | 0.0146 | 0.033 | 0.09 | 3/3 | 2.3 | 0.14 | 0.01 | beats DC |
| case14_ieee | 14 | 2026-10-10 | 3 | 0.411 | 0.0189 | 0.230 | 0.56 | 3/3 | 12.2 | 0.23 | 0.09 | beats DC |
| case24_ieee_rts | 24 | 2026-10-21 | 3 | 0.272 | 0.0101 | 0.123 | 0.45 | 3/3 | 12.1 | 0.19 | 0.07 | beats DC |
| case30_ieee | 30 | 2026-10-10 | 3 | 0.288 | 0.0085 | 0.178 | 0.62 | 3/3 | 21.0 | 0.33 | 0.17 | beats DC |
| case39_epri | 39 | 2026-10-21 | 3 | 0.254 | 0.0234 | 0.390 | 1.53 | 0/3 | 16.7 | 0.43 | 1.35 | loses to DC |
| case57_ieee | 57 | 2026-10-10 | 3 | 0.389 | 0.0308 | 0.288 | 0.74 | 3/3 | 9.3 | 0.31 | 0.27 | beats DC |
| case73_ieee_rts | 73 | 2026-10-21 | 3 | 0.311 | 0.0101 | 0.238 | 0.76 | 3/3 | 23.5 | 0.07 | 0.11 | beats DC |
| case89_pegase | 89 | 2026-10-21 | 3 | 0.360 | 0.0168 | 0.316 | 0.88 | 3/3 | 18.8 | 0.37 | 0.05 | beats DC |
| case118_ieee | 118 | 2026-10-10 | 3 | 0.354 | 0.0209 | 0.432 | 1.22 | 0/3 | 20.7 | 0.05 | 0.11 | loses to DC |
| case162_ieee_dtc | 162 | 2026-10-21 | 3 | 0.240 | 0.0204 | 0.278 | 1.16 | 0/3 | 13.7 | 0.40 | 0.38 | loses to DC |
| case240_pserc | 240 | 2026-10-21 | 3 | 0.968 | 0.4503 | 1.337 | 1.38 | 0/3 | 3.0 | 0.27 | 1.24 | loses to DC |
| case300_ieee | 300 | 2026-10-21 | 3 | 0.567 | 0.0273 | 0.970 | 1.71 | 0/3 | 35.5 | 0.37 | 0.25 | loses to DC |
| case1354_pegase | 1354 | - | 0 | | | | | | | | | NOT RUN: datakit labels collapse (fewshot.py: 69/97 eval, ~76% train); fewshot study excluded it by design |
| case2736sp_k | 2736 | - | 0 | | | | | | | | | NOT RUN: training-pool generation (1200 scen) projected well over the 1 h budget (ladder: 2,735 s for 200 scenarios, only 74 survive) |
| case3120sp_k | 3120 | - | 0 | | | | | | | | | NOT RUN: same (1,070 s for 200 scenarios, 99 survive) |
