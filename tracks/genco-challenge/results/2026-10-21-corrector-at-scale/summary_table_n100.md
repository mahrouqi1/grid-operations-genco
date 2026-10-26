Branch dVa (deg), |V| MAE (pu), active residual (MW/bus) and time (ms/scenario) per grid and method, corrector n = 100.


**case2869pegase** (217 scored test scenarios, GPU batch 32)

| method | branch dVa deg | |V| MAE pu | active res MW/bus | ms/scenario |
|---|---:|---:|---:|---:|
| dc | 0.23022 | 2.74e-02 | 136 | 9.2 |
| fd2_input (corrector's start; guarded fd2) | 0.01708 | 2.17e-04 | 6.74 | 18.1 |
| fd2 | 0.01708 | 2.17e-04 | 6.74 | 18.1 |
| fd3 | 0.00652 | 8.02e-05 | 0.449 | 19.1 |
| fd4 | 0.00394 | 4.83e-05 | 0.083 | 20.2 |
| fd5 | 0.00269 | 3.33e-05 | 0.0412 | 21.2 |
| fd30 (time-matched) | 0.00002 | 3.10e-07 | 0.00015 | 48.0 |
| corrector | 0.02569 | 1.02e-03 | 9.93 | 133.7 |

**case9241pegase** (217 scored test scenarios, GPU batch 8)

| method | branch dVa deg | |V| MAE pu | active res MW/bus | ms/scenario |
|---|---:|---:|---:|---:|
| dc | 0.41389 | 2.60e-02 | 94.6 | 28.0 |
| fd2_input (corrector's start; guarded fd2) | 0.02502 | 2.78e-04 | 5.6 | 64.3 |
| fd2 | 0.02502 | 2.78e-04 | 5.6 | 64.3 |
| fd3 | 0.00543 | 6.82e-05 | 0.465 | 68.2 |
| fd4 | 0.00179 | 2.31e-05 | 0.076 | 72.4 |
| fd5 | 0.00080 | 1.05e-05 | 0.023 | 76.2 |
| fd30 (time-matched) | 0.00000 | 2.86e-08 | 1.52e-06 | 175.4 |
| corrector | 0.09284 | 7.10e-04 | 18.9 | 248.6 |

**case6470rte** (217 scored test scenarios, GPU batch 64)

| method | branch dVa deg | |V| MAE pu | active res MW/bus | ms/scenario |
|---|---:|---:|---:|---:|
| dc | 0.20107 | 3.74e-02 | 93.1 | 17.1 |
| fd2_input (corrector's start; guarded fd2) | 0.20107 | 3.74e-02 | 93.1 | 41.9 |
| fd2 | 0.10742 | 5.82e-03 | 5.54 | 41.9 |
| fd3 | 0.04826 | 3.40e-03 | 0.734 | 44.3 |
| fd4 | 0.02816 | 2.29e-03 | 0.364 | 46.7 |
| fd5 | 0.01902 | 1.73e-03 | 0.351 | 49.2 |
| fd30 (time-matched) | 0.00002 | 3.55e-06 | 0.000366 | 110.9 |
| corrector | 0.12886 | 1.29e-02 | 4.66 | 148.3 |
