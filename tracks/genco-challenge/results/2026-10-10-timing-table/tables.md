## T0. Blocks and the load average they ran under

| block | grid | ckpt | load1 before | load1 after | load1 max during | seconds | noisy (>25) | error |
|---|---|---|---:|---:|---:|---:|---|---|
| classical | case14 |  | 10.7 | 10.7 | 10.7 | 1 | False |  |
| powermodels | case14 |  | 10.7 | 10.8 | 10.8 | 33 | False |  |
| genco | case14 | pretrained | 10.9 | 11.6 | 11.9 | 42 | False |  |
| classical | case57 |  | 11.2 | 11.2 | 11.2 | 2 | False |  |
| powermodels | case57 |  | 11.1 | 11.0 | 11.4 | 33 | False |  |
| genco | case57 | pretrained | 10.6 | 10.4 | 10.9 | 47 | False |  |
| classical | case118 |  | 10.1 | 10.0 | 10.1 | 2 | False |  |
| powermodels | case118 |  | 10.0 | 10.7 | 10.7 | 37 | False |  |
| genco | case118 | pretrained | 11.3 | 10.8 | 11.4 | 57 | False |  |
| classical | case300 |  | 10.3 | 10.3 | 10.3 | 3 | False |  |
| powermodels | case300 |  | 10.6 | 11.3 | 11.3 | 42 | False |  |
| genco | case300 | pretrained | 11.2 | 11.3 | 12.0 | 78 | False |  |
| classical | case1354 |  | 11.6 | 11.2 | 11.6 | 7 | False |  |
| powermodels | case1354 |  | 11.2 | 10.7 | 11.3 | 75 | False |  |
| genco | case1354 | pretrained | 10.6 | 12.2 | 12.9 | 155 | False |  |
| classical | goc2000 |  | 12.1 | 11.9 | 12.1 | 11 | False |  |
| powermodels | goc2000 |  | 11.7 | 12.1 | 13.6 | 98 | False |  |
| genco | goc2000 | tiny_s0 | 12.0 | 11.4 | 12.0 | 64 | False |  |
| genco | goc2000 | small_s0 | 10.7 | 11.5 | 11.5 | 79 | False |  |
| classical | goc10000 |  | 11.3 | 10.3 | 11.3 | 163 | False |  |
| powermodels | goc10000 |  | 10.2 | 9.5 | 10.3 | 84 | False | JuliaError(<jl KeyError("bus")>, <jl Union{Ptr{Nothing}, Base.InterpreterIP}[Ptr{Nothing}(0x00007fc1f853aa4c), Ptr{Nothing}(0x00007fc1f83b5ab2), Ptr{Nothing}(0x00007fc1fe35e7a4), Ptr{Nothing}(0x00007fc1fe35eb3a), Ptr{Nothing}(0x00007fc225aebe6f), Ptr{Nothing}(0x00007fc1fe35d907), Ptr{Nothing}(0x00007fc1f72f9497), Ptr{Nothing}(0x00007fc1f72d1e53), Ptr{Nothing}(0x00007fc1f72ed200), Ptr{Nothing}(0x000055c7e0a9dda7), Ptr{Nothing}(0x000055c7e0a78db3), Ptr{Nothing}(0x000055c7e0a876c5), Ptr{Nothing}(0x000055c7e0aaed44), Ptr{Nothing}(0x000055c7e0a7dbac), Ptr{Nothing}(0x000055c7e0ab7643), Ptr{Nothing}(0x000055c7e0b91ee1), Ptr{Nothing}(0x000055c7e0a78db3), Ptr{Nothing}(0x000055c7e0a876c5), Ptr{Nothing}(0x000055c7e0b4bdf9), Ptr{Nothing}(0x000055c7e0b4b566), Ptr{Nothing}(0x000055c7e0b698d8), Ptr{Nothing}(0x000055c7e0b65294), Ptr{Nothing}(0x000055c7e0b7c02f), Ptr{Nothing}(0x000055c7e0b7b0b0), Ptr{Nothing}(0x000055c7e0b7adc3), Ptr{Nothing}(0x000055c7e0b74c36), Ptr{Nothing}(0x000055c7e0b3b0e4), Ptr{Nothing}(0x00007fc345790082), Ptr{Nothing}(0x000055c7e0b3af60)]>) |
| genco | goc10000 | tiny_s0 | 9.3 | 10.6 | 11.0 | 161 | False |  |
| genco | goc10000 | small_s0 | 10.2 | 11.2 | 12.1 | 207 | False |  |
| powermodels | goc10000 |  | 11.2 | 9.7 | 14.3 | 3203 | False |  |
| powermodels | goc2000 |  | 9.4 | 9.6 | 9.6 | 54 | False |  |
| powermodels | goc10000 |  | 9.6 | 11.1 | 11.2 | 80 | False |  |

Max load1 over all blocks: 14.3; blocks above 25: 0; blocks with an error: 1.

Machine (first block): {"host": "<HOST>", "cpu": "Intel(R) Core(TM) i9-10920X CPU @ 3.50GHz", "n_cpus": 24, "gpus_now": ["0, NVIDIA GeForce RTX 3090, 36 MiB, 0 %", "1, NVIDIA GeForce RTX 3090, 13633 MiB, 88 %"], "kernel": "5.15.0-194-generic", "python": "3.11.17", "affinity": [16, 17, 18, 19], "torch": "2.8.0+cu128", "cuda": "12.8"}


## T1. Per-scenario wall time, ms: median (Q1 to Q3) over scenarios (n in tables.json); one scenario at a time unless a batch is named


### case14 (14 buses)

| item | ms median (Q1 to Q3) | n | note |
|---|---:|---:|---|
| setup: Ybus csr from table | 0.23 [0.22, 0.25] | 40 | part of the assembly |
| setup: assembly (Ybus + Sbus + B' + B'') | 1.45 [1.38, 1.50] | 40 |  |
| DC solve (ours) | 0.08 [0.08, 0.08] | 40 |  |
| fd2 iterations incl. 2 LU | 0.17 [0.17, 0.18] | 40 |  |
| **fd2 total** (assembly + DC + 2 iter) | 1.70 [1.63, 1.77] | 40 |  |
| **fd3 total** | 1.74 [1.66, 1.80] | 40 |  |
| sparse Newton from flat, solve only | 7.40 [7.29, 7.69] | 40 |  |
| **sparse Newton from flat, total** (assembly + solve) | 8.84 [8.73, 9.13] | 40 | converged frac 1.00; median iters 4.0 |
| sparse Newton from DC, solve only | 5.52 [5.42, 5.83] | 40 |  |
| **sparse Newton from DC, total** (assembly + DC + solve) | 7.05 [6.93, 7.35] | 40 | converged frac 1.00; median iters 3.0 |
| fd2nr solve only (2 XB iter then Newton from DC) | 3.88 [3.85, 4.06] | 40 |  |
| **fd2nr total** (assembly + DC + fd2nr) | 5.45 [5.33, 5.58] | 40 | converged frac 1.00; median iters 4.0 |
| PowerModels parse_file (setup) | 5.24 [4.63, 7.63] | 30 |  |
| PowerModels compute_ac_pf, wall | 1.29 [1.10, 1.58] | 30 | status {'true': 30} |
| PowerModels compute_ac_pf, its own reported solve_time | 0.86 [0.74, 1.11] | 30 |  |
| PowerModels datakit fast path (parse + solve + update + flows) | 6.33 [5.35, 7.22] | 30 |  |
| PowerModels compute_ac_pf from solved start (lower bound) | 1.08 [0.97, 1.32] | 30 |  |
| (python: tables to .m file, not PowerModels) | 6.37 [5.66, 7.17] | 30 |  |
| PowerModels Ipopt path (solve_ac_pf, parse + flows) | 14.54 [13.74, 15.54] | 10 | status {'LOCALLY_SOLVED': 10} |
| PowerModels AC time recorded at generation (load 50 to 70), mean / median | 2.89 / 1.12 | 200 | datakit `solve_time`, single worker |
| **GENCO pretrained_zeroshot** | | | model load once 0.71 s (+ first CUDA call 0.19 s), 20.07M params |
| GENCO pretrained_zeroshot (a) forward only, GPU, batch 1, data resident | 36.69 [35.49, 37.74] | 30 | peak GPU 0.08 GB |
| GENCO pretrained_zeroshot (a) forward only, GPU, batch 16, data resident | 2.38 [2.34, 2.43] | 30 | peak GPU 0.09 GB |
| GENCO pretrained_zeroshot (a) forward only, GPU, batch 64, data resident | 0.63 [0.61, 0.73] | 30 | peak GPU 0.12 GB |
| GENCO pretrained_zeroshot (a) forward only, CPU 4 thr, batch 1, data resident | 37.72 [36.57, 38.53] | 30 |  |
| GENCO pretrained_zeroshot (a) forward only, CPU 4 thr, batch 16, data resident | 3.99 [3.95, 4.07] | 30 |  |
| GENCO pretrained_zeroshot (b) B=1 component: merge + build graph (in memory) | 6.24 [6.03, 6.53] | 30 | |
| GENCO pretrained_zeroshot (b) B=1 component: normalise + mask | 1.80 [1.77, 1.87] | 30 | |
| GENCO pretrained_zeroshot (b) B=1 component: collate | 1.54 [1.51, 1.57] | 30 | |
| GENCO pretrained_zeroshot (b) B=1 component: host to device | 0.42 [0.40, 0.43] | 30 | |
| GENCO pretrained_zeroshot (b) B=1 component: forward | 37.15 [36.73, 37.83] | 30 | |
| GENCO pretrained_zeroshot (b) B=1 component: device to host | 0.05 [0.04, 0.06] | 30 | |
| **GENCO pretrained_zeroshot (b) B=1 conversion + transfer + forward + D2H** | 47.45 [46.57, 48.19] | 30 | |
| **GENCO pretrained_zeroshot (b) batch 16 conversion + transfer + forward + D2H, per scenario** | 10.19 [9.96, 10.35] | 30 | |
| **GENCO pretrained_zeroshot (b) batch 64 conversion + transfer + forward + D2H, per scenario** | 8.01 [7.92, 8.14] | 30 | |
| GENCO pretrained_zeroshot graphkit shipped conversion (parquet read + process_scenarios to .pt), one scenario | 58.04 [55.96, 59.41] | 12 | read 34.01 + build/save 22.96 |
| **(f) corrector pipeline = fd2 + (b) B=1 + add** | 49.33 [48.38, 50.43] | 30 | GENCO pretrained_zeroshot; cost-equal stand-in for the trained corrector |
| **(c) Newton from vmdc total** = assembly + DC + (b) B=1 + Newton | 55.20 [54.05, 56.09] | 30 | Newton alone 5.92 [5.80, 6.11]; converged frac 1.00, median iters 3.0 (zero-shot/released weights) |

### case57 (57 buses)

| item | ms median (Q1 to Q3) | n | note |
|---|---:|---:|---|
| setup: Ybus csr from table | 0.23 [0.22, 0.24] | 40 | part of the assembly |
| setup: assembly (Ybus + Sbus + B' + B'') | 1.47 [1.42, 1.51] | 40 |  |
| DC solve (ours) | 0.13 [0.13, 0.14] | 40 |  |
| fd2 iterations incl. 2 LU | 0.28 [0.28, 0.29] | 40 |  |
| **fd2 total** (assembly + DC + 2 iter) | 1.90 [1.84, 1.94] | 40 |  |
| **fd3 total** | 1.94 [1.89, 1.99] | 40 |  |
| sparse Newton from flat, solve only | 8.43 [8.26, 8.85] | 40 |  |
| **sparse Newton from flat, total** (assembly + solve) | 9.91 [9.69, 10.33] | 40 | converged frac 1.00; median iters 4.0 |
| sparse Newton from DC, solve only | 8.37 [8.24, 8.66] | 40 |  |
| **sparse Newton from DC, total** (assembly + DC + solve) | 10.01 [9.81, 10.26] | 40 | converged frac 1.00; median iters 4.0 |
| fd2nr solve only (2 XB iter then Newton from DC) | 4.63 [4.53, 4.90] | 40 |  |
| **fd2nr total** (assembly + DC + fd2nr) | 6.25 [6.13, 6.52] | 40 | converged frac 1.00; median iters 4.0 |
| PowerModels parse_file (setup) | 11.74 [9.73, 14.42] | 30 |  |
| PowerModels compute_ac_pf, wall | 2.99 [2.69, 3.60] | 30 | status {'true': 30} |
| PowerModels compute_ac_pf, its own reported solve_time | 2.11 [1.86, 2.52] | 30 |  |
| PowerModels datakit fast path (parse + solve + update + flows) | 11.66 [11.18, 12.92] | 30 |  |
| PowerModels compute_ac_pf from solved start (lower bound) | 2.57 [2.42, 2.84] | 30 |  |
| (python: tables to .m file, not PowerModels) | 7.22 [6.31, 8.02] | 30 |  |
| PowerModels Ipopt path (solve_ac_pf, parse + flows) | 35.32 [34.08, 36.59] | 10 | status {'LOCALLY_SOLVED': 10} |
| PowerModels AC time recorded at generation (load 50 to 70), mean / median | 3.67 / 2.84 | 198 | datakit `solve_time`, single worker |
| **GENCO pretrained_zeroshot** | | | model load once 0.71 s (+ first CUDA call 0.15 s), 20.07M params |
| GENCO pretrained_zeroshot (a) forward only, GPU, batch 1, data resident | 37.01 [36.11, 38.20] | 30 | peak GPU 0.09 GB |
| GENCO pretrained_zeroshot (a) forward only, GPU, batch 16, data resident | 2.36 [2.35, 2.41] | 30 | peak GPU 0.12 GB |
| GENCO pretrained_zeroshot (a) forward only, GPU, batch 64, data resident | 0.59 [0.59, 0.60] | 30 | peak GPU 0.23 GB |
| GENCO pretrained_zeroshot (a) forward only, CPU 4 thr, batch 1, data resident | 43.44 [43.17, 43.85] | 30 |  |
| GENCO pretrained_zeroshot (a) forward only, CPU 4 thr, batch 16, data resident | 11.78 [11.68, 11.99] | 30 |  |
| GENCO pretrained_zeroshot (b) B=1 component: merge + build graph (in memory) | 6.34 [6.20, 6.53] | 30 | |
| GENCO pretrained_zeroshot (b) B=1 component: normalise + mask | 1.84 [1.80, 1.95] | 30 | |
| GENCO pretrained_zeroshot (b) B=1 component: collate | 1.65 [1.55, 1.70] | 30 | |
| GENCO pretrained_zeroshot (b) B=1 component: host to device | 0.41 [0.40, 0.46] | 30 | |
| GENCO pretrained_zeroshot (b) B=1 component: forward | 38.08 [37.63, 39.68] | 30 | |
| GENCO pretrained_zeroshot (b) B=1 component: device to host | 0.06 [0.05, 0.07] | 30 | |
| **GENCO pretrained_zeroshot (b) B=1 conversion + transfer + forward + D2H** | 48.30 [47.80, 50.46] | 30 | |
| **GENCO pretrained_zeroshot (b) batch 16 conversion + transfer + forward + D2H, per scenario** | 9.86 [9.82, 10.02] | 30 | |
| **GENCO pretrained_zeroshot (b) batch 64 conversion + transfer + forward + D2H, per scenario** | 8.04 [7.95, 8.13] | 30 | |
| GENCO pretrained_zeroshot graphkit shipped conversion (parquet read + process_scenarios to .pt), one scenario | 61.39 [57.68, 63.47] | 12 | read 36.56 + build/save 23.18 |
| **(f) corrector pipeline = fd2 + (b) B=1 + add** | 50.59 [49.84, 52.63] | 30 | GENCO pretrained_zeroshot; cost-equal stand-in for the trained corrector |
| **(c) Newton from vmdc total** = assembly + DC + (b) B=1 + Newton | 57.83 [56.45, 59.72] | 30 | Newton alone 7.05 [6.70, 7.32]; converged frac 1.00, median iters 3.0 (zero-shot/released weights) |

### case118 (118 buses)

| item | ms median (Q1 to Q3) | n | note |
|---|---:|---:|---|
| setup: Ybus csr from table | 0.23 [0.22, 0.25] | 40 | part of the assembly |
| setup: assembly (Ybus + Sbus + B' + B'') | 1.47 [1.39, 1.54] | 40 |  |
| DC solve (ours) | 0.21 [0.21, 0.21] | 40 |  |
| fd2 iterations incl. 2 LU | 0.36 [0.35, 0.37] | 40 |  |
| **fd2 total** (assembly + DC + 2 iter) | 2.04 [1.96, 2.10] | 40 |  |
| **fd3 total** | 2.10 [2.02, 2.16] | 40 |  |
| sparse Newton from flat, solve only | 9.07 [9.00, 9.24] | 40 |  |
| **sparse Newton from flat, total** (assembly + solve) | 10.50 [10.41, 10.74] | 40 | converged frac 1.00; median iters 4.0 |
| sparse Newton from DC, solve only | 9.10 [9.00, 9.25] | 40 |  |
| **sparse Newton from DC, total** (assembly + DC + solve) | 10.78 [10.65, 11.03] | 40 | converged frac 1.00; median iters 4.0 |
| fd2nr solve only (2 XB iter then Newton from DC) | 4.97 [4.94, 5.04] | 40 |  |
| **fd2nr total** (assembly + DC + fd2nr) | 6.67 [6.56, 6.76] | 40 | converged frac 1.00; median iters 4.0 |
| PowerModels parse_file (setup) | 26.42 [21.65, 29.69] | 30 |  |
| PowerModels compute_ac_pf, wall | 6.18 [5.92, 6.59] | 30 | status {'true': 30} |
| PowerModels compute_ac_pf, its own reported solve_time | 4.66 [4.50, 4.84] | 30 |  |
| PowerModels datakit fast path (parse + solve + update + flows) | 25.54 [25.01, 27.25] | 30 |  |
| PowerModels compute_ac_pf from solved start (lower bound) | 6.03 [5.89, 6.17] | 30 |  |
| (python: tables to .m file, not PowerModels) | 8.44 [7.73, 9.56] | 30 |  |
| PowerModels Ipopt path (solve_ac_pf, parse + flows) | 81.51 [78.51, 83.60] | 10 | status {'LOCALLY_SOLVED': 10} |
| PowerModels AC time recorded at generation (load 50 to 70), mean / median | 10.58 / 7.50 | 200 | datakit `solve_time`, single worker |
| **GENCO pretrained_zeroshot** | | | model load once 0.74 s (+ first CUDA call 0.16 s), 20.07M params |
| GENCO pretrained_zeroshot (a) forward only, GPU, batch 1, data resident | 39.12 [38.60, 39.90] | 30 | peak GPU 0.09 GB |
| GENCO pretrained_zeroshot (a) forward only, GPU, batch 16, data resident | 2.40 [2.36, 2.47] | 30 | peak GPU 0.17 GB |
| GENCO pretrained_zeroshot (a) forward only, GPU, batch 64, data resident | 0.83 [0.83, 0.83] | 30 | peak GPU 0.43 GB |
| GENCO pretrained_zeroshot (a) forward only, CPU 4 thr, batch 1, data resident | 55.13 [54.02, 56.16] | 30 |  |
| GENCO pretrained_zeroshot (a) forward only, CPU 4 thr, batch 16, data resident | 24.61 [24.32, 25.08] | 30 |  |
| GENCO pretrained_zeroshot (b) B=1 component: merge + build graph (in memory) | 6.38 [6.31, 6.46] | 30 | |
| GENCO pretrained_zeroshot (b) B=1 component: normalise + mask | 1.96 [1.94, 1.98] | 30 | |
| GENCO pretrained_zeroshot (b) B=1 component: collate | 1.57 [1.55, 1.62] | 30 | |
| GENCO pretrained_zeroshot (b) B=1 component: host to device | 0.42 [0.41, 0.43] | 30 | |
| GENCO pretrained_zeroshot (b) B=1 component: forward | 38.89 [38.57, 39.42] | 30 | |
| GENCO pretrained_zeroshot (b) B=1 component: device to host | 0.06 [0.06, 0.07] | 30 | |
| **GENCO pretrained_zeroshot (b) B=1 conversion + transfer + forward + D2H** | 49.55 [48.92, 50.03] | 30 | |
| **GENCO pretrained_zeroshot (b) batch 16 conversion + transfer + forward + D2H, per scenario** | 10.19 [10.09, 10.36] | 30 | |
| **GENCO pretrained_zeroshot (b) batch 64 conversion + transfer + forward + D2H, per scenario** | 8.41 [8.37, 8.52] | 30 | |
| GENCO pretrained_zeroshot graphkit shipped conversion (parquet read + process_scenarios to .pt), one scenario | 74.15 [72.44, 76.55] | 12 | read 51.14 + build/save 23.29 |
| **(f) corrector pipeline = fd2 + (b) B=1 + add** | 51.81 [51.23, 52.42] | 30 | GENCO pretrained_zeroshot; cost-equal stand-in for the trained corrector |
| **(c) Newton from vmdc total** = assembly + DC + (b) B=1 + Newton | 58.65 [58.16, 59.58] | 30 | Newton alone 7.36 [7.25, 7.46]; converged frac 1.00, median iters 3.0 (zero-shot/released weights) |

### case300 (300 buses)

| item | ms median (Q1 to Q3) | n | note |
|---|---:|---:|---|
| setup: Ybus csr from table | 0.28 [0.27, 0.29] | 40 | part of the assembly |
| setup: assembly (Ybus + Sbus + B' + B'') | 1.63 [1.59, 1.70] | 40 |  |
| DC solve (ours) | 0.42 [0.41, 0.42] | 40 |  |
| fd2 iterations incl. 2 LU | 0.83 [0.82, 0.84] | 40 |  |
| **fd2 total** (assembly + DC + 2 iter) | 2.88 [2.85, 2.96] | 40 |  |
| **fd3 total** | 2.98 [2.95, 3.06] | 40 |  |
| sparse Newton from flat, solve only | 16.27 [16.15, 16.40] | 40 |  |
| **sparse Newton from flat, total** (assembly + solve) | 17.90 [17.80, 18.16] | 40 | converged frac 1.00; median iters 5.0 |
| sparse Newton from DC, solve only | 16.28 [16.20, 16.47] | 40 |  |
| **sparse Newton from DC, total** (assembly + DC + solve) | 18.37 [18.23, 18.59] | 40 | converged frac 1.00; median iters 5.0 |
| fd2nr solve only (2 XB iter then Newton from DC) | 10.67 [7.59, 10.76] | 40 |  |
| **fd2nr total** (assembly + DC + fd2nr) | 12.69 [9.65, 12.82] | 40 | converged frac 1.00; median iters 5.0 |
| PowerModels parse_file (setup) | 40.20 [38.27, 44.70] | 30 |  |
| PowerModels compute_ac_pf, wall | 15.51 [15.01, 17.39] | 30 | status {'true': 30} |
| PowerModels compute_ac_pf, its own reported solve_time | 12.18 [11.91, 14.27] | 30 |  |
| PowerModels datakit fast path (parse + solve + update + flows) | 54.92 [53.56, 59.86] | 30 |  |
| PowerModels compute_ac_pf from solved start (lower bound) | 15.34 [14.92, 19.81] | 30 |  |
| (python: tables to .m file, not PowerModels) | 12.11 [11.25, 13.44] | 30 |  |
| PowerModels Ipopt path (solve_ac_pf, parse + flows) | 180 [173, 318] | 10 | status {'LOCALLY_SOLVED': 10} |
| PowerModels AC time recorded at generation (load 50 to 70), mean / median | 29.15 / 20.35 | 182 | datakit `solve_time`, single worker |
| **GENCO pretrained_zeroshot** | | | model load once 0.70 s (+ first CUDA call 0.17 s), 20.07M params |
| GENCO pretrained_zeroshot (a) forward only, GPU, batch 1, data resident | 37.63 [37.18, 38.50] | 30 | peak GPU 0.09 GB |
| GENCO pretrained_zeroshot (a) forward only, GPU, batch 16, data resident | 2.30 [2.29, 2.32] | 30 | peak GPU 0.28 GB |
| GENCO pretrained_zeroshot (a) forward only, GPU, batch 64, data resident | 1.64 [1.64, 1.64] | 30 | peak GPU 0.85 GB |
| GENCO pretrained_zeroshot (a) forward only, CPU 4 thr, batch 1, data resident | 73.61 [72.94, 74.54] | 30 |  |
| GENCO pretrained_zeroshot (a) forward only, CPU 4 thr, batch 16, data resident | 54.07 [52.61, 55.83] | 30 |  |
| GENCO pretrained_zeroshot (b) B=1 component: merge + build graph (in memory) | 6.58 [6.43, 6.73] | 30 | |
| GENCO pretrained_zeroshot (b) B=1 component: normalise + mask | 2.04 [1.99, 2.14] | 30 | |
| GENCO pretrained_zeroshot (b) B=1 component: collate | 1.62 [1.55, 1.80] | 30 | |
| GENCO pretrained_zeroshot (b) B=1 component: host to device | 0.46 [0.44, 0.49] | 30 | |
| GENCO pretrained_zeroshot (b) B=1 component: forward | 39.42 [38.08, 40.35] | 30 | |
| GENCO pretrained_zeroshot (b) B=1 component: device to host | 0.07 [0.06, 0.08] | 30 | |
| **GENCO pretrained_zeroshot (b) B=1 conversion + transfer + forward + D2H** | 50.36 [48.84, 51.92] | 30 | |
| **GENCO pretrained_zeroshot (b) batch 16 conversion + transfer + forward + D2H, per scenario** | 10.37 [10.21, 10.61] | 30 | |
| **GENCO pretrained_zeroshot (b) batch 64 conversion + transfer + forward + D2H, per scenario** | 9.45 [9.34, 9.61] | 30 | |
| GENCO pretrained_zeroshot graphkit shipped conversion (parquet read + process_scenarios to .pt), one scenario | 71.32 [67.07, 75.88] | 12 | read 49.39 + build/save 22.55 |
| **(f) corrector pipeline = fd2 + (b) B=1 + add** | 53.60 [52.28, 55.20] | 30 | GENCO pretrained_zeroshot; cost-equal stand-in for the trained corrector |
| **(c) Newton from vmdc total** = assembly + DC + (b) B=1 + Newton | 69.61 [68.27, 71.34] | 30 | Newton alone 17.19 [16.83, 17.61]; converged frac 1.00, median iters 5.0 (zero-shot/released weights) |

### case1354 (1354 buses)

| item | ms median (Q1 to Q3) | n | note |
|---|---:|---:|---|
| setup: Ybus csr from table | 0.38 [0.38, 0.40] | 40 | part of the assembly |
| setup: assembly (Ybus + Sbus + B' + B'') | 2.35 [2.30, 2.45] | 40 |  |
| DC solve (ours) | 1.59 [1.57, 1.63] | 40 |  |
| fd2 iterations incl. 2 LU | 3.11 [3.02, 3.15] | 40 |  |
| **fd2 total** (assembly + DC + 2 iter) | 7.05 [6.93, 7.21] | 40 |  |
| **fd3 total** | 7.38 [7.27, 7.57] | 40 |  |
| sparse Newton from flat, solve only | 48.85 [47.90, 49.46] | 40 |  |
| **sparse Newton from flat, total** (assembly + solve) | 51.21 [50.24, 51.82] | 40 | converged frac 1.00; median iters 6.0 |
| sparse Newton from DC, solve only | 48.78 [47.72, 49.51] | 40 |  |
| **sparse Newton from DC, total** (assembly + DC + solve) | 52.75 [51.60, 53.54] | 40 | converged frac 1.00; median iters 6.0 |
| fd2nr solve only (2 XB iter then Newton from DC) | 28.03 [27.36, 28.64] | 40 |  |
| **fd2nr total** (assembly + DC + fd2nr) | 32.15 [31.27, 32.64] | 40 | converged frac 1.00; median iters 5.0 |
| PowerModels parse_file (setup) | 181 [176, 190] | 30 |  |
| PowerModels compute_ac_pf, wall | 97.42 [85.97, 113] | 30 | status {'true': 30} |
| PowerModels compute_ac_pf, its own reported solve_time | 80.46 [63.08, 95.91] | 30 |  |
| PowerModels datakit fast path (parse + solve + update + flows) | 303 [290, 328] | 30 |  |
| PowerModels compute_ac_pf from solved start (lower bound) | 102 [88.84, 117] | 30 |  |
| (python: tables to .m file, not PowerModels) | 28.59 [27.58, 30.50] | 30 |  |
| PowerModels Ipopt path (solve_ac_pf, parse + flows) | 1243 [903, 1285] | 5 | status {'LOCALLY_SOLVED': 5} |
| PowerModels AC time recorded at generation (load 50 to 70), mean / median | 531 / 284 | 97 | datakit `solve_time`, single worker |
| **GENCO pretrained_zeroshot** | | | model load once 0.83 s (+ first CUDA call 0.16 s), 20.07M params |
| GENCO pretrained_zeroshot (a) forward only, GPU, batch 1, data resident | 38.04 [37.50, 38.76] | 30 | peak GPU 0.14 GB |
| GENCO pretrained_zeroshot (a) forward only, GPU, batch 16, data resident | 7.59 [7.57, 7.60] | 30 | peak GPU 1.00 GB |
| GENCO pretrained_zeroshot (a) forward only, GPU, batch 64, data resident | 6.92 [6.90, 6.95] | 30 | peak GPU 3.73 GB |
| GENCO pretrained_zeroshot (a) forward only, CPU 4 thr, batch 1, data resident | 232 [221, 237] | 30 |  |
| GENCO pretrained_zeroshot (a) forward only, CPU 4 thr, batch 16, data resident | 246 [243, 253] | 12 |  |
| GENCO pretrained_zeroshot (b) B=1 component: merge + build graph (in memory) | 7.00 [6.95, 7.07] | 30 | |
| GENCO pretrained_zeroshot (b) B=1 component: normalise + mask | 2.33 [2.30, 2.37] | 30 | |
| GENCO pretrained_zeroshot (b) B=1 component: collate | 1.71 [1.70, 1.77] | 30 | |
| GENCO pretrained_zeroshot (b) B=1 component: host to device | 0.56 [0.56, 0.58] | 30 | |
| GENCO pretrained_zeroshot (b) B=1 component: forward | 38.62 [38.49, 39.05] | 30 | |
| GENCO pretrained_zeroshot (b) B=1 component: device to host | 0.07 [0.07, 0.08] | 30 | |
| **GENCO pretrained_zeroshot (b) B=1 conversion + transfer + forward + D2H** | 50.58 [50.24, 51.03] | 30 | |
| **GENCO pretrained_zeroshot (b) batch 16 conversion + transfer + forward + D2H, per scenario** | 16.41 [16.33, 16.55] | 30 | |
| **GENCO pretrained_zeroshot (b) batch 64 conversion + transfer + forward + D2H, per scenario** | 15.46 [15.40, 15.54] | 30 | |
| GENCO pretrained_zeroshot graphkit shipped conversion (parquet read + process_scenarios to .pt), one scenario | 83.87 [81.44, 84.82] | 12 | read 54.03 + build/save 29.09 |
| **(f) corrector pipeline = fd2 + (b) B=1 + add** | 57.86 [57.42, 58.23] | 30 | GENCO pretrained_zeroshot; cost-equal stand-in for the trained corrector |
| **(c) Newton from vmdc total** = assembly + DC + (b) B=1 + Newton | 87.33 [86.73, 88.04] | 30 | Newton alone 32.55 [32.36, 32.81]; converged frac 1.00, median iters 4.0 (zero-shot/released weights) |

### goc2000 (2000 buses)

| item | ms median (Q1 to Q3) | n | note |
|---|---:|---:|---|
| setup: Ybus csr from table | 0.42 [0.42, 0.43] | 40 | part of the assembly |
| setup: assembly (Ybus + Sbus + B' + B'') | 2.69 [2.66, 2.75] | 40 |  |
| DC solve (ours) | 2.76 [2.74, 2.79] | 40 |  |
| fd2 iterations incl. 2 LU | 6.05 [6.00, 6.09] | 40 |  |
| **fd2 total** (assembly + DC + 2 iter) | 11.49 [11.41, 11.63] | 40 |  |
| **fd3 total** | 12.01 [11.91, 12.16] | 40 |  |
| sparse Newton from flat, solve only | 74.09 [73.82, 74.92] | 40 |  |
| **sparse Newton from flat, total** (assembly + solve) | 76.78 [76.60, 77.62] | 40 | converged frac 1.00; median iters 6.0 |
| sparse Newton from DC, solve only | 74.42 [73.86, 75.12] | 40 |  |
| **sparse Newton from DC, total** (assembly + DC + solve) | 79.87 [79.41, 80.61] | 40 | converged frac 1.00; median iters 6.0 |
| fd2nr solve only (2 XB iter then Newton from DC) | 43.78 [43.37, 44.00] | 40 |  |
| **fd2nr total** (assembly + DC + fd2nr) | 49.24 [48.86, 49.50] | 40 | converged frac 1.00; median iters 5.0 |
| PowerModels parse_file (setup) | 298 [292, 317] | 30 |  |
| PowerModels compute_ac_pf, wall | 141 [136, 162] | 30 | status {'true': 30} |
| PowerModels compute_ac_pf, its own reported solve_time | 112 [91.97, 129] | 30 |  |
| PowerModels datakit fast path (parse + solve + update + flows) | 573 [468, 589] | 30 |  |
| PowerModels compute_ac_pf from solved start (lower bound) | 143 [138, 220] | 30 |  |
| (python: tables to .m file, not PowerModels) | 43.77 [42.60, 45.06] | 30 |  |
| PowerModels Ipopt path (solve_ac_pf, parse + flows) | 2195 [2058, 2414] | 3 | status {'LOCALLY_SOLVED': 3} |
| PowerModels AC time recorded at generation (load 50 to 70), mean / median | 2065 / 1328 | 264 | datakit `solve_time`, single worker |
| **GENCO case2000_goc_small_s0** | | | model load once 0.89 s (+ first CUDA call 0.14 s), 5.05M params |
| GENCO case2000_goc_small_s0 (a) forward only, GPU, batch 1, data resident | 37.38 [36.72, 38.12] | 30 | peak GPU 0.08 GB |
| GENCO case2000_goc_small_s0 (a) forward only, GPU, batch 16, data resident | 5.81 [5.79, 5.82] | 30 | peak GPU 0.87 GB |
| GENCO case2000_goc_small_s0 (a) forward only, GPU, batch 64, data resident | 5.53 [5.52, 5.56] | 30 | peak GPU 3.37 GB |
| GENCO case2000_goc_small_s0 (a) forward only, CPU 4 thr, batch 1, data resident | 155 [153, 157] | 30 |  |
| GENCO case2000_goc_small_s0 (b) B=1 component: merge + build graph (in memory) | 7.51 [7.34, 7.76] | 30 | |
| GENCO case2000_goc_small_s0 (b) B=1 component: normalise + mask | 2.57 [2.54, 2.69] | 30 | |
| GENCO case2000_goc_small_s0 (b) B=1 component: collate | 1.88 [1.78, 2.00] | 30 | |
| GENCO case2000_goc_small_s0 (b) B=1 component: host to device | 0.64 [0.63, 0.66] | 30 | |
| GENCO case2000_goc_small_s0 (b) B=1 component: forward | 38.94 [38.06, 39.46] | 30 | |
| GENCO case2000_goc_small_s0 (b) B=1 component: device to host | 0.07 [0.06, 0.07] | 30 | |
| **GENCO case2000_goc_small_s0 (b) B=1 conversion + transfer + forward + D2H** | 52.05 [50.84, 52.52] | 30 | |
| **GENCO case2000_goc_small_s0 (b) batch 16 conversion + transfer + forward + D2H, per scenario** | 15.13 [15.00, 15.36] | 30 | |
| **GENCO case2000_goc_small_s0 (b) batch 64 conversion + transfer + forward + D2H, per scenario** | 14.89 [14.74, 15.14] | 30 | |
| GENCO case2000_goc_small_s0 graphkit shipped conversion (parquet read + process_scenarios to .pt), one scenario | 97.24 [94.56, 100] | 12 | read 64.64 + build/save 32.30 |
| **(f) corrector pipeline = fd2 + (b) B=1 + add** | 64.09 [62.78, 64.75] | 30 | GENCO case2000_goc_small_s0; cost-equal stand-in for the trained corrector |
| **(c) Newton from vmdc total** = assembly + DC + (b) B=1 + Newton | 97.35 [95.53, 98.33] | 30 | Newton alone 39.14 [38.28, 39.92]; converged frac 1.00, median iters 3.0 (zero-shot/released weights) |
| **GENCO case2000_goc_tiny_s0** | | | model load once 0.58 s (+ first CUDA call 0.15 s), 1.28M params |
| GENCO case2000_goc_tiny_s0 (a) forward only, GPU, batch 1, data resident | 36.32 [36.10, 36.91] | 30 | peak GPU 0.04 GB |
| GENCO case2000_goc_tiny_s0 (a) forward only, GPU, batch 16, data resident | 3.40 [3.40, 3.40] | 30 | peak GPU 0.48 GB |
| GENCO case2000_goc_tiny_s0 (a) forward only, GPU, batch 64, data resident | 3.19 [3.19, 3.20] | 30 | peak GPU 1.88 GB |
| GENCO case2000_goc_tiny_s0 (a) forward only, CPU 4 thr, batch 1, data resident | 98.01 [95.86, 99.55] | 30 |  |
| GENCO case2000_goc_tiny_s0 (b) B=1 component: merge + build graph (in memory) | 7.75 [7.46, 8.12] | 30 | |
| GENCO case2000_goc_tiny_s0 (b) B=1 component: normalise + mask | 2.65 [2.56, 2.74] | 30 | |
| GENCO case2000_goc_tiny_s0 (b) B=1 component: collate | 1.88 [1.80, 1.96] | 30 | |
| GENCO case2000_goc_tiny_s0 (b) B=1 component: host to device | 0.65 [0.63, 0.70] | 30 | |
| GENCO case2000_goc_tiny_s0 (b) B=1 component: forward | 39.00 [37.57, 40.00] | 30 | |
| GENCO case2000_goc_tiny_s0 (b) B=1 component: device to host | 0.07 [0.06, 0.08] | 30 | |
| **GENCO case2000_goc_tiny_s0 (b) B=1 conversion + transfer + forward + D2H** | 52.53 [50.52, 53.44] | 30 | |
| **GENCO case2000_goc_tiny_s0 (b) batch 16 conversion + transfer + forward + D2H, per scenario** | 13.15 [12.97, 13.29] | 30 | |
| **GENCO case2000_goc_tiny_s0 (b) batch 64 conversion + transfer + forward + D2H, per scenario** | 12.38 [12.28, 12.64] | 30 | |
| GENCO case2000_goc_tiny_s0 graphkit shipped conversion (parquet read + process_scenarios to .pt), one scenario | 97.38 [93.08, 102] | 12 | read 62.76 + build/save 33.03 |
| **(f) corrector pipeline = fd2 + (b) B=1 + add** | 64.95 [63.04, 66.12] | 30 | GENCO case2000_goc_tiny_s0; cost-equal stand-in for the trained corrector |
| **(c) Newton from vmdc total** = assembly + DC + (b) B=1 + Newton | 98.30 [95.80, 100] | 30 | Newton alone 39.83 [38.97, 40.84]; converged frac 1.00, median iters 3.0 (zero-shot/released weights) |

### goc10000 (10000 buses)

| item | ms median (Q1 to Q3) | n | note |
|---|---:|---:|---|
| setup: Ybus csr from table | 0.95 [0.92, 0.99] | 30 | part of the assembly |
| setup: assembly (Ybus + Sbus + B' + B'') | 5.89 [5.79, 5.99] | 30 |  |
| DC solve (ours) | 14.03 [13.89, 14.28] | 30 |  |
| fd2 iterations incl. 2 LU | 29.57 [29.24, 30.11] | 30 |  |
| **fd2 total** (assembly + DC + 2 iter) | 49.46 [49.11, 50.18] | 30 |  |
| **fd3 total** | 51.98 [51.61, 52.76] | 30 |  |
| sparse Newton from flat, solve only | 2133 [2105, 2152] | 30 |  |
| **sparse Newton from flat, total** (assembly + solve) | 2139 [2111, 2157] | 30 | converged frac 0.00; median iters 30.0 |
| sparse Newton from DC, solve only | 2132 [2121, 2143] | 30 |  |
| **sparse Newton from DC, total** (assembly + DC + solve) | 2152 [2141, 2164] | 30 | converged frac 0.00; median iters 30.0 |
| fd2nr solve only (2 XB iter then Newton from DC) | 219 [215, 221] | 30 |  |
| **fd2nr total** (assembly + DC + fd2nr) | 239 [235, 241] | 30 | converged frac 1.00; median iters 5.0 |
| PowerModels parse_file (setup) | 1341 [1330, 1387] | 12 |  |
| PowerModels compute_ac_pf, wall | 60757 [60359, 61489] | 12 | status {'false': 12} |
| PowerModels compute_ac_pf, its own reported solve_time | 60538 [60107, 61330] | 12 |  |
| PowerModels datakit fast path (parse + solve + update + flows) | 62613 [61337, 63247] | 12 |  |
| PowerModels compute_ac_pf from solved start (lower bound) | 61630 [60914, 61744] | 12 |  |
| (python: tables to .m file, not PowerModels) | 162 [159, 168] | 12 |  |
| PowerModels Ipopt path (solve_ac_pf, parse + flows) | 9598 [9160, 9964] | 3 | status {'LOCALLY_SOLVED': 3} |
| PowerModels AC time recorded at generation (load 50 to 70), mean / median | 7803 / 7669 | 132 | datakit `solve_time`, single worker |
| **GENCO case10000_goc_small_s0** | | | model load once 0.98 s (+ first CUDA call 0.17 s), 5.05M params |
| GENCO case10000_goc_small_s0 (a) forward only, GPU, batch 1, data resident | 37.61 [37.21, 38.78] | 30 | peak GPU 0.24 GB |
| GENCO case10000_goc_small_s0 (a) forward only, GPU, batch 16, data resident | 26.39 [26.28, 26.53] | 30 | peak GPU 3.35 GB |
| GENCO case10000_goc_small_s0 (a) forward only, GPU, batch 64, data resident | 27.70 [27.21, 27.86] | 12 | peak GPU 13.27 GB |
| GENCO case10000_goc_small_s0 (a) forward only, CPU 4 thr, batch 1, data resident | 592 [590, 594] | 30 |  |
| GENCO case10000_goc_small_s0 (b) B=1 component: merge + build graph (in memory) | 10.11 [9.97, 10.36] | 30 | |
| GENCO case10000_goc_small_s0 (b) B=1 component: normalise + mask | 4.61 [4.52, 4.73] | 30 | |
| GENCO case10000_goc_small_s0 (b) B=1 component: collate | 2.22 [2.19, 2.33] | 30 | |
| GENCO case10000_goc_small_s0 (b) B=1 component: host to device | 1.21 [1.20, 1.27] | 30 | |
| GENCO case10000_goc_small_s0 (b) B=1 component: forward | 38.33 [37.72, 39.11] | 30 | |
| GENCO case10000_goc_small_s0 (b) B=1 component: device to host | 0.09 [0.08, 0.09] | 30 | |
| **GENCO case10000_goc_small_s0 (b) B=1 conversion + transfer + forward + D2H** | 56.79 [56.18, 57.84] | 30 | |
| **GENCO case10000_goc_small_s0 (b) batch 16 conversion + transfer + forward + D2H, per scenario** | 40.04 [39.79, 40.23] | 30 | |
| **GENCO case10000_goc_small_s0 (b) batch 64 conversion + transfer + forward + D2H, per scenario** | 40.22 [40.02, 40.37] | 30 | |
| GENCO case10000_goc_small_s0 graphkit shipped conversion (parquet read + process_scenarios to .pt), one scenario | 136 [134, 143] | 12 | read 80.50 + build/save 54.82 |
| **(f) corrector pipeline = fd2 + (b) B=1 + add** | 108 [107, 109] | 30 | GENCO case10000_goc_small_s0; cost-equal stand-in for the trained corrector |
| **(c) Newton from vmdc total** = assembly + DC + (b) B=1 + Newton | 317 [314, 320] | 30 | Newton alone 238 [237, 241]; converged frac 1.00, median iters 4.0 (zero-shot/released weights) |
| **GENCO case10000_goc_tiny_s0** | | | model load once 0.56 s (+ first CUDA call 0.18 s), 1.28M params |
| GENCO case10000_goc_tiny_s0 (a) forward only, GPU, batch 1, data resident | 36.23 [35.98, 36.60] | 30 | peak GPU 0.13 GB |
| GENCO case10000_goc_tiny_s0 (a) forward only, GPU, batch 16, data resident | 15.08 [15.06, 15.13] | 30 | peak GPU 1.88 GB |
| GENCO case10000_goc_tiny_s0 (a) forward only, GPU, batch 64, data resident | 14.91 [14.78, 15.03] | 30 | peak GPU 7.44 GB |
| GENCO case10000_goc_tiny_s0 (a) forward only, CPU 4 thr, batch 1, data resident | 324 [321, 325] | 30 |  |
| GENCO case10000_goc_tiny_s0 (b) B=1 component: merge + build graph (in memory) | 9.97 [9.81, 10.18] | 30 | |
| GENCO case10000_goc_tiny_s0 (b) B=1 component: normalise + mask | 4.53 [4.44, 4.64] | 30 | |
| GENCO case10000_goc_tiny_s0 (b) B=1 component: collate | 2.22 [2.15, 2.30] | 30 | |
| GENCO case10000_goc_tiny_s0 (b) B=1 component: host to device | 1.19 [1.18, 1.22] | 30 | |
| GENCO case10000_goc_tiny_s0 (b) B=1 component: forward | 38.24 [37.43, 38.99] | 30 | |
| GENCO case10000_goc_tiny_s0 (b) B=1 component: device to host | 0.09 [0.08, 0.09] | 30 | |
| **GENCO case10000_goc_tiny_s0 (b) B=1 conversion + transfer + forward + D2H** | 56.45 [55.49, 57.60] | 30 | |
| **GENCO case10000_goc_tiny_s0 (b) batch 16 conversion + transfer + forward + D2H, per scenario** | 28.77 [28.63, 29.00] | 30 | |
| **GENCO case10000_goc_tiny_s0 (b) batch 64 conversion + transfer + forward + D2H, per scenario** | 28.04 [27.93, 28.29] | 30 | |
| GENCO case10000_goc_tiny_s0 graphkit shipped conversion (parquet read + process_scenarios to .pt), one scenario | 138 [128, 140] | 12 | read 79.17 + build/save 56.04 |
| **(f) corrector pipeline = fd2 + (b) B=1 + add** | 107 [106, 110] | 30 | GENCO case10000_goc_tiny_s0; cost-equal stand-in for the trained corrector |
| **(c) Newton from vmdc total** = assembly + DC + (b) B=1 + Newton | 315 [313, 321] | 30 | Newton alone 238 [235, 240]; converged frac 1.00, median iters 4.0 (zero-shot/released weights) |

## T2. Speedup ratios (reference time / method time; > 1 means the method is faster)

NOTE: on goc2000 and goc10000 the reference columns use the Ipopt path (the solver that produced those labels, config pf_fast: false; compute_ac_pf on the rebuilt case lands on another root at 2,000 buses and does not converge at 10,000), so cpf and fast columns are equal there. Reference PowerModels = `compute_ac_pf` wall (solve only), the datakit fast path (parse + solve + flows) and the Ipopt path, single process, measured here at low load; `/64` is the graphkit convention (reference divided by 64). Methods: our fd2 total, our fd2nr total, GENCO forward only (best GPU batch, resident), GENCO B=1 (b), GENCO best-batch (b).

| grid | GENCO ckpt | PM cpf / GENCO fwd best-batch | PM cpf/64 / GENCO fwd best-batch | PM fast / GENCO fwd best-batch | PM fast/64 / GENCO fwd best-batch | PM fast / GENCO (b) B=1 | PM fast / GENCO (b) best batch | PM fast / fd2 total | PM fast / fd2nr total | PM fast / Newton best (flat or DC) | fd2nr total / GENCO (b) best batch | Newton best / GENCO fwd best-batch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case14 | pretrained_zeroshot | 2.03 | 0.03 | 9.97 | 0.16 | 0.13 | 0.79 | 3.71 | 1.16 | 0.90 | 0.68 | 11.11 |
| case57 | pretrained_zeroshot | 5.05 | 0.08 | 19.66 | 0.31 | 0.24 | 1.45 | 6.14 | 1.87 | 1.18 | 0.78 | 16.71 |
| case118 | pretrained_zeroshot | 7.46 | 0.12 | 30.82 | 0.48 | 0.52 | 3.04 | 12.51 | 3.83 | 2.43 | 0.79 | 12.67 |
| case300 | pretrained_zeroshot | 9.46 | 0.15 | 33.47 | 0.52 | 1.09 | 5.81 | 19.07 | 4.33 | 3.07 | 1.34 | 10.91 |
| case1354 | pretrained_zeroshot | 14.08 | 0.22 | 43.83 | 0.68 | 6.00 | 19.63 | 43.06 | 9.44 | 5.92 | 2.08 | 7.40 |
| goc2000 | case2000_goc_small_s0 | 396.91 | 6.20 | 396.91 | 6.20 | 42.18 | 147.45 | 191.05 | 44.59 | 28.59 | 3.31 | 13.88 |
| goc2000 | case2000_goc_tiny_s0 | 687.36 | 10.74 | 687.36 | 10.74 | 41.79 | 177.38 | 191.05 | 44.59 | 28.59 | 3.98 | 24.04 |
| goc10000 | case10000_goc_small_s0 | 363.65 | 5.68 | 363.65 | 5.68 | 169.01 | 239.73 | 194.06 | 40.15 | n/a | 5.97 | n/a |
| goc10000 | case10000_goc_tiny_s0 | 643.61 | 10.06 | 643.61 | 10.06 | 170.01 | 342.32 | 194.06 | 40.15 | n/a | 8.53 | n/a |

Recorded-at-generation reference (mean, load 50 to 70) over the same GENCO times, and the Ipopt path:

| grid | GENCO ckpt | recorded PM mean / GENCO fwd best-batch | recorded PM mean/64 / GENCO fwd best-batch | recorded PM mean/64 / GENCO (b) best batch | Ipopt path / GENCO fwd best-batch | Ipopt path / fd2nr total |
|---|---|---:|---:|---:|---:|---:|
| case14 | pretrained_zeroshot | 4.55 | 0.07 | 0.01 | 22.92 | 2.67 |
| case57 | pretrained_zeroshot | 6.19 | 0.10 | 0.01 | 59.56 | 5.65 |
| case118 | pretrained_zeroshot | 12.76 | 0.20 | 0.02 | 98.36 | 12.21 |
| case300 | pretrained_zeroshot | 17.77 | 0.28 | 0.05 | 109.90 | 14.21 |
| case1354 | pretrained_zeroshot | 76.66 | 1.20 | 0.54 | 179.59 | 38.66 |
| goc2000 | case2000_goc_small_s0 | 373.32 | 5.83 | 2.17 | 396.91 | 44.59 |
| goc2000 | case2000_goc_tiny_s0 | 646.52 | 10.10 | 2.61 | 687.36 | 44.59 |
| goc10000 | case10000_goc_small_s0 | 295.65 | 4.62 | 3.05 | 363.65 | 40.15 |
| goc10000 | case10000_goc_tiny_s0 | 523.25 | 8.18 | 4.35 | 643.61 | 40.15 |

## T3. Fidelity of the rebuilt PowerModels inputs (max abs difference of the solved state against the datakit label, scenario 0)

- pm_fidelity case14: status true |dVm| 4.944489262470597e-12 |dVa| deg 1.447375552743324e-10 
- pm_fidelity case57: status true |dVm| 2.2875035199376725e-12 |dVa| deg 3.6844305384420295e-11 
- pm_fidelity case118: status true |dVm| 4.880762460857113e-12 |dVa| deg 1.1512568676153023e-10 
- pm_fidelity case300: status true |dVm| 4.9535930912725235e-12 |dVa| deg 3.936975190299563e-10 
- pm_fidelity case1354: status true |dVm| 4.9611426078399745e-12 |dVa| deg 1.4913803525473668e-10 
- pm_fidelity goc2000: status true |dVm| 1.042619019875363 |dVa| deg 0.6576844957990069 
- pm_fidelity goc10000: status false |dVm| n/a |dVa| deg n/a compute_ac_pf did not converge from this start
- pm_fidelity_ipopt goc2000: status LOCALLY_SOLVED |dVm| 4.986011603591578e-12 |dVa| deg 1.503783764178479e-10 
- pm_fidelity_ipopt goc10000: status LOCALLY_SOLVED |dVm| 4.9968917892329046e-12 |dVa| deg 1.0219736168437521e-10 
