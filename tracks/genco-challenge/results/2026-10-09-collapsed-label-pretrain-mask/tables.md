### Pretraining subgrids

| grid | coll. scen. (roots) | hidden collapsed buses | PRE: frac pred < 0.5 (3 seeds) | PRE: median pred / median label | PRE: MAE vs label | PRE: MAE vs root (label vs root) | PF: frac < 0.5 | PF: median pred | PF: MAE vs label | normal hidden PQ MAE vs label: PRE / PF |
|---|---|---|---|---|---|---|---|---|---|---|
| case4837_goc_idx_35 | 659 (69) | 525 of 665 | 0.97/0.98/0.98 | -0.050 / -0.061 | 0.097 | 1.129 (1.128) | 0.00 | 1.093 | 1.145 | 0.0017 / 0.0029 |
| case20758_epigrids_idx_49 | 272 (140) | 232 of 293 | 0.97/0.98/0.98 | 0.015 / 0.022 | 0.095 | 0.979 (0.995) | 0.00 | 1.010 | 1.016 | 0.0018 / 0.0039 |
| case9591_goc_idx_12 | 70 (3) | 59 of 72 | 0.85/0.81/0.75 | 0.474 / 0.477 | 0.069 | 0.226 (0.896) | 0.00 | 1.020 | 0.597 | 0.0037 / 0.0061 |
| case5658_epigrids_idx_8 | 28 (28) | 44 of 55 | 0.97/0.96/0.98 | -0.000 / 0.001 | 0.031 | 0.980 (0.997) | 0.00 | 1.065 | 0.967 | 0.0008 / 0.0044 |
| case4837_goc_idx_28 | 16 (16) | 11 of 16 | 0.80/0.82/1.00 | -0.037 / -0.128 | 0.218 | 0.822 (0.969) | 0.00 | 0.902 | 0.957 | 0.0014 / 0.0041 |
| case19402_goc_idx_47 | 10 (10) | 8 of 11 | 0.89/0.75/0.89 | -0.047 / -0.043 | 0.408 | 0.959 (1.240) | 0.00 | 1.097 | 1.281 | 0.0015 / 0.0029 |
| case10480_goc_idx_40 | 9 (9) | 8 of 9 | 1.00/1.00/1.00 | 0.004 / -0.004 | 0.013 | 1.077 (1.084) | 0.00 | 1.077 | 1.066 | 0.0016 / 0.0059 |
| case4837_goc_idx_37 | 6 (6) | 15 of 19 | 0.93/0.78/0.92 | 0.228 / 0.073 | 0.176 | 0.841 (1.008) | 0.00 | 1.092 | 1.021 | 0.0018 / 0.0043 |
| case10480_goc_idx_26 | 3 (0) | 2 of 3 | 1.00/1.00/1.00 | 0.014 / 0.009 | 0.013 | - | 0.00 | 1.086 | 1.082 | 0.0013 / 0.0015 |

### Held-out ladder grids

| grid | coll. scen. (roots) | hidden collapsed buses | PRE: frac pred < 0.5 (3 seeds) | PRE: median pred / median label | PRE: MAE vs label | PRE: MAE vs root (label vs root) | PF: frac < 0.5 | PF: median pred | PF: MAE vs label | normal hidden PQ MAE vs label: PRE / PF |
|---|---|---|---|---|---|---|---|---|---|---|
| case1354_pegase | 69 (69) | 56 of 73 | 0.88/0.91/0.86 | 0.009 / -0.000 | 0.134 | 0.964 (1.096) | 0.00 | 1.098 | 1.095 | 0.0034 / 0.0081 |
| case2736sp_k | 8 (8) | 6 of 8 | 0.00/0.00/0.00 | 0.991 / -0.000 | 0.992 | 0.047 (1.039) | 0.00 | 1.027 | 1.030 | 0.0036 / 0.0140 |
| case240_pserc | 3 (1) | 1 of 3 | 0.50/0.50/0.00 | 0.583 / -0.000 | 0.583 | - | 0.00 | 1.075 | 1.062 | 0.0652 / 0.0413 |
| case1888_rte | 5 (0) | 29 of 35 | 0.21/0.14/0.14 | 0.850 / -0.030 | 0.751 | - | 0.00 | 1.064 | 1.063 | - / - |
| case1951_rte | 5 (0) | 45 of 56 | 0.52/0.36/0.43 | 0.761 / 0.016 | 0.478 | - | 0.00 | 1.072 | 0.999 | - / - |

### Pooled over the 9 pretraining subgrids (weights = hidden collapsed buses)

- PRE seed 1000: hidden collapsed buses 892; frac pred < 0.5 = 0.961; MAE vs label = 0.102
- PRE seed 1001: hidden collapsed buses 921; frac pred < 0.5 = 0.960; MAE vs label = 0.096
- PRE seed 1002: hidden collapsed buses 904; frac pred < 0.5 = 0.965; MAE vs label = 0.091
- PF task: hidden collapsed buses 1143; frac pred < 0.5 = 0.000; MAE vs label = 1.065
- Grids where every seed has frac < 0.5 >= 0.5 AND mean PRE MAE vs label < half the PF MAE: 9 of 9
- lowest per-seed frac < 0.5 by subgrid: case4837_goc_idx_35 0.97, case20758_epigrids_idx_49 0.97, case9591_goc_idx_12 0.75, case5658_epigrids_idx_8 0.96, case4837_goc_idx_28 0.80, case19402_goc_idx_47 0.75, case10480_goc_idx_40 1.00, case4837_goc_idx_37 0.78, case10480_goc_idx_26 1.00
