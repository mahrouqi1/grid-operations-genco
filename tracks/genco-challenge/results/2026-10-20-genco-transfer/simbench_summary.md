Released median ratio 3.692; released median |V| error 0.00651 pu.

| arm / basis | median seed-mean ratio | per-seed medians | grids better than released (seed-mean; per seed) | beats DC on \|V\| | median \|V\| MAE (pu) | selected ckpts |
|---|---|---|---|---|---|---|
| H_12k | 4.402 | 4.79 / 4.48 / 3.92 | 1; 3 / 2 / 0 | 13 | 0.00701 | 12000 |
| D_12k | 5.703 | 7.85 / 4.84 / 4.60 | 0; 0 / 1 / 2 | 12 | 0.02078 | 12000 |
| C00_12k | 4.477 | 4.23 / 4.92 / 4.41 | 1; 1 / 1 / 3 | 12 | 0.00965 | 12000 |
| C01_12k | 4.451 | 4.68 / 4.45 / 4.80 | 3; 3 / 3 / 3 | 15 | 0.00873 | 12000 |
| H_selected | 4.145 | 3.98 / 4.42 / 4.09 | 2; 4 / 4 / 2 | 13 | 0.00880 | final, 10000, final |
| D_selected | 3.899 | 5.38 / 3.69 / 3.69 | 2; 2 / 0 / 0 | 13 | 0.01148 | 4000, 0, 0 |
| C00_selected | 3.985 | 3.90 / 3.91 / 4.14 | 3; 5 / 4 / 2 | 13 | 0.00931 | 6000, 6000, final |
| C01_selected | 3.692 | 3.69 / 3.69 / 3.69 | 0; 0 / 0 / 0 | 13 | 0.00651 | 0, 0, 0 |

H criteria 1-4 on the validation-selected checkpoints: {'1_median<=2.77': False, '2_breadth>=11': False, '3_each_seed_drop>=15%': False, '4_V>=12': True}
H criteria 1-4 at 12,000 updates (informational): {'1_median<=2.77': False, '2_breadth>=11': False, '3_each_seed_drop>=15%': False, '4_V>=12': True}
