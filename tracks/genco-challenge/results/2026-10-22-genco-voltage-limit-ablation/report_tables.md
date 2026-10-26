# GENCO |V| inference: generator-limit / cost ablation (2026-10-22)

> **UNAUDITED.** Pre-registered control (`protocol.json`, sha256 in `protocol.sha256`, written 2026-10-22 00:52 EDT before any inference; amendments 1 and 2 written 01:16 and 01:37 EDT, before any whole grid ran). Generated 2026-10-22 07:14:23. CPU only.

## Bottom line

Threshold (pre-registered): crossed if any ablated arm loses >= 3 wins vs the released arm on the pieces, or on the whole grids pooled. **THRESHOLD_CROSSED = False.**
- A1 limits constant: wins lost on pieces 0, on pooled whole grids 1 -> not crossed
- A2 costs zeroed: wins lost on pieces 1, on pooled whole grids 0 -> not crossed
- A3 both: wins lost on pieces 0, on pooled whole grids 1 -> not crossed

Coverage: pieces 36/36, seen whole grids 6/15, unseen whole grids 10/17. Inconclusive flags (fewer than half run): {'pieces': False, 'whole_pooled': False}.

## Win counts per arm

Win = GENCO median error below the best of five interpolators and the upper end of the 95% scenario-bootstrap CI of the ratio below 1, 40% measured, 3 mask seeds pooled. Counts are on the grids that ran (same grids for every arm).

### Carved pieces (datakit-generated)

| arm | wins on the 36 grids that ran (of 36) | wins lost vs A0 | wins gained vs A0 | median ratio | median shift factor (arm/A0) |
|---|---|---|---|---|---|
| A0 released | 33 | - | - | 0.650 | 1.000 |
| A1 limits constant | 33 | 0 | 0 | 0.648 | 1.010 |
| A2 costs zeroed | 32 | 1 | 0 | 0.646 | 1.008 |
| A3 both | 33 | 0 | 0 | 0.651 | 1.024 |

Original study wins on these same grids: 33 (recomputed A0: 33).

### Seen-family whole grids

| arm | wins on the 6 grids that ran (of 15) | wins lost vs A0 | wins gained vs A0 | median ratio | median shift factor (arm/A0) |
|---|---|---|---|---|---|
| A0 released | 6 | - | - | 0.775 | 1.000 |
| A1 limits constant | 6 | 0 | 0 | 0.740 | 1.017 |
| A2 costs zeroed | 6 | 0 | 0 | 0.777 | 0.994 |
| A3 both | 6 | 0 | 0 | 0.780 | 1.029 |

Original study wins on these same grids: 6 (recomputed A0: 6).

### Unseen-family whole grids

| arm | wins on the 10 grids that ran (of 17) | wins lost vs A0 | wins gained vs A0 | median ratio | median shift factor (arm/A0) |
|---|---|---|---|---|---|
| A0 released | 6 | - | - | 0.928 | 1.000 |
| A1 limits constant | 5 | 1 | 0 | 0.974 | 1.041 |
| A2 costs zeroed | 6 | 0 | 0 | 0.921 | 1.000 |
| A3 both | 5 | 1 | 0 | 0.979 | 1.042 |

Original study wins on these same grids: 6 (recomputed A0: 6).

### Whole grids pooled

| arm | wins on the 16 grids that ran (of 32) | wins lost vs A0 | wins gained vs A0 | median ratio | median shift factor (arm/A0) |
|---|---|---|---|---|---|
| A0 released | 12 | - | - | 0.883 | 1.000 |
| A1 limits constant | 11 | 1 | 0 | 0.891 | 1.022 |
| A2 costs zeroed | 12 | 0 | 0 | 0.889 | 0.999 |
| A3 both | 11 | 1 | 0 | 0.910 | 1.038 |

## Not run

- **seen_whole** (9): pglib_opf_case3970_goc (not started); pglib_opf_case4020_goc (not started); pglib_opf_case4601_goc (not started); pglib_opf_case4619_goc (not started); pglib_opf_case4837_goc (not started); pglib_opf_case4917_goc (not started); pglib_opf_case9591_goc (not started); pglib_opf_case5658_epigrids (not started); pglib_opf_case7336_epigrids (not started)
- **unseen_whole** (7): case6470rte (not started); case6495rte (not started); case6515rte (not started); pglib_opf_case8387_pegase (not started); case9241pegase (not started); pglib_opf_case3012wp_k (not started); pglib_opf_case3375wp_k (not started)

## Per-case results

`per_case_pieces.csv`, `per_case_seen_whole.csv`, `per_case_unseen_whole.csv`: one row per grid per arm (ratio, CI, win).

### pieces

| grid | buses | ratio A0 | A1 | A2 | A3 | wins A0/A1/A2/A3 |
|---|---|---|---|---|---|---|
| carved_case2383wp_k_c42 | 159 | 0.780 | 0.786 | 0.790 | 0.829 | 1/1/1/1 |
| carved_case20758_epigrids_c0 | 165 | 0.475 | 0.580 | 0.479 | 0.562 | 1/1/1/1 |
| carved_case3375wp_k_c53 | 165 | 0.721 | 0.721 | 0.710 | 0.719 | 1/1/1/1 |
| carved_case2848_rte_c20 | 167 | 0.857 | 0.831 | 0.885 | 0.864 | 1/1/1/1 |
| carved_case10480_goc_c12 | 185 | 0.460 | 0.476 | 0.461 | 0.474 | 1/1/1/1 |
| carved_case2383wp_k_c4 | 202 | 0.676 | 0.679 | 0.662 | 0.683 | 1/1/1/1 |
| carved_case4837_goc_c1 | 235 | 0.438 | 0.540 | 0.460 | 0.572 | 1/1/1/1 |
| carved_case2383wp_k_c40 | 241 | 0.689 | 0.690 | 0.690 | 0.687 | 1/1/1/1 |
| carved_case2848_rte_c1 | 242 | 0.755 | 0.724 | 0.769 | 0.740 | 1/1/1/1 |
| carved_case30000_goc_c0 | 247 | 0.858 | 0.852 | 0.874 | 0.853 | 1/1/1/1 |
| carved_case19402_goc_c10 | 259 | 0.370 | 0.374 | 0.374 | 0.380 | 1/1/1/1 |
| carved_case10480_goc_c13 | 268 | 0.247 | 0.246 | 0.248 | 0.247 | 1/1/1/1 |
| carved_case4837_goc_c0 | 279 | 0.452 | 0.496 | 0.437 | 0.466 | 1/1/1/1 |
| carved_case30000_goc_c1 | 299 | 0.768 | 0.727 | 0.748 | 0.719 | 1/1/1/1 |
| carved_case2848_rte_c22 | 323 | 0.755 | 0.733 | 0.765 | 0.741 | 1/1/1/1 |
| carved_case2869_pegase_c2 | 327 | 0.857 | 0.869 | 0.868 | 0.877 | 1/1/1/1 |
| carved_case8387_pegase_c2 | 327 | 0.436 | 0.429 | 0.447 | 0.455 | 1/1/1/1 |
| carved_case2383wp_k_c2 | 353 | 0.576 | 0.595 | 0.579 | 0.596 | 1/1/1/1 |
| carved_case5658_epigrids_c0 | 360 | 0.354 | 0.401 | 0.346 | 0.401 | 1/1/1/1 |
| carved_case20758_epigrids_c2 | 405 | 0.405 | 0.466 | 0.408 | 0.440 | 1/1/1/1 |
| carved_case8387_pegase_c3 | 405 | 1.513 | 1.290 | 1.737 | 1.664 | 0/0/0/0 |
| carved_case9241_pegase_c1 | 439 | 0.816 | 0.840 | 0.815 | 0.867 | 1/1/1/1 |
| carved_case30000_goc_c2 | 453 | 0.697 | 0.707 | 0.704 | 0.704 | 1/1/1/1 |
| carved_case9241_pegase_c2 | 488 | 1.305 | 1.319 | 1.305 | 1.342 | 0/0/0/0 |
| carved_case4837_goc_c3 | 493 | 0.468 | 0.469 | 0.466 | 0.472 | 1/1/1/1 |
| carved_case5658_epigrids_c3 | 494 | 0.511 | 0.541 | 0.510 | 0.546 | 1/1/1/1 |
| carved_case7336_epigrids_c1 | 501 | 0.487 | 0.540 | 0.493 | 0.534 | 1/1/1/1 |
| carved_case2869_pegase_c1 | 513 | 0.877 | 0.801 | 0.857 | 0.854 | 1/1/1/1 |
| carved_case2848_rte_c4 | 554 | 0.970 | 0.961 | 0.991 | 0.984 | 1/1/0/1 |
| carved_case7336_epigrids_c2 | 610 | 0.597 | 0.620 | 0.620 | 0.612 | 1/1/1/1 |
| carved_case7336_epigrids_c0 | 632 | 0.374 | 0.382 | 0.363 | 0.380 | 1/1/1/1 |
| carved_case2383wp_k_c41 | 660 | 0.626 | 0.631 | 0.630 | 0.636 | 1/1/1/1 |
| carved_case7336_epigrids_c4 | 673 | 0.343 | 0.381 | 0.334 | 0.366 | 1/1/1/1 |
| carved_case2848_rte_c21 | 735 | 1.116 | 1.099 | 1.159 | 1.155 | 0/0/0/0 |
| carved_case30000_goc_c3 | 792 | 0.675 | 0.666 | 0.684 | 0.666 | 1/1/1/1 |
| carved_case4837_goc_c2 | 903 | 0.499 | 0.515 | 0.497 | 0.501 | 1/1/1/1 |

### seen_whole

| grid | buses | ratio A0 | A1 | A2 | A3 | wins A0/A1/A2/A3 |
|---|---|---|---|---|---|---|
| pglib_opf_case500_goc | 500 | 0.702 | 0.617 | 0.693 | 0.670 | 1/1/1/1 |
| pglib_opf_case793_goc | 793 | 0.847 | 0.863 | 0.860 | 0.890 | 1/1/1/1 |
| pglib_opf_case2000_goc | 2000 | 0.515 | 0.513 | 0.513 | 0.504 | 1/1/1/1 |
| pglib_opf_case2312_goc | 2312 | 0.936 | 0.951 | 0.928 | 0.957 | 1/1/1/1 |
| pglib_opf_case2742_goc | 2742 | 0.201 | 0.216 | 0.201 | 0.225 | 1/1/1/1 |
| pglib_opf_case3022_goc | 3022 | 0.943 | 0.968 | 0.935 | 0.977 | 1/1/1/1 |

### unseen_whole

| grid | buses | ratio A0 | A1 | A2 | A3 | wins A0/A1/A2/A3 |
|---|---|---|---|---|---|---|
| 1-HV-mixed--0-no_sw | 64 | 0.918 | 1.028 | 0.918 | 1.028 | 1/0/1/0 |
| 1-HV-urban--0-no_sw | 82 | 0.694 | 0.769 | 0.694 | 0.769 | 1/1/1/1 |
| 1-HV-mixed--1-no_sw | 94 | 1.035 | 1.117 | 1.035 | 1.117 | 0/0/0/0 |
| 1-HV-urban--1-no_sw | 100 | 0.818 | 0.861 | 0.818 | 0.861 | 1/1/1/1 |
| 1-HV-mixed--2-no_sw | 114 | 1.060 | 1.103 | 1.060 | 1.103 | 0/0/0/0 |
| 1-HV-urban--2-no_sw | 120 | 0.808 | 0.803 | 0.808 | 0.803 | 1/1/1/1 |
| case1888rte | 1888 | 0.939 | 0.920 | 0.924 | 0.930 | 1/1/1/1 |
| pglib_opf_case2383wp_k | 2383 | 0.723 | 0.728 | 0.722 | 0.735 | 1/1/1/1 |
| case2848rte | 2848 | 1.285 | 1.289 | 1.271 | 1.259 | 0/0/0/0 |
| case2869pegase | 2869 | 1.130 | 1.176 | 1.150 | 1.179 | 0/0/0/0 |

