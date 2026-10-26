| model | mask seed | hidden collapsed buses | frac predicted < 0.5 | MAE vs collapsed label | MAE vs physical root (buses with a root) |
|---|---|---|---|---|---|
| released | 1000 | 892 | 0.961 | 0.102 | 0.992 |
| released | 1001 | 921 | 0.960 | 0.096 | 0.972 |
| released | 1002 | 904 | 0.965 | 0.091 | 1.005 |
| retrain_fixed | 1000 | 892 | 0.000 | 0.921 | 0.148 |
| retrain_fixed | 1001 | 921 | 0.000 | 0.922 | 0.147 |
| retrain_fixed | 1002 | 904 | 0.000 | 0.931 | 0.157 |
| retrain_published | 1000 | 892 | 0.028 | 0.875 | 0.168 |
| retrain_published | 1001 | 921 | 0.025 | 0.879 | 0.155 |
| retrain_published | 1002 | 904 | 0.019 | 0.882 | 0.167 |
