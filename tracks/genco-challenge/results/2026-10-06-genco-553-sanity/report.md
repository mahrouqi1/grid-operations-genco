# GENCO section 5.5.3 sanity reproduction

2026-10-06, `run.sh` (env `grid-operations-genco`, graphkit branch `genco-paper-repro` 0.0.7, datakit
branch `genco-paper-repro` 1.0.4, torch 2.8.0+cu128, GPU0 RTX 3090 on <HOST>). Released
checkpoints from `gridfm/genco-pf-transfer`, evaluated on GENCO's own 9,952-scenario case118_ieee
eval set with GENCO's own `case118_eval.yaml`. Raw MLflow runs are local only (gitignored);
`metrics.json` holds every number below (regenerated from mlruns by the last step of `run.sh`).

**Result: reproduction to within 1e-5 MW (float32 nondeterminism).** Mean absolute active-power residual (MW), the paper's Figure 15
metric:

| model | ours | published (`transfer_case118.csv`) |
|---|---:|---:|
| zero-shot (pretrained on subgrids, no case118 data) | 11.0782 | 11.0783 |
| from scratch, 1,000 case118 samples | 3.8104 | 3.8104 |
| fine-tuned, 1,000 case118 samples | 1.9317 | 1.9317 |
| DC power flow (computed in the same run) | 2.2951 | 2.2951 |
| reference AC labels themselves | 5.9e-11 | n/a |

Also measured (not in the published CSV): reactive residual 1.26 / 0.73 / 0.39 MVar and max per-bus
power-balance error 197 / 217 / 150 MVA for zero-shot / scratch / fine-tuned.

**What this establishes.** The env, the checkpoints and the evaluation path are GENCO's own, and they reproduce
the published CSV to within 1e-5 MW (zero-shot 11.07824 here against 11.078253 published, which rounds to
11.0783; the DC value is identical; the differences are float32 nondeterminism). Starting point for the challenge, as framed at the start: zero-shot GENCO is 4.8x
worse than DC on case118 by GENCO's own metric, and 1,000 fine-tuning samples bring it 16% below DC.
The target is to beat DC on grids GENCO never saw, with as few samples as possible.

**Caveat.** The metric is GENCO's power-balance residual of the predicted state, not voltage error
against the exact solution. The ladder evaluation reports both.

Side note: graphkit raises harmless `OSError: Device or resource busy: '.nfs...'` at exit when its
temp dir sits on NFS; every run still wrote its metrics.
