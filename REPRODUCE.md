# Reproduce

Three levels, from cheapest to most expensive. The entry's figures and tables need only level 1.

| level | needs | command | what it proves |
|---|---|---|---|
| 1 | the pinned env, no GPU, no external data | `bash reproduce.sh` | every file matches `CHECKSUMS.sha256`; the env matches the lock; the two post figures (figures-v4) regenerate from the committed result files; the unit tests pass; prints PASS or FAIL per step |
| 2 | level 1 plus Julia/PowerModels, PGLib-OPF v23.07, the authors' datakit and graphkit branches | commands in `EXTERNAL-DATA.md` | the big data sets regenerate from public inputs; compare with `tools/check_external.py` |
| 3 | level 2 plus the released GENCO checkpoints and a GPU | module command lines in `OMITTED.md` | the scored numbers themselves (the result files are the committed record) |

## 1. Build the pinned environment

`environment-genco.lock.yml` holds exact versions (conda packages with build strings, then a `pip:` section).
`lock/conda-explicit.txt` (`conda list --explicit`) and `lock/pip-freeze.txt` carry the same pins in the two
other standard forms. `environment-genco.yml` is the looser, human-edited spec the lock was taken from.

```
conda env create -f environment-genco.lock.yml      # builds the GPU torch stack; use mamba if conda's solver is slow
conda activate grid-operations-genco
export PYTHONNOUSERSITE=1
```

Three steps a plain `env create` does not do:

1. **torch-scatter from source (glibc 2.31 hosts).** The prebuilt `pt28cu128` wheel needs glibc 2.32, so on a
   host with glibc 2.31 (RHEL 8 class, `ldd --version`) it fails to import. Rebuild it against the installed torch:
   ```
   CUDA_HOME=/usr/local/cuda-12.3 TORCH_CUDA_ARCH_LIST=8.6 FORCE_CUDA=1 MAX_JOBS=6 \
   pip install --no-build-isolation --no-binary torch-scatter --force-reinstall --no-deps torch-scatter==2.1.2
   ```
   Adjust `CUDA_HOME` and `TORCH_CUDA_ARCH_LIST` (8.6 is an RTX 3090) to the host. On a host with glibc 2.32 or
   newer the wheel from the PyG index in the lock works and this step is not needed. Level 1 does not import
   torch-scatter.
2. **gridfm-datakit and gridfm-graphkit.** Both are pinned to a commit of the authors' `genco-paper-repro` branch
   in the lock (git URLs). Install datakit with `--no-deps` so pip keeps the pinned torch stack:
   `pip install --no-deps "git+https://github.com/gridfm/gridfm-datakit.git@<commit in the lock>"`.
3. **Julia and PowerModels (levels 2 and 3 only; datakit labels come from PowerModels `compute_ac_pf`).**
   `lock/julia/Project.toml` and `lock/julia/Manifest.toml` pin Julia 1.13.1, PowerModels 0.21.6, Ipopt 1.16.0 and
   PythonCall 0.9.36 (datakit drives Julia through `juliapkg`). Put them where juliapkg looks for its project
   (`<env prefix>/julia_env/`, or the directory named by `PYTHON_JULIAPKG_PROJECT`) before the first datakit
   call, and point the depot at a large writable directory, not a small home or `/tmp`:
   ```
   export JULIA_DEPOT_PATH=/path/to/julia_depot       # several GB after the first resolve
   python -c "import juliapkg; juliapkg.resolve()"     # downloads and precompiles from the manifest
   ```
   Parallel datakit workers share one depot; precompile once before launching workers.

Check the result: `python tools/check_env.py` compares every pinned package with what is installed (exit 1 on a
mismatch; `ALLOW_ENV_DRIFT=1` reports without failing). `reproduce.sh` runs it as its second step.

## 2. One command

```
bash reproduce.sh              # checksums, env check, figures, unit tests, then a PASS or FAIL summary
bash reproduce.sh checksums    # one step; also: env, figures, tests
```

Exit code 0 only if every step passed. `PY=/path/to/python bash reproduce.sh` picks the interpreter. Figures are
rebuilt in place, so run the checksum step first (the default order does) or restore the files afterwards.

## 3. Checksums

`CHECKSUMS.sha256` lists every file of this repo except itself (`sha256sum -c CHECKSUMS.sha256`). The big files
that are not shipped are in `EXTERNAL-DATA.md` with size, sha256 and the command that regenerates each. The
regeneration is not seed-free end to end (datakit draws scenarios and Julia solves them on the host), so a byte
difference in a regenerated parquet file is possible; compare tables before calling it a failure.
