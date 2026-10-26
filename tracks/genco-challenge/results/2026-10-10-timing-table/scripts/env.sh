# source me: env for the H3 timing blocks (CPU work pinned by the caller with taskset -c 16-19)
R=<REPO>
eval "$(conda shell.bash hook)"
conda activate <CONDA_ENVS>/grid-operations-genco
export PYTHONPATH=$R/tracks/genco-challenge/src:$R/tracks/genco-challenge/results/2026-10-10-timing-table/scripts
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 JULIA_CPU_TARGET=generic
export JULIA_DEPOT_PATH=<JULIA_DEPOT>
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
cd $R
