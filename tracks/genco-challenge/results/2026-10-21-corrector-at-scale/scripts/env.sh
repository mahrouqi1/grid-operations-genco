eval "$(conda shell.bash hook)"
conda activate <CONDA_ENVS>/grid-operations-genco
cd <REPO>
export PYTHONPATH=tracks/genco-challenge/src PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 MLFLOW_ALLOW_FILE_STORE=true CUDA_VISIBLE_DEVICES=1
