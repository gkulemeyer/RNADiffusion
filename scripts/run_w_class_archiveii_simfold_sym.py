#!/usr/bin/env python
from __future__ import annotations

import copy
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.run_loop import run_training_and_evaluation

# ------------------------------------------------------------
# setup
# ------------------------------------------------------------
BASE_CONFIG_PATH = "configs/train/archiveII_simfold_weighted.yaml" 


SIMFOLDS = ["sim40", "sim50", "sim60", "sim70", "sim80", "sim90"]  
FOLDS = [0] 
TIMESTEPS = [10] 
EPOCHS = 200




BATCH_SIZE = 4
ACCUMULATE_GRAD_BATCHES = 4
VAL_EVERY_N_EPOCHS = 25 



diffuser = "symmetric_unet" # simple_unet | symmetric_unet | symmetric_resnet_unet
EXPERIMENT_NAME = f"ArchiveII_simfold_weighted_{diffuser}"

model_kwargs = { "balance_bp": "class",   "w": 0.02, "lambda_valid": 1.0} 

RESUME = True
PRECISION = "16-mixed"
BASE_DIM = [32]
CONTINUE_ON_OOM = True
EVALUATE_PERIODIC = True
KEEP_SAMPLES = False


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def build_job_dir(experiment_name, sim, fold, timestep):
    return Path("logs") / experiment_name / sim / str(fold) / f"t{timestep}" 

def simfold_path(sim):
    return f"data/simfolds/bootstrap/archiveII_similars_splits_less_than_512_k10_boostrap_{sim}-fold.csv"

def build_run_config(base_config, experiment_name, fold, sim, timestep, epochs, base_dim, diffuser, model_kwargs):
    config = copy.deepcopy(base_config) 
    job_dir = build_job_dir(experiment_name, sim, fold, timestep)

    config["experiment"]["name"] = job_dir.name
    config["experiment"]["note"] = (
        f"{experiment_name} | sim={sim}, {fold}, t={timestep}, bd={base_dim}, e={epochs}, "
        f"val={VAL_EVERY_N_EPOCHS}"
    )
    config["data"]["fold"] = fold
    config["data"]["partition_path"] = simfold_path(sim)
    config["model"]["diffuser"] = diffuser
    config["model"]["timesteps"] = timestep
    config["model"]["base_dim"] = base_dim
    config["model"]["load_pretrained"] = False
    config["model"]["model_kwargs"] = model_kwargs

    config["training"]["max_epochs"] = epochs
    config["training"]["check_val_every_n_epoch"] = VAL_EVERY_N_EPOCHS
    config["training"]["precision"] = PRECISION
    config["training"]["batch_size"] = BATCH_SIZE
    config["training"]["accumulate_grad_batches"] = ACCUMULATE_GRAD_BATCHES
    return config, job_dir

# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    repo_root = REPO_ROOT
    base_config = load_config(repo_root / BASE_CONFIG_PATH)

    jobs = [(f,s,  t, d) for f in FOLDS for s in SIMFOLDS for t in TIMESTEPS for d in BASE_DIM]

    for i, (fold, sim, timestep, base_dim) in enumerate(jobs, start=1):
        config, job_dir = build_run_config(
            base_config,
            EXPERIMENT_NAME,
            fold,
            sim,
            timestep,
            EPOCHS,
            base_dim,
            diffuser,
            model_kwargs
        )

        print(f"\n[{i}/{len(jobs)}] {job_dir}")

        run_training_and_evaluation(
            config,
            repo_root / job_dir,
            resume=RESUME,
            retry_on_oom=CONTINUE_ON_OOM,
            evaluate_periodic=EVALUATE_PERIODIC,
            keep_samples=KEEP_SAMPLES,
        )


if __name__ == "__main__":
    main()
