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
BASE_CONFIG_PATH = "configs/train/famfold.yaml"
RESUME = True
PRECISION = "16-mixed"
BASE_DIM = [32]

BATCH_SIZE = 4
ACCUMULATE_GRAD_BATCHES = 4

VAL_EVERY_N_EPOCHS = 3
 
FOLDS = ["srp", "tRNA", "tmRNA", "RNaseP", "telomerase", "grp1", "16s", "23s", "5s"]
# TIMESTEPS = [10, 15, 25, 50, 75, 100, 150, 200, 250]
TIMESTEPS = [10]
# TIMESTEPS = [2]
# EPOCHS = 100
EPOCHS = 15


EXPERIMENT_NAME = "ArchiveII_famfold"
CONTINUE_ON_OOM = True
EVALUATE_PERIODIC = True
KEEP_SAMPLES = False


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def build_job_dir(experiment_name, fold, timestep, seed):
    return Path("logs") / experiment_name / fold / f"t{timestep}" / f"seed{seed}"



def build_run_config(base_config, experiment_name, fold, timestep, epochs, base_dim):
    config = copy.deepcopy(base_config)
    seed = int(config["experiment"]["seed"])
    job_dir = build_job_dir(experiment_name, fold, timestep, seed)

    config["experiment"]["name"] = job_dir.name
    config["experiment"]["note"] = (
        f"{experiment_name} | {fold}, t={timestep}, bd={base_dim}, e={epochs}, "
        f"val={VAL_EVERY_N_EPOCHS}"
    )
    config["data"]["fold"] = fold

    config["model"]["timesteps"] = timestep
    config["model"]["base_dim"] = base_dim
    config["model"]["load_pretrained"] = False

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

    jobs = [(f, t, d) for f in FOLDS for t in TIMESTEPS for d in BASE_DIM]

    for i, (fold, timestep, base_dim) in enumerate(jobs, start=1):
        config, job_dir = build_run_config(
            base_config,
            EXPERIMENT_NAME,
            fold,
            timestep,
            EPOCHS,
            base_dim,
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
