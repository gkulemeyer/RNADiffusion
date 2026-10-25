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

BASE_CONFIG_PATH = "configs/train/archiveII_simfold_weighted.yaml"
EXPERIMENT_NAME = "ArchiveII_simfold_w_exp"

# Solo varían el simfold y el peso w.
SIMFOLDS = ["sim40", "sim50", "sim60", "sim70", "sim80", "sim90"] 
W = [0.1, 0.5, 1]

FOLD = 0
TIMESTEPS = 10
BASE_DIM = 32
EPOCHS = 500
BATCH_SIZE = 4
ACCUMULATE_GRAD_BATCHES = 4
VAL_EVERY_N_EPOCHS = 25
RESUME = True
PRECISION = "16-mixed"
CONTINUE_ON_OOM = True
EVALUATE_PERIODIC = True
KEEP_SAMPLES = False


def build_run_config(base_config, sim, w):
    config = copy.deepcopy(base_config)
    job_dir = (
        Path("logs") / EXPERIMENT_NAME / sim / str(FOLD)
        / f"t{TIMESTEPS}" / f"w{w}"
    )
    config["experiment"]["name"] = f"{EXPERIMENT_NAME}_{sim}_w{w}"
    config["experiment"]["note"] = (
        f"{EXPERIMENT_NAME} | sim={sim}, fold={FOLD}, w={w}, "
        f"t={TIMESTEPS}, bd={BASE_DIM}, e={EPOCHS}, val={VAL_EVERY_N_EPOCHS}"
    )
    config["data"]["fold"] = FOLD
    config["data"]["partition_path"] = str(
        REPO_ROOT / "data/simfolds/bootstrap"
        / f"archiveII_similars_splits_less_than_512_k10_boostrap_{sim}-fold.csv"
    )

    config["model"]["timesteps"] = TIMESTEPS
    config["model"]["base_dim"] = BASE_DIM
    config["model"]["load_pretrained"] = False
    config["model"]["model_kwargs"].update(
        balance_bp="class", w=w, lambda_valid=1.0
    )

    config["training"]["max_epochs"] = EPOCHS
    config["training"]["check_val_every_n_epoch"] = VAL_EVERY_N_EPOCHS
    config["training"]["precision"] = PRECISION
    config["training"]["batch_size"] = BATCH_SIZE
    config["training"]["accumulate_grad_batches"] = ACCUMULATE_GRAD_BATCHES
    return config, job_dir


def main():
    base_config = load_config(REPO_ROOT / BASE_CONFIG_PATH)
    jobs = [(sim, w) for sim in SIMFOLDS for w in W]

    for i, (sim, w) in enumerate(jobs, start=1):
        config, job_dir = build_run_config(base_config, sim, w)

        print(f"\n[{i}/{len(jobs)}] {job_dir}")

        run_training_and_evaluation(
            config,
            REPO_ROOT / job_dir,
            resume=RESUME,
            retry_on_oom=CONTINUE_ON_OOM,
            evaluate_periodic=EVALUATE_PERIODIC,
            keep_samples=KEEP_SAMPLES,
        )


if __name__ == "__main__":
    main()
