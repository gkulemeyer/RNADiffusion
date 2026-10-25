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
EXPERIMENT_NAME = "ArchiveII_simfold_l_exp"

# Solo varían el simfold y el peso w.
SIMFOLDS = ["sim40", "sim70"]
L = [0.01, 0.1, 0.5, 1, 3]

FOLD = 0
TIMESTEPS = 10
BASE_DIM = 32
EPOCHS = 250
BATCH_SIZE = 4
ACCUMULATE_GRAD_BATCHES = 4
VAL_EVERY_N_EPOCHS = 10
RESUME = True
PRECISION = "16-mixed"
CONTINUE_ON_OOM = True
EVALUATE_PERIODIC = True
KEEP_SAMPLES = False


def build_run_config(base_config, sim, l):
    config = copy.deepcopy(base_config)
    job_dir = (
        Path("logs") / EXPERIMENT_NAME / sim / str(FOLD)
        / f"t{TIMESTEPS}" / f"L{l}"
    )
    config["experiment"]["name"] = f"{EXPERIMENT_NAME}_{sim}_w{l}"
    config["experiment"]["note"] = (
        f"{EXPERIMENT_NAME} | sim={sim}, fold={FOLD}, w={l}, "
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
        balance_bp="class", w=0.02, lambda_valid=l
    )

    config["training"]["max_epochs"] = EPOCHS
    config["training"]["check_val_every_n_epoch"] = VAL_EVERY_N_EPOCHS
    config["training"]["precision"] = PRECISION
    config["training"]["batch_size"] = BATCH_SIZE
    config["training"]["accumulate_grad_batches"] = ACCUMULATE_GRAD_BATCHES
    return config, job_dir


def main():
    base_config = load_config(REPO_ROOT / BASE_CONFIG_PATH)
    jobs = [(sim, l) for sim in SIMFOLDS for l in L]

    for i, (sim, l) in enumerate(jobs, start=1):
        config, job_dir = build_run_config(base_config, sim, l)

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
