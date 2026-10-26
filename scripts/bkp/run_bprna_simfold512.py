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


BASE_CONFIG_PATH = "configs/train/simfold_bprna_pretrain.yaml"
EXPERIMENT_NAME = "bpRNA_simfold_testing_pretrain"
PARTITIONS = ["sim40"]
TIMESTEPS = [5]
LOSS_TYPES = ["vb_stochastic"]

EPOCHS = 150
BATCH_SIZE = 1
ACCUMULATE_GRAD_BATCHES = 4
VAL_EVERY_N_EPOCHS = 1

RESUME = False
CONTINUE_ON_OOM = True


def repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def make_run(base_config, partition, timestep, loss_type):
    job_dir = Path("logs") / EXPERIMENT_NAME / partition / f"t{timestep}" / loss_type
    config = copy.deepcopy(base_config)
    partition_path = config["data"]["partition_path"].format(partition=partition)

    config["experiment"]["name"] = job_dir.name
    config["experiment"]["note"] = (
        f"{EXPERIMENT_NAME} | {partition}, t={timestep}, loss={loss_type}, "
        f"e={EPOCHS}, val={VAL_EVERY_N_EPOCHS}"
    )
    config["data"]["base_path"] = str(repo_path(config["data"]["base_path"]))
    config["data"]["partition_path"] = str(repo_path(partition_path))
    config["model"]["timesteps"] = timestep
    config["model"]["loss_type"] = loss_type
    config["training"]["max_epochs"] = EPOCHS
    config["training"]["batch_size"] = BATCH_SIZE
    config["training"]["accumulate_grad_batches"] = ACCUMULATE_GRAD_BATCHES
    config["training"]["check_val_every_n_epoch"] = VAL_EVERY_N_EPOCHS
    return config, job_dir


def main():
    base_config = load_config(REPO_ROOT / BASE_CONFIG_PATH)
    jobs = [
        (partition, timestep, loss_type)
        for partition in PARTITIONS
        for timestep in TIMESTEPS
        for loss_type in LOSS_TYPES
    ]

    for index, (partition, timestep, loss_type) in enumerate(jobs, start=1):
        config, job_dir = make_run(base_config, partition, timestep, loss_type)
        print(f"\n[{index}/{len(jobs)}] {job_dir}")
        run_training_and_evaluation(
            config,
            REPO_ROOT / job_dir,
            resume=RESUME,
            retry_on_oom=CONTINUE_ON_OOM,
        )


if __name__ == "__main__":
    main()
