#!/usr/bin/env python
from __future__ import annotations

from itertools import product
from pathlib import Path
import sys

import torch as tr

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import clone_config, load_config
from src.run_loop import run_training_and_evaluation

CONFIG_PATH = REPO_ROOT / "configs/train/simfold_bprna.yaml"

# Edit these values for each run.
EXPERIMENT_NAME = "bpRNA_simfold_test_best_and_worst"
PARTITIONS = ["sim40"]
TIMESTEPS = [5]
LOSS_TYPES = ["vb_stochastic"]

EPOCHS = 150
BATCH_SIZE = 1
ACCUMULATE_GRAD_BATCHES = 4
VAL_EVERY_N_EPOCHS = 1
CHECKPOINT_EVERY_N_EPOCHS = 1

RESUME = False
CONTINUE_ON_OOM = True


def repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def make_run(base_config, partition, timestep, loss_type):
    job_dir = Path("logs") / EXPERIMENT_NAME / partition / f"t{timestep}" / loss_type
    config = clone_config(base_config)
    partition_path = str(config.data.partition_path).format(partition=partition)

    config.experiment.note = (
        f"{EXPERIMENT_NAME} | {partition}, t={timestep}, loss={loss_type}, "
        f"e={EPOCHS}, val={VAL_EVERY_N_EPOCHS}, ckpt={CHECKPOINT_EVERY_N_EPOCHS}"
    )
    config.data.base_path = str(repo_path(config.data.base_path))
    config.data.partition_path = str(repo_path(partition_path))
    config.model.timesteps = timestep
    config.model.loss_type = loss_type
    config.training.max_epochs = EPOCHS
    config.training.batch_size = BATCH_SIZE
    config.training.accumulate_grad_batches = ACCUMULATE_GRAD_BATCHES
    config.training.check_val_every_n_epoch = VAL_EVERY_N_EPOCHS
    config.logging.checkpoint_every_n_epochs = CHECKPOINT_EVERY_N_EPOCHS
    config.logging.save_dir = str(job_dir.parent)

    return config, job_dir


def validate_data_files(config):
    paths = [
        ("Base data file", config.data.base_path),
        ("Partition file", config.data.partition_path),
    ]
    for label, path in paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")


def main():
    base_config = load_config(CONFIG_PATH)
    jobs = list(product(PARTITIONS, TIMESTEPS, LOSS_TYPES))

    for index, (partition, timestep, loss_type) in enumerate(jobs, start=1):
        config, job_dir = make_run(base_config, partition, timestep, loss_type)

        print(f"\n[{index}/{len(jobs)}] {job_dir}")
        validate_data_files(config)

        print(config)
        try:
            run_training_and_evaluation(
                config,
                REPO_ROOT / job_dir,
                resume=RESUME,
            )
        except tr.cuda.OutOfMemoryError:
            print(f"[OOM] partition={partition}, timesteps={timestep}, loss={loss_type}")
            if not CONTINUE_ON_OOM:
                raise


if __name__ == "__main__":
    main()
