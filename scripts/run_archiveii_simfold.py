#!/usr/bin/env python
from __future__ import annotations

import copy
from pathlib import Path
import sys

from ml_collections import ConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.run_loop import run_training_and_evaluation

# ------------------------------------------------------------
# setup
# ------------------------------------------------------------
BASE_CONFIG_PATH = "configs/train/default.yaml"
RESUME = True
PRECISION = "16-mixed"
BASE_DIM = [32]

BATCH_SIZE = 1
ACCUMULATE_GRAD_BATCHES = 4

VAL_EVERY_N_EPOCHS = 1
CHECKPOINT_EVERY_N_EPOCHS = 5 
PARTITIONS = ["sim60", "sim70"] 
TIMESTEPS = [10, 25, 50, 75, 100, 150] 
EPOCHS = 65 
FOLD = 1

EXPERIMENT_NAME = f"ArchiveII_simfold_fold{FOLD}"


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def build_job_dir(experiment_name, partition, timestep, base_dim, seed, fold):
    return Path("logs") / experiment_name / partition / f"t{timestep}" / f"bd{base_dim}" / f"bs{BATCH_SIZE}_acc{ACCUMULATE_GRAD_BATCHES}" / f"seed{seed}" / f"fold{fold}"


def build_run_config(base_config, experiment_name, partition, timestep, epochs, base_dim):
    config = ConfigDict(copy.deepcopy(base_config.to_dict()))
    seed = int(config.experiment.seed)
    job_dir = build_job_dir(experiment_name, partition, timestep, base_dim, seed, FOLD)

    config.experiment.name = job_dir.name
    config.experiment.note = (
        f"{experiment_name} | {partition}, t={timestep}, bd={base_dim}, e={epochs}, "
        f"val={VAL_EVERY_N_EPOCHS}, ckpt={CHECKPOINT_EVERY_N_EPOCHS}"
    )
    config.data.partition_path = f"data/simfolds/simfolds_max128/ArchiveII_partitions_{partition}.csv"
    config.data.fold = FOLD
    config.model.timesteps = timestep
    config.model.base_dim = base_dim
    config.training.max_epochs = epochs
    config.training.check_val_every_n_epoch = VAL_EVERY_N_EPOCHS
    config.training.precision = PRECISION
    config.training.batch_size = BATCH_SIZE
    config.training.accumulate_grad_batches = ACCUMULATE_GRAD_BATCHES
    config.logging.save_dir = str(job_dir.parent)
    config.logging.checkpoint_every_n_epochs = CHECKPOINT_EVERY_N_EPOCHS

    return config, job_dir

# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    repo_root = REPO_ROOT
    base_config = ConfigDict(load_config(repo_root / BASE_CONFIG_PATH))

    jobs = [(p, t, d) for p in PARTITIONS for t in TIMESTEPS for d in BASE_DIM]

    for i, (partition, timestep, base_dim) in enumerate(jobs, start=1):
        config, job_dir = build_run_config(
            base_config,
            EXPERIMENT_NAME,
            partition,
            timestep,
            EPOCHS,
            base_dim,
        )

        print(f"\n[{i}/{len(jobs)}] {job_dir}")

        partition_path = repo_root / config.data.partition_path
        if not partition_path.exists():
            raise FileNotFoundError(f"Partition file not found: {partition_path}")

        run_training_and_evaluation(config, repo_root / job_dir, resume=RESUME)


if __name__ == "__main__":
    main()
