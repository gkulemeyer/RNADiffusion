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


BATCH_SIZE = 1
ACCUMULATE_GRAD_BATCHES = 4
VAL_EVERY_N_EPOCHS = 25

CHECKPOINT_EVERY_N_EPOCHS = 100
PARTITIONS = ["sim60"] 
# TIMESTEPS = [100] 
TIMESTEPS = [500, 1000, 4000] 
EPOCHS = 5000
# LOSS_TYPE = ["vb_all"] 
LOSS_TYPE = ["vb_stochastic"] 

EXPERIMENT_NAME = f"ArchiveII_simfold_loss_comparison"


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def build_run_config(base_config, experiment_name, partition, timestep, epochs, loss_type):
    logs = Path("logs")
    job_dir = logs / experiment_name / partition / f"t{timestep}" / loss_type 
    note = f"""
    Compare different loss types on the simfolds dataset, using the same training setup as for famfold.

    {experiment_name} | {partition}, t={timestep}, loss={loss_type}, e={epochs}, 
     ckpt={CHECKPOINT_EVERY_N_EPOCHS}
    """

    config = ConfigDict(copy.deepcopy(base_config.to_dict()))
    seed = int(config.experiment.seed) 

    config.experiment.note = note
    config.data.partition_path = f"data/simfolds/simfolds_max128/ArchiveII_partitions_{partition}.csv"
    config.model.timesteps = timestep
    config.model.loss_type = loss_type
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

    jobs = [(p, t, loss) for p in PARTITIONS for t in TIMESTEPS for loss in LOSS_TYPE]

    for i, (partition, timestep, loss_type) in enumerate(jobs, start=1):
        config, job_dir = build_run_config(
            base_config,
            EXPERIMENT_NAME,
            partition,
            timestep,
            EPOCHS,
            loss_type,
        )

        print(f"\n[{i}/{len(jobs)}] {job_dir}")

        partition_path = repo_root / config.data.partition_path
        if not partition_path.exists():
            raise FileNotFoundError(f"Partition file not found: {partition_path}")
        print(config)
        run_training_and_evaluation(config, repo_root / job_dir, resume=RESUME)


if __name__ == "__main__":
    main()
