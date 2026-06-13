#!/usr/bin/env python
from __future__ import annotations

from itertools import product
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import clone_config, load_config
from src.run_loop import run_training_and_evaluation


CONFIG_PATH = REPO_ROOT / "configs/train/simfold_bprna_pretrain.yaml" 
# Edit these values for each run.
EXPERIMENT_NAME = "pretrained_bpRNA_simfold_test"
PARTITIONS = ["sim40"]
TIMESTEPS = [5] 

EPOCHS = 2
BATCH_SIZE = 16
ACCUMULATE_GRAD_BATCHES = 1
VAL_EVERY_N_EPOCHS = 1
CHECKPOINT_EVERY_N_EPOCHS = 1

RESUME = True
EVALUATE = True
CONTINUE_ON_OOM = True
 
def repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def make_run(base_config, partition, timestep):
    job_dir = Path("logs") / EXPERIMENT_NAME / partition / f"t{timestep}"  

    config = clone_config(base_config)
    partition_path = str(config.data.partition_path).format(partition=partition)

    config.experiment.note = (
        f"{EXPERIMENT_NAME} | {partition}, t={timestep}, "
        f"e={EPOCHS}, val={VAL_EVERY_N_EPOCHS}, ckpt={CHECKPOINT_EVERY_N_EPOCHS}"
    )
    # Update data paths and training parameters based on input arguments
    config.data.base_path = str(repo_path(config.data.base_path))
    config.data.partition_path = str(repo_path(partition_path))

    # update model and training config based on input arguments
    config.model.evaluate = EVALUATE
    config.model.timesteps = timestep
    config.model.load_pretrained = False
    config.model.loss_type = "vb_stochastic"

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
    jobs = list(product(PARTITIONS, TIMESTEPS))

    for index, (partition, timestep) in enumerate(jobs, start=1):
        config, job_dir = make_run(base_config, partition, timestep)

        print(f"\n[{index}/{len(jobs)}] {job_dir}")
        validate_data_files(config)

        print(config) 
        run_training_and_evaluation(
            config,
            REPO_ROOT / job_dir,
            resume=RESUME,
            CONTINUE_ON_OOM=CONTINUE_ON_OOM,
        ) 

if __name__ == "__main__":
    main()
