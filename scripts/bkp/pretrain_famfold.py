#!/usr/bin/env python
from __future__ import annotations

import copy
from pathlib import Path
import sys

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.run_loop import run_training_and_evaluation


BASE_CONFIG_PATH = "configs/train/famfold_archiveii_insideBE.yaml"
EXPERIMENT_NAME = "test_famfold"
FOLDS = ["srp", "telomerase", "16s", "tRNA"]
TIMESTEPS = [5, 25, 100]

EPOCHS = 10
BATCH_SIZE = 16
ACCUMULATE_GRAD_BATCHES = 1
VAL_EVERY_N_EPOCHS = 1

RESUME = True
EVALUATE = True
CONTINUE_ON_OOM = True


def repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def make_run(base_config, fold, timestep):
    job_dir = Path("logs") / EXPERIMENT_NAME / fold / f"t{timestep}"
    config = copy.deepcopy(base_config)

    config["experiment"]["name"] = job_dir.name
    config["experiment"]["note"] = (
        f"{EXPERIMENT_NAME} | {fold}, t={timestep}, "
        f"e={EPOCHS}, val={VAL_EVERY_N_EPOCHS}"
    )
    config["data"]["base_path"] = str(repo_path(config["data"]["base_path"]))
    config["data"]["fold"] = fold
    config["model"]["evaluate"] = EVALUATE
    config["model"]["timesteps"] = timestep
    config["model"]["load_pretrained"] = False
    config["model"]["loss_type"] = "vb_stochastic"
    config["training"]["max_epochs"] = EPOCHS
    config["training"]["batch_size"] = BATCH_SIZE
    config["training"]["accumulate_grad_batches"] = ACCUMULATE_GRAD_BATCHES
    config["training"]["check_val_every_n_epoch"] = VAL_EVERY_N_EPOCHS
    return config, job_dir


def main():
    base_config = load_config(REPO_ROOT / BASE_CONFIG_PATH)
    jobs = [(fold, timestep) for fold in FOLDS for timestep in TIMESTEPS]

    for index, (fold, timestep) in enumerate(tqdm(jobs, desc="Running jobs"), start=1):
        config, job_dir = make_run(base_config, fold, timestep)
        print(f"\n[{index}/{len(jobs)}] {job_dir}")
        run_training_and_evaluation(
            config,
            REPO_ROOT / job_dir,
            resume=RESUME,
            retry_on_oom=CONTINUE_ON_OOM,
        )


if __name__ == "__main__":
    main()
