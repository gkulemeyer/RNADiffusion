#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.experiment import run_experiment
from src.sweeps import (
    clone_config,
    completed_epochs,
    from_config_dict,
    last_checkpoint_path,
    latest_run_dir,
    run_completed,
    to_config_dict,
)

# ------------------------------------------------------------
# setup (edit this block)
# ------------------------------------------------------------
EXPERIMENT_NAME = "ArchiveII_test0"
BASE_CONFIG_PATH = "configs/train/default.yaml"
PARTITIONS = ["sim60", "sim70", "sim80", "sim90"]
# PARTITIONS = ["sim60"]
# TIMESTEPS = [5, 10]
TIMESTEPS = [8] 

EPOCHS = 3
FOLD = 0
RESUME = True
DRY_RUN = False
RUN_NAME_TEMPLATE = "{partition}_t{timestep}"


def build_run_config(base_config, experiment_name, partition, timestep, epochs, fold):
    config = clone_config(base_config)
    run_name = RUN_NAME_TEMPLATE.format(
        partition=partition,
        timestep=timestep,
        epochs=epochs,
        fold=fold,
    )
    config.experiment.name = run_name
    config.experiment.note = f"{experiment_name} | {partition} fold={fold}, t={timestep}, e={epochs}"
    config.data.partition_path = (
        f"data/simfolds/simfolds_max128/ArchiveII_partitions_{partition}.csv"
    )
    config.data.fold = fold
    config.model.timesteps = timestep
    config.training.max_epochs = epochs
    config.logging.save_dir = f"logs/{experiment_name}/{partition}/t{timestep}"
    return config, run_name


def main():
    repo_root = REPO_ROOT
    base_config = to_config_dict(load_config(repo_root / BASE_CONFIG_PATH))
    jobs = [(partition, timestep) for partition in PARTITIONS for timestep in TIMESTEPS]

    for index, (partition, timestep) in enumerate(jobs, start=1):
        config, run_name = build_run_config(
            base_config=base_config,
            experiment_name=EXPERIMENT_NAME,
            partition=partition,
            timestep=timestep,
            epochs=EPOCHS,
            fold=FOLD,
        )
        print(f"\n[{index}/{len(jobs)}] {run_name}")

        partition_path = repo_root / config.data.partition_path
        if not partition_path.exists():
            raise FileNotFoundError(f"Partition file not found: {partition_path}")

        plain_config = from_config_dict(config)
        target_dir = Path(plain_config["logging"]["save_dir"]) / run_name
        existing_run = latest_run_dir(config) if RESUME else None
        resume_checkpoint = None

        if RESUME and existing_run is not None:
            finished_epochs = completed_epochs(existing_run)
            completed_run = run_completed(existing_run)
            if completed_run is not None and finished_epochs is not None and finished_epochs >= EPOCHS:
                print(f"[SKIP] {completed_run} already reached {finished_epochs} epochs")
                continue

            resume_checkpoint = last_checkpoint_path(existing_run)
            if resume_checkpoint is not None:
                target_dir = existing_run

        if DRY_RUN:
            if resume_checkpoint is not None:
                print(f"[DRY-RUN] would resume {run_name} -> {target_dir} from {resume_checkpoint}")
            else:
                print(f"[DRY-RUN] would run {run_name} -> {target_dir}")
            continue

        if resume_checkpoint is not None:
            print(f"[RESUME] {run_name} -> {target_dir} from {resume_checkpoint}")
        run_kwargs = {"resume_from_checkpoint": resume_checkpoint}
        if resume_checkpoint is not None:
            run_kwargs["experiment_dir"] = target_dir

        result = run_experiment(plain_config, **run_kwargs)
        print(f"[DONE] {result['experiment_dir']}")


if __name__ == "__main__":
    main()
