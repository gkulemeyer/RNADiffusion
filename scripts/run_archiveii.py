#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
from datetime import datetime
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
# setup
# ------------------------------------------------------------
BASE_CONFIG_PATH = "configs/train/default.yaml"
RESUME = True
DRY_RUN = False
PRECISION = "16-mixed"
BASE_DIM = [32]

BATCH_SIZE = 1
ACCUMULATE_GRAD_BATCHES = 4

# PARTITIONS = ["sim70"]
# PARTITIONS = ["sim60", "sim70", "sim80", "sim90"]
PARTITIONS = ["sim60"]
# TIMESTEPS = [10, 15, 25, 50, 75, 100, 150, 200, 250]
TIMESTEPS = [5]
# EPOCHS = 100
EPOCHS = 4


# EXPERIMENT_NAME = "ArchiveII_100epochs"
EXPERIMENT_NAME = "ArchiveII_resume_test"
RUN_NAME_TEMPLATE = "ts_{timesteps}_dt{dt}"

# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def build_run_config(base_config, experiment_name, partition, timestep, epochs, base_dim):
    config = clone_config(base_config)

    dt = datetime.now().strftime("%Y%m%d-%H%M")
    run_name = RUN_NAME_TEMPLATE.format(dt=dt, timesteps=timestep)

    config.experiment.name = run_name
    config.experiment.note = f"{experiment_name} | {partition}, t={timestep}, e={epochs}"
    config.data.partition_path = f"data/simfolds/simfolds_max128/ArchiveII_partitions_{partition}.csv"
    config.model.timesteps = timestep
    config.model.base_dim = base_dim
    config.training.max_epochs = epochs
    config.training.precision = PRECISION
    config.training.batch_size = BATCH_SIZE
    config.training.accumulate_grad_batches = ACCUMULATE_GRAD_BATCHES
    config.logging.save_dir = f"logs/{experiment_name}/{partition}/t{timestep}/bs{BATCH_SIZE}_acc{ACCUMULATE_GRAD_BATCHES}"

    return config, run_name


def resolve_resume(config, epochs):
    """Decide si se resume, se salta o se corre desde cero."""
    existing = latest_run_dir(
        config,
        run_pattern=f"*_Fold{config.data.fold}*",
    )

    if existing is None:
        return None, None  # fresh run

    finished_epochs = completed_epochs(existing)
    is_complete = run_completed(existing)

    if is_complete and finished_epochs is not None and finished_epochs >= epochs:
        return "skip", existing

    ckpt = last_checkpoint_path(existing)
    return ckpt, existing


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    repo_root = REPO_ROOT
    base_config = to_config_dict(load_config(repo_root / BASE_CONFIG_PATH))

    jobs = [(p, t, d) for p in PARTITIONS for t in TIMESTEPS for d in BASE_DIM]

    for i, (partition, timestep, base_dim) in enumerate(jobs, start=1):
        config, run_name = build_run_config(
            base_config,
            EXPERIMENT_NAME,
            partition,
            timestep,
            EPOCHS,
            base_dim,
        )

        print(f"\n[{i}/{len(jobs)}] {run_name}")

        partition_path = repo_root / config.data.partition_path
        if not partition_path.exists():
            raise FileNotFoundError(f"Partition file not found: {partition_path}")

        plain_config = from_config_dict(config)

        resume_ckpt, existing_dir = (None, None)
        if RESUME:
            resume_ckpt, existing_dir = resolve_resume(config, EPOCHS)

        # ---- decisiones ----
        if resume_ckpt == "skip":
            print(f"[SKIP] {existing_dir} already completed")
            continue

        target_dir = existing_dir if resume_ckpt else None

        if DRY_RUN:
            if resume_ckpt:
                print(f"[DRY-RUN] resume → {target_dir} from {resume_ckpt}")
            else:
                print(f"[DRY-RUN] new run → {run_name}")
            continue

        if resume_ckpt:
            print(f"[RESUME] {run_name} → {target_dir}")

        result = run_experiment(
            plain_config,
            experiment_dir=target_dir,
            resume=resume_ckpt,
        )

        print(f"[DONE] {result['experiment_dir']}")


if __name__ == "__main__":
    main()