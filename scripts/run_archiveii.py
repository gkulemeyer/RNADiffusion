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
EXPERIMENT_NAME = "ArchiveII_test_max_ts"

BASE_CONFIG_PATH = "configs/train/default.yaml"
PARTITIONS = ["sim70"]
TIMESTEPS = [50, 75, 100]
BASE_DIM = [64]
EPOCHS = 2

PRECISION = "16-mixed"

FOLD = 0
RESUME = True
DRY_RUN = False
RUN_NAME_TEMPLATE = "{dt}_model_{base_dim}_Fold{fold}"


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def build_run_config(base_config, experiment_name, partition, timestep, epochs, fold, base_dim):
    config = clone_config(base_config)

    dt = datetime.now().strftime("%Y%m%d-%H%M")
    run_name = RUN_NAME_TEMPLATE.format(dt=dt, fold=fold, base_dim=base_dim)

    config.experiment.name = run_name
    config.experiment.note = f"{experiment_name} | {partition} fold={fold}, t={timestep}, e={epochs}"
    config.data.partition_path = f"data/simfolds/simfolds_max128/ArchiveII_partitions_{partition}.csv"
    config.data.fold = fold
    config.model.timesteps = timestep
    config.model.base_dim = base_dim
    config.training.max_epochs = epochs
    config.training.precision = PRECISION
    config.logging.save_dir = f"logs/{experiment_name}/{partition}/t{timestep}/m{base_dim}"

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
            FOLD,
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