#!/usr/bin/env python
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    clone_config,
    latest_run_dir,
    load_config,
    to_config_dict,
)
from src.run_io import RunIO

from supervised.backbone_experiment import (
    evaluate_backbone_checkpoint,
    last_backbone_checkpoint,
    milestone_complete,
    snapshot_milestone,
    run_backbone_experiment,
)

# ------------------------------------------------------------
# setup
# ------------------------------------------------------------
BASE_CONFIG_PATH = "configs/train/default.yaml"
PRECISION = "16-mixed"
BASE_DIM = [32]

BATCH_SIZE = 1
ACCUMULATE_GRAD_BATCHES = 4
LR = 0.001

PARTITIONS = ["sim60", "sim70", "sim80", "sim90"]
EPOCH_MILESTONES = list(range(5, 101, 5))

EXPERIMENT_NAME = "ArchiveII_backbone_simfold512"
RUN_NAME_TEMPLATE = "timestamp_{dt}"
BACKBONE_IN_CHANNELS = 16


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def build_run_config(base_config, experiment_name, repo_root, partition, base_dim):
    config = clone_config(base_config)
    repo_root = Path(repo_root)

    dt = datetime.now().strftime("%Y%m%d-%H%M")
    run_name = RUN_NAME_TEMPLATE.format(dt=dt)
    run_pattern = RUN_NAME_TEMPLATE.format(dt="*")

    config.experiment.name = run_name
    config.experiment.note = (
        f"{experiment_name} | {partition}, base_dim={base_dim}"
    )
    config.data.base_path = str(repo_root / "data" / "ArchiveII.csv")
    config.data.partition_path = str(
        REPO_ROOT / f"data/simfolds/6_sim_folds/archiveII_similars_splits_less_than_512_k10_NOboostrap_{partition}-fold.csv"
        )
    config.model.in_channels = BACKBONE_IN_CHANNELS
    config.model.out_channels = 2
    config.model.base_dim = base_dim
    config.training.max_epochs = 0
    config.training.precision = PRECISION
    config.training.batch_size = BATCH_SIZE
    config.training.accumulate_grad_batches = ACCUMULATE_GRAD_BATCHES
    config.training.lr = LR
    config.logging.save_dir = str(
        repo_root / "supervised" / "logs" / experiment_name / partition / f"base_dim{base_dim}"
    )

    return config, run_name, run_pattern


def normalize_milestones(values):
    milestones = sorted({int(value) for value in values if int(value) > 0})
    if not milestones:
        raise ValueError("EPOCH_MILESTONES must contain at least one positive epoch")
    return milestones


def resolve_run_dir(config, run_pattern):
    existing_dir = latest_run_dir(config, run_pattern=run_pattern)
    if existing_dir is None:
        return None
    if not RunIO(existing_dir).train_dir.exists():
        return None

    config.experiment.name = existing_dir.name
    return Path(existing_dir)


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    repo_root = REPO_ROOT
    base_config = to_config_dict(load_config(repo_root / BASE_CONFIG_PATH))
    milestones = normalize_milestones(EPOCH_MILESTONES)

    jobs = [(partition, base_dim) for partition in PARTITIONS for base_dim in BASE_DIM]

    for index, (partition, base_dim) in enumerate(jobs, start=1):
        config, run_name, run_pattern = build_run_config(
            base_config,
            EXPERIMENT_NAME,
            repo_root,
            partition,
            base_dim,
        )

        print(f"\n[{index}/{len(jobs)}] {run_name}")

        base_path = Path(config.data.base_path)
        partition_path = Path(config.data.partition_path)
        if not base_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {base_path}")
        if not partition_path.exists():
            raise FileNotFoundError(f"Partition file not found: {partition_path}")

        run_root = resolve_run_dir(config, run_pattern)
        run = RunIO(run_root) if run_root is not None else None
        resume_ckpt = run.last_checkpoint() if run is not None else None
        current_epoch_count = run.completed_epoch_count() if run is not None else 0

        if run_root is not None:
            print(f"[RESUME] {run_root} from {current_epoch_count} completed epochs")

        for milestone in milestones:
            if milestone_complete(run_dir, milestone) if run_dir is not None else False:
                print(f"[SKIP] epoch_{milestone:03d} already snapshotted")
                continue

            if current_epoch > milestone:
                print(
                    f"[WARN] current epoch {current_epoch} is past milestone {milestone}; "
                    "cannot recreate exact checkpoint"
                )
                continue

            if current_epoch < milestone:
                config.training.max_epochs = milestone
                result = run_backbone_experiment(
                    config.to_dict(),
                    run_root=run_root,
                    resume=str(resume_ckpt) if resume_ckpt is not None else None,
                )
                run_root = Path(result["experiment_dir"])
                run = RunIO(run_root)
                resume_ckpt = Path(result["last_checkpoint"])
                current_epoch_count = run.completed_epoch_count()
                print(f"[TRAIN] reached {current_epoch_count} completed epochs in {run_root}")

            if resume_ckpt is None:
                raise FileNotFoundError("Expected last checkpoint after training milestone")

            summary = evaluate_backbone_checkpoint(
                load_config(run.train_dir / "config.yaml"),
                resume_ckpt,
                run_dir=run_dir,
                epoch=milestone,
            )
            snapshot_dir = snapshot_milestone(run_dir, milestone, resume_ckpt, summary)
            print(f"[TEST] wrote {snapshot_dir / 'test_summary.csv'}")

        if run_root is not None:
            print(f"[DONE] {run_root}")


if __name__ == "__main__":
    main()
