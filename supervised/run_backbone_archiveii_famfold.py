#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import clone_config, load_config, to_config_dict
from src.run_io import RunIO

from supervised.backbone_experiment import (
    evaluate_backbone_checkpoint,
    run_backbone_experiment,
)

# ------------------------------------------------------------
# setup
# ------------------------------------------------------------
BASE_CONFIG_PATH = "configs/train/famfold.yaml"
RESUME = True
PRECISION = "16-mixed"
BASE_DIM = [32]

BATCH_SIZE = 8
ACCUMULATE_GRAD_BATCHES = 1
LR = 0.001

VAL_EVERY_N_EPOCHS = 1
CHECKPOINT_EVERY_N_EPOCHS = 1

FOLDS = ["grp1", "tmRNA", "23s", "telomerase", "RNaseP", "5s", "srp", "tRNA", "16s"] 
EPOCHS = 20

EXPERIMENT_NAME = "ArchiveII_backbone_famfold_20e"
BACKBONE_IN_CHANNELS = 16


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def build_job_dir(experiment_name, fold, base_dim, seed):
    return Path("supervised") / "logs" / experiment_name / fold / f"bd{base_dim}" 


def latest_attempt_dir(job_dir):
    attempt_dirs = sorted(
        path for path in Path(job_dir).glob("attempt_*") if path.is_dir()
    )
    if not attempt_dirs:
        return None
    return attempt_dirs[-1]


def next_attempt_dir(job_dir):
    index = 1
    while True:
        attempt_dir = Path(job_dir) / f"attempt_{index:03d}"
        if not attempt_dir.exists():
            return attempt_dir
        index += 1


def resolve_run_dir(job_dir, resume):
    job_dir = Path(job_dir)

    if resume:
        latest_attempt = latest_attempt_dir(job_dir)
        if latest_attempt is not None and RunIO(latest_attempt).train_dir.exists():
            return latest_attempt
        return job_dir

    if not job_dir.exists():
        return job_dir
    if RunIO(job_dir).train_dir.exists() or latest_attempt_dir(job_dir) is not None:
        return next_attempt_dir(job_dir)
    return job_dir


def build_run_config(base_config, experiment_name, repo_root, fold, epochs, base_dim):
    config = clone_config(base_config)
    repo_root = Path(repo_root)

    seed = int(config.experiment.seed)
    job_dir = build_job_dir(experiment_name, fold, base_dim, seed)

    config.experiment.name = job_dir.name
    config.experiment.note = (
        f"{experiment_name} | {fold}, bd={base_dim}, e={epochs}, "
        f"val={VAL_EVERY_N_EPOCHS}, ckpt={CHECKPOINT_EVERY_N_EPOCHS}"
    )
    config.data.base_path = str(repo_root / "data" / "ArchiveII.csv")
    config.data.partition_path = str(repo_root / "data" / "famfold" / "ArchiveII_famfold.csv")
    config.data.partition_scheme = "famfold"
    config.data.fold = fold
    config.model.in_channels = BACKBONE_IN_CHANNELS
    config.model.out_channels = 2
    config.model.base_dim = base_dim
    config.training.max_epochs = epochs
    config.training.check_val_every_n_epoch = VAL_EVERY_N_EPOCHS
    config.training.precision = PRECISION
    config.training.batch_size = BATCH_SIZE
    config.training.accumulate_grad_batches = ACCUMULATE_GRAD_BATCHES
    config.training.lr = LR
    config.logging.checkpoint_every_n_epochs = CHECKPOINT_EVERY_N_EPOCHS
    config.logging.save_dir = str((repo_root / job_dir.parent).resolve())

    return config, repo_root / job_dir


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    repo_root = REPO_ROOT
    base_config = to_config_dict(load_config(repo_root / BASE_CONFIG_PATH))

    jobs = [(fold, base_dim) for fold in FOLDS for base_dim in BASE_DIM]

    for index, (fold, base_dim) in enumerate(jobs, start=1):
        config, job_dir = build_run_config(
            base_config,
            EXPERIMENT_NAME,
            repo_root,
            fold,
            EPOCHS,
            base_dim,
        )
        run_dir = resolve_run_dir(job_dir, RESUME)

        print(f"\n[{index}/{len(jobs)}] {run_dir}")

        base_path = Path(config.data.base_path)
        partition_path = Path(config.data.partition_path)
        if not base_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {base_path}")
        if not partition_path.exists():
            raise FileNotFoundError(f"Partition file not found: {partition_path}")

        config.experiment.name = run_dir.name

        run = RunIO(run_dir)
        resume_ckpt = run.last_checkpoint() if run.train_dir.exists() else None
        current_epoch_count = run.completed_epoch_count() if run.train_dir.exists() else 0

        if run.train_dir.exists():
            print(f"[RESUME] {run_dir} from {current_epoch_count} completed epochs")

        if current_epoch_count < EPOCHS:
            result = run_backbone_experiment(
                config.to_dict(),
                run_root=run_dir,
                resume=str(resume_ckpt) if resume_ckpt is not None else None,
            )
            run_dir = Path(result["experiment_dir"])
            run = RunIO(run_dir)
            resume_ckpt = Path(result["last_checkpoint"])
            current_epoch_count = run.completed_epoch_count()
            print(f"[TRAIN] reached {current_epoch_count} completed epochs in {run_dir}")
        else:
            print(f"[TRAIN] already complete at {current_epoch_count} completed epochs")

        if resume_ckpt is None:
            raise FileNotFoundError(f"Last checkpoint not found in {run.checkpoint_dir}")

        summary_path = run.best_eval_dir / "test_summary.csv"
        evaluate_backbone_checkpoint(
            load_config(run.train_dir / "config.yaml"),
            resume_ckpt,
            run_dir=run_dir,
            output_path=summary_path,
        )
        print(f"[TEST] wrote {summary_path}")
        print(f"[DONE] {run_dir}")


if __name__ == "__main__":
    main()
