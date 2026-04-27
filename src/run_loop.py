from __future__ import annotations

from pathlib import Path

from ml_collections import ConfigDict
import yaml

from src.config import load_config, prepare_experiment_config
from src.experiment import (
    checkpoint_completed_epoch,
    evaluate,
    list_periodic_checkpoints,
    prepare_run,
    train,
)
from src.io import configure_logger
from src.sweeps import completed_epochs, last_checkpoint_path


def _save_dir_structure(run_root, legacy=False):
    run_root = Path(run_root)
    return run_root / "train", run_root / "eval" / "best", run_root / "eval" / "periodic"


def _latest_attempt_dir(job_dir):
    attempt_dirs = sorted(
        path for path in Path(job_dir).glob("attempt_*") if path.is_dir()
    )
    if not attempt_dirs:
        return None
    return attempt_dirs[-1]


def _next_attempt_dir(job_dir):
    index = 1
    while True:
        attempt_dir = Path(job_dir) / f"attempt_{index:03d}"
        if not attempt_dir.exists():
            return attempt_dir
        index += 1


def _resolve_run_root(config, job_dir, resume):
    job_dir = Path(job_dir)

    if resume:
        if (job_dir / "train").exists():
            config.experiment.name = job_dir.name
            return job_dir

        latest_attempt = _latest_attempt_dir(job_dir)
        if latest_attempt is not None and (latest_attempt / "train").exists():
            config.experiment.name = latest_attempt.name
            return latest_attempt

        config.experiment.name = job_dir.name
        return job_dir

    if not job_dir.exists():
        config.experiment.name = job_dir.name
        return job_dir

    if (job_dir / "train").exists() or _latest_attempt_dir(job_dir) is not None:
        attempt_dir = _next_attempt_dir(job_dir)
        config.experiment.name = attempt_dir.name
        return attempt_dir

    config.experiment.name = job_dir.name
    return job_dir

def _best_eval_matches_checkpoint(best_eval_dir, best_ckpt):
    metadata_path = Path(best_eval_dir) / "ensemble_metadata.yaml"
    if not metadata_path.exists():
        return False

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle) or {}

    if metadata.get("checkpoint_path") != str(best_ckpt):
        return False

    stored_epoch = metadata.get("checkpoint_epoch")
    if stored_epoch is None:
        return True
    return int(stored_epoch) == checkpoint_completed_epoch(best_ckpt)


def _evaluate_periodic_checkpoints(train_dir, periodic_eval_root, logger):
    run_config = load_config(train_dir / "config.yaml")
    checkpoint_paths = list_periodic_checkpoints(train_dir)

    if not checkpoint_paths:
        print(f"[WARN] no periodic checkpoints found in {train_dir / 'checkpoints' / 'periodic'}")
        return

    for checkpoint_path in checkpoint_paths:
        epoch = checkpoint_completed_epoch(checkpoint_path)
        target_dir = Path(periodic_eval_root) / f"epoch_{epoch:03d}"
        if (target_dir / "ensemble_stats.csv").exists():
            print(f"[SKIP] epoch_{epoch:03d} already evaluated")
            continue

        eval_config = prepare_experiment_config(run_config, target_dir)
        evaluate(
            eval_config,
            target_dir,
            checkpoint_path,
            logger,
            cleanup_raw_samples=True,
        )
        print(f"[EVAL] wrote {target_dir}")


def _evaluate_best_checkpoint(train_dir, best_eval_dir, logger):
    best_ckpt = Path(train_dir) / "checkpoints" / "best.ckpt"
    if not best_ckpt.exists():
        print(f"[WARN] best checkpoint not found in {Path(train_dir) / 'checkpoints'}")
        return

    if (Path(best_eval_dir) / "ensemble_stats.csv").exists() and _best_eval_matches_checkpoint(
        best_eval_dir, best_ckpt
    ):
        print(f"[SKIP] best checkpoint already evaluated in {best_eval_dir}")
        return

    eval_config = load_config(Path(train_dir) / "config.yaml")
    eval_config = prepare_experiment_config(eval_config, best_eval_dir)
    evaluate(eval_config, best_eval_dir, best_ckpt, logger, cleanup_raw_samples=False)
    print(f"[EVAL] wrote best checkpoint outputs in {best_eval_dir}")

def run_training_and_evaluation(config, job_dir, resume=True):
    requested_epochs = int(config.training.max_epochs)
    requested_val_every = int(config.training.check_val_every_n_epoch)
    requested_ckpt_every = int(config.logging.checkpoint_every_n_epochs)

    run_root = _resolve_run_root(config, job_dir, resume)
    train_dir, best_eval_dir, periodic_eval_root = _save_dir_structure(run_root)

    effective_config = config
    config_path = train_dir / "config.yaml"
    previous_epochs = requested_epochs

    if config_path.exists():
        effective_config = ConfigDict(load_config(config_path))
        previous_epochs = int(effective_config.training.max_epochs)

    effective_config.training.max_epochs = requested_epochs
    effective_config.training.check_val_every_n_epoch = requested_val_every
    effective_config.logging.checkpoint_every_n_epochs = requested_ckpt_every

    resume_ckpt = last_checkpoint_path(train_dir) if train_dir.exists() else None
    done_epochs = 0

    if train_dir.exists():
        completed = completed_epochs(train_dir)
        if completed is not None:
            done_epochs = min(int(completed), previous_epochs)
        print(f"[RESUME] {train_dir} from epoch {done_epochs}")

    if done_epochs < requested_epochs:
        prepared_config, train_dir, logger = prepare_run(
            effective_config.to_dict(),
            experiment_dir=train_dir,
        )
        train(
            prepared_config,
            train_dir,
            logger,
            resume=str(resume_ckpt) if resume_ckpt else None,
        )
        completed = completed_epochs(train_dir)
        done_epochs = 0 if completed is None else min(int(completed), requested_epochs)
        print(f"[TRAIN] reached epoch {done_epochs} in {train_dir}")
    else:
        logger = configure_logger(Path(train_dir) / "run.log")
        print(f"[TRAIN] already complete at epoch {done_epochs}")

    _evaluate_periodic_checkpoints(train_dir, periodic_eval_root, logger)
    _evaluate_best_checkpoint(train_dir, best_eval_dir, logger)
    print(f"[DONE] {run_root}")

    return run_root