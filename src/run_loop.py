from __future__ import annotations

from src.config import clone_config
from src.experiment import evaluate_checkpoint, prepare_run, train
from src.run_io import RunIO, RunResolver
from src.io import configure_logger


def evaluate_periodic_checkpoints(run, logger):
    run = run if isinstance(run, RunIO) else RunIO(run)
    checkpoints = run.periodic_checkpoints()

    if not checkpoints:
        print(f"[WARN] no periodic checkpoints found in {run.periodic_ckpt_dir}")
        return

    for checkpoint_path in checkpoints:
        epoch = RunIO.checkpoint_epoch(checkpoint_path)
        target_dir = run.periodic_eval_dir(epoch)
        if (target_dir / "ensemble_stats.csv").exists():
            print(f"[SKIP] epoch_{epoch:03d} already evaluated")
            continue

        evaluate_checkpoint(
            run.root,
            checkpoint=checkpoint_path,
            output_dir=target_dir,
            keep_samples=False,
            logger=logger,
        )
        print(f"[EVAL] wrote {target_dir}")


def evaluate_run(run, logger=None, include_periodic=True):
    run = run if isinstance(run, RunIO) else RunIO(run)
    logger = logger or configure_logger(run.log_path)

    if include_periodic:
        evaluate_periodic_checkpoints(run, logger)

    if not run.best_ckpt_path.exists():
        print(f"[WARN] best checkpoint not found in {run.checkpoint_dir}")
        return run.root

    if run.best_eval_is_current():
        print(f"[SKIP] best checkpoint already evaluated in {run.best_eval_dir}")
        return run.root

    evaluate_checkpoint(run.root, checkpoint="best", logger=logger)
    print(f"[EVAL] wrote best checkpoint outputs in {run.best_eval_dir}")
    return run.root


def run_training_and_evaluation(config, job_dir, resume=True):
    config = clone_config(config)
    state = RunResolver(job_dir, resume=resume).build_state(config)
    run = state.run

    if run.run_is_complete(state.total_epochs):
        print(f"[DONE] already complete: {run.root}")
        return run.root

    if state.should_train:
        prepared_config, train_dir, logger = prepare_run(
            state.config.to_dict(),
            experiment_dir=run.train_dir,
        )
        train(
            prepared_config,
            train_dir,
            logger,
            resume=str(state.checkpoint_path) if state.checkpoint_path else None,
        )
    else:
        logger = configure_logger(run.log_path)
        print(f"[TRAIN] already complete at epoch {state.done_epochs}")

    evaluate_run(run, logger=logger, include_periodic=True)

    print(f"[DONE] {run.root}")
    return run.root 