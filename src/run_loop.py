from __future__ import annotations

import copy

import torch as tr

from src.experiment import evaluate_checkpoint, prepare_run, train
from src.run_io import RunIO


def _train(config, run, logger, resume, retry_on_oom):
    try:
        return train(config, run, logger, resume=resume)
    except tr.cuda.OutOfMemoryError:
        if not retry_on_oom:
            raise

    print("[OOM] training failed, retrying with batch size 1")
    retry_config = copy.deepcopy(config)
    retry_config["training"]["batch_size"] = 1
    return train(retry_config, run, logger, resume=resume)


def _evaluate(config, checkpoint, output_dir, retry_on_oom, keep_samples):
    try:
        return evaluate_checkpoint(
            config,
            checkpoint,
            output_dir,
            keep_samples=keep_samples,
        )
    except tr.cuda.OutOfMemoryError:
        if not retry_on_oom:
            raise

    print("[OOM] evaluation failed, retrying with batch size 1")
    return evaluate_checkpoint(
        config,
        checkpoint,
        output_dir,
        keep_samples=keep_samples,
        batch_size1=True,
    )


def evaluate_periodic_checkpoints(
    config,
    run,
    retry_on_oom=True,
):
    for checkpoint in run.periodic_checkpoints():
        epoch = run.checkpoint_epoch(checkpoint)
        _evaluate(
            config,
            checkpoint,
            run.periodic_eval_dir(epoch),
            retry_on_oom,
            False,
        )


def run_training_and_evaluation(
    config,
    job_dir,
    resume=True,
    retry_on_oom=True,
    evaluate_periodic=False,
    keep_samples=False,
):
    config = copy.deepcopy(config)
    run = RunIO(job_dir)
    logger = prepare_run(config, run)

    resume_checkpoint = run.last_checkpoint() if resume else None
    completed_epochs = run.completed_epoch_count() if resume_checkpoint else 0

    if completed_epochs < config["training"]["max_epochs"]:
        _train(config, run, logger, resume_checkpoint, retry_on_oom)
    else:
        print(f"[TRAIN] already complete at epoch {completed_epochs}")

    if config["model"]["evaluate"]:
        if evaluate_periodic:
            evaluate_periodic_checkpoints(
                config,
                run,
                retry_on_oom=retry_on_oom,
            )
        _evaluate(
            config,
            run.best_ckpt_path,
            run.best_eval_dir,
            retry_on_oom,
            keep_samples,
        )

    print(f"[DONE] {run.root}")
    return run.root
