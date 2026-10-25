#!/usr/bin/env python
from __future__ import annotations

from copy import deepcopy
from itertools import product
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.run_io import RunIO

from supervised.backbone_experiment_weighted_loss import (
    evaluate_backbone_checkpoint,
    run_backbone_experiment,
)

# Configuración
BASE_CONFIG_PATH = "configs/train/default.yaml"
RESUME = True
PRECISION = "16-mixed"

BASE_DIM = [32]
W = 0.02
LAMBDA_INV = 1.0

BATCH_SIZE = 8
ACCUMULATE_GRAD_BATCHES = 1
LR = 0.001

VAL_EVERY_N_EPOCHS = 1
CHECKPOINT_EVERY_N_EPOCHS = 1

PARTITIONS = ["sim40", "sim50", "sim60", "sim70", "sim80", "sim90"]
# PARTITIONS = ["sim40"]
FOLDS = [0]
EPOCHS = 100
# EPOCHS = 2

EXPERIMENT_NAME = f"test_ArchiveII_backbone_simfold_w_{W}_lambda_{LAMBDA_INV}_100e"
BACKBONE_IN_CHANNELS = 16


def resolve_run_dir(job_dir, resume):
    """Reanuda el último intento o crea uno nuevo sin pisar resultados."""
    attempts = sorted(path for path in job_dir.glob("attempt_*") if path.is_dir())

    if resume:
        if attempts and RunIO(attempts[-1]).train_dir.exists():
            return attempts[-1]
        return job_dir

    if not RunIO(job_dir).train_dir.exists() and not attempts:
        return job_dir

    index = 1
    while (job_dir / f"attempt_{index:03d}").exists():
        index += 1
    return job_dir / f"attempt_{index:03d}"


def main():
    base_config = load_config(REPO_ROOT / BASE_CONFIG_PATH)
    jobs = list(product(FOLDS, PARTITIONS, BASE_DIM))

    for index, (fold, sim, base_dim) in enumerate(jobs, start=1):
        job_dir = (
            REPO_ROOT / "supervised" / "logs" / EXPERIMENT_NAME
            / sim / str(fold) / f"bd{base_dim}"
        )
        run_dir = resolve_run_dir(job_dir, RESUME)
        run = RunIO(run_dir)
        print(f"\n[{index}/{len(jobs)}] {run_dir}")

        # Configuración de esta combinación de fold, similitud y modelo.
        config = deepcopy(base_config)
        config["experiment"].update(
            name=run_dir.name,
            note=f"{EXPERIMENT_NAME} | {sim}, fold={fold}, bd={base_dim}, e={EPOCHS}",
        )
        config["data"].update(
            base_path=str(REPO_ROOT / "data" / "ArchiveII.csv"),
            partition_path=str(
                REPO_ROOT / "data" / "simfolds" / "bootstrap"
                / f"archiveII_similars_splits_less_than_512_k10_boostrap_{sim}-fold.csv"
            ),
            partition_scheme="simfold",
            fold=fold,
        )
        config["model"].update(
            in_channels=BACKBONE_IN_CHANNELS,
            out_channels=2,
            base_dim=base_dim,
            w=W,
            lambda_inv=LAMBDA_INV,
        )
        config["training"].update(
            max_epochs=EPOCHS,
            check_val_every_n_epoch=VAL_EVERY_N_EPOCHS,
            precision=PRECISION,
            batch_size=BATCH_SIZE,
            accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
            lr=LR,
        )
        config["logging"].update(
            checkpoint_every_n_epochs=CHECKPOINT_EVERY_N_EPOCHS,
            save_dir=str(run_dir.parent),
        )

        # Entrenamiento desde cero o desde el último checkpoint.
        checkpoint = run.last_checkpoint()
        completed_epochs = run.completed_epoch_count()
        if completed_epochs < EPOCHS:
            print(f"[TRAIN] {completed_epochs} -> {EPOCHS} épocas")
            result = run_backbone_experiment(
                config,
                run_root=run_dir,
                resume=str(checkpoint) if checkpoint else None,
            )
            checkpoint = result["last_checkpoint"]
        else:
            print(f"[TRAIN] completo: {completed_epochs} épocas")

        # Evaluación del último checkpoint en test.
        summary_path = run.best_eval_dir / "test_summary.csv"
        evaluate_backbone_checkpoint(
            load_config(run.config_path),
            checkpoint,
            run_dir=run_dir,
            output_path=summary_path,
        )
        print(f"[TEST] wrote {summary_path}")


if __name__ == "__main__":
    main()
