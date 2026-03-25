from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import torch as tr
from lightning.pytorch import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

from src.config import build_experiment_name, prepare_experiment_config, save_config
from src.data import RNADataModule, build_dataloader
from src.ensemble import (
    generate_raw_samples,
    evaluate_samples_dir,
    evaluate_samples_stats,
)
from src.io import (
    build_loggers,
    configure_logger,
    load_model_checkpoint,
    write_ensemble_metadata,
    write_samples_metadata,
)
from src.train_module import RNADiffusionModule


def write_metrics_csv(raw_metrics_path, output_path):
    raw_metrics_path = Path(raw_metrics_path)
    output_path = Path(output_path)

    with raw_metrics_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    grouped = {}
    metric_names = []
    for row in rows:
        key = (row.get("epoch", ""), row.get("step", ""))
        if key not in grouped:
            grouped[key] = {"epoch": key[0], "step": key[1]}

        for metric_name, metric_value in row.items():
            if metric_name in ("epoch", "step"):
                continue
            if metric_name not in metric_names:
                metric_names.append(metric_name)
            if metric_value not in ("", None):
                grouped[key][metric_name] = metric_value

    fieldnames = ["epoch", "step"] + metric_names
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(grouped.values())


def merge_csv_files(source_paths, output_path):
    rows = []
    fieldnames = []

    for source_path in source_paths:
        source_path = Path(source_path)
        if not source_path.exists():
            continue

        with source_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                for fieldname in row.keys():
                    if fieldname not in fieldnames:
                        fieldnames.append(fieldname)
                rows.append(row)

    if not rows:
        return

    with Path(output_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _export_lightning_metrics(loggers, config):
    csv_logger = next((logger for logger in loggers if logger.__class__.__name__ == "CSVLogger"), None)
    if csv_logger is None:
        raise FileNotFoundError("CSVLogger not found in configured loggers")

    source_metrics_path = Path(csv_logger.log_dir) / "metrics.csv"
    if not source_metrics_path.exists():
        raise FileNotFoundError(f"CSVLogger did not produce metrics.csv at {source_metrics_path}")

    lightning_dir = Path(config["logging"]["lightning_dir"])
    lightning_dir.mkdir(parents=True, exist_ok=True)

    raw_metrics_path = lightning_dir / "metrics_raw.csv"
    if raw_metrics_path.exists():
        raw_metrics_path.unlink()
    shutil.move(str(source_metrics_path), raw_metrics_path)

    metrics_path = Path(config["logging"]["metrics_path"])
    write_metrics_csv(raw_metrics_path, metrics_path)
    return metrics_path, raw_metrics_path


def _resolve_checkpoint_path(checkpoint_callback):
    checkpoint_path = checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
    if not checkpoint_path:
        raise FileNotFoundError("Lightning did not produce a best or last checkpoint")
    return checkpoint_path


def completed_epochs_from_checkpoint(checkpoint_path):

    checkpoint = tr.load(checkpoint_path, map_location="cpu")
    epoch = checkpoint.get("epoch")
    if epoch is None:
        return None
    return int(epoch) + 1


def archive_epoch_artifacts(experiment_dir, epoch_count, checkpoint_path, run_logger):
    if epoch_count is None:
        return checkpoint_path

    experiment_dir = Path(experiment_dir)
    archive_dir = experiment_dir / f"epoch_{epoch_count}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    stats_path = experiment_dir / "ensemble_stats.csv"
    if stats_path.exists():
        archived_stats_path = archive_dir / "ensemble_stats.csv"
        if archived_stats_path.exists():
            archived_stats_path.unlink()
        shutil.move(str(stats_path), archived_stats_path)
        run_logger.info("Archived previous ensemble summary stats to %s", archived_stats_path)

    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        archived_checkpoint_path = archive_dir / checkpoint_path.name
        if archived_checkpoint_path.exists():
            archived_checkpoint_path.unlink()
        shutil.move(str(checkpoint_path), archived_checkpoint_path)
        run_logger.info("Archived previous last checkpoint to %s", archived_checkpoint_path)
        return archived_checkpoint_path

    return checkpoint_path


def build_experiment_dir(config):
    base_dir = Path(config["logging"]["save_dir"])
    experiment_name = build_experiment_name(config)
    experiment_dir = base_dir / experiment_name

    if not experiment_dir.exists():
        return experiment_dir

    suffix = 1
    while True:
        candidate = Path(f"{experiment_dir}_{suffix}")
        if not candidate.exists():
            return candidate
        suffix += 1


def prepare_run(config, experiment_dir=None):
    config = prepare_experiment_config(config)

    if experiment_dir is None:
        experiment_dir = build_experiment_dir(config)
    else:
        experiment_dir = Path(experiment_dir)
        
    config = prepare_experiment_config(config, experiment_dir)

    experiment_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, experiment_dir)
    run_logger = configure_logger(config["logging"]["train_log_path"])

    seed_everything(config["experiment"]["seed"], workers=True)
    run_logger.info("Starting experiment in %s", experiment_dir)
    run_logger.info("Resolved config: %s", json.dumps(config, indent=2))
    return config, experiment_dir, run_logger


def train_run(config, experiment_dir, run_logger, resume_from_checkpoint=None):
    data_module = RNADataModule(config)
    model = RNADiffusionModule(config)
    loggers = build_loggers(config, experiment_dir)
    previous_raw_metrics_backup = None
    if resume_from_checkpoint is not None:
        previous_raw_metrics = Path(config["logging"]["lightning_dir"]) / "metrics_raw.csv"
        if previous_raw_metrics.exists():
            previous_raw_metrics_backup = experiment_dir / "lightning" / "metrics_raw_previous.csv"
            shutil.copy2(previous_raw_metrics, previous_raw_metrics_backup)
    checkpoint_callback = ModelCheckpoint(
        dirpath=experiment_dir / "checkpoints",
        filename="best",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    trainer = Trainer(
        max_epochs=config["training"]["max_epochs"],
        accelerator=config["training"]["accelerator"],
        devices=config["training"]["devices"],
        precision=config["training"]["precision"],
        logger=loggers,
        callbacks=[checkpoint_callback],
        log_every_n_steps=config["logging"]["log_every_n_steps"],
    )
    trainer.fit(model=model, datamodule=data_module, ckpt_path=resume_from_checkpoint)

    metrics_path, raw_metrics_path = _export_lightning_metrics(loggers, config)
    if previous_raw_metrics_backup is not None and previous_raw_metrics_backup.exists():
        merge_csv_files([previous_raw_metrics_backup, raw_metrics_path], raw_metrics_path)
        write_metrics_csv(raw_metrics_path, metrics_path)
        previous_raw_metrics_backup.unlink()
    run_logger.info("Training metrics saved to %s", metrics_path)
    run_logger.info("Raw Lightning metrics saved to %s", raw_metrics_path)

    checkpoint_path = _resolve_checkpoint_path(checkpoint_callback)
    return checkpoint_path


def finalize_run(config, experiment_dir, checkpoint_path, run_logger):
    ensemble_config = config["ensemble"]
    run_logger.info("Generating ensemble samples from %s", checkpoint_path)

    raw_samples_dir = Path(config["logging"]["raw_samples_dir"])
    raw_samples_dir.mkdir(parents=True, exist_ok=True)
    for sample_path in raw_samples_dir.glob("*.pt"):
        sample_path.unlink()
    samples_metadata_path = raw_samples_dir / "samples_metadata.yaml"
    if samples_metadata_path.exists():
        samples_metadata_path.unlink()

    loader = build_dataloader(config, partition="test", shuffle=False)
    best_model = load_model_checkpoint(config, checkpoint_path, eval_mode=True)
    generate_raw_samples(
        model=best_model,
        loader=loader,
        output_dir=config["logging"]["raw_samples_dir"],
        num_samples=ensemble_config["num_samples"],
        base_seed=ensemble_config["base_seed"],
        chunk_size=ensemble_config["chunk_size"],
    )
    write_samples_metadata(
        samples_dir=config["logging"]["raw_samples_dir"],
        checkpoint_path=checkpoint_path,
        num_samples=ensemble_config["num_samples"],
        base_seed=ensemble_config["base_seed"],
        chunk_size=ensemble_config["chunk_size"],
    )

    run_logger.info("Evaluating ensemble statistics")
    ensemble_df = evaluate_samples_dir(
        samples_dir=config["logging"]["raw_samples_dir"],
        consensus_sizes=ensemble_config["consensus_sizes"],
        trials=ensemble_config["trials"],
        seed=ensemble_config["base_seed"],
    )
    ensemble_df.to_csv(config["logging"]["ensemble_path"], index=False)

    ensemble_stats_path = experiment_dir / "ensemble_stats.csv"
    ensemble_stats_df = evaluate_samples_stats(
        samples_csv=config["logging"]["ensemble_path"],
        consensus_sizes=ensemble_config["consensus_sizes"],
    )
    ensemble_stats_df.to_csv(ensemble_stats_path, index=False)

    write_ensemble_metadata(
        output_path=config["logging"]["ensemble_path"],
        samples_dir=config["logging"]["raw_samples_dir"],
        trials=ensemble_config["trials"],
        consensus_sizes=ensemble_config["consensus_sizes"],
        seed=ensemble_config["base_seed"],
        metadata_path=config["logging"]["ensemble_metadata_path"],
    )
    run_logger.info("Ensemble analysis saved to %s", config["logging"]["ensemble_path"])
    run_logger.info("Ensemble summary stats saved to %s", ensemble_stats_path)
    run_logger.info("Ensemble metadata saved to %s", config["logging"]["ensemble_metadata_path"])
    run_logger.info("Experiment finished. Logs saved to %s", experiment_dir)


def run_experiment(config, experiment_dir=None, resume_from_checkpoint=None):
    config, experiment_dir, run_logger = prepare_run(config, experiment_dir=experiment_dir)
    if resume_from_checkpoint is not None:
        completed_epochs = completed_epochs_from_checkpoint(resume_from_checkpoint)
        resume_from_checkpoint = archive_epoch_artifacts(
            experiment_dir,
            completed_epochs,
            resume_from_checkpoint,
            run_logger,
        )
    checkpoint_path = train_run(
        config,
        experiment_dir,
        run_logger,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    finalize_run(config, experiment_dir, checkpoint_path, run_logger)
    return {
        "config": config,
        "experiment_dir": str(experiment_dir),
        "metrics_path": config["logging"]["metrics_path"],
        "checkpoint_path": str(checkpoint_path),
        "ensemble_path": config["logging"]["ensemble_path"],
        "ensemble_stats_path": str(experiment_dir / "ensemble_stats.csv"),
        "ensemble_metadata_path": config["logging"]["ensemble_metadata_path"],
        "raw_samples_dir": config["logging"]["raw_samples_dir"],
    }
