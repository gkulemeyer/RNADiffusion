import argparse
import csv
import json
import shutil
from pathlib import Path

from lightning.pytorch import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

from src.config import load_config, prepare_experiment_config, save_config
from src.data import RNADataModule, build_dataloader
from src.ensemble import (
    DEFAULT_BASE_SEED,
    DEFAULT_CONSENSUS,
    DEFAULT_TRIALS,
    evaluate_samples_dir,
    generate_raw_samples,
)
from src.io import (
    build_experiment_dir,
    build_loggers,
    configure_logger,
    load_model_checkpoint,
)
from src.train_module import RNADiffusionModule


def parse_args():
    parser = argparse.ArgumentParser(description="Train RNADiffusion with PyTorch Lightning.")
    parser.add_argument(
        "--config",
        default="configs/train.yaml",
        help="Path to a YAML config file.",
    )
    return parser.parse_args()


def write_metrics_summary(metrics_path, summary_path):
    metrics_path = Path(metrics_path)
    summary_path = Path(summary_path)

    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
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
    merged_rows = list(grouped.values())
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)


def main():
    args = parse_args()
    config = load_config(args.config)
    config = prepare_experiment_config(config)
    experiment_dir = Path(build_experiment_dir(config))
    config = prepare_experiment_config(config, experiment_dir)

    experiment_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, experiment_dir)
    run_logger = configure_logger(config["logging"]["train_log_path"])

    seed_everything(config["experiment"]["seed"], workers=True)
    run_logger.info("Starting experiment in %s", experiment_dir)
    run_logger.info("Resolved config: %s", json.dumps(config, indent=2))

    data_module = RNADataModule(config)
    model = RNADiffusionModule(config)
    loggers = build_loggers(config, experiment_dir)
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
    trainer.fit(model=model, datamodule=data_module)

    csv_logger = next((logger for logger in loggers if logger.__class__.__name__ == "CSVLogger"), None)
    if csv_logger is None:
        raise FileNotFoundError("CSVLogger not found in configured loggers")
    source_metrics_path = Path(csv_logger.log_dir) / "metrics.csv"
    if not source_metrics_path.exists():
        raise FileNotFoundError(f"CSVLogger did not produce metrics.csv at {source_metrics_path}")
    metrics_path = experiment_dir / "metrics.csv"
    shutil.copyfile(source_metrics_path, metrics_path)
    run_logger.info("Training metrics saved to %s", metrics_path)
    metrics_summary_path = experiment_dir / "metrics_summary.csv"
    write_metrics_summary(metrics_path, metrics_summary_path)
    run_logger.info("Training metrics summary saved to %s", metrics_summary_path)

    best_checkpoint = checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
    if not best_checkpoint:
        raise FileNotFoundError("Lightning did not produce a best or last checkpoint")
    run_logger.info("Generating ensemble samples from %s", best_checkpoint)

    loader = build_dataloader(config, partition="test", shuffle=False)
    best_model = load_model_checkpoint(config, best_checkpoint, eval_mode=True)
    generate_raw_samples(
        model=best_model,
        loader=loader,
        output_dir=config["logging"]["raw_samples_dir"],
        base_seed=DEFAULT_BASE_SEED,
    )

    run_logger.info("Evaluating ensemble statistics")
    ensemble_df = evaluate_samples_dir(
        samples_dir=config["logging"]["raw_samples_dir"],
        consensus_sizes=DEFAULT_CONSENSUS,
        trials=DEFAULT_TRIALS,
        seed=DEFAULT_BASE_SEED,
    )
    ensemble_df.to_csv(config["logging"]["ensemble_path"], index=False)
    run_logger.info("Ensemble analysis saved to %s", config["logging"]["ensemble_path"])
    run_logger.info("Experiment finished. Logs saved to %s", experiment_dir)


if __name__ == "__main__":
    main()
