from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import torch as tr
from lightning.pytorch import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

from src.config import build_experiment_dir, prepare_experiment_config, save_config
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

def handle_metrics(loggers, config, resume=False):
    csv_logger = next(
        (l for l in loggers if l.__class__.__name__ == "CSVLogger"), None
    )
    if csv_logger is None:
        raise RuntimeError("CSVLogger not found")

    log_cfg = config["logging"]

    src = Path(csv_logger.log_dir) / "metrics.csv"
    dst_raw = Path(log_cfg["lightning_dir"]) / "metrics_raw.csv"
    dst = Path(log_cfg["metrics_path"])

    dst_raw.parent.mkdir(parents=True, exist_ok=True)

    if dst_raw.exists() and resume:
        prev = dst_raw.read_text()
        new = src.read_text()
        dst_raw.write_text(prev + new)
        src.unlink()
    else:
        shutil.move(src, dst_raw)

    # compact CSV 
    with dst_raw.open() as f:
        rows = list(csv.DictReader(f))

    grouped = {}
    keys = []
    for r in rows:
        k = (r.get("epoch"), r.get("step"))
        grouped.setdefault(k, {"epoch": k[0], "step": k[1]})
        for m, v in r.items():
            if m in ("epoch", "step") or v in ("", None):
                continue
            grouped[k][m] = v
            if m not in keys:
                keys.append(m)

    with dst.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "step"] + keys)
        writer.writeheader()
        writer.writerows(grouped.values())

    return dst, dst_raw


def train(config, experiment_dir, logger, resume=None):
    log_cfg = config["logging"]
    train_cfg = config["training"]

    data = RNADataModule(config)
    model = RNADiffusionModule(config)
    loggers = build_loggers(config, experiment_dir)

    ckpt_cb = ModelCheckpoint(
        dirpath=experiment_dir / "checkpoints",
        filename="best",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )

    trainer = Trainer(
        max_epochs=train_cfg["max_epochs"],
        accelerator=train_cfg["accelerator"],
        devices=train_cfg["devices"],
        precision=train_cfg["precision"],
        logger=loggers,
        callbacks=[ckpt_cb],
        log_every_n_steps=log_cfg["log_every_n_steps"],
    )

    trainer.fit(model, datamodule=data, ckpt_path=resume)

    handle_metrics(loggers, config, resume=resume is not None)

    ckpt = ckpt_cb.best_model_path or ckpt_cb.last_model_path
    if not ckpt:
        raise RuntimeError("No checkpoint produced")

    logger.info("Training done. Checkpoint: %s", ckpt)
    return ckpt

def evaluate(config, experiment_dir, checkpoint, logger):
    log_cfg = config["logging"]
    ens_cfg = config["ensemble"]

    samples_dir = Path(log_cfg["raw_samples_dir"])
    samples_dir.mkdir(parents=True, exist_ok=True)

    # eliminar esto en el futuro
    for f in samples_dir.glob("*.pt"):
        f.unlink()

    model = load_model_checkpoint(config, checkpoint, eval_mode=True)
    loader = build_dataloader(config, partition="test", shuffle=False)

    generate_raw_samples(
        model=model,
        loader=loader,
        output_dir=samples_dir,
        num_samples=ens_cfg["num_samples"],
        base_seed=ens_cfg["base_seed"],
        chunk_size=ens_cfg["chunk_size"],
    )

    write_samples_metadata(
        samples_dir=samples_dir,
        checkpoint_path=checkpoint,
        num_samples=ens_cfg["num_samples"],
        base_seed=ens_cfg["base_seed"],
        chunk_size=ens_cfg["chunk_size"],
    )

    df = evaluate_samples_dir(
        samples_dir=samples_dir,
        consensus_sizes=ens_cfg["consensus_sizes"],
        trials=ens_cfg["trials"],
        seed=ens_cfg["base_seed"],
    )
    df.to_csv(log_cfg["ensemble_path"], index=False)

    stats_path = experiment_dir / "ensemble_stats.csv"
    stats = evaluate_samples_stats(
        samples_csv=log_cfg["ensemble_path"],
        consensus_sizes=ens_cfg["consensus_sizes"],
    )
    stats.to_csv(stats_path, index=False)

    write_ensemble_metadata(
        output_path=log_cfg["ensemble_path"],
        samples_dir=samples_dir,
        trials=ens_cfg["trials"],
        consensus_sizes=ens_cfg["consensus_sizes"],
        seed=ens_cfg["base_seed"],
        metadata_path=log_cfg["ensemble_metadata_path"],
    )

    logger.info("Evaluation done")
    
def run_experiment(config, experiment_dir=None, resume=None):
    config, exp_dir, logger = prepare_run(config, experiment_dir)

    ckpt = train(config, exp_dir, logger, resume)

    evaluate(config, exp_dir, ckpt, logger)

    return {
        "experiment_dir": str(exp_dir),
        "checkpoint": str(ckpt),
        "metrics": config["logging"]["metrics_path"],
        "ensemble": config["logging"]["ensemble_path"],
    }