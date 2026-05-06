from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
import time

import torch as tr
from lightning.pytorch import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, Callback

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


def periodic_checkpoint_dir(experiment_dir):
    return Path(experiment_dir) / "checkpoints" / "periodic"


def checkpoint_completed_epoch(checkpoint_path):
    checkpoint = tr.load(checkpoint_path, map_location="cpu")
    epoch = checkpoint.get("epoch")
    if epoch is None:
        raise ValueError(f"Checkpoint does not contain epoch metadata: {checkpoint_path}")
    return int(epoch) + 1


def list_periodic_checkpoints(experiment_dir):
    checkpoint_paths = sorted(periodic_checkpoint_dir(experiment_dir).glob("*.ckpt"))
    return sorted(checkpoint_paths, key=checkpoint_completed_epoch)


def normalize_periodic_checkpoint_names(experiment_dir):
    checkpoint_dir = periodic_checkpoint_dir(experiment_dir)
    for checkpoint_path in sorted(checkpoint_dir.glob("*.ckpt")):
        epoch = checkpoint_completed_epoch(checkpoint_path)
        target_path = checkpoint_dir / f"epoch{epoch:03d}.ckpt"
        if checkpoint_path == target_path or target_path.exists():
            continue
        checkpoint_path.rename(target_path)

def prepare_run(config, experiment_dir=None):
    if experiment_dir is None:
        base_config = prepare_experiment_config(config)
        experiment_dir = Path(build_experiment_dir(base_config))
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
    dst = Path(log_cfg["metrics_path"])

    dst.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    if resume and dst.exists():
        with dst.open() as handle:
            rows.extend(csv.DictReader(handle))

    with src.open() as handle:
        rows.extend(csv.DictReader(handle))
    src.unlink()

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

    return dst


class TimingCallback(Callback):
    def __init__(self, logger): 
        self.logger = logger
        self.train_time_s = None
        self.valid_time_s = None
    
    def on_train_epoch_start(self, trainer, pl_module):
        self.train_epoch_start = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        if self.train_epoch_start is not None:
            duration = time.perf_counter() - self.train_epoch_start
            self.logger.info(
                "Epoch %d train time: %.2f min",
                trainer.current_epoch, 
                duration / 60,
            )
    
    def on_validation_epoch_start(self, trainer, pl_module):
        self.valid_epoch_start = time.perf_counter()
        
    def on_validation_epoch_end(self, trainer, pl_module):
        if self.valid_epoch_start is not None:
            duration = time.perf_counter() - self.valid_epoch_start
            self.logger.info(
                "Epoch %d valid time: %.2f min",
                trainer.current_epoch,
                duration / 60,
            )

def train(config, experiment_dir, logger, resume=None):
    log_cfg = config["logging"]
    train_cfg = config["training"]
    checkpoint_interval = max(1, int(log_cfg.get("checkpoint_every_n_epochs", 1)))

    data = RNADataModule(config)
    model = RNADiffusionModule(config)
    loggers = build_loggers(config, experiment_dir)

    best_ckpt_cb = ModelCheckpoint(
        dirpath=experiment_dir / "checkpoints",
        filename="best",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        save_last=False,
        auto_insert_metric_name=False,
    )
    periodic_ckpt_cb = ModelCheckpoint(
        dirpath=periodic_checkpoint_dir(experiment_dir),
        filename="epoch{epoch:03d}",
        monitor=None,
        save_top_k=-1,
        save_last=False,
        every_n_epochs=checkpoint_interval,
        auto_insert_metric_name=False,
    )
    timer = TimingCallback(logger)
    
 
    trainer = Trainer(
        max_epochs=train_cfg["max_epochs"],
        accelerator=train_cfg["accelerator"],
        devices=train_cfg["devices"],
        precision=train_cfg["precision"],
        accumulate_grad_batches=train_cfg["accumulate_grad_batches"],
        check_val_every_n_epoch=max(1, int(train_cfg.get("check_val_every_n_epoch", 1))),
        logger=loggers,
        callbacks=[best_ckpt_cb, periodic_ckpt_cb, timer],
        log_every_n_steps=log_cfg["log_every_n_steps"],
    )
    trainer.fit(model, datamodule=data, ckpt_path=resume) 

    last_ckpt_path = Path(experiment_dir) / "checkpoints" / "last.ckpt"
    last_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(last_ckpt_path)
    normalize_periodic_checkpoint_names(experiment_dir)

    handle_metrics(loggers, config, resume=resume is not None)

    ckpt = best_ckpt_cb.best_model_path or str(last_ckpt_path)
    if not ckpt:
        raise RuntimeError("No checkpoint produced")

    logger.info("Training done. Checkpoint: %s", ckpt)
    return ckpt

def evaluate(config, experiment_dir, checkpoint, logger, cleanup_raw_samples=False):
    log_cfg = config["logging"]
    ens_cfg = config["ensemble"]

    samples_dir = Path(log_cfg["raw_samples_dir"])
    samples_dir.mkdir(parents=True, exist_ok=True)

    # eliminar esto en el futuro
    for f in samples_dir.glob("*.pt"):
        f.unlink()

    model = load_model_checkpoint(config, checkpoint, eval_mode=True)
    loader = build_dataloader(config, partition="test", shuffle=False)
    
    samples_start = time.perf_counter()
    generate_raw_samples(
        model=model,
        loader=loader,
        output_dir=samples_dir,
        num_samples=ens_cfg["num_samples"],
        base_seed=ens_cfg["base_seed"],
        chunk_size=ens_cfg["chunk_size"],
    )

    samples_time = time.perf_counter() - samples_start

    logger.info(
        "Timing: samples_generation_time=%.2fs (%.2f min), num_samples=%s, checkpoint=%s",
        samples_time,
        samples_time / 60,
        ens_cfg["num_samples"],
        Path(checkpoint).stem, 
    )

    write_samples_metadata(
        samples_dir=samples_dir,
        checkpoint_path=checkpoint,
        checkpoint_epoch=checkpoint_completed_epoch(checkpoint),
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

    if cleanup_raw_samples:
        shutil.rmtree(samples_dir, ignore_errors=True)

    logger.info("Evaluation done")


def periodic_eval_dir(run_dir, epoch):
    return Path(run_dir) / f"epoch_{int(epoch):03d}"
