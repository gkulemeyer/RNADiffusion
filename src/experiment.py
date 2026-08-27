from __future__ import annotations

import copy
import csv
import json
import shutil
from pathlib import Path
import time

import torch as tr
from lightning.pytorch import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, Callback

from src.data import RNADataModule, build_dataloader
from src.train_module import (
    RNADiffusionModule,
    TestMetricsCollector,
    load_rna_module_checkpoint,
)
from src.ensemble import (
    evaluate_samples_dir,
    evaluate_samples_stats,
    export_db_ensemble,
    save_ensemble_samples,
)

from src.config import (
    load_ensemble_defaults,
    save_config,
)
from src.run_io import RunIO
from src.io import (
    build_loggers,
    configure_logger,
    handle_metrics,
    load_model_checkpoint,
    write_ensemble_metadata,
    write_samples_metadata,
)


def prepare_run(config, run):
    run.train_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run.train_dir)

    run_logger = configure_logger(run.log_path)
    seed_everything(config["experiment"]["seed"], workers=True)

    run_logger.info("Starting experiment in %s", run.train_dir)
    run_logger.info("Resolved config: %s", json.dumps(config, indent=2))
    return run_logger

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

def train(config, run, logger, resume=None):
    log_cfg = config["logging"]
    train_cfg = config["training"]
    validation_interval = int(train_cfg["check_val_every_n_epoch"])

    data = RNADataModule(config)
    model = RNADiffusionModule(config)
    loggers = build_loggers(config, run.train_dir)

    best_ckpt_cb = ModelCheckpoint(
        dirpath=run.checkpoint_dir,
        filename="best",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        save_last=False,
        auto_insert_metric_name=False,
    )
    periodic_ckpt_cb = ModelCheckpoint(
        dirpath=run.periodic_ckpt_dir,
        filename="epoch{epoch:03d}",
        monitor=None,
        save_top_k=-1,
        save_last=False,
        every_n_epochs=validation_interval,
        auto_insert_metric_name=False,
    )
    timer = TimingCallback(logger)
    
 
    trainer = Trainer(
        max_epochs=train_cfg["max_epochs"],
        accelerator=train_cfg["accelerator"],
        devices=train_cfg["devices"],
        precision=train_cfg["precision"],
        accumulate_grad_batches=train_cfg["accumulate_grad_batches"],
        check_val_every_n_epoch=int(train_cfg["check_val_every_n_epoch"]),
        logger=loggers,
        callbacks=[best_ckpt_cb, periodic_ckpt_cb, timer],
        log_every_n_steps=log_cfg["log_every_n_steps"],
    )
    trainer.fit(model, datamodule=data, ckpt_path=resume) 

    last_ckpt_path = run.last_ckpt_path
    last_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(last_ckpt_path)

    handle_metrics(loggers, run.metrics_path, resume=resume is not None)

    ckpt = best_ckpt_cb.best_model_path or str(last_ckpt_path)
    if not ckpt:
        raise RuntimeError("No checkpoint produced")

    logger.info("Training done. Checkpoint: %s", ckpt)
    return ckpt


def _stats_output_path(output_path):
    output_path = Path(output_path)
    if output_path.name.endswith("_ensemble_metrics.csv"):
        return output_path.with_name(
            output_path.name.replace("_ensemble_metrics.csv", "_ensemble_stats.csv")
        )
    return output_path.with_name(f"{output_path.stem}_stats.csv")


def generate_ensemble_samples(
    config,
    checkpoint,
    samples_dir,
    num_samples=None,
    base_seed=None,
    batch_size=None,
    clear_samples=False,
    threshold=None,
):
    ens_cfg = config["ensemble"]
    samples_dir = Path(samples_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)

    if clear_samples:
        for sample_path in samples_dir.glob("*.pt"):
            sample_path.unlink()

    model = load_model_checkpoint(config, checkpoint, eval_mode=True)
    loader = build_dataloader(
        config,
        partition="test",
        batch_size=batch_size,
        shuffle=False,
    )

    num_samples = int(ens_cfg["num_samples"] if num_samples is None else num_samples)
    base_seed = int(ens_cfg["base_seed"] if base_seed is None else base_seed)
    threshold = float(ens_cfg["threshold"] if threshold is None else threshold)
    sample_seeds = [
        base_seed + sample_id
        for sample_id in range(num_samples)
    ]

    save_ensemble_samples(
        model=model,
        loader=loader,
        output_dir=samples_dir,
        sample_seeds=sample_seeds,
        threshold=threshold,
    )

    write_samples_metadata(
        samples_dir=samples_dir,
        checkpoint_path=checkpoint,
        checkpoint_epoch=RunIO.checkpoint_epoch(checkpoint),
        num_samples=num_samples,
        base_seed=base_seed,
        sample_seeds=sample_seeds,
        threshold=threshold,
    )
    return samples_dir


def evaluate_ensemble_samples(
    samples_dir,
    output_path=None,
    trials=None,
    consensus_sizes=None,
    seed=None,
    metadata_path=None,
    get_best_and_worst=None,
    sample_type="raw",
):
    samples_dir = Path(samples_dir)
    output_path = Path(
        output_path
        or samples_dir.parent / f"{sample_type}_ensemble_metrics.csv"
    )
    ens_cfg = load_ensemble_defaults()

    trials = int(trials if trials is not None else ens_cfg["trials"])
    consensus_sizes = list(consensus_sizes or ens_cfg["consensus_sizes"])
    seed = int(seed if seed is not None else ens_cfg["base_seed"])
    get_best_and_worst = bool(
        ens_cfg["get_best_and_worst"]
        if get_best_and_worst is None
        else get_best_and_worst
    )

    df = evaluate_samples_dir(
        samples_dir=samples_dir,
        consensus_sizes=consensus_sizes,
        trials=trials,
        seed=seed,
        get_best_and_worst=get_best_and_worst,
        sample_type=sample_type,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    stats_path = _stats_output_path(output_path)
    stats = evaluate_samples_stats(
        samples_csv=output_path,
        consensus_sizes=consensus_sizes,
        get_best_and_worst=get_best_and_worst,
    )
    stats.to_csv(stats_path, index=False)

    write_ensemble_metadata(
        output_path=output_path,
        samples_dir=samples_dir,
        trials=trials,
        consensus_sizes=consensus_sizes,
        seed=seed,
        get_best_and_worst=get_best_and_worst,
        metadata_path=metadata_path,
    )
    return output_path, stats_path


def evaluate_checkpoint_ensemble(
    config,
    checkpoint,
    output_dir,
    logger,
    keep_samples=False,
):
    ens_cfg = config["ensemble"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"

    samples_start = time.perf_counter()
    generate_ensemble_samples(
        config=config,
        checkpoint=checkpoint,
        samples_dir=samples_dir,
        num_samples=ens_cfg["num_samples"],
        base_seed=ens_cfg["base_seed"],
        threshold=ens_cfg["threshold"],
        clear_samples=True,
    )
    samples_time = time.perf_counter() - samples_start

    logger.info(
        "Timing: samples_generation_time=%.2fs (%.2f min), num_samples=%s, checkpoint=%s",
        samples_time,
        samples_time / 60,
        ens_cfg["num_samples"],
        Path(checkpoint).stem,
    )

    for sample_type in ("raw", "processed"):
        evaluate_ensemble_samples(
            samples_dir=samples_dir,
            output_path=output_dir / f"{sample_type}_ensemble_metrics.csv",
            trials=ens_cfg["trials"],
            consensus_sizes=ens_cfg["consensus_sizes"],
            seed=ens_cfg["base_seed"],
            metadata_path=output_dir / "ensemble_metadata.yaml",
            get_best_and_worst=ens_cfg["get_best_and_worst"],
            sample_type=sample_type,
        )

    export_db_ensemble(
        samples_dir=samples_dir,
        output_csv=output_dir / "generated_ensemble.csv",
    )

    if not keep_samples:
        shutil.rmtree(samples_dir, ignore_errors=True)

    logger.info("Evaluation done")


def evaluate_checkpoint(
    config,
    checkpoint,
    output_dir,
    keep_samples=False,
    logger=None,
    batch_size1=False,
):
    config = copy.deepcopy(config)
    checkpoint = Path(checkpoint)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if batch_size1:
        print("[EVAL] setting batch size to 1 for evaluation")
        config["training"]["batch_size"] = 1
    logger = logger or configure_logger(output_dir / "eval.log")

    evaluate_checkpoint_ensemble(
        config,
        checkpoint,
        output_dir,
        logger,
        keep_samples=keep_samples,
    )
    return output_dir


def test_checkpoint(config, checkpoint, output_path=None, batch_size=None):
    loader = build_dataloader(
        config,
        partition="test",
        batch_size=batch_size,
        shuffle=False,
    )
    model = load_rna_module_checkpoint(config, checkpoint, eval_mode=True)
    collector = TestMetricsCollector()
    trainer = Trainer(
        accelerator=config["training"]["accelerator"],
        devices=config["training"]["devices"],
        precision=config["training"]["precision"],
        logger=False,
        enable_checkpointing=False,
        callbacks=[collector],
    )
    trainer.test(model, dataloaders=loader, verbose=False)

    losses = tr.tensor(collector.losses, dtype=tr.float)
    f1_scores = tr.tensor(collector.f1_scores, dtype=tr.float)
    summary = {
        "checkpoint": str(checkpoint),
        "timesteps": config["model"]["timesteps"],
        "epochs": config["training"]["max_epochs"],
        "test_loss": float(losses.mean().item()),
        "test_f1": float(f1_scores.mean().item()),
        "test_f1_std": float(f1_scores.std(unbiased=False).item()),
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
            writer.writeheader()
            writer.writerow(summary)

    return summary
