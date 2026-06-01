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

from src.data import RNADataModule, build_dataloader
from src.train_module import (
    RNADiffusionModule,
    TestMetricsCollector,
    load_rna_module_checkpoint,
)
from src.ensemble import generate_raw_samples, evaluate_samples_dir, evaluate_samples_stats

from src.config import (
    build_experiment_dir,
    load_config,
    load_ensemble_defaults,
    prepare_experiment_config,
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


def prepare_run(config, experiment_dir=None):
    if experiment_dir is None:
        base_config = prepare_experiment_config(config)
        run_root = Path(build_experiment_dir(base_config))
        experiment_dir = RunIO(run_root).train_dir
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

    run = RunIO.from_train_dir(experiment_dir)
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

    last_ckpt_path = run.last_ckpt_path
    last_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(last_ckpt_path)
    run.normalize_periodic_checkpoint_names()

    handle_metrics(loggers, config, resume=resume is not None)

    ckpt = best_ckpt_cb.best_model_path or str(last_ckpt_path)
    if not ckpt:
        raise RuntimeError("No checkpoint produced")

    logger.info("Training done. Checkpoint: %s", ckpt)
    return ckpt


def _stats_output_path(output_path):
    output_path = Path(output_path)
    if output_path.name == "ensemble.csv":
        return output_path.with_name("ensemble_stats.csv")
    return output_path.with_name(f"{output_path.stem}_stats.csv")


def generate_ensemble_samples(
    config,
    checkpoint,
    samples_dir=None,
    num_samples=None,
    base_seed=None,
    batch_size=None,
    clear_samples=False,
):
    ens_cfg = config["ensemble"]
    samples_dir = Path(samples_dir or config["logging"].get("raw_samples_dir"))
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

    num_samples = int(num_samples or ens_cfg["num_samples"])
    base_seed = int(ens_cfg["base_seed"] if base_seed is None else base_seed)
    chunk_size = int(ens_cfg["chunk_size"])

    generate_raw_samples(
        model=model,
        loader=loader,
        output_dir=samples_dir,
        num_samples=num_samples,
        base_seed=base_seed,
        chunk_size=chunk_size,
    )

    write_samples_metadata(
        samples_dir=samples_dir,
        checkpoint_path=checkpoint,
        checkpoint_epoch=RunIO.checkpoint_epoch(checkpoint),
        num_samples=num_samples,
        base_seed=base_seed,
        chunk_size=chunk_size,
    )
    return samples_dir


def evaluate_ensemble_samples(
    samples_dir,
    output_path=None,
    trials=None,
    consensus_sizes=None,
    seed=None,
    metadata_path=None,
    get_best_and_worst=False,
):
    samples_dir = Path(samples_dir)
    output_path = Path(output_path or samples_dir.parent / "ensemble.csv")
    ens_cfg = load_ensemble_defaults()

    trials = int(trials if trials is not None else ens_cfg["trials"])
    consensus_sizes = list(consensus_sizes or ens_cfg["consensus_sizes"])
    seed = int(seed if seed is not None else ens_cfg["base_seed"])
    get_best_and_worst = bool(get_best_and_worst or ens_cfg["get_best_and_worst"])

    df = evaluate_samples_dir(
        samples_dir=samples_dir,
        consensus_sizes=consensus_sizes,
        trials=trials,
        seed=seed,
        get_best_and_worst=get_best_and_worst,
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
        metadata_path=metadata_path,
    )
    return output_path, stats_path


def evaluate_checkpoint_ensemble(config, checkpoint, logger, keep_samples=False):
    log_cfg = config["logging"]
    ens_cfg = config["ensemble"]

    samples_dir = Path(log_cfg["raw_samples_dir"])

    samples_start = time.perf_counter()
    generate_ensemble_samples(
        config=config,
        checkpoint=checkpoint,
        samples_dir=samples_dir,
        num_samples=ens_cfg["num_samples"],
        base_seed=ens_cfg["base_seed"],
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

    evaluate_ensemble_samples(
        samples_dir=samples_dir,
        output_path=log_cfg["ensemble_path"],
        trials=ens_cfg["trials"],
        consensus_sizes=ens_cfg["consensus_sizes"],
        seed=ens_cfg["base_seed"],
        metadata_path=log_cfg["ensemble_metadata_path"],
    )

    if not keep_samples:
        shutil.rmtree(samples_dir, ignore_errors=True)

    logger.info("Evaluation done")


def _run_from_checkpoint_path(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.parent.name == "checkpoints":
        return RunIO.from_train_dir(checkpoint_path.parent.parent)
    if (
        checkpoint_path.parent.name == "periodic"
        and checkpoint_path.parent.parent.name == "checkpoints"
    ):
        return RunIO.from_train_dir(checkpoint_path.parent.parent.parent)
    raise ValueError(
        "Could not infer run directory from checkpoint path. "
        "Expected <run>/train/checkpoints/*.ckpt."
    )


def _resolve_checkpoint(run, checkpoint):
    if checkpoint == "best":
        return run.best_ckpt_path
    if checkpoint == "last":
        return run.last_ckpt_path
    return Path(checkpoint)


def _default_eval_dir(run, checkpoint, checkpoint_path):
    if checkpoint_path == run.best_ckpt_path:
        return run.best_eval_dir
    if checkpoint_path == run.last_ckpt_path:
        return run.root / "eval" / "last"
    if checkpoint_path.parent.name == "periodic":
        return run.periodic_eval_dir(RunIO.checkpoint_epoch(checkpoint_path))
    if checkpoint == "best":
        return run.best_eval_dir
    if checkpoint == "last":
        return run.root / "eval" / "last"
    raise ValueError("output_dir is required for explicit checkpoint paths")


def evaluate_checkpoint(
    target,
    checkpoint="best",
    output_dir=None,
    keep_samples=False,
    logger=None,
):
    target = Path(target)
    if target.suffix == ".ckpt":
        run = _run_from_checkpoint_path(target)
        checkpoint_path = target
    else:
        run = RunIO(target)
        checkpoint_path = _resolve_checkpoint(run, checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = Path(output_dir) if output_dir is not None else _default_eval_dir(
        run,
        checkpoint,
        checkpoint_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    config = prepare_experiment_config(load_config(run.config_path), output_dir)
    logger = logger or configure_logger(output_dir / "eval.log")

    evaluate_checkpoint_ensemble(
        config,
        checkpoint_path,
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


def evaluate(config, experiment_dir, checkpoint, logger, keep_samples=False):
    config = prepare_experiment_config(config, experiment_dir)
    return evaluate_checkpoint_ensemble(
        config,
        checkpoint,
        logger,
        keep_samples=keep_samples,
    )
