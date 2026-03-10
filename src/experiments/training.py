"""Training workflow."""
from __future__ import annotations

import logging
import tempfile
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch as tr
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from ..data.collate import pad_batch
from ..data.datasets import load_dataset
from ..training.callbacks import BestModelSaverCallback
from ..training.lightning_module import RNAContactLightningModule
from ..utils.io import resolve_dataset_config
from ..utils.mlflow_io import load_run_info, prepare_local_tracking_dir, resolve_tracking_dir
from ..utils.reporting import build_run_metrics_dataframe

logger = logging.getLogger(__name__)


def _load_dataset_from_cfg(dataset_cfg):
    return load_dataset(**dataset_cfg)


def _build_model_summary(model):
    total = 0
    trainable = 0
    grouped = {}
    for name, param in model.named_parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
        key = ".".join(name.split(".")[:2]) if "." in name else name
        grouped[key] = grouped.get(key, 0) + count

    lines = [
        "Model Summary",
        "=" * 50,
        f"Total parameters: {total:,}",
        f"Trainable parameters: {trainable:,}",
        "",
        "Parameters by module:",
    ]
    for key in sorted(grouped):
        lines.append(f"  {key}: {grouped[key]:,}")
    return "\n".join(lines), total, trainable


def _build_run_name(config):
    run_name = config.get("run_name") or config["mlflow"].get("run_name")
    if run_name:
        return run_name
    fold_number = config.get("fold_number")
    wrapper_cfg = config["network"]["wrapper"]
    prefix = config.get("run_name_prefix") or "train"
    epochs = config["training"]["epochs"]
    timesteps = wrapper_cfg.get("timesteps")
    model_tag = f"timesteps-{timesteps}" if timesteps is not None else f"wrapper-{wrapper_cfg['name']}"
    fold_tag = f"_fold-{fold_number}" if fold_number is not None else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_epochs-{epochs}_{model_tag}{fold_tag}_{ts}"


def _resolve_training_config(cfg):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    fold_number = cfg_dict.get("fold_number")
    cfg_dict["datasets"] = {
        "train": resolve_dataset_config(cfg_dict["data"], "train", fold_number, for_prediction=False),
        "val": resolve_dataset_config(cfg_dict["data"], "val", fold_number, for_prediction=False),
    }
    cfg_dict["fold_number"] = fold_number
    return cfg_dict


def _log_training_metadata(mlflow_logger, model, train_dataset, val_dataset, config):
    if mlflow_logger is None:
        return
    try:
        mlflow_logger.experiment.set_tag(mlflow_logger.run_id, "exp_name", config["mlflow"]["experiment_name"])
        mlflow_logger.experiment.set_tag(mlflow_logger.run_id, "run_name", config["run_name"])
        mlflow_logger.log_hyperparams(config)
        summary_text, total, trainable = _build_model_summary(model)
        mlflow_logger.log_hyperparams(
            {
                "num_parameters_total": total,
                "num_parameters_trainable": trainable,
                "train_size": len(train_dataset),
                "val_size": len(val_dataset),
                "fold_number": config.get("fold_number"),
                "partitioned": config["datasets"]["train"]["partitioned"],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "model_summary.txt"
            summary_path.write_text(summary_text)
            mlflow_logger.experiment.log_artifact(
                mlflow_logger.run_id,
                str(summary_path),
                artifact_path="summaries",
            )
    except Exception as exc:
        logger.warning("Failed to log metadata to MLFlow: %s", exc)


def run_training(cfg: DictConfig):
    """Run a single training job described by the Hydra config."""
    config = _resolve_training_config(cfg)
    config["run_name"] = _build_run_name(config)
    training_cfg = config["training"]

    train_dataset = _load_dataset_from_cfg(config["datasets"]["train"])
    val_dataset = _load_dataset_from_cfg(config["datasets"]["val"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=True,
        collate_fn=pad_batch,
        num_workers=training_cfg.get("num_workers", 2),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        collate_fn=pad_batch,
        num_workers=training_cfg.get("num_workers", 2),
    )

    model = RNAContactLightningModule(config=config)

    mlflow_logger = None
    try:
        from pytorch_lightning.loggers import MLFlowLogger

        prepare_local_tracking_dir(config["mlflow"]["tracking_uri"])
        mlflow_logger = MLFlowLogger(
            experiment_name=config["mlflow"]["experiment_name"],
            tracking_uri=config["mlflow"]["tracking_uri"],
            run_name=config["run_name"],
        )
    except Exception as exc:
        logger.warning("Failed to initialize MLFlow logger: %s. Continuing without MLFlow.", exc)

    _log_training_metadata(mlflow_logger, model, train_dataset, val_dataset, config)

    best_callback = BestModelSaverCallback(config, mlflow_logger, logger)
    trainer = pl.Trainer(
        max_epochs=training_cfg["epochs"],
        accelerator="gpu" if tr.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=10,
        enable_checkpointing=False,
        logger=mlflow_logger,
        callbacks=[best_callback],
        num_sanity_val_steps=0,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    run_id = getattr(mlflow_logger, "run_id", None) if mlflow_logger is not None else None
    if run_id is not None:
        tracking_dir = resolve_tracking_dir(config["mlflow"]["tracking_uri"])
        run_info = load_run_info(tracking_dir, run_id)
        if run_info is not None:
            metrics_df = build_run_metrics_dataframe(run_info["run_dir"])
            if not metrics_df.empty:
                artifact_metrics_path = run_info["run_dir"] / "artifacts" / "metrics.csv"
                artifact_metrics_path.parent.mkdir(parents=True, exist_ok=True)
                metrics_df.to_csv(artifact_metrics_path, index=False)
    return config["run_name"], best_callback.best_val_f1
