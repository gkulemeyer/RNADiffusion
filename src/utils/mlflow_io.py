"""Small MLflow helpers oriented to experiment workflows."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf

from .io import load_config, save_config

logger = logging.getLogger(__name__)
DEFAULT_EXPERIMENT_ID = "0"
DEFAULT_EXPERIMENT_NAME = "Default"


def _to_dict(config):
    if config is None:
        return {}
    if isinstance(config, DictConfig):
        plain = OmegaConf.to_container(config, resolve=False, enum_to_str=True)
        return plain if isinstance(plain, dict) else {}
    return dict(config)


def _iter_run_dirs(tracking_dir, checkpoints_only=False):
    if not tracking_dir.exists():
        return []
    run_dirs = []
    for experiment_dir in tracking_dir.iterdir():
        if not experiment_dir.is_dir():
            continue
        for run_dir in experiment_dir.iterdir():
            if not run_dir.is_dir():
                continue
            if checkpoints_only and not (run_dir / "artifacts" / "checkpoints").exists():
                continue
            run_dirs.append(run_dir)
    return sorted(run_dirs)


def _build_run_info(tracking_dir, run_dir):
    artifacts_dir = run_dir / "artifacts"
    checkpoint_dir = artifacts_dir / "checkpoints"
    best = checkpoint_dir / "best_model.pt"
    if best.exists():
        checkpoint_path = best
    else:
        checkpoint_path = None
        for checkpoint_path in checkpoint_dir.glob("*.pt"):
            break
    config_path = artifacts_dir / "config.json"
    config = None
    if config_path.exists():
        try:
            config = load_config(config_path)
        except Exception as exc:
            logger.warning("Failed to load config for run %s: %s", run_dir.name, exc)
    tags = {"exp_name": None, "run_name": None}
    if (run_dir / "meta.yaml").exists():
        try:
            client = MlflowClient(tracking_uri=f"file:{Path(tracking_dir).resolve()}")
            mlflow_tags = client.get_run(run_dir.name).data.tags or {}
            tags = {
                "exp_name": mlflow_tags.get("exp_name"),
                "run_name": mlflow_tags.get("mlflow.runName") or mlflow_tags.get("run_name"),
            }
        except Exception as exc:
            logger.warning("Failed to fetch run tags for %s: %s", run_dir.name, exc)
    samples_dir = artifacts_dir / "ensembles" / "raw_samples"
    return {
        "run_id": run_dir.name,
        "run_dir": run_dir,
        "checkpoint_path": checkpoint_path if checkpoint_path and checkpoint_path.exists() else None,
        "config": config,
        "samples_dir": samples_dir if samples_dir.exists() else None,
        "tags": tags,
    }


def resolve_tracking_dir(tracking_uri=None):
    tracking_uri = tracking_uri or mlflow.get_tracking_uri()
    if tracking_uri and str(tracking_uri).startswith("file:"):
        return Path(str(tracking_uri).split("file:", 1)[1]).resolve()
    return Path("mlruns").resolve()


def prepare_local_tracking_dir(tracking_uri=None):
    tracking_uri = tracking_uri or mlflow.get_tracking_uri()
    tracking_dir = resolve_tracking_dir(tracking_uri)
    if tracking_uri and not str(tracking_uri).startswith("file:"):
        return tracking_dir

    tracking_dir.mkdir(parents=True, exist_ok=True)
    default_experiment_dir = tracking_dir / DEFAULT_EXPERIMENT_ID
    default_experiment_dir.mkdir(parents=True, exist_ok=True)

    meta_path = default_experiment_dir / "meta.yaml"
    if meta_path.exists():
        return tracking_dir

    now_ms = int(time.time() * 1000)
    meta_path.write_text(
        "\n".join(
            [
                f"artifact_location: {default_experiment_dir.resolve().as_uri()}",
                f"creation_time: {now_ms}",
                f"experiment_id: '{DEFAULT_EXPERIMENT_ID}'",
                f"last_update_time: {now_ms}",
                "lifecycle_stage: active",
                f"name: {DEFAULT_EXPERIMENT_NAME}",
                "",
            ]
        )
    )
    return tracking_dir


def iter_run_info(tracking_dir, checkpoints_only=False):
    for run_dir in _iter_run_dirs(Path(tracking_dir), checkpoints_only=checkpoints_only):
        run_info = _build_run_info(tracking_dir, run_dir)
        if checkpoints_only and run_info["checkpoint_path"] is None:
            continue
        yield run_info


def load_run_info(tracking_dir, run_id):
    return next((run_info for run_info in iter_run_info(tracking_dir) if run_info["run_id"] == run_id), None)


def save_run_config(tracking_dir, run_id, config):
    run_info = load_run_info(tracking_dir, run_id)
    if run_info is None:
        return False
    destination = run_info["run_dir"] / "artifacts"
    destination.mkdir(parents=True, exist_ok=True)
    try:
        save_config(config, destination)
        return True
    except Exception as exc:
        logger.warning("Failed to save config for run %s: %s", run_id, exc)
        return False


def resolve_run_dataset(tracking_dir, run_id, default_dataset_cfg):
    default_dataset_cfg = dict(default_dataset_cfg or {})
    run_info = run_id if isinstance(run_id, dict) else load_run_info(tracking_dir, run_id)
    if run_info is None or run_info["config"] is None:
        return default_dataset_cfg

    run_cfg = _to_dict(run_info["config"])
    test_cfg = dict((run_cfg.get("datasets") or {}).get("test") or {})
    if test_cfg:
        test_cfg.setdefault("for_prediction", default_dataset_cfg.get("for_prediction", True))
        test_cfg.setdefault("dataset_path", None)
        test_cfg.setdefault("partitioned", False)
        test_cfg.setdefault("main_path", None)
        test_cfg.setdefault("partition_path", None)
        test_cfg.setdefault("partition_value", None)
        test_cfg.setdefault("fold_number", None)
        test_cfg.setdefault("min_len", default_dataset_cfg.get("min_len", 0))
        test_cfg.setdefault("max_len", default_dataset_cfg.get("max_len", 512))
        return test_cfg

    data_cfg = dict(run_cfg.get("data") or {})
    if data_cfg:
        partitioned = bool(data_cfg.get("use_partitions", False))
        dataset_cfg = {
            "dataset_path": None,
            "for_prediction": default_dataset_cfg.get("for_prediction", True),
            "partitioned": partitioned,
            "main_path": data_cfg.get("main_path"),
            "partition_path": None,
            "partition_value": None,
            "fold_number": run_cfg.get("fold_number"),
            "min_len": data_cfg.get("min_len", default_dataset_cfg.get("min_len", 0)),
            "max_len": data_cfg.get("max_len", default_dataset_cfg.get("max_len", 512)),
        }
        partition_path = data_cfg.get("partition_file_template") or data_cfg.get("partition_path")
        if partition_path:
            split_name = str(data_cfg.get("split", "default"))
            dataset_cfg["partition_path"] = str(partition_path).replace("${split}", split_name)
        if partitioned:
            dataset_cfg["partition_value"] = data_cfg.get("test_partition", "test")
            return dataset_cfg
        dataset_cfg["dataset_path"] = data_cfg.get("test_path", default_dataset_cfg.get("dataset_path"))
        return dataset_cfg

    if run_cfg.get("partitioned") and run_cfg.get("main_path") and run_cfg.get("partition_path"):
        return {
            "dataset_path": None,
            "for_prediction": default_dataset_cfg.get("for_prediction", True),
            "partitioned": True,
            "main_path": run_cfg["main_path"],
            "partition_path": run_cfg["partition_path"],
            "partition_value": run_cfg.get("test_partition", "test"),
            "fold_number": run_cfg.get("fold_number"),
            "min_len": run_cfg.get("min_len", default_dataset_cfg.get("min_len", 0)),
            "max_len": run_cfg.get("max_len", default_dataset_cfg.get("max_len", 512)),
        }

    resolved = dict(default_dataset_cfg)
    resolved["dataset_path"] = run_cfg.get("test_path") or resolved.get("dataset_path")
    return resolved


def log_summary_metrics(tracking_dir, run_id, summary):
    try:
        client = MlflowClient(tracking_uri=f"file:{Path(tracking_dir).resolve()}")
        row = summary.iloc[0].to_dict()
        metrics = {key: value for key, value in row.items() if key != "run_id" and isinstance(value, (int, float))}
        if metrics:
            client.log_metrics(run_id, metrics)
    except Exception as exc:
        logger.warning("Failed to log summary metrics for run %s: %s", run_id, exc)
