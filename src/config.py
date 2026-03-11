from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import yaml


DEFAULT_CONFIG = {
    "experiment": {
        "name": "",
        "note": "",
        "seed": 42,
        "uuid": "",
        "timestamp": "",
    },
    "data": {
        "base_path": "",
        "partition_path": "",
        "fold": 0,
    },
    "model": {
        "timesteps": 5,
        "num_classes": 2,
        "in_channels": 18,
        "out_channels": 2,
        "base_dim": 64,
    },
    "training": {
        "max_epochs": 10,
        "batch_size": 4,
        "lr": 1e-3,
        "num_workers": 2,
        "accelerator": "auto",
        "devices": 1,
        "precision": 32,
    },
    "logging": {
        "save_dir": "logs/RNADiffusion",
        "tensorboard": True,
        "log_every_n_steps": 1,
    },
}

def deep_merge(base, override):
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path):
    config_path = Path(path)
    if not config_path.is_file():
        config_path = config_path / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config path not found: {path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    return deep_merge(DEFAULT_CONFIG, raw_config)


def save_config(config, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def prepare_experiment_config(config, experiment_dir=""):
    prepared = deep_merge(DEFAULT_CONFIG, config)
    experiment = prepared["experiment"]
    logging_config = prepared["logging"]

    if not experiment["uuid"]:
        experiment["uuid"] = str(uuid4())
    if not experiment["timestamp"]:
        experiment["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    if experiment_dir:
        experiment_path = Path(experiment_dir)
        logging_config["experiment_dir"] = str(experiment_path)
        logging_config["checkpoint_dir"] = str(experiment_path / "checkpoints")
        logging_config["metrics_path"] = str(experiment_path / "metrics.csv")
        logging_config["ensemble_path"] = str(experiment_path / "ensemble.csv")
        logging_config["train_log_path"] = str(experiment_path / "train.log")
        logging_config["raw_samples_dir"] = str(experiment_path / "raw_samples")
    return prepared


def build_experiment_name(config):
    explicit_name = config["experiment"]["name"].strip()
    if explicit_name:
        return explicit_name

    timestamp = config["experiment"].get("timestamp") or datetime.now().strftime("%Y%m%d_%H%M%S")
    timesteps = config["model"]["timesteps"]
    epochs = config["training"]["max_epochs"]
    return f"exp_T{timesteps}_E{epochs}_{timestamp}"
