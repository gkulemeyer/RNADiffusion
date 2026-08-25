from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from ml_collections import ConfigDict

from src.io import read_yaml, write_yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/train/default.yaml"
ENSEMBLE_DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/ensemble/default.yaml"


def load_train_defaults():
    defaults = read_yaml(TRAIN_DEFAULT_CONFIG_PATH)
    defaults.setdefault("experiment", {})
    defaults["experiment"].setdefault("uuid", "")
    defaults["experiment"].setdefault("timestamp", "")
    return defaults


def load_ensemble_defaults():
    return read_yaml(ENSEMBLE_DEFAULT_CONFIG_PATH)


def load_base_defaults():
    defaults = load_train_defaults()
    defaults["ensemble"] = load_ensemble_defaults()
    return defaults

def deep_merge(base, override):
    """
    Recursively merges two dictionaries.
    """
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

    raw_config = read_yaml(config_path)

    return deep_merge(load_base_defaults(), raw_config)


def save_config(config, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    write_yaml(config, output_path / "config.yaml")

def prepare_experiment_config(config, experiment_dir=None):
    prepared = deep_merge(load_base_defaults(), config)

    experiment = prepared["experiment"]
    logging_config = prepared["logging"]

    if not experiment.get("uuid"):
        experiment["uuid"] = str(uuid4())
    if not experiment.get("timestamp"):
        experiment["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    if experiment_dir is not None:
        experiment_dir = Path(experiment_dir)
        logging_config["experiment_dir"] = str(experiment_dir)
        logging_config["checkpoint_dir"] = str(experiment_dir / "checkpoints")
        logging_config["metrics_path"] = str(experiment_dir / "metrics.csv")
        logging_config["raw_ensemble_metrics_path"] = str(
            experiment_dir / "raw_ensemble_metrics.csv"
        )
        logging_config["processed_ensemble_metrics_path"] = str(
            experiment_dir / "processed_ensemble_metrics.csv"
        )
        logging_config["generated_ensemble_path"] = str(
            experiment_dir / "generated_ensemble.csv"
        )
        logging_config["ensemble_metadata_path"] = str(experiment_dir / "ensemble_metadata.yaml")
        logging_config["train_log_path"] = str(experiment_dir / "run.log")
        logging_config["samples_dir"] = str(experiment_dir / "samples")
        logging_config["lightning_dir"] = str(experiment_dir / "lightning")

    return prepared


def build_experiment_name(config):
    explicit_name = config["experiment"]["name"].strip()
    if explicit_name:
        return explicit_name

    else:
        timestamp = config["experiment"].get("timestamp") or datetime.now().strftime("%Y%m%d_%H%M%S")
        timesteps = config["model"]["timesteps"]
        epochs = config["training"]["max_epochs"]
        return f"exp_T{timesteps}_E{epochs}_{timestamp}"

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


def to_config_dict(config):
    return ConfigDict(copy.deepcopy(config))


def from_config_dict(config):
    if hasattr(config, "to_dict"):
        return config.to_dict()
    return copy.deepcopy(config)


def clone_config(config):
    return to_config_dict(from_config_dict(config))


def latest_run_dir(config, run_pattern=None):
    config_dict = from_config_dict(config)
    save_dir = Path(config_dict["logging"]["save_dir"])
    run_name = build_experiment_name(config_dict)
    pattern = run_pattern or f"{run_name}*"

    matches = [path for path in save_dir.glob(pattern) if path.is_dir()]
    if not matches:
        return None
    matches.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0]
