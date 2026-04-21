from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/train/default.yaml"
ENSEMBLE_DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/ensemble/default.yaml"


def _read_yaml_file(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_train_defaults():
    defaults = _read_yaml_file(TRAIN_DEFAULT_CONFIG_PATH)
    defaults.setdefault("experiment", {})
    defaults["experiment"].setdefault("uuid", "")
    defaults["experiment"].setdefault("timestamp", "")
    return defaults


def load_ensemble_defaults():
    return _read_yaml_file(ENSEMBLE_DEFAULT_CONFIG_PATH)


def load_base_defaults():
    defaults = load_train_defaults()
    defaults["ensemble"] = load_ensemble_defaults()
    return defaults

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

    raw_config = _read_yaml_file(config_path)

    return deep_merge(load_base_defaults(), raw_config)


def save_config(config, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def prepare_experiment_config(config, experiment_dir=""):
    prepared = deep_merge(load_base_defaults(), config)
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
        logging_config["ensemble_metadata_path"] = str(experiment_path / "ensemble_metadata.yaml")
        logging_config["train_log_path"] = str(experiment_path / "run.log")
        logging_config["raw_samples_dir"] = str(experiment_path / "raw_samples")
        logging_config["lightning_dir"] = str(experiment_path / "lightning")
    return prepared


def build_experiment_name(config):
    explicit_name = config["experiment"]["name"].strip()
    if explicit_name:
        return explicit_name

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
