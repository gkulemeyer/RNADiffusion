from __future__ import annotations

import copy
from pathlib import Path

from src.config import build_experiment_name
from ml_collections import ConfigDict 
import torch as tr


COMPLETED_RUN_FILES = [
    "config.yaml",
    "metrics.csv",
    "ensemble.csv",
    "ensemble_metadata.yaml",
    "train.log",
    "raw_samples",
    "checkpoints/best.ckpt",
    "checkpoints/last.ckpt",
]


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


def run_completed(config_or_path):
    if isinstance(config_or_path, (str, Path)):
        run_dir = Path(config_or_path)
    else:
        run_dir = latest_run_dir(config_or_path)
        if run_dir is None:
            return None

    if all((run_dir / relative_path).exists() for relative_path in COMPLETED_RUN_FILES):
        return run_dir
    return None


def last_checkpoint_path(run_dir):
    checkpoint_path = Path(run_dir) / "checkpoints" / "last.ckpt"
    if checkpoint_path.exists():
        return checkpoint_path
    return None


def completed_epochs(run_dir):
    checkpoint_path = last_checkpoint_path(run_dir)
    if checkpoint_path is None:
        return None 
    checkpoint = tr.load(checkpoint_path, map_location="cpu")
    epoch = checkpoint.get("epoch")
    if epoch is None:
        return None
    return int(epoch) + 1
