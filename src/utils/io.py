"""Small I/O helpers for configs and dataset runtime resolution."""
from __future__ import annotations

import json
import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def _to_dict(config):
    if config is None:
        return {}
    if isinstance(config, DictConfig):
        plain = OmegaConf.to_container(config, resolve=True, enum_to_str=True)
        return plain if isinstance(plain, dict) else {}
    return dict(config)


def load_config(path):
    path = Path(path)
    config_path = path / "config.json" if path.is_dir() else path
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")
    return OmegaConf.create(json.loads(config_path.read_text()))


def save_config(config, path) -> None:
    output_path = Path(path) / "config.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True, enum_to_str=True)
    output_path.write_text(json.dumps(config, indent=4, default=str))


def resolve_workspace_paths(paths_cfg=None, app_root=None):
    if paths_cfg is not None:
        paths_cfg = _to_dict(paths_cfg)
        return {
            "workspace_root": Path(str(paths_cfg["workspace_root"])).resolve(),
            "data_root": Path(str(paths_cfg["data_root"])).resolve(),
            "logs_root": Path(str(paths_cfg["logs_root"])).resolve(),
            "mlruns_root": Path(str(paths_cfg["mlruns_root"])).resolve(),
        }

    workspace_root = Path(app_root).resolve() if app_root is not None else Path.cwd().resolve()
    return {
        "workspace_root": workspace_root,
        "data_root": workspace_root / "data",
        "logs_root": workspace_root / "logs",
        "mlruns_root": workspace_root / "mlruns",
    }


def resolve_dataset_config(data_cfg, dataset_kind=None, fold_number=None, for_prediction=False):
    data_cfg = _to_dict(data_cfg)
    split_name = str(data_cfg.get("split", "default"))
    partition_path = data_cfg.get("partition_file_template") or data_cfg.get("partition_path")
    if partition_path:
        partition_path = str(partition_path).replace("${split}", split_name)

    partitioned = bool(data_cfg.get("use_partitions", False))
    dataset_cfg = {
        "dataset_path": None,
        "min_len": data_cfg.get("min_len", 0),
        "max_len": data_cfg.get("max_len", 512),
        "for_prediction": for_prediction,
        "partitioned": partitioned,
        "main_path": data_cfg.get("main_path"),
        "partition_path": partition_path,
        "partition_value": None,
        "fold_number": fold_number,
    }

    if dataset_kind is None:
        return dataset_cfg

    if partitioned:
        dataset_cfg["partition_value"] = data_cfg.get(f"{dataset_kind}_partition", dataset_kind)
        return dataset_cfg

    dataset_cfg["dataset_path"] = data_cfg.get(f"{dataset_kind}_path")
    return dataset_cfg


def make_file_tracking_uri(path) -> str:
    return f"file:{Path(path).resolve()}"


def prepare_env_with_pythonpath(project_dir):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_dir)
    return env
