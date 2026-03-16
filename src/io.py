from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path

import torch as tr
import yaml
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from src.model import build_model


def configure_logger(log_file, logger_name="rnadiffusion", level=logging.INFO):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def build_loggers(config, experiment_dir):
    lightning_root = experiment_dir / "lightning"
    loggers = [CSVLogger(save_dir=experiment_dir, name="lightning", version="")]
    if config["logging"].get("tensorboard", False):
        loggers.append(TensorBoardLogger(save_dir=lightning_root, name="tensorboard", version=""))
    return loggers


def load_model_checkpoint(config, checkpoint_path, eval_mode=True):
    checkpoint = tr.load(checkpoint_path, map_location="cpu")
    if "state_dict" not in checkpoint:
        raise ValueError(f"Expected a Lightning .ckpt file, got: {checkpoint_path}")

    state_dict = checkpoint["state_dict"]
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            cleaned_state_dict[key[len("model."):]] = value
        else:
            cleaned_state_dict[key] = value

    model = build_model(config)
    model.load_state_dict(cleaned_state_dict)

    device = tr.device("cuda" if tr.cuda.is_available() else "cpu")
    model.to(device)
    if eval_mode:
        model.eval()
    return model



### ENSEMBLE IO
def _write_yaml(data, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def load_samples_metadata(samples_dir):
    metadata_path = Path(samples_dir) / "samples_metadata.yaml"
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}

def write_samples_metadata(
    samples_dir,
    checkpoint_path,
    num_samples,
    base_seed,
    chunk_size,
):
    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "samples_dir": str(samples_dir),
        "num_samples": int(num_samples),
        "base_seed": int(base_seed),
        "chunk_size": int(chunk_size),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_yaml(metadata, Path(samples_dir) / "samples_metadata.yaml")
    return metadata


def write_ensemble_metadata(
    output_path,
    samples_dir,
    trials,
    consensus_sizes,
    seed,
    metadata_path=None,
):
    sample_metadata = load_samples_metadata(samples_dir)
    metadata = {
        "checkpoint_path": sample_metadata.get("checkpoint_path", ""),
        "samples_dir": str(samples_dir),
        "num_samples": sample_metadata.get("num_samples"),
        "base_seed": sample_metadata.get("base_seed"),
        "trials": int(trials),
        "consensus_sizes": [int(size) for size in consensus_sizes],
        "seed": int(seed),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    
    if metadata_path is not None:
        target_path = Path(metadata_path)
    else:
        output_path = Path(output_path)
        if output_path.name == "ensemble.csv":
            target_path = output_path.with_name("ensemble_metadata.yaml")
        else:
            target_path = output_path.with_name(f"{output_path.stem}_metadata.yaml")
    _write_yaml(metadata, target_path)
    return metadata
