from __future__ import annotations

import csv
import yaml
import logging
from pathlib import Path
from datetime import datetime
import torch as tr
from lightning.pytorch.loggers import CSVLogger

from src.model import build_model


def read_yaml(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(data, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


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
    return [CSVLogger(save_dir=experiment_dir, name="lightning", version="")]


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



def load_samples_metadata(samples_dir):
    metadata_path = Path(samples_dir) / "samples_metadata.yaml"
    if not metadata_path.exists():
        return {}
    return read_yaml(metadata_path)

def write_samples_metadata(
    samples_dir,
    checkpoint_path,
    num_samples,
    base_seed,
    chunk_size,
    checkpoint_epoch=None,
):
    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "samples_dir": str(samples_dir),
        "num_samples": int(num_samples),
        "base_seed": int(base_seed),
        "chunk_size": int(chunk_size),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if checkpoint_epoch is not None:
        metadata["checkpoint_epoch"] = int(checkpoint_epoch)
    write_yaml(metadata, Path(samples_dir) / "samples_metadata.yaml")
    return metadata


def write_ensemble_metadata(
    output_path,
    samples_dir,
    trials,
    consensus_sizes,
    seed,
    get_best_and_worst=False,
    metadata_path=None,
):
    sample_metadata = load_samples_metadata(samples_dir)
    metadata = {
        "checkpoint_path": sample_metadata.get("checkpoint_path", ""),
        "checkpoint_epoch": sample_metadata.get("checkpoint_epoch"),
        "samples_dir": str(samples_dir),
        "num_samples": sample_metadata.get("num_samples"),
        "base_seed": sample_metadata.get("base_seed"),
        "trials": int(trials),
        "consensus_sizes": [int(size) for size in consensus_sizes],
        "seed": int(seed),
        "get_best_and_worst": bool(get_best_and_worst),
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
    write_yaml(metadata, target_path)
    return metadata


def handle_metrics(loggers, config, resume=False):
    csv_logger = next(
        (l for l in loggers if l.__class__.__name__ == "CSVLogger"), None
    )
    if csv_logger is None:
        raise RuntimeError("CSVLogger not found")

    log_cfg = config["logging"]

    src = Path(csv_logger.log_dir) / "metrics.csv"
    dst = Path(log_cfg["metrics_path"])

    dst.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    if resume and dst.exists():
        with dst.open() as handle:
            rows.extend(csv.DictReader(handle))

    with src.open() as handle:
        rows.extend(csv.DictReader(handle))
    src.unlink()

    grouped = {}
    keys = []
    for r in rows:
        k = (r.get("epoch"), r.get("step"))
        grouped.setdefault(k, {"epoch": k[0], "step": k[1]})
        for m, v in r.items():
            if m in ("epoch", "step") or v in ("", None):
                continue
            grouped[k][m] = v
            if m not in keys:
                keys.append(m)

    with dst.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "step"] + keys)
        writer.writeheader()
        writer.writerows(grouped.values())

    return dst
