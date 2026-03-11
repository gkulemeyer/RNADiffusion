from __future__ import annotations

import logging
from pathlib import Path

import torch as tr
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from src.config import build_experiment_name
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


def build_experiment_dir(config):
    base_dir = Path(config["logging"]["save_dir"])
    experiment_name = build_experiment_name(config)
    experiment_dir = base_dir / experiment_name

    if not experiment_dir.exists():
        return str(experiment_dir)

    suffix = 1
    while True:
        candidate = f"{experiment_dir}_{suffix}"
        if not Path(candidate).exists():
            return candidate
        suffix += 1


def build_loggers(config, experiment_dir):
    loggers = [CSVLogger(save_dir=experiment_dir, name="csv", version="")]
    if config["logging"].get("tensorboard", False):
        loggers.append(TensorBoardLogger(save_dir=experiment_dir, name="tensorboard", version=""))
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
