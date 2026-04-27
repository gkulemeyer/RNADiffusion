## config.py

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

    raw_config = _read_yaml_file(config_path)

    return deep_merge(load_base_defaults(), raw_config)


def save_config(config, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

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
        logging_config["ensemble_path"] = str(experiment_dir / "ensemble.csv")
        logging_config["ensemble_metadata_path"] = str(experiment_dir / "ensemble_metadata.yaml")
        logging_config["train_log_path"] = str(experiment_dir / "run.log")
        logging_config["raw_samples_dir"] = str(experiment_dir / "raw_samples")
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

### io.py 
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
        "checkpoint_epoch": sample_metadata.get("checkpoint_epoch"),
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


## sweeps.py

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


## experiment.py

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import torch as tr
from lightning.pytorch import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

from src.config import build_experiment_dir, prepare_experiment_config, save_config
from src.data import RNADataModule, build_dataloader
from src.ensemble import (
    generate_raw_samples,
    evaluate_samples_dir,
    evaluate_samples_stats,
)
from src.io import (
    build_loggers,
    configure_logger,
    load_model_checkpoint,
    write_ensemble_metadata,
    write_samples_metadata,
)
from src.train_module import RNADiffusionModule


def periodic_checkpoint_dir(experiment_dir):
    return Path(experiment_dir) / "checkpoints" / "periodic"


def checkpoint_completed_epoch(checkpoint_path):
    checkpoint = tr.load(checkpoint_path, map_location="cpu")
    epoch = checkpoint.get("epoch")
    if epoch is None:
        raise ValueError(f"Checkpoint does not contain epoch metadata: {checkpoint_path}")
    return int(epoch) + 1


def list_periodic_checkpoints(experiment_dir):
    checkpoint_paths = sorted(periodic_checkpoint_dir(experiment_dir).glob("*.ckpt"))
    return sorted(checkpoint_paths, key=checkpoint_completed_epoch)


def normalize_periodic_checkpoint_names(experiment_dir):
    checkpoint_dir = periodic_checkpoint_dir(experiment_dir)
    for checkpoint_path in sorted(checkpoint_dir.glob("*.ckpt")):
        epoch = checkpoint_completed_epoch(checkpoint_path)
        target_path = checkpoint_dir / f"epoch{epoch:03d}.ckpt"
        if checkpoint_path == target_path or target_path.exists():
            continue
        checkpoint_path.rename(target_path)

def prepare_run(config, experiment_dir=None):
    if experiment_dir is None:
        base_config = prepare_experiment_config(config)
        experiment_dir = Path(build_experiment_dir(base_config))
    else:
        experiment_dir = Path(experiment_dir)

    config = prepare_experiment_config(config, experiment_dir)

    experiment_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, experiment_dir)

    run_logger = configure_logger(config["logging"]["train_log_path"])
    seed_everything(config["experiment"]["seed"], workers=True)

    run_logger.info("Starting experiment in %s", experiment_dir)
    run_logger.info("Resolved config: %s", json.dumps(config, indent=2))

    return config, experiment_dir, run_logger

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


def train(config, experiment_dir, logger, resume=None):
    log_cfg = config["logging"]
    train_cfg = config["training"]
    checkpoint_interval = max(1, int(log_cfg.get("checkpoint_every_n_epochs", 1)))

    data = RNADataModule(config)
    model = RNADiffusionModule(config)
    loggers = build_loggers(config, experiment_dir)

    best_ckpt_cb = ModelCheckpoint(
        dirpath=experiment_dir / "checkpoints",
        filename="best",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        save_last=False,
        auto_insert_metric_name=False,
    )
    periodic_ckpt_cb = ModelCheckpoint(
        dirpath=periodic_checkpoint_dir(experiment_dir),
        filename="epoch{epoch:03d}",
        monitor=None,
        save_top_k=-1,
        save_last=False,
        every_n_epochs=checkpoint_interval,
        auto_insert_metric_name=False,
    )

    trainer = Trainer(
        max_epochs=train_cfg["max_epochs"],
        accelerator=train_cfg["accelerator"],
        devices=train_cfg["devices"],
        precision=train_cfg["precision"],
        accumulate_grad_batches=train_cfg["accumulate_grad_batches"],
        check_val_every_n_epoch=max(1, int(train_cfg.get("check_val_every_n_epoch", 1))),
        logger=loggers,
        callbacks=[best_ckpt_cb, periodic_ckpt_cb],
        log_every_n_steps=log_cfg["log_every_n_steps"],
    )
    trainer.fit(model, datamodule=data, ckpt_path=resume)

    last_ckpt_path = Path(experiment_dir) / "checkpoints" / "last.ckpt"
    last_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(last_ckpt_path)
    normalize_periodic_checkpoint_names(experiment_dir)

    handle_metrics(loggers, config, resume=resume is not None)

    ckpt = best_ckpt_cb.best_model_path or str(last_ckpt_path)
    if not ckpt:
        raise RuntimeError("No checkpoint produced")

    logger.info("Training done. Checkpoint: %s", ckpt)
    return ckpt

def evaluate(config, experiment_dir, checkpoint, logger, cleanup_raw_samples=False):
    log_cfg = config["logging"]
    ens_cfg = config["ensemble"]

    samples_dir = Path(log_cfg["raw_samples_dir"])
    samples_dir.mkdir(parents=True, exist_ok=True)

    # eliminar esto en el futuro
    for f in samples_dir.glob("*.pt"):
        f.unlink()

    model = load_model_checkpoint(config, checkpoint, eval_mode=True)
    loader = build_dataloader(config, partition="test", shuffle=False)

    generate_raw_samples(
        model=model,
        loader=loader,
        output_dir=samples_dir,
        num_samples=ens_cfg["num_samples"],
        base_seed=ens_cfg["base_seed"],
        chunk_size=ens_cfg["chunk_size"],
    )

    write_samples_metadata(
        samples_dir=samples_dir,
        checkpoint_path=checkpoint,
        checkpoint_epoch=checkpoint_completed_epoch(checkpoint),
        num_samples=ens_cfg["num_samples"],
        base_seed=ens_cfg["base_seed"],
        chunk_size=ens_cfg["chunk_size"],
    )

    df = evaluate_samples_dir(
        samples_dir=samples_dir,
        consensus_sizes=ens_cfg["consensus_sizes"],
        trials=ens_cfg["trials"],
        seed=ens_cfg["base_seed"],
    )
    df.to_csv(log_cfg["ensemble_path"], index=False)

    stats_path = experiment_dir / "ensemble_stats.csv"
    stats = evaluate_samples_stats(
        samples_csv=log_cfg["ensemble_path"],
        consensus_sizes=ens_cfg["consensus_sizes"],
    )
    stats.to_csv(stats_path, index=False)

    write_ensemble_metadata(
        output_path=log_cfg["ensemble_path"],
        samples_dir=samples_dir,
        trials=ens_cfg["trials"],
        consensus_sizes=ens_cfg["consensus_sizes"],
        seed=ens_cfg["base_seed"],
        metadata_path=log_cfg["ensemble_metadata_path"],
    )

    if cleanup_raw_samples:
        shutil.rmtree(samples_dir, ignore_errors=True)

    logger.info("Evaluation done")


def periodic_eval_dir(run_dir, epoch):
    return Path(run_dir) / f"epoch_{int(epoch):03d}"


## run_loop.py

from __future__ import annotations

from pathlib import Path

from ml_collections import ConfigDict
import yaml

from src.config import load_config, prepare_experiment_config
from src.experiment import (
    checkpoint_completed_epoch,
    evaluate,
    list_periodic_checkpoints,
    prepare_run,
    train,
)
from src.io import configure_logger
from src.sweeps import completed_epochs, last_checkpoint_path


def _save_dir_structure(run_root, legacy=False):
    run_root = Path(run_root)
    return run_root / "train", run_root / "eval" / "best", run_root / "eval" / "periodic"


def _latest_attempt_dir(job_dir):
    attempt_dirs = sorted(
        path for path in Path(job_dir).glob("attempt_*") if path.is_dir()
    )
    if not attempt_dirs:
        return None
    return attempt_dirs[-1]


def _next_attempt_dir(job_dir):
    index = 1
    while True:
        attempt_dir = Path(job_dir) / f"attempt_{index:03d}"
        if not attempt_dir.exists():
            return attempt_dir
        index += 1


def _resolve_run_root(config, job_dir, resume):
    job_dir = Path(job_dir)

    if resume:
        if (job_dir / "train").exists():
            config.experiment.name = job_dir.name
            return job_dir

        latest_attempt = _latest_attempt_dir(job_dir)
        if latest_attempt is not None and (latest_attempt / "train").exists():
            config.experiment.name = latest_attempt.name
            return latest_attempt

        config.experiment.name = job_dir.name
        return job_dir

    if not job_dir.exists():
        config.experiment.name = job_dir.name
        return job_dir

    if (job_dir / "train").exists() or _latest_attempt_dir(job_dir) is not None:
        attempt_dir = _next_attempt_dir(job_dir)
        config.experiment.name = attempt_dir.name
        return attempt_dir

    config.experiment.name = job_dir.name
    return job_dir

def _best_eval_matches_checkpoint(best_eval_dir, best_ckpt):
    metadata_path = Path(best_eval_dir) / "ensemble_metadata.yaml"
    if not metadata_path.exists():
        return False

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle) or {}

    if metadata.get("checkpoint_path") != str(best_ckpt):
        return False

    stored_epoch = metadata.get("checkpoint_epoch")
    if stored_epoch is None:
        return True
    return int(stored_epoch) == checkpoint_completed_epoch(best_ckpt)


def _evaluate_periodic_checkpoints(train_dir, periodic_eval_root, logger):
    run_config = load_config(train_dir / "config.yaml")
    checkpoint_paths = list_periodic_checkpoints(train_dir)

    if not checkpoint_paths:
        print(f"[WARN] no periodic checkpoints found in {train_dir / 'checkpoints' / 'periodic'}")
        return

    for checkpoint_path in checkpoint_paths:
        epoch = checkpoint_completed_epoch(checkpoint_path)
        target_dir = Path(periodic_eval_root) / f"epoch_{epoch:03d}"
        if (target_dir / "ensemble_stats.csv").exists():
            print(f"[SKIP] epoch_{epoch:03d} already evaluated")
            continue

        eval_config = prepare_experiment_config(run_config, target_dir)
        evaluate(
            eval_config,
            target_dir,
            checkpoint_path,
            logger,
            cleanup_raw_samples=True,
        )
        print(f"[EVAL] wrote {target_dir}")


def _evaluate_best_checkpoint(train_dir, best_eval_dir, logger):
    best_ckpt = Path(train_dir) / "checkpoints" / "best.ckpt"
    if not best_ckpt.exists():
        print(f"[WARN] best checkpoint not found in {Path(train_dir) / 'checkpoints'}")
        return

    if (Path(best_eval_dir) / "ensemble_stats.csv").exists() and _best_eval_matches_checkpoint(
        best_eval_dir, best_ckpt
    ):
        print(f"[SKIP] best checkpoint already evaluated in {best_eval_dir}")
        return

    eval_config = load_config(Path(train_dir) / "config.yaml")
    eval_config = prepare_experiment_config(eval_config, best_eval_dir)
    evaluate(eval_config, best_eval_dir, best_ckpt, logger, cleanup_raw_samples=False)
    print(f"[EVAL] wrote best checkpoint outputs in {best_eval_dir}")

def run_training_and_evaluation(config, job_dir, resume=True):
    requested_epochs = int(config.training.max_epochs)
    requested_val_every = int(config.training.check_val_every_n_epoch)
    requested_ckpt_every = int(config.logging.checkpoint_every_n_epochs)

    run_root = _resolve_run_root(config, job_dir, resume)
    train_dir, best_eval_dir, periodic_eval_root = _save_dir_structure(run_root)

    effective_config = config
    config_path = train_dir / "config.yaml"
    previous_epochs = requested_epochs

    if config_path.exists():
        effective_config = ConfigDict(load_config(config_path))
        previous_epochs = int(effective_config.training.max_epochs)

    effective_config.training.max_epochs = requested_epochs
    effective_config.training.check_val_every_n_epoch = requested_val_every
    effective_config.logging.checkpoint_every_n_epochs = requested_ckpt_every

    resume_ckpt = last_checkpoint_path(train_dir) if train_dir.exists() else None
    done_epochs = 0

    if train_dir.exists():
        completed = completed_epochs(train_dir)
        if completed is not None:
            done_epochs = min(int(completed), previous_epochs)
        print(f"[RESUME] {train_dir} from epoch {done_epochs}")

    if done_epochs < requested_epochs:
        prepared_config, train_dir, logger = prepare_run(
            effective_config.to_dict(),
            experiment_dir=train_dir,
        )
        train(
            prepared_config,
            train_dir,
            logger,
            resume=str(resume_ckpt) if resume_ckpt else None,
        )
        completed = completed_epochs(train_dir)
        done_epochs = 0 if completed is None else min(int(completed), requested_epochs)
        print(f"[TRAIN] reached epoch {done_epochs} in {train_dir}")
    else:
        logger = configure_logger(Path(train_dir) / "run.log")
        print(f"[TRAIN] already complete at epoch {done_epochs}")

    _evaluate_periodic_checkpoints(train_dir, periodic_eval_root, logger)
    _evaluate_best_checkpoint(train_dir, best_eval_dir, logger)
    print(f"[DONE] {run_root}")

    return run_root
