"""Ensemble generation and analysis workflows."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import mlflow
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from ..data.collate import pad_batch
from ..data.datasets import load_dataset
from ..evaluation.ensemble import EnsembleAnalyzer, EnsembleGenerator
from ..models.factory import create_model
from ..utils.io import resolve_dataset_config
from ..utils.mlflow_io import (
    iter_run_info,
    log_summary_metrics,
    resolve_run_dataset,
    resolve_tracking_dir,
)
from ..utils.reporting import export_run_logs, get_run_metadata

logger = logging.getLogger(__name__)


def _load_dataset_from_cfg(dataset_cfg):
    return load_dataset(**dataset_cfg)


def _write_ensemble_summary(results, output_path, run_id):
    numeric = results.select_dtypes(include="number")
    row = {"run_id": run_id}
    means = numeric.mean(numeric_only=True)
    stds = numeric.std(numeric_only=True)
    for col in numeric.columns:
        row[f"{col}_mean"] = means[col]
        row[f"{col}_std"] = stds[col]
    summary = results.__class__([row])
    summary.to_csv(output_path, index=False)
    return summary


def _attach_run_metadata(results, run_id, exp_name=None, run_name=None):
    results = results.copy()
    results.insert(0, "run_id", run_id)
    results.insert(1, "exp_name", exp_name)
    results.insert(2, "run_name", run_name)
    return results


def _matches_target_run(run_info, target_run_name=None):
    if target_run_name is None:
        return True
    return get_run_metadata(run_info)["run_name"] == target_run_name


def _generate_for_run(tracking_dir, run_info, dataset_cfg, ensemble_cfg):
    if run_info["checkpoint_path"] is None or run_info["config"] is None:
        logger.warning("Skipping run %s: missing checkpoint or config", run_info["run_id"])
        return

    model = create_model(run_info["config"], checkpoint_path=run_info["checkpoint_path"], eval_mode=True)
    dataset = _load_dataset_from_cfg(dataset_cfg)
    loader = DataLoader(
        dataset,
        batch_size=ensemble_cfg["batch_size"],
        shuffle=False,
        collate_fn=pad_batch,
        num_workers=ensemble_cfg["num_workers"],
    )
    generator = EnsembleGenerator(
        model=model,
        save_dir=None,
        num_samples=ensemble_cfg["num_samples"],
        base_seed=ensemble_cfg["base_seed"],
        mlflow_tracking_dir=tracking_dir,
        mlflow_run_id=run_info["run_id"],
    )
    generator.generate(loader)


def _analyze_run(tracking_dir, run_info, ensemble_stats_cfg, log_root=None):
    if run_info["samples_dir"] is None or not run_info["samples_dir"].exists():
        logger.warning("Skipping run %s: no raw ensemble samples found", run_info["run_id"])
        return False

    analyzer = EnsembleAnalyzer(
        samples_dir=run_info["samples_dir"],
        consensus_sizes=ensemble_stats_cfg["consensus_sizes"],
        trials=ensemble_stats_cfg["trials"],
    )
    results = analyzer.analyze()
    run_tags = run_info["tags"]
    results = _attach_run_metadata(results, run_info["run_id"], run_tags.get("exp_name"), run_tags.get("run_name"))

    output_dir = run_info["run_dir"] / "artifacts" / "ensembles"
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "ensemble_stats.csv"
    results.to_csv(stats_path, index=False)
    trial_stats_name = f"ensemble_stats_{ensemble_stats_cfg['trials']}_trials.csv"
    results.to_csv(output_dir / trial_stats_name, index=False)
    summary = _write_ensemble_summary(results, output_dir / "ensemble_stats_mean.csv", run_info["run_id"])
    log_summary_metrics(tracking_dir, run_info["run_id"], summary)
    if log_root is not None:
        export_run_logs(run_info, log_root, ensemble_trials=ensemble_stats_cfg["trials"])
    if run_info["samples_dir"].exists():
        shutil.rmtree(run_info["samples_dir"])
    logger.info("Saved stats to MLFlow run %s -> ensembles/ensemble_stats.csv", run_info["run_id"])
    return True


def run_ensemble_generation(cfg: DictConfig, target_run_name=None):
    """Generate ensembles for all runs with checkpoints."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    data_cfg = cfg_dict["data"]
    mlflow_cfg = cfg_dict["mlflow"]
    ensemble_cfg = cfg_dict["ensemble"]

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    tracking_dir = resolve_tracking_dir(mlflow_cfg["tracking_uri"])
    default_dataset_cfg = resolve_dataset_config(data_cfg, "test", cfg_dict.get("fold_number"), for_prediction=True)

    for run_info in iter_run_info(tracking_dir, checkpoints_only=True):
        if not _matches_target_run(run_info, target_run_name=target_run_name):
            continue
        _generate_for_run(tracking_dir, run_info, resolve_run_dataset(tracking_dir, run_info, default_dataset_cfg), ensemble_cfg)


def run_ensemble_analysis(cfg: DictConfig, target_run_name=None):
    """Analyze ensemble stats for all runs in MLflow."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    tracking_dir = resolve_tracking_dir(cfg_dict["mlflow"]["tracking_uri"])
    log_root = Path(cfg_dict["logging"]["root"]) / cfg_dict["logging"]["group_name"]
    log_root.mkdir(parents=True, exist_ok=True)
    for run_info in iter_run_info(tracking_dir):
        if not _matches_target_run(run_info, target_run_name=target_run_name):
            continue
        _analyze_run(tracking_dir, run_info, cfg_dict["ensemble_stats"], log_root=log_root)
