"""Testing workflow."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import torch as tr
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from ..data.collate import pad_batch
from ..data.datasets import load_dataset
from ..evaluation.metrics import contact_f1
from ..models.factory import create_model
from ..utils.io import resolve_dataset_config
from ..utils.mlflow_io import iter_run_info, resolve_tracking_dir
from ..utils.reporting import (
    build_run_metric_summaries,
    build_run_metrics_dataframe,
    export_run_logs,
    get_run_metadata,
)

logger = logging.getLogger(__name__)


def _load_dataset_from_cfg(dataset_cfg):
    return load_dataset(**dataset_cfg)


def _save_test_metrics(run_info, result, metrics_extra=None):
    testing_dir = run_info["run_dir"] / "artifacts" / "testing"
    testing_dir.mkdir(parents=True, exist_ok=True)
    save_data = {
        "run_id": run_info["run_id"],
        "run_name": result.get("run_name"),
        "exp_name": result.get("exp_name"),
        "experiment": result.get("experiment"),
        "test_loss": float(result.get("test_loss", 0.0)),
        "test_f1": float(result.get("test_f1", 0.0)),
        "timesteps": result.get("timesteps"),
        "epochs": result.get("epochs"),
    }
    if metrics_extra:
        save_data.update(metrics_extra)
    (testing_dir / "test_metrics.json").write_text(json.dumps(save_data, indent=2))
    pd.DataFrame([save_data]).to_csv(testing_dir / "test_metrics.csv", index=False)


def evaluate_single_experiment(
    run_info,
    *,
    dataset,
    batch_size=4,
    num_workers=2,
    persist_metrics=True,
    metrics_extra=None,
):
    """Evaluate a single experiment checkpoint directory."""
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_batch,
        num_workers=num_workers,
    )

    model = create_model(run_info["config"], checkpoint_path=run_info["checkpoint_path"], eval_mode=True)
    device = tr.device("cuda" if tr.cuda.is_available() else "cpu")
    model.to(device)

    all_f1 = []
    all_loss = []
    with tr.no_grad():
        for batch in test_loader:
            cond = batch["outer"].to(device)
            target = batch["contact_oh"].to(device)
            mask = batch["mask"].to(device)
            lengths = batch["length"]
            all_loss.append(model.forward_all_timesteps(target, cond, mask=mask).item())
            all_f1.append(contact_f1(model._sample(cond), target, lengths=lengths, reduce=True))

    meta = get_run_metadata(run_info)
    result = {
        "experiment": meta["run_name"],
        "run_id": run_info["run_id"],
        "run_name": meta["run_name"],
        "exp_name": meta["exp_name"],
        "test_loss": sum(all_loss) / len(all_loss),
        "test_f1": sum(all_f1) / len(all_f1),
        "timesteps": meta["timesteps"],
        "epochs": meta["epochs"],
    }
    if persist_metrics:
        _save_test_metrics(run_info, result, metrics_extra)
    return result


def _matches_target_run(run_info, target_run_name=None):
    if target_run_name is None:
        return True
    return get_run_metadata(run_info)["run_name"] == target_run_name


def run_testing(cfg: DictConfig, target_run_name=None):
    """Run testing for all experiments with checkpoints in MLflow."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    logging_cfg = cfg_dict["logging"]
    mlflow_cfg = cfg_dict["mlflow"]
    data_cfg = cfg_dict["data"]
    testing_cfg = cfg_dict["testing"]

    log_root = Path(logging_cfg["root"]) / logging_cfg["group_name"]
    log_root.mkdir(parents=True, exist_ok=True)

    data_cfg = dict(data_cfg)
    if not data_cfg.get("use_partitions", False) and not data_cfg.get("test_path") and testing_cfg.get("fallback_test_path"):
        data_cfg["test_path"] = testing_cfg["fallback_test_path"]
    dataset_cfg = resolve_dataset_config(data_cfg, "test", cfg_dict.get("fold_number"), for_prediction=False)
    test_dataset = _load_dataset_from_cfg(dataset_cfg)

    tracking_dir = resolve_tracking_dir(mlflow_cfg["tracking_uri"])
    results = []
    last_metric_rows = []
    best_metric_rows = []
    for run_info in iter_run_info(tracking_dir, checkpoints_only=True):
        if run_info["config"] is None or not _matches_target_run(run_info, target_run_name=target_run_name):
            continue
        result = evaluate_single_experiment(
            run_info,
            dataset=test_dataset,
            batch_size=testing_cfg["batch_size"],
            num_workers=testing_cfg["num_workers"],
        )
        results.append(result)
        metrics_df = build_run_metrics_dataframe(run_info["run_dir"])
        if not metrics_df.empty:
            artifact_metrics_path = run_info["run_dir"] / "artifacts" / "metrics.csv"
            artifact_metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_df.to_csv(artifact_metrics_path, index=False)
        export_run_logs(run_info, log_root, metrics_df=metrics_df)
        last_summary, best_summary = build_run_metric_summaries(run_info, metrics_df)
        last_metric_rows.append(last_summary)
        best_metric_rows.append(best_summary)
        logger.info(
            "run %s | timesteps=%s | epochs=%s | test_f1=%.4f",
            result["run_name"],
            result["timesteps"],
            result["epochs"],
            result["test_f1"],
        )

    if not results:
        logger.info("No experiments found to evaluate.")
        return

    df = pd.DataFrame(results).sort_values(by="test_f1", ascending=False)
    ordered_cols = [col for col in ["run_name", "exp_name", "timesteps", "epochs", "test_loss", "test_f1", "run_id"] if col in df.columns]
    df = df[ordered_cols + [col for col in df.columns if col not in ordered_cols]]
    if last_metric_rows:
        pd.DataFrame(last_metric_rows).sort_values(by=["timesteps", "epochs"], na_position="last").to_csv(
            log_root / "last_metrics.csv",
            index=False,
        )
    if best_metric_rows:
        pd.DataFrame(best_metric_rows).sort_values(by=["timesteps", "epochs"], na_position="last").to_csv(
            log_root / "best_metrics.csv",
            index=False,
        )
    if testing_cfg.get("save_summary_csv", False):
        output_path = log_root / testing_cfg.get("summary_filename", "all_test_summary.csv")
        df.to_csv(output_path, index=False)
        logger.info("Testing Complete : %s", output_path)
    logger.info("\n%s", df)
