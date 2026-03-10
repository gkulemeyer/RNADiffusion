"""Reporting helpers for local MLflow experiment summaries."""
from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path

import pandas as pd


def _read_meta(path):
    if not path.exists():
        return {}
    meta = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip().strip("'\"")
    return meta


def _read_metric_rows(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        try:
            rows.append((int(float(parts[0])), float(parts[1]), int(float(parts[2]))))
        except ValueError:
            continue
    return rows


def _read_test_metrics(run_dir):
    metrics_path = run_dir / "artifacts" / "testing" / "test_metrics.json"
    if not metrics_path.exists():
        return {}
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _collect_params_row(run_dir, run_name, include_ids, test_metrics):
    row = {"run_id": run_dir.name, "run_name": run_name} if include_ids else {}
    for directory, prefix in ((run_dir / "params", ""), (run_dir / "tags", "tag_")):
        if not directory.exists():
            continue
        for file_path in sorted(directory.iterdir()):
            if not file_path.is_file():
                continue
            try:
                row[f"{prefix}{file_path.name}"] = file_path.read_text(encoding="utf-8").strip()
            except Exception:
                row[f"{prefix}{file_path.name}"] = ""

    for key, value in test_metrics.items():
        if str(key).startswith("test_"):
            row[key] = value

    partition_sources = (
        str(row.get("tag_partition_name", "")),
        str(row.get("tag_partition_file", "")),
        str(row.get("partition_path", "")),
    )
    row["partition"] = next(
        (match.group(1) for source in partition_sources if (match := re.search(r"(sim\d+)", source))),
        "",
    )
    return row


def _collect_metric_rows(run_dir, run_name, include_ids, test_metrics):
    summary_row = {"run_id": run_dir.name, "run_name": run_name} if include_ids else {}
    long_rows = []
    metrics_dir = run_dir / "metrics"
    if metrics_dir.exists():
        for metric_file in sorted(metrics_dir.iterdir()):
            if not metric_file.is_file():
                continue
            metric_rows = _read_metric_rows(metric_file)
            if not metric_rows:
                continue
            long_rows.extend(
                {
                    "epoch": step,
                    "metric": metric_file.name,
                    "value": value,
                    "run_id": run_dir.name,
                    "run_name": run_name,
                }
                for _ts, value, step in metric_rows
            )
            _ts, latest_value, _latest_step = max(metric_rows, key=lambda row: (row[2], row[0]))
            summary_row[metric_file.name] = latest_value

    for key, value in test_metrics.items():
        if not str(key).startswith("test_"):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        long_rows.append(
            {
                "epoch": -1,
                "metric": key,
                "value": numeric,
                "run_id": run_dir.name,
                "run_name": run_name,
            }
        )
        summary_row[key] = numeric

    return long_rows, summary_row if summary_row else None


def _collect_tables(experiment_dir, include_ids):
    params_rows = []
    metrics_long_rows = []
    metrics_summary_rows = []

    run_dirs = sorted(
        path
        for path in experiment_dir.iterdir()
        if path.is_dir() and (path / "meta.yaml").exists() and (path / "params").exists()
    )
    for run_dir in run_dirs:
        meta = _read_meta(run_dir / "meta.yaml")
        run_name = meta.get("run_name", "")
        test_metrics = _read_test_metrics(run_dir)

        params_rows.append(_collect_params_row(run_dir, run_name, include_ids, test_metrics))
        metric_rows, summary_row = _collect_metric_rows(run_dir, run_name, include_ids, test_metrics)
        metrics_long_rows.extend(metric_rows)
        if summary_row is not None:
            metrics_summary_rows.append(summary_row)

    return params_rows, metrics_long_rows, metrics_summary_rows


def _ordered_fieldnames(rows, preferred):
    seen = set()
    ordered = []
    for key in preferred:
        if any(key in row for row in rows):
            ordered.append(key)
            seen.add(key)
    for row in rows:
        for key in row:
            if key not in seen:
                ordered.append(key)
                seen.add(key)
    return ordered


def _write_csv(path, rows, preferred):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _ordered_fieldnames(rows, preferred)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def get_run_metadata(run_info):
    run_dir = Path(run_info["run_dir"])
    config = run_info.get("config") or {}
    tags = run_info.get("tags") or {}
    meta = _read_meta(run_dir / "meta.yaml")

    wrapper_cfg = (config.get("network") or {}).get("wrapper") or {}
    training_cfg = config.get("training") or {}
    run_name = tags.get("run_name") or meta.get("run_name") or config.get("run_name") or run_info["run_id"]
    exp_name = tags.get("exp_name") or (config.get("mlflow") or {}).get("experiment_name")

    return {
        "run_id": run_info["run_id"],
        "run_name": run_name,
        "exp_name": exp_name,
        "timesteps": wrapper_cfg.get("timesteps"),
        "epochs": training_cfg.get("epochs"),
        "fold_number": config.get("fold_number"),
    }


def build_run_metrics_dataframe(run_dir):
    metrics_dir = Path(run_dir) / "metrics"
    if not metrics_dir.exists():
        return pd.DataFrame()

    metric_series = {}
    all_steps = set()
    for metric_file in sorted(metrics_dir.iterdir()):
        if not metric_file.is_file():
            continue
        by_step = {}
        for ts, value, step in _read_metric_rows(metric_file):
            current = by_step.get(step)
            if current is None or ts >= current[0]:
                by_step[step] = (ts, value)
        if not by_step:
            continue
        metric_series[metric_file.name] = {step: value for step, (_ts, value) in by_step.items()}
        all_steps.update(by_step)

    if not all_steps:
        return pd.DataFrame()

    epoch_series = metric_series.get("epoch", {})
    rows = []
    for step in sorted(all_steps):
        row = {"step": step}
        epoch_value = epoch_series.get(step)
        if epoch_value is not None:
            row["epoch"] = int(round(epoch_value))
        for metric_name, series in metric_series.items():
            if metric_name == "epoch":
                continue
            if step in series:
                row[metric_name] = series[step]
        rows.append(row)

    df = pd.DataFrame(rows)
    ordered = [column for column in ("epoch", "step", "train_loss", "val_loss", "val_f1") if column in df.columns]
    ordered.extend(column for column in df.columns if column not in ordered)
    return df[ordered].sort_values(by=[column for column in ("epoch", "step") if column in df.columns]).reset_index(drop=True)


def build_run_metric_summaries(run_info, metrics_df):
    meta = get_run_metadata(run_info)
    base = {
        "experiment": meta["run_name"],
        "run_id": meta["run_id"],
        "exp_name": meta["exp_name"],
        "timesteps": meta["timesteps"],
        "epochs": meta["epochs"],
    }

    last_summary = dict(base)
    best_summary = dict(base)

    if metrics_df.empty:
        return last_summary, best_summary

    if "train_loss" in metrics_df:
        last_summary["final_train_loss"] = float(metrics_df["train_loss"].dropna().iloc[-1])
    if "val_loss" in metrics_df:
        val_loss_series = metrics_df["val_loss"].dropna()
        if not val_loss_series.empty:
            last_summary["final_val_loss"] = float(val_loss_series.iloc[-1])
            best_idx = val_loss_series.idxmin()
            best_summary["best_val_loss"] = float(metrics_df.loc[best_idx, "val_loss"])
            if "epoch" in metrics_df:
                best_summary["best_val_loss_epoch"] = int(metrics_df.loc[best_idx, "epoch"])
    if "val_f1" in metrics_df:
        val_f1_series = metrics_df["val_f1"].dropna()
        if not val_f1_series.empty:
            last_summary["final_val_f1"] = float(val_f1_series.iloc[-1])
            best_idx = val_f1_series.idxmax()
            best_summary["best_val_f1"] = float(metrics_df.loc[best_idx, "val_f1"])
            if "epoch" in metrics_df:
                best_summary["best_val_f1_epoch"] = int(metrics_df.loc[best_idx, "epoch"])

    return last_summary, best_summary


def export_run_logs(
    run_info,
    log_root,
    *,
    metrics_df=None,
    ensemble_trials=None,
):
    log_root = Path(log_root)
    meta = get_run_metadata(run_info)
    run_output_dir = log_root / meta["run_name"]
    run_output_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = Path(run_info["run_dir"]) / "artifacts"
    config_path = artifacts_dir / "config.json"
    if config_path.exists():
        shutil.copy2(config_path, run_output_dir / "config.json")

    if metrics_df is not None and not metrics_df.empty:
        metrics_df.to_csv(run_output_dir / "metrics.csv", index=False)

    test_metrics_csv = artifacts_dir / "testing" / "test_metrics.csv"
    if test_metrics_csv.exists():
        shutil.copy2(test_metrics_csv, run_output_dir / "test_metrics.csv")

    ensembles_dir = artifacts_dir / "ensembles"
    if ensemble_trials is not None:
        source_name = f"ensemble_stats_{ensemble_trials}_trials.csv"
        source_path = ensembles_dir / source_name
        if source_path.exists():
            shutil.copy2(source_path, run_output_dir / source_name)
            shutil.copy2(source_path, run_output_dir / f"enemble_stats_{ensemble_trials}_trials.csv")


def write_experiment_summary_csvs(experiment_dir, output_dir=None, include_ids=False):
    experiment_dir = Path(experiment_dir).resolve()
    if not experiment_dir.exists() or not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    output_dir = Path(output_dir).resolve() if output_dir is not None else experiment_dir
    params_rows, metrics_long_rows, metrics_summary_rows = _collect_tables(experiment_dir, include_ids)

    params_path = output_dir / "params_summary_long.csv"
    metrics_long_path = output_dir / "metrics_long_summary_long.csv"
    metrics_summary_path = output_dir / "metrics_summary_long.csv"

    _write_csv(params_path, params_rows, ["fold_number", "lr", "epochs", "timesteps", "partition", "run_id", "run_name"])
    _write_csv(metrics_long_path, metrics_long_rows, ["epoch", "metric", "value", "run_id", "run_name"])
    _write_csv(metrics_summary_path, metrics_summary_rows, ["run_id", "run_name"])

    return {
        "params_path": params_path,
        "metrics_long_path": metrics_long_path,
        "metrics_summary_path": metrics_summary_path,
        "params_rows": len(params_rows),
        "metrics_long_rows": len(metrics_long_rows),
        "metrics_summary_rows": len(metrics_summary_rows),
    }
