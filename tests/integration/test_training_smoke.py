import json
import pandas as pd
from omegaconf import OmegaConf

from src.experiments.training import run_training


def _write_csv(path, rows):
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_training_runner_smoke(tmp_path):
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    rows = [
        {"id": "a", "sequence": "ACGU", "base_pairs": json.dumps([])},
        {"id": "b", "sequence": "ACG", "base_pairs": json.dumps([])},
    ]
    _write_csv(train_csv, rows)
    _write_csv(val_csv, rows)

    cfg = OmegaConf.create(
        {
            "project_name": "test",
            "exp_name": "debug_experiment",
            "run_name": None,
            "run_name_prefix": "debug",
            "data": {
                "split": "sim",
                "use_partitions": False,
                "train_path": str(train_csv),
                "val_path": str(val_csv),
                "max_len": 8,
                "num_classes": 4,
            },
            "training": {"batch_size": 1, "lr": 1e-3, "epochs": 1, "num_workers": 0},
            "network": {
                "denoiser": {"name": "simple_unet", "params": {"in_channels": 18, "out_channels": 2, "base_dim": 8}},
                "wrapper": {"name": "diffusion", "schedule": "cosine", "timesteps": 2},
            },
            "mlflow": {
                "tracking_uri": f"file:{tmp_path / 'mlruns'}",
                "experiment_name": "debug_experiment",
                "run_name": None,
            },
            "logging": {"root": str(tmp_path / "logs"), "group_name": "test"},
        }
    )

    run_name, best_f1 = run_training(cfg)
    assert isinstance(run_name, str)
    assert run_name.startswith("debug_epochs-1_timesteps-2_")
    assert best_f1 is not None


def test_training_runner_respects_explicit_run_name(tmp_path):
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    rows = [
        {"id": "a", "sequence": "ACGU", "base_pairs": json.dumps([])},
        {"id": "b", "sequence": "ACG", "base_pairs": json.dumps([])},
    ]
    _write_csv(train_csv, rows)
    _write_csv(val_csv, rows)

    cfg = OmegaConf.create(
        {
            "project_name": "test",
            "exp_name": "debug_experiment",
            "run_name": "manual_debug_name",
            "run_name_prefix": "ignored",
            "data": {
                "split": "sim",
                "use_partitions": False,
                "train_path": str(train_csv),
                "val_path": str(val_csv),
                "max_len": 8,
                "num_classes": 4,
            },
            "training": {"batch_size": 1, "lr": 1e-3, "epochs": 1, "num_workers": 0},
            "network": {
                "denoiser": {"name": "simple_unet", "params": {"in_channels": 18, "out_channels": 2, "base_dim": 8}},
                "wrapper": {"name": "diffusion", "schedule": "cosine", "timesteps": 2},
            },
            "mlflow": {
                "tracking_uri": f"file:{tmp_path / 'mlruns'}",
                "experiment_name": "debug_experiment",
                "run_name": "manual_debug_name",
            },
            "logging": {"root": str(tmp_path / "logs"), "group_name": "test"},
        }
    )

    run_name, best_f1 = run_training(cfg)
    assert run_name == "manual_debug_name"
    assert best_f1 is not None
