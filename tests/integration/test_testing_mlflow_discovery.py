import json

import pandas as pd
import torch as tr
from omegaconf import OmegaConf

from src.experiments.evaluation import run_testing
from src.models.factory import create_model
from src.utils.io import save_config


def _write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_testing_discovers_runs_from_mlflow(tmp_path):
    tracking_dir = tmp_path / "mlruns"
    run_dir = tracking_dir / "0" / "run1"
    artifacts_dir = run_dir / "artifacts"
    checkpoint_dir = artifacts_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "network": {
            "denoiser": {
                "name": "simple_unet",
                "params": {"in_channels": 18, "out_channels": 2, "base_dim": 8},
            },
            "wrapper": {"name": "diffusion", "schedule": "cosine", "timesteps": 2},
        },
        "training": {"epochs": 3},
        "data": {"num_classes": 4},
        "run_name": "demo-run",
    }
    save_config(config, artifacts_dir)
    (run_dir / "meta.yaml").write_text("run_name: demo-run\n")
    (metrics_dir / "epoch").write_text("0 0 0\n1 1 1\n2 2 2\n")
    (metrics_dir / "train_loss").write_text("0 0.8 0\n1 0.5 1\n2 0.3 2\n")
    (metrics_dir / "val_loss").write_text("0 0.7 0\n1 0.4 1\n2 0.2 2\n")
    (metrics_dir / "val_f1").write_text("0 0.1 0\n1 0.4 1\n2 0.9 2\n")

    model = create_model(config, move_to_device=False)
    tr.save({"model_state": model.state_dict()}, checkpoint_dir / "best_model.pt")

    test_csv = tmp_path / "test.csv"
    rows = [
        {"id": "a", "sequence": "ACGU", "base_pairs": json.dumps([])},
        {"id": "b", "sequence": "ACG", "base_pairs": json.dumps([])},
    ]
    _write_csv(test_csv, rows)

    cfg = OmegaConf.create(
        {
            "paths": {
                "workspace_root": str(tmp_path),
                "data_root": str(tmp_path / "data"),
                "logs_root": str(tmp_path / "logs"),
                "mlruns_root": str(tracking_dir),
            },
            "project_name": "test",
            "data": {"split": "sim", "use_partitions": False, "test_path": str(test_csv), "max_len": 8},
            "mlflow": {"tracking_uri": f"file:{tracking_dir}", "experiment_name": "test"},
            "logging": {"root": str(tmp_path / "logs"), "group_name": "test"},
            "testing": {
                "batch_size": 1,
                "num_workers": 0,
                "fallback_test_path": None,
                "save_summary_csv": True,
                "summary_filename": "summary.csv",
            },
        }
    )

    run_testing(cfg)

    assert (tmp_path / "logs" / "test" / "summary.csv").exists()
    assert (tmp_path / "logs" / "test" / "demo-run" / "metrics.csv").exists()
    assert (tmp_path / "logs" / "test" / "last_metrics.csv").exists()
    assert (tmp_path / "logs" / "test" / "best_metrics.csv").exists()
    assert (artifacts_dir / "metrics.csv").exists()
    assert (artifacts_dir / "testing" / "test_metrics.json").exists()

    summary = pd.read_csv(tmp_path / "logs" / "test" / "summary.csv")
    assert summary.loc[0, "timesteps"] == 2
    assert summary.loc[0, "epochs"] == 3
