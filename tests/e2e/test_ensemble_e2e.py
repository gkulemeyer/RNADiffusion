import json
import pandas as pd
from omegaconf import OmegaConf

from src.models.factory import create_model
from src.utils.io import save_config
from src.experiments.ensemble import run_ensemble_generation, run_ensemble_analysis


def _write_csv(path, rows):
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_ensemble_end_to_end(tmp_path):
    tracking_dir = tmp_path / "mlruns"
    exp_dir = tracking_dir / "0"
    run_id = "run1"
    run_dir = exp_dir / run_id
    artifacts_dir = run_dir / "artifacts"
    ckpt_dir = artifacts_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "network": {
            "denoiser": {"name": "simple_unet", "params": {"in_channels": 18, "out_channels": 2, "base_dim": 8}},
            "wrapper": {"name": "diffusion", "schedule": "cosine", "timesteps": 2},
        },
        "training": {"epochs": 3},
        "data": {"num_classes": 4},
        "run_name": "demo-run",
    }
    save_config(config, artifacts_dir)
    (run_dir / "meta.yaml").write_text("run_name: demo-run\n")

    model = create_model(config, move_to_device=False)
    ckpt_path = ckpt_dir / "best_model.pt"
    model_state = {"model_state": model.state_dict()}
    import torch as tr

    tr.save(model_state, ckpt_path)

    test_csv = tmp_path / "test.csv"
    rows = [
        {"id": "a", "sequence": "ACGU", "base_pairs": json.dumps([])},
        {"id": "b", "sequence": "ACG", "base_pairs": json.dumps([])},
    ]
    _write_csv(test_csv, rows)

    cfg = OmegaConf.create(
        {
            "data": {"split": "sim", "use_partitions": False, "test_path": str(test_csv), "max_len": 8},
            "mlflow": {"tracking_uri": f"file:{tracking_dir}", "experiment_name": "test"},
            "logging": {"root": str(tmp_path / "logs"), "group_name": "test"},
            "ensemble": {"num_samples": 2, "base_seed": 0, "batch_size": 1, "num_workers": 0},
            "ensemble_stats": {"consensus_sizes": [1], "trials": 1},
        }
    )

    missing_run_dir = exp_dir / "run_missing" / "artifacts" / "checkpoints"
    missing_run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, missing_run_dir.parent)
    tr.save(model_state, missing_run_dir / "best_model.pt")

    run_ensemble_generation(cfg)
    run_ensemble_analysis(cfg)

    stats_path = run_dir / "artifacts" / "ensembles" / "ensemble_stats.csv"
    assert stats_path.exists()
    assert (run_dir / "artifacts" / "ensembles" / "ensemble_stats_1_trials.csv").exists()
    assert (tmp_path / "logs" / "test" / "demo-run" / "ensemble_stats_1_trials.csv").exists()
    assert (tmp_path / "logs" / "test" / "demo-run" / "enemble_stats_1_trials.csv").exists()
