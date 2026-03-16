from pathlib import Path

import pytest
import torch

from src.sweeps import (
    COMPLETED_RUN_FILES,
    clone_config,
    completed_epochs,
    from_config_dict,
    last_checkpoint_path,
    run_completed,
    to_config_dict,
)


def write_completed_run(run_dir: Path):
    (run_dir / "checkpoints").mkdir(parents=True)
    for relative_path in COMPLETED_RUN_FILES:
        target = run_dir / relative_path
        if target.suffix:
            target.write_text("", encoding="utf-8")
        else:
            target.mkdir(parents=True, exist_ok=True)


def test_run_completed_accepts_path(tmp_path: Path):
    run_dir = tmp_path / "run"
    write_completed_run(run_dir)

    assert run_completed(run_dir) == run_dir


def test_run_completed_returns_none_for_incomplete_run(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text("", encoding="utf-8")

    assert run_completed(run_dir) is None


def test_run_completed_with_config_uses_latest_matching_run(tmp_path: Path):
    pytest.importorskip("ml_collections")

    save_dir = tmp_path / "logs"
    run_dir = save_dir / "sim60_t5_e1"
    write_completed_run(run_dir)

    config = to_config_dict(
        {
            "experiment": {"name": "sim60_t5_e1"},
            "logging": {"save_dir": str(save_dir)},
            "model": {"timesteps": 5},
            "training": {"max_epochs": 1},
        }
    )
    assert run_completed(config) == run_dir


def test_config_dict_roundtrip_and_clone():
    pytest.importorskip("ml_collections")

    config = {
        "experiment": {"name": "test"},
        "model": {"timesteps": 5},
        "training": {"max_epochs": 2},
    }

    config_dict = to_config_dict(config)
    config_dict.model.timesteps = 10

    cloned = clone_config(config_dict)
    cloned.training.max_epochs = 4

    plain = from_config_dict(config_dict)
    cloned_plain = from_config_dict(cloned)

    assert plain["model"]["timesteps"] == 10
    assert plain["training"]["max_epochs"] == 2
    assert cloned_plain["training"]["max_epochs"] == 4


def test_completed_epochs_reads_last_checkpoint(tmp_path: Path):
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_path = checkpoint_dir / "last.ckpt"
    torch.save({"epoch": 1}, checkpoint_path)

    assert last_checkpoint_path(run_dir) == checkpoint_path
    assert completed_epochs(run_dir) == 2
