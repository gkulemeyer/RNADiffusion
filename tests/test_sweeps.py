from pathlib import Path
import os

import torch
from ml_collections import ConfigDict

from src.sweeps import (
    completed_epochs,
    last_checkpoint_path,
    latest_run_dir,
)


def test_latest_run_dir_uses_latest_matching_run(tmp_path: Path):
    save_dir = tmp_path / "logs"
    older_run_dir = save_dir / "sim60_t5_e1_dt20200101-0000"
    newer_run_dir = save_dir / "sim60_t5_e1_dt20200101-0001"
    older_run_dir.mkdir(parents=True)
    newer_run_dir.mkdir(parents=True)
    os.utime(older_run_dir, (1, 1))
    os.utime(newer_run_dir, (2, 2))

    config = ConfigDict(
        {
            "experiment": {"name": "sim60_t5_e1"},
            "logging": {"save_dir": str(save_dir)},
            "model": {"timesteps": 5},
            "training": {"max_epochs": 1},
        }
    )
    assert latest_run_dir(config) == newer_run_dir


def test_completed_epochs_reads_last_checkpoint(tmp_path: Path):
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_path = checkpoint_dir / "last.ckpt"
    torch.save({"epoch": 1}, checkpoint_path)

    assert last_checkpoint_path(run_dir) == checkpoint_path
    assert completed_epochs(run_dir) == 2
