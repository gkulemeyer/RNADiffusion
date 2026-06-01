from pathlib import Path
import os

from ml_collections import ConfigDict

from src.config import (
    clone_config,
    latest_run_dir,
    to_config_dict,
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


def test_clone_config_returns_independent_configdict():
    config = {"experiment": {"name": "original"}}

    cloned = clone_config(config)
    cloned.experiment.name = "changed"

    assert isinstance(to_config_dict(config), ConfigDict)
    assert config["experiment"]["name"] == "original"
    assert cloned.experiment.name == "changed"
