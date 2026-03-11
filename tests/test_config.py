from pathlib import Path

import pytest

from src.config import load_config, prepare_experiment_config


def test_load_yaml_config(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  name: test",
                "data:",
                "  base_path: dataset.csv",
                "  partition_path: partitions.csv",
                "  fold: 0",
                "model:",
                "  timesteps: 5",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(str(config_path))
    assert config["experiment"]["name"] == "test"
    assert config["training"]["batch_size"] == 4


def test_prepare_experiment_config_adds_metadata(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  name: ''",
                "data:",
                "  base_path: dataset.csv",
                "  partition_path: partitions.csv",
                "  fold: 0",
                "model:",
                "  timesteps: 5",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(str(config_path))
    prepared = prepare_experiment_config(config, "logs/RNADiffusion/exp_test")
    assert prepared["experiment"]["uuid"]
    assert prepared["experiment"]["timestamp"]
    assert prepared["logging"]["ensemble_path"].endswith("ensemble.csv")


def test_load_config_keeps_invalid_values_without_strict_validation(tmp_path: Path):
    config_path = tmp_path / "loose.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  name: loose",
                "model:",
                "  timesteps: 0",
                "  num_classes: 2",
                "  in_channels: 17",
                "  out_channels: 1",
                "  base_dim: 10",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(str(config_path))
    assert config["model"]["timesteps"] == 0
    assert config["model"]["in_channels"] == 17


def test_load_config_missing_path_raises(tmp_path: Path):
    missing_path = tmp_path / "does-not-exist.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(str(missing_path))
