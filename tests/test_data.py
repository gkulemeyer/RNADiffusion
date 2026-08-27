from pathlib import Path

import pandas as pd
import pytest

import torch

from src.data import (
    RNADataModule,
    SeqDataset,
    _load_partitioned_data,
    build_dataloader,
    mat2bp,
    mat2db,
)


def test_mat2bp_reads_only_upper_triangle_and_uses_one_based_indices():
    matrix = torch.zeros((4, 4), dtype=torch.int8)
    matrix[0, 3] = 1
    matrix[2, 1] = 1
    matrix[2, 2] = 1

    assert mat2bp(matrix) == [[1, 4]]


def test_mat2db_uses_processed_base_pairs():
    matrix = torch.zeros((4, 4), dtype=torch.int8)
    matrix[0, 3] = matrix[3, 0] = 1
    matrix[1, 2] = matrix[2, 1] = 1

    assert mat2db(matrix) == "(())"


def test_load_partitioned_data_filters_partition_and_fold(tmp_path: Path):
    data_path = tmp_path / "data.csv"
    partitions_path = tmp_path / "partitions.csv"

    pd.DataFrame(
        [
            {"id": "a", "sequence": "AC", "base_pairs": "[]"},
            {"id": "b", "sequence": "GU", "base_pairs": "[]"},
            {"id": "c", "sequence": "CG", "base_pairs": "[]"},
        ]
    ).to_csv(data_path, index=False)
    pd.DataFrame(
        [
            {"id": "a", "partition": "train", "fold_number": 0},
            {"id": "b", "partition": "valid", "fold_number": 0},
            {"id": "c", "partition": "train", "fold_number": 1},
        ]
    ).to_csv(partitions_path, index=False)

    filtered = _load_partitioned_data(
        data_path,
        partitions_path,
        "train",
        fold_value=0,
        partition_scheme="simfold",
    )
    assert filtered["id"].tolist() == ["a"]


def test_load_partitioned_data_requires_columns(tmp_path: Path):
    data_path = tmp_path / "data.csv"
    partitions_path = tmp_path / "partitions.csv"

    pd.DataFrame([{"id": "a", "sequence": "AC", "base_pairs": "[]"}]).to_csv(data_path, index=False)
    pd.DataFrame([{"id": "a"}]).to_csv(partitions_path, index=False)

    with pytest.raises(ValueError):
        _load_partitioned_data(
            data_path,
            partitions_path,
            "train",
            fold_value=0,
            partition_scheme="simfold",
        )


def test_load_partitioned_data_errors_when_partition_missing(tmp_path: Path):
    data_path = tmp_path / "data.csv"
    partitions_path = tmp_path / "partitions.csv"

    pd.DataFrame([{"id": "a", "sequence": "AC", "base_pairs": "[]"}]).to_csv(data_path, index=False)
    pd.DataFrame([{"id": "a", "partition": "valid", "fold_number": 0}]).to_csv(partitions_path, index=False)

    with pytest.raises(ValueError, match="Available partitions"):
        _load_partitioned_data(
            data_path,
            partitions_path,
            "train",
            fold_value=0,
            partition_scheme="simfold",
        )


def test_load_partitioned_data_filters_partition_and_family_for_famfold(tmp_path: Path):
    data_path = tmp_path / "data.csv"
    partitions_path = tmp_path / "famfold.csv"

    pd.DataFrame(
        [
            {"id": "a", "sequence": "AC", "base_pairs": "[]"},
            {"id": "b", "sequence": "GU", "base_pairs": "[]"},
            {"id": "c", "sequence": "CG", "base_pairs": "[]"},
        ]
    ).to_csv(data_path, index=False)
    pd.DataFrame(
        [
            {"id": "a", "partition": "train", "fold": "23s"},
            {"id": "b", "partition": "valid", "fold": "23s"},
            {"id": "c", "partition": "train", "fold": "16s"},
        ]
    ).to_csv(partitions_path, index=False)

    filtered = _load_partitioned_data(
        data_path,
        partitions_path,
        "train",
        fold_value="23s",
        partition_scheme="famfold",
    )
    assert filtered["id"].tolist() == ["a"]


def test_seq_dataset_builds_base_pairs_from_structure(tmp_path: Path):
    data_path = tmp_path / "data.csv"
    partitions_path = tmp_path / "partitions.csv"

    pd.DataFrame(
        [
            {"id": "a", "sequence": "AC", "structure": "()"},
            {"id": "b", "sequence": "GU", "structure": ".."},
        ]
    ).to_csv(data_path, index=False)
    pd.DataFrame(
        [
            {"id": "a", "partition": "train", "fold_number": 0},
            {"id": "b", "partition": "train", "fold_number": 0},
        ]
    ).to_csv(partitions_path, index=False)

    dataset = SeqDataset(
        base_path=data_path,
        partition_path=partitions_path,
        partition_value="train",
        fold_value=0,
        partition_scheme="simfold",
    )
    assert dataset.base_pairs[0] == [[1, 2]]
    assert dataset.base_pairs[1] == []


def test_build_dataloader_and_datamodule_contract_are_consistent(tmp_path: Path):
    data_path = tmp_path / "data.csv"
    partitions_path = tmp_path / "partitions.csv"

    pd.DataFrame(
        [
            {"id": "a", "sequence": "AC", "base_pairs": "[]"},
            {"id": "b", "sequence": "GU", "base_pairs": "[]"},
            {"id": "c", "sequence": "CG", "base_pairs": "[]"},
        ]
    ).to_csv(data_path, index=False)
    pd.DataFrame(
        [
            {"id": "a", "partition": "train", "fold_number": 0},
            {"id": "b", "partition": "valid", "fold_number": 0},
            {"id": "c", "partition": "test", "fold_number": 0},
        ]
    ).to_csv(partitions_path, index=False)

    config = {
        "experiment": {"name": "test", "note": "", "seed": 42},
        "data": {
            "base_path": str(data_path),
            "partition_path": str(partitions_path),
            "partition_scheme": "simfold",
            "fold": 0,
        },
        "model": {
            "timesteps": 2,
            "num_classes": 2,
            "in_channels": 18,
            "out_channels": 2,
            "base_dim": 8,
        },
        "training": {
            "max_epochs": 1,
            "batch_size": 1,
            "lr": 1e-3,
            "num_workers": 0,
            "accelerator": "cpu",
            "devices": 1,
            "precision": 32,
        },
        "logging": {
            "save_dir": "logs/test",
            "log_every_n_steps": 1,
        },
    }

    loader = build_dataloader(config, partition="train", shuffle=False)
    batch_from_builder = next(iter(loader))

    data_module = RNADataModule(config)
    data_module.setup("fit")
    batch_from_module = next(iter(data_module.train_dataloader()))

    assert set(batch_from_builder.keys()) == set(batch_from_module.keys())
    assert batch_from_builder["embedding"].shape == batch_from_module["embedding"].shape
    assert batch_from_builder["conditioning"].shape == batch_from_module["conditioning"].shape
    assert batch_from_builder["contact_oh"].shape == batch_from_module["contact_oh"].shape
