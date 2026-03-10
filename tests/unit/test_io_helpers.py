import pandas as pd

from src.utils.io import resolve_dataset_config


def test_resolve_dataset_config_defaults():
    dataset_cfg = resolve_dataset_config({"split": "simfolds"}, "test")

    assert dataset_cfg == {
        "dataset_path": None,
        "min_len": 0,
        "max_len": 512,
        "for_prediction": False,
        "partitioned": False,
        "main_path": None,
        "partition_path": None,
        "partition_value": None,
        "fold_number": None,
    }


def test_resolve_dataset_config_partitioned(tmp_path):
    partition_path = tmp_path / "simfolds_partitions.csv"
    pd.DataFrame({"fold_number": [2, 1, 2]}).to_csv(partition_path, index=False)

    data_cfg = {
        "split": "simfolds",
        "main_path": str(tmp_path / "dataset.csv"),
        "partition_file_template": str(tmp_path / "${split}_partitions.csv"),
        "use_partitions": True,
        "test_partition": "heldout",
        "min_len": 4,
        "max_len": 32,
    }

    dataset_cfg = resolve_dataset_config(data_cfg, "test", fold_number=3, for_prediction=True)

    assert dataset_cfg == {
        "dataset_path": None,
        "min_len": 4,
        "max_len": 32,
        "for_prediction": True,
        "partitioned": True,
        "main_path": str(tmp_path / "dataset.csv"),
        "partition_path": str(partition_path),
        "partition_value": "heldout",
        "fold_number": 3,
    }


def test_resolve_dataset_config_direct_path(tmp_path):
    test_path = tmp_path / "test.csv"
    dataset_cfg = resolve_dataset_config({"test_path": str(test_path), "max_len": 64}, "test")

    assert dataset_cfg == {
        "dataset_path": str(test_path),
        "min_len": 0,
        "max_len": 64,
        "for_prediction": False,
        "partitioned": False,
        "main_path": None,
        "partition_path": None,
        "partition_value": None,
        "fold_number": None,
    }
