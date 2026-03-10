import json
import pandas as pd

from src.data.datasets import load_dataset


def _write_csv(path, rows):
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_load_dataset_basic(tmp_path):
    csv_path = tmp_path / "data.csv"
    rows = [
        {"id": "a", "sequence": "ACGU", "base_pairs": json.dumps([])},
        {"id": "b", "sequence": "ACG", "base_pairs": json.dumps([])},
    ]
    _write_csv(csv_path, rows)

    ds = load_dataset(dataset_path=str(csv_path), max_len=10, for_prediction=False)
    assert len(ds) == 2
    item = ds[0]
    assert item["contact_oh"] is not None


def test_load_dataset_partitioned(tmp_path):
    main_csv = tmp_path / "main.csv"
    part_csv = tmp_path / "parts.csv"
    rows = [
        {"id": "a", "sequence": "ACGU", "base_pairs": json.dumps([])},
        {"id": "b", "sequence": "ACG", "base_pairs": json.dumps([])},
    ]
    _write_csv(main_csv, rows)
    part_rows = [
        {"id": "a", "partition": "train", "fold_number": 0},
        {"id": "b", "partition": "test", "fold_number": 0},
    ]
    _write_csv(part_csv, part_rows)

    ds = load_dataset(
        partitioned=True,
        main_path=str(main_csv),
        partition_path=str(part_csv),
        partition_value="train",
        fold_number=0,
        max_len=10,
        for_prediction=False,
    )
    assert len(ds) == 1
