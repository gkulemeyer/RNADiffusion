import json
import pandas as pd
import pytest

from src.data.datasets import load_dataset


def _write_csv(path, rows):
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_partition_missing_columns(tmp_path):
    main_csv = tmp_path / "main.csv"
    part_csv = tmp_path / "parts.csv"
    rows = [
        {"id": "a", "sequence": "ACGU", "base_pairs": json.dumps([])},
    ]
    _write_csv(main_csv, rows)
    _write_csv(part_csv, [{"id": "a"}])

    with pytest.raises(ValueError):
        load_dataset(
            partitioned=True,
            main_path=str(main_csv),
            partition_path=str(part_csv),
            partition_value="train",
            fold_number=0,
            max_len=10,
            for_prediction=False,
        )
