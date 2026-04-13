from pathlib import Path

import torch

from src.experiment import handle_metrics, normalize_periodic_checkpoint_names


def make_csv_logger(log_dir: Path):
    csv_logger_cls = type("CSVLogger", (), {})
    logger = csv_logger_cls()
    logger.log_dir = str(log_dir)
    return logger


def test_handle_metrics_writes_single_compacted_metrics_file(tmp_path: Path):
    lightning_dir = tmp_path / "lightning" / "version_0"
    lightning_dir.mkdir(parents=True)
    metrics_path = tmp_path / "metrics.csv"

    (lightning_dir / "metrics.csv").write_text(
        "\n".join(
            [
                "epoch,step,train_loss,val_f1",
                "0,1,1.0,",
                "0,1,,0.8",
                "1,2,0.7,",
            ]
        ),
        encoding="utf-8",
    )

    config = {
        "logging": {
            "lightning_dir": str(tmp_path / "lightning"),
            "metrics_path": str(metrics_path),
        }
    }

    handle_metrics([make_csv_logger(lightning_dir)], config, resume=False)

    assert metrics_path.exists()
    assert not (tmp_path / "lightning" / "metrics_raw.csv").exists()
    assert not (lightning_dir / "metrics.csv").exists()
    assert metrics_path.read_text(encoding="utf-8") == "\n".join(
        [
            "epoch,step,train_loss,val_f1",
            "0,1,1.0,0.8",
            "1,2,0.7,",
        ]
    ) + "\n"


def test_handle_metrics_merges_resume_into_same_metrics_file(tmp_path: Path):
    lightning_dir = tmp_path / "lightning" / "version_0"
    lightning_dir.mkdir(parents=True)
    metrics_path = tmp_path / "metrics.csv"

    metrics_path.write_text(
        "\n".join(
            [
                "epoch,step,train_loss,val_f1",
                "0,1,1.0,0.8",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (lightning_dir / "metrics.csv").write_text(
        "\n".join(
            [
                "epoch,step,train_loss,val_f1",
                "1,2,0.7,",
                "1,2,,0.9",
            ]
        ),
        encoding="utf-8",
    )

    config = {
        "logging": {
            "lightning_dir": str(tmp_path / "lightning"),
            "metrics_path": str(metrics_path),
        }
    }

    handle_metrics([make_csv_logger(lightning_dir)], config, resume=True)

    assert metrics_path.read_text(encoding="utf-8") == "\n".join(
        [
            "epoch,step,train_loss,val_f1",
            "0,1,1.0,0.8",
            "1,2,0.7,0.9",
        ]
    ) + "\n"


def test_normalize_periodic_checkpoint_names_uses_one_based_names(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints" / "periodic"
    checkpoint_dir.mkdir(parents=True)
    source_path = checkpoint_dir / "epoch004.ckpt"
    torch.save({"epoch": 4}, source_path)

    normalize_periodic_checkpoint_names(tmp_path)

    assert not source_path.exists()
    assert (checkpoint_dir / "epoch005.ckpt").exists()
