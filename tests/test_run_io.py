from pathlib import Path

import torch

from src.run_io import RunIO


def save_ckpt(path: Path, epoch: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch}, path)


def test_run_io_layout(tmp_path):
    run = RunIO(tmp_path / "run")

    assert run.train_dir == tmp_path / "run/train"
    assert run.config_path == tmp_path / "run/train/config.yaml"
    assert run.log_path == tmp_path / "run/train/run.log"
    assert run.metrics_path == tmp_path / "run/train/metrics.csv"
    assert run.checkpoint_dir == tmp_path / "run/train/checkpoints"
    assert run.periodic_ckpt_dir == tmp_path / "run/train/checkpoints/periodic"
    assert run.best_ckpt_path == tmp_path / "run/train/checkpoints/best.ckpt"
    assert run.last_ckpt_path == tmp_path / "run/train/checkpoints/last.ckpt"
    assert run.best_eval_dir == tmp_path / "run/eval/best"
    assert run.periodic_eval_dir(4) == tmp_path / "run/eval/periodic/epoch_004"


def test_periodic_checkpoints_are_sorted_by_checkpoint_epoch(tmp_path):
    run = RunIO(tmp_path / "run")
    save_ckpt(run.periodic_ckpt_dir / "second.ckpt", epoch=8)
    save_ckpt(run.periodic_ckpt_dir / "first.ckpt", epoch=2)

    assert run.periodic_checkpoints() == [
        run.periodic_ckpt_dir / "first.ckpt",
        run.periodic_ckpt_dir / "second.ckpt",
    ]


def test_completed_epoch_count_reads_last_checkpoint(tmp_path):
    run = RunIO(tmp_path / "run")

    assert run.last_checkpoint() is None
    assert run.last_completed_epoch() is None
    assert run.completed_epoch_count() == 0

    save_ckpt(run.last_ckpt_path, epoch=9)

    assert run.last_checkpoint() == run.last_ckpt_path
    assert run.last_completed_epoch() == 9
    assert run.completed_epoch_count() == 10


def test_checkpoint_epoch_reads_checkpoint_metadata(tmp_path):
    checkpoint = tmp_path / "model.ckpt"
    save_ckpt(checkpoint, epoch=12)

    assert RunIO.checkpoint_epoch(checkpoint) == 12


def test_from_train_dir_keeps_run_layout(tmp_path):
    run = RunIO.from_train_dir(tmp_path / "run/train")

    assert run.root == tmp_path / "run"
