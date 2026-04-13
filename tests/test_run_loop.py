from pathlib import Path

from ml_collections import ConfigDict
import yaml

from src.run_loop import (
    _best_eval_matches_checkpoint,
    _layout_dirs,
    _next_attempt_dir,
    _resolve_run_root,
)


def make_config(tmp_path: Path):
    return ConfigDict(
        {
            "experiment": {"name": "seed42", "seed": 42},
            "data": {"partition_path": str(tmp_path / "ArchiveII_partitions_sim60.csv")},
            "model": {"timesteps": 10, "base_dim": 32},
            "training": {"batch_size": 1, "accumulate_grad_batches": 4, "max_epochs": 15},
            "logging": {"save_dir": str(tmp_path / "logs")},
        }
    )


def test_layout_dirs_for_new_runs(tmp_path: Path):
    train_dir, best_eval_dir, periodic_eval_root = _layout_dirs(tmp_path / "seed42")

    assert train_dir == tmp_path / "seed42" / "train"
    assert best_eval_dir == tmp_path / "seed42" / "eval" / "best"
    assert periodic_eval_root == tmp_path / "seed42" / "eval" / "periodic"


def test_next_attempt_dir_uses_incremental_suffix(tmp_path: Path):
    job_dir = tmp_path / "seed42"
    (job_dir / "attempt_001").mkdir(parents=True)
    (job_dir / "attempt_002").mkdir(parents=True)

    assert _next_attempt_dir(job_dir) == job_dir / "attempt_003"


def test_resolve_run_root_creates_attempt_when_job_dir_exists_and_resume_false(tmp_path: Path):
    config = make_config(tmp_path)
    job_dir = tmp_path / "seed42"
    (job_dir / "train").mkdir(parents=True)

    run_root = _resolve_run_root(config, job_dir, resume=False)

    assert run_root == job_dir / "attempt_001"


def test_resolve_run_root_prefers_latest_attempt_when_resuming(tmp_path: Path):
    config = make_config(tmp_path)
    job_dir = tmp_path / "seed42"
    (job_dir / "attempt_001" / "train").mkdir(parents=True)
    (job_dir / "attempt_002" / "train").mkdir(parents=True)

    run_root = _resolve_run_root(config, job_dir, resume=True)

    assert run_root == job_dir / "attempt_002"


def test_best_eval_matches_checkpoint_uses_metadata_path(tmp_path: Path):
    best_eval_dir = tmp_path / "eval" / "best"
    best_eval_dir.mkdir(parents=True)
    metadata_path = best_eval_dir / "ensemble_metadata.yaml"
    best_ckpt = tmp_path / "train" / "checkpoints" / "best.ckpt"
    best_ckpt.parent.mkdir(parents=True)
    import torch

    torch.save({"epoch": 1}, best_ckpt)

    with metadata_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {"checkpoint_path": str(best_ckpt), "checkpoint_epoch": 2},
            handle,
            sort_keys=False,
        )

    assert _best_eval_matches_checkpoint(best_eval_dir, best_ckpt) is True
    assert _best_eval_matches_checkpoint(best_eval_dir, best_ckpt.with_name("other.ckpt")) is False


def test_best_eval_matches_checkpoint_rejects_stale_epoch(tmp_path: Path):
    best_eval_dir = tmp_path / "eval" / "best"
    best_eval_dir.mkdir(parents=True)
    metadata_path = best_eval_dir / "ensemble_metadata.yaml"
    best_ckpt = tmp_path / "train" / "checkpoints" / "best.ckpt"
    best_ckpt.parent.mkdir(parents=True)

    import torch

    torch.save({"epoch": 4}, best_ckpt)

    with metadata_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {"checkpoint_path": str(best_ckpt), "checkpoint_epoch": 2},
            handle,
            sort_keys=False,
        )

    assert _best_eval_matches_checkpoint(best_eval_dir, best_ckpt) is False
