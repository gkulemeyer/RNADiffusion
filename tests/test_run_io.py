from pathlib import Path
import pytest
import torch
from ml_collections import ConfigDict

from src.io import write_yaml
from src.run_io import RunIO, RunResolver


def save_ckpt(path: Path, epoch: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch}, path)


def make_config(tmp_path: Path, max_epochs=10):
    return ConfigDict(
        {
            "experiment": {"name": "seed42", "seed": 42},
            "training": {
                "max_epochs": max_epochs,
                "check_val_every_n_epoch": 1,
            },
            "logging": {
                "checkpoint_every_n_epochs": 5,
                "save_dir": str(tmp_path / "logs"),
            },
        }
    )


def test_run_io_layout_paths(tmp_path: Path):
    run = RunIO(tmp_path / "run")

    assert run.train_dir == tmp_path / "run" / "train"
    assert run.config_path == tmp_path / "run" / "train" / "config.yaml"
    assert run.metrics_path == tmp_path / "run" / "train" / "metrics.csv"
    assert run.best_ckpt_path == tmp_path / "run" / "train" / "checkpoints" / "best.ckpt"
    assert run.last_ckpt_path == tmp_path / "run" / "train" / "checkpoints" / "last.ckpt"
    assert run.periodic_ckpt_dir == tmp_path / "run" / "train" / "checkpoints" / "periodic"
    assert run.best_eval_dir == tmp_path / "run" / "eval" / "best"
    assert run.periodic_eval_dir(0) == tmp_path / "run" / "eval" / "periodic" / "epoch_000"
    assert run.periodic_eval_dir(5) == tmp_path / "run" / "eval" / "periodic" / "epoch_005"


def test_checkpoint_epoch_is_zero_based(tmp_path: Path):
    ckpt = tmp_path / "last.ckpt"
    save_ckpt(ckpt, epoch=4)

    assert RunIO.checkpoint_epoch(ckpt) == 4


def test_completed_epoch_count_reads_last_checkpoint(tmp_path: Path):
    run = RunIO(tmp_path / "run")
    save_ckpt(run.last_ckpt_path, epoch=9)

    assert run.last_checkpoint() == run.last_ckpt_path
    assert run.last_completed_epoch() == 9
    assert run.completed_epoch_count() == 10
    assert run.completed_epochs() == 10


def test_periodic_checkpoints_sorted_by_checkpoint_epoch(tmp_path: Path):
    run = RunIO(tmp_path / "run")
    save_ckpt(run.periodic_ckpt_dir / "epoch099.ckpt", epoch=99)
    save_ckpt(run.periodic_ckpt_dir / "epoch004.ckpt", epoch=4)
    save_ckpt(run.periodic_ckpt_dir / "epoch024.ckpt", epoch=24)

    assert run.periodic_checkpoints() == [
        run.periodic_ckpt_dir / "epoch004.ckpt",
        run.periodic_ckpt_dir / "epoch024.ckpt",
        run.periodic_ckpt_dir / "epoch099.ckpt",
    ]


def test_best_eval_is_current_accepts_matching_metadata(tmp_path: Path):
    run = RunIO(tmp_path / "run")
    save_ckpt(run.best_ckpt_path, epoch=6)

    run.best_eval_dir.mkdir(parents=True)
    (run.best_eval_dir / "ensemble.csv").write_text("x\n", encoding="utf-8")
    (run.best_eval_dir / "ensemble_stats.csv").write_text("x\n", encoding="utf-8")
    write_yaml(
        {
            "checkpoint_path": str(run.best_ckpt_path),
            "checkpoint_epoch": 6,
        },
        run.best_eval_dir / "ensemble_metadata.yaml",
    )

    assert run.best_eval_is_current() is True


def test_best_eval_is_current_rejects_stale_or_incomplete_metadata(tmp_path: Path):
    run = RunIO(tmp_path / "run")
    save_ckpt(run.best_ckpt_path, epoch=6)

    run.best_eval_dir.mkdir(parents=True)
    (run.best_eval_dir / "ensemble.csv").write_text("x\n", encoding="utf-8")
    (run.best_eval_dir / "ensemble_stats.csv").write_text("x\n", encoding="utf-8")
    write_yaml(
        {
            "checkpoint_path": str(run.best_ckpt_path),
            "checkpoint_epoch": 7,
        },
        run.best_eval_dir / "ensemble_metadata.yaml",
    )

    assert run.best_eval_is_current() is False


def test_run_is_complete_requires_training_and_current_best_eval(tmp_path: Path):
    run = RunIO(tmp_path / "run")
    run.train_dir.mkdir(parents=True)
    run.config_path.write_text("experiment:\n  name: test\n", encoding="utf-8")
    run.metrics_path.write_text("epoch,step\n", encoding="utf-8")
    save_ckpt(run.last_ckpt_path, epoch=19)
    save_ckpt(run.best_ckpt_path, epoch=6)

    run.best_eval_dir.mkdir(parents=True)
    (run.best_eval_dir / "ensemble.csv").write_text("x\n", encoding="utf-8")
    (run.best_eval_dir / "ensemble_stats.csv").write_text("x\n", encoding="utf-8")
    write_yaml(
        {
            "checkpoint_path": str(run.best_ckpt_path),
            "checkpoint_epoch": 6,
        },
        run.best_eval_dir / "ensemble_metadata.yaml",
    )

    assert run.run_is_complete(total_epochs=20) is True
    assert run.run_is_complete(total_epochs=21) is False

def test_normalize_periodic_checkpoint_names_uses_zero_based_names(tmp_path: Path):
    run = RunIO(tmp_path / "run")
    source_path = run.periodic_ckpt_dir / "lightning-name.ckpt"
    save_ckpt(source_path, epoch=4)

    run.normalize_periodic_checkpoint_names()

    assert not source_path.exists()
    assert (run.periodic_ckpt_dir / "epoch004.ckpt").exists()


def test_normalize_periodic_checkpoint_names_rejects_duplicate_epoch(tmp_path: Path):
    run = RunIO(tmp_path / "run")
    save_ckpt(run.periodic_ckpt_dir / "epoch004.ckpt", epoch=4)
    save_ckpt(run.periodic_ckpt_dir / "lightning-name.ckpt", epoch=4)

    with pytest.raises(RuntimeError, match="Duplicate periodic checkpoint"):
        run.normalize_periodic_checkpoint_names()


def test_run_resolver_next_attempt_dir_uses_incremental_suffix(tmp_path: Path):
    job_dir = tmp_path / "seed42"
    (job_dir / "attempt_001").mkdir(parents=True)
    (job_dir / "attempt_002").mkdir(parents=True)

    resolver = RunResolver(job_dir, resume=False)

    assert resolver.next_attempt_dir() == job_dir / "attempt_003"


def test_run_resolver_uses_job_dir_when_resuming_existing_run(tmp_path: Path):
    config = make_config(tmp_path)
    job_dir = tmp_path / "seed42"
    (job_dir / "train").mkdir(parents=True)

    root = RunResolver(job_dir, resume=True).resolve_root(config)

    assert root == job_dir
    assert config.experiment.name == "seed42"


def test_run_resolver_uses_latest_attempt_when_resuming(tmp_path: Path):
    config = make_config(tmp_path)
    job_dir = tmp_path / "seed42"
    (job_dir / "attempt_001" / "train").mkdir(parents=True)
    (job_dir / "attempt_002" / "train").mkdir(parents=True)

    root = RunResolver(job_dir, resume=True).resolve_root(config)

    assert root == job_dir / "attempt_002"
    assert config.experiment.name == "attempt_002"


def test_run_resolver_creates_attempt_when_resume_false_and_run_exists(tmp_path: Path):
    config = make_config(tmp_path)
    job_dir = tmp_path / "seed42"
    (job_dir / "train").mkdir(parents=True)

    root = RunResolver(job_dir, resume=False).resolve_root(config)

    assert root == job_dir / "attempt_001"
    assert config.experiment.name == "attempt_001"


def test_run_resolver_build_state_uses_requested_config_for_new_run(tmp_path: Path):
    config = make_config(tmp_path, max_epochs=10)
    job_dir = tmp_path / "seed42"

    state = RunResolver(job_dir, resume=True).build_state(config)

    assert state.run.root == job_dir
    assert state.config.training.max_epochs == 10 
    assert state.done_epochs == 0
    assert state.checkpoint_path is None
    assert state.should_train is True


def test_run_resolver_build_state_reads_last_checkpoint(tmp_path: Path):
    config = make_config(tmp_path, max_epochs=10)
    job_dir = tmp_path / "seed42"
    run = RunIO(job_dir)
    run.train_dir.mkdir(parents=True)
    write_yaml(config.to_dict(), run.config_path)
    save_ckpt(run.last_ckpt_path, epoch=3)

    state = RunResolver(job_dir, resume=True).build_state(config)

    assert state.run.root == job_dir
    assert state.done_epochs == 4
    assert state.checkpoint_path == run.last_ckpt_path
    assert state.should_train is True


def test_run_resolver_build_state_skips_train_when_complete(tmp_path: Path):
    config = make_config(tmp_path, max_epochs=4)
    job_dir = tmp_path / "seed42"
    run = RunIO(job_dir)
    run.train_dir.mkdir(parents=True)
    write_yaml(config.to_dict(), run.config_path)
    save_ckpt(run.last_ckpt_path, epoch=3)

    state = RunResolver(job_dir, resume=True).build_state(config)

    assert state.done_epochs == 4
    assert state.should_train is False


def test_run_resolver_build_state_keeps_requested_epoch_over_saved_config(tmp_path: Path):
    requested_config = make_config(tmp_path, max_epochs=20)
    saved_config = make_config(tmp_path, max_epochs=5)
    job_dir = tmp_path / "seed42"
    run = RunIO(job_dir)
    run.train_dir.mkdir(parents=True)
    write_yaml(saved_config.to_dict(), run.config_path)

    state = RunResolver(job_dir, resume=True).build_state(requested_config)

    assert state.config.training.max_epochs == 20
    assert state.total_epochs == 20
