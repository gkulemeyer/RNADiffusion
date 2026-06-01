from pathlib import Path

import torch

from src.io import write_yaml
from src.run_io import RunIO
from src.run_loop import evaluate_periodic_checkpoints, evaluate_run


def save_ckpt(path: Path, epoch: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch}, path)


def test_evaluate_best_checkpoint_skips_when_metadata_is_current(tmp_path, monkeypatch):
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

    calls = []
    monkeypatch.setattr("src.run_loop.evaluate_checkpoint", lambda *args, **kwargs: calls.append((args, kwargs)))

    evaluate_run(run, logger=object(), include_periodic=False)

    assert calls == []


def test_evaluate_best_checkpoint_runs_when_metadata_is_stale(tmp_path, monkeypatch):
    run = RunIO(tmp_path / "run")
    save_ckpt(run.best_ckpt_path, epoch=6)
    write_yaml({"experiment": {"name": "test"}}, run.config_path)
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

    calls = []
    logger = object()

    def fake_evaluate(target, checkpoint="best", **kwargs):
        calls.append(
            {
                "target": target,
                "checkpoint": checkpoint,
                "logger": kwargs.get("logger"),
            }
        )

    monkeypatch.setattr("src.run_loop.evaluate_checkpoint", fake_evaluate)

    evaluate_run(run, logger=logger, include_periodic=False)

    assert calls == [
        {
            "target": run.root,
            "checkpoint": "best",
            "logger": logger,
        }
    ]


def test_evaluate_best_checkpoint_returns_when_best_checkpoint_missing(tmp_path, monkeypatch):
    run = RunIO(tmp_path / "run")
    calls = []
    monkeypatch.setattr("src.run_loop.evaluate_checkpoint", lambda *args, **kwargs: calls.append((args, kwargs)))

    evaluate_run(run, logger=object(), include_periodic=False)

    assert calls == []


def test_evaluate_periodic_checkpoints_uses_zero_based_eval_dir(tmp_path, monkeypatch):
    run = RunIO(tmp_path / "run")
    checkpoint = run.periodic_ckpt_dir / "epoch000.ckpt"
    save_ckpt(checkpoint, epoch=0)
    write_yaml({"experiment": {"name": "test"}}, run.config_path)

    calls = []

    def fake_evaluate(target, checkpoint="best", output_dir=None, keep_samples=True, logger=None):
        calls.append(
            {
                "target": target,
                "eval_dir": output_dir,
                "checkpoint": checkpoint,
                "keep_samples": keep_samples,
            }
        )

    monkeypatch.setattr("src.run_loop.evaluate_checkpoint", fake_evaluate)

    evaluate_periodic_checkpoints(run, logger=None)

    assert calls == [
        {
            "target": run.root,
            "eval_dir": run.periodic_eval_dir(0),
            "checkpoint": checkpoint,
            "keep_samples": False,
        }
    ]
