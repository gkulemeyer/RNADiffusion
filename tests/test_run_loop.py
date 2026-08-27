import copy
from pathlib import Path

import pytest
import torch

from src.run_io import RunIO
from src.run_loop import evaluate_periodic_checkpoints, run_training_and_evaluation


def save_ckpt(path: Path, epoch: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch}, path)


def make_config(epochs=10, batch_size=8, evaluate=False):
    return {
        "experiment": {"name": "current"},
        "model": {"evaluate": evaluate},
        "training": {"max_epochs": epochs, "batch_size": batch_size},
    }


def test_resume_uses_last_checkpoint_and_current_config(tmp_path, monkeypatch):
    run = RunIO(tmp_path / "run")
    save_ckpt(run.last_ckpt_path, epoch=4)
    run.config_path.write_text("experiment:\n  name: old\n", encoding="utf-8")
    config = make_config(epochs=8)
    calls = {}

    monkeypatch.setattr(
        "src.run_loop.prepare_run",
        lambda received, received_run: calls.update(
            prepared=copy.deepcopy(received), run=received_run
        ) or object(),
    )
    monkeypatch.setattr(
        "src.run_loop.train",
        lambda received, received_run, logger, resume=None: calls.update(
            trained=copy.deepcopy(received), resume=resume
        ),
    )

    run_training_and_evaluation(config, run.root, resume=True)

    assert calls["prepared"]["experiment"]["name"] == "current"
    assert calls["trained"]["training"]["max_epochs"] == 8
    assert calls["resume"] == run.last_ckpt_path


def test_complete_training_is_skipped_but_best_is_evaluated(tmp_path, monkeypatch):
    run = RunIO(tmp_path / "run")
    save_ckpt(run.last_ckpt_path, epoch=9)
    config = make_config(epochs=10, evaluate=True)
    calls = []

    monkeypatch.setattr("src.run_loop.prepare_run", lambda config, run: object())
    monkeypatch.setattr(
        "src.run_loop.train",
        lambda *args, **kwargs: pytest.fail("training should be skipped"),
    )
    monkeypatch.setattr(
        "src.run_loop.evaluate_checkpoint",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    result = run_training_and_evaluation(config, run.root, resume=True)

    assert result == run.root
    assert calls == [
        ((config, run.best_ckpt_path, run.best_eval_dir), {"keep_samples": False})
    ]


def test_resume_false_trains_from_scratch_in_same_run(tmp_path, monkeypatch):
    run = RunIO(tmp_path / "run")
    save_ckpt(run.last_ckpt_path, epoch=9)
    calls = []

    monkeypatch.setattr("src.run_loop.prepare_run", lambda config, run: object())
    monkeypatch.setattr(
        "src.run_loop.train",
        lambda config, run, logger, resume=None: calls.append(resume),
    )

    run_training_and_evaluation(make_config(), run.root, resume=False)

    assert calls == [None]
    assert not list(run.root.glob("attempt_*"))


def test_training_oom_retries_once_with_batch_size_one(tmp_path, monkeypatch):
    config = make_config(batch_size=16)
    original = copy.deepcopy(config)
    calls = []

    monkeypatch.setattr("src.run_loop.prepare_run", lambda config, run: object())

    def fake_train(received, run, logger, resume=None):
        calls.append((received["training"]["batch_size"], resume))
        if len(calls) == 1:
            raise torch.cuda.OutOfMemoryError

    monkeypatch.setattr("src.run_loop.train", fake_train)

    run_training_and_evaluation(config, tmp_path / "run")

    assert calls == [(16, None), (1, None)]
    assert config == original


def test_training_second_oom_is_propagated(tmp_path, monkeypatch):
    monkeypatch.setattr("src.run_loop.prepare_run", lambda config, run: object())
    monkeypatch.setattr(
        "src.run_loop.train",
        lambda *args, **kwargs: (_ for _ in ()).throw(torch.cuda.OutOfMemoryError()),
    )

    with pytest.raises(torch.cuda.OutOfMemoryError):
        run_training_and_evaluation(make_config(), tmp_path / "run")


def test_training_oom_is_not_retried_when_disabled(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("src.run_loop.prepare_run", lambda config, run: object())

    def fail(*args, **kwargs):
        calls.append(1)
        raise torch.cuda.OutOfMemoryError

    monkeypatch.setattr("src.run_loop.train", fail)

    with pytest.raises(torch.cuda.OutOfMemoryError):
        run_training_and_evaluation(
            make_config(),
            tmp_path / "run",
            retry_on_oom=False,
        )

    assert len(calls) == 1


def test_evaluation_oom_retries_with_batch_size_one(tmp_path, monkeypatch):
    config = make_config(epochs=0, evaluate=True)
    calls = []
    monkeypatch.setattr("src.run_loop.prepare_run", lambda config, run: object())

    def fake_evaluate(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) == 1:
            raise torch.cuda.OutOfMemoryError

    monkeypatch.setattr("src.run_loop.evaluate_checkpoint", fake_evaluate)

    run_training_and_evaluation(config, tmp_path / "run")

    assert calls[0][1] == {"keep_samples": False}
    assert calls[1][1] == {"keep_samples": False, "batch_size1": True}
    assert config["training"]["batch_size"] == 8


def test_periodic_evaluation_uses_every_checkpoint_and_explicit_dirs(
    tmp_path,
    monkeypatch,
):
    run = RunIO(tmp_path / "run")
    first = run.periodic_ckpt_dir / "first.ckpt"
    second = run.periodic_ckpt_dir / "second.ckpt"
    save_ckpt(first, epoch=2)
    save_ckpt(second, epoch=5)
    calls = []

    monkeypatch.setattr(
        "src.run_loop.evaluate_checkpoint",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    config = make_config()

    evaluate_periodic_checkpoints(config, run)

    assert calls == [
        ((config, first, run.periodic_eval_dir(2)), {"keep_samples": False}),
        ((config, second, run.periodic_eval_dir(5)), {"keep_samples": False}),
    ]


def test_run_can_evaluate_periodic_and_keep_all_samples(tmp_path, monkeypatch):
    run = RunIO(tmp_path / "run")
    periodic = run.periodic_ckpt_dir / "epoch002.ckpt"
    save_ckpt(periodic, epoch=2)
    calls = []

    monkeypatch.setattr("src.run_loop.prepare_run", lambda config, run: object())
    monkeypatch.setattr(
        "src.run_loop.evaluate_checkpoint",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    config = make_config(epochs=0, evaluate=True)
    run_training_and_evaluation(
        config,
        run.root,
        evaluate_periodic=True,
        keep_samples=True,
    )

    assert calls == [
        (
            (config, periodic, run.periodic_eval_dir(2)),
            {"keep_samples": False},
        ),
        (
            (config, run.best_ckpt_path, run.best_eval_dir),
            {"keep_samples": True},
        ),
    ]
