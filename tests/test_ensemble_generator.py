from pathlib import Path

import ensemble_generator
from src.run_io import RunIO


def test_ensemble_generator_uses_runio_best_samples_dir(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run = RunIO(run_dir)
    config = {"experiment": {"name": "test"}}
    calls = []

    monkeypatch.setattr(ensemble_generator, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ensemble_generator, "RUN_DIR", Path("run"))
    monkeypatch.setattr(ensemble_generator, "CHECKPOINT_PATH", None)
    monkeypatch.setattr(ensemble_generator, "SAMPLES_DIR", None)
    monkeypatch.setattr(ensemble_generator, "load_config", lambda path: config)
    monkeypatch.setattr(
        ensemble_generator,
        "generate_ensemble_samples",
        lambda **kwargs: calls.append(kwargs),
    )

    ensemble_generator.main()

    assert calls[0]["config"] == config
    assert calls[0]["checkpoint"] == run.best_ckpt_path
    assert calls[0]["samples_dir"] == run.best_eval_dir / "samples"
