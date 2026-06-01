from pathlib import Path

from src.experiment import evaluate_checkpoint, prepare_run
from src.io import handle_metrics
from src.io import write_yaml
from src.run_io import RunIO
import torch


def make_csv_logger(log_dir: Path):
    csv_logger_cls = type("CSVLogger", (), {})
    logger = csv_logger_cls()
    logger.log_dir = str(log_dir)
    return logger


def make_config(tmp_path: Path):
    return {
        "experiment": {"name": "seed42", "seed": 42},
        "logging": {
            "save_dir": str(tmp_path / "logs"),
            "log_every_n_steps": 1,
            "checkpoint_every_n_epochs": 1,
        },
        "training": {
            "max_epochs": 1,
            "check_val_every_n_epoch": 1,
        },
    }


def save_ckpt(path: Path, epoch: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch}, path)


def test_prepare_run_without_experiment_dir_uses_train_layout(tmp_path: Path):
    config = make_config(tmp_path)

    prepared, experiment_dir, _logger = prepare_run(config)

    expected_train_dir = tmp_path / "logs" / "seed42" / "train"
    assert experiment_dir == expected_train_dir
    assert (expected_train_dir / "config.yaml").exists()
    assert (expected_train_dir / "run.log").exists()
    assert prepared["logging"]["experiment_dir"] == str(expected_train_dir)
    assert prepared["logging"]["checkpoint_dir"] == str(expected_train_dir / "checkpoints")
    assert prepared["logging"]["metrics_path"] == str(expected_train_dir / "metrics.csv")


def test_prepare_run_with_explicit_train_dir_preserves_runio_layout(tmp_path: Path):
    config = make_config(tmp_path)
    run = RunIO(tmp_path / "manual_run")

    prepared, experiment_dir, _logger = prepare_run(config, experiment_dir=run.train_dir)

    assert experiment_dir == run.train_dir
    assert run.config_path.exists()
    assert run.log_path.exists()
    assert prepared["logging"]["experiment_dir"] == str(run.train_dir)
    assert prepared["logging"]["checkpoint_dir"] == str(run.checkpoint_dir)
    assert prepared["logging"]["train_log_path"] == str(run.log_path)


def test_evaluate_checkpoint_uses_best_checkpoint_by_default(tmp_path: Path, monkeypatch):
    run = RunIO(tmp_path / "run")
    write_yaml(make_config(tmp_path), run.config_path)
    save_ckpt(run.best_ckpt_path, epoch=3)
    calls = []

    def fake_evaluate(config, checkpoint, logger, keep_samples=False):
        calls.append((Path(config["logging"]["experiment_dir"]), checkpoint, keep_samples))

    monkeypatch.setattr("src.experiment.evaluate_checkpoint_ensemble", fake_evaluate)

    output_dir = evaluate_checkpoint(run.root, logger=object())

    assert output_dir == run.best_eval_dir
    assert calls == [(run.best_eval_dir, run.best_ckpt_path, False)]


def test_evaluate_checkpoint_can_use_last_checkpoint(tmp_path: Path, monkeypatch):
    run = RunIO(tmp_path / "run")
    write_yaml(make_config(tmp_path), run.config_path)
    save_ckpt(run.last_ckpt_path, epoch=4)
    calls = []
    monkeypatch.setattr(
        "src.experiment.evaluate_checkpoint_ensemble",
        lambda config, checkpoint, logger, keep_samples=False: calls.append(
            (Path(config["logging"]["experiment_dir"]), checkpoint)
        ),
    )

    output_dir = evaluate_checkpoint(run.root, checkpoint="last", logger=object())

    assert output_dir == run.root / "eval" / "last"
    assert calls == [(run.root / "eval" / "last", run.last_ckpt_path)]


def test_evaluate_checkpoint_accepts_explicit_checkpoint_path(tmp_path: Path, monkeypatch):
    run = RunIO(tmp_path / "run")
    write_yaml(make_config(tmp_path), run.config_path)
    checkpoint = run.periodic_ckpt_dir / "epoch005.ckpt"
    save_ckpt(checkpoint, epoch=5)
    calls = []
    monkeypatch.setattr(
        "src.experiment.evaluate_checkpoint_ensemble",
        lambda config, checkpoint, logger, keep_samples=False: calls.append(
            (Path(config["logging"]["experiment_dir"]), checkpoint, keep_samples)
        ),
    )

    output_dir = evaluate_checkpoint(checkpoint, logger=object(), keep_samples=True)

    assert output_dir == run.periodic_eval_dir(5)
    assert calls == [(run.periodic_eval_dir(5), checkpoint, True)]


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
