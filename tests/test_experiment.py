from pathlib import Path

import pandas as pd

from src.experiment import (
    evaluate_checkpoint,
    evaluate_checkpoint_ensemble,
    evaluate_ensemble_samples,
    generate_ensemble_samples,
    prepare_run,
)
from src.io import handle_metrics
from src.run_io import RunIO


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


def test_prepare_run_uses_explicit_run_without_derived_config_paths(tmp_path: Path):
    config = make_config(tmp_path)
    run = RunIO(tmp_path / "manual_run")

    prepare_run(config, run)

    assert run.config_path.exists()
    assert run.log_path.exists()
    assert "experiment_dir" not in config["logging"]
    assert "metrics_path" not in config["logging"]


def test_evaluate_checkpoint_uses_explicit_inputs_and_local_batch_copy(
    tmp_path: Path,
    monkeypatch,
):
    config = make_config(tmp_path)
    config["training"]["batch_size"] = 8
    checkpoint = tmp_path / "model.ckpt"
    output_dir = tmp_path / "evaluation"
    calls = []

    def fake_evaluate(received, checkpoint, output_dir, logger, keep_samples=False):
        calls.append(
            (received["training"]["batch_size"], checkpoint, output_dir, keep_samples)
        )

    monkeypatch.setattr("src.experiment.evaluate_checkpoint_ensemble", fake_evaluate)

    result = evaluate_checkpoint(
        config,
        checkpoint,
        output_dir,
        logger=object(),
        keep_samples=True,
        batch_size1=True,
    )

    assert result == output_dir
    assert calls == [(1, checkpoint, output_dir, True)]
    assert config["training"]["batch_size"] == 8


def test_generate_ensemble_samples_builds_seeds_once_and_passes_threshold(
    tmp_path: Path,
    monkeypatch,
):
    calls = {}
    model = object()
    loader = object()
    config = {
        "ensemble": {"num_samples": 3, "base_seed": 42, "threshold": 0.6},
    }
    monkeypatch.setattr("src.experiment.load_model_checkpoint", lambda *args, **kwargs: model)
    monkeypatch.setattr("src.experiment.build_dataloader", lambda *args, **kwargs: loader)
    monkeypatch.setattr("src.experiment.RunIO.checkpoint_epoch", lambda checkpoint: 7)

    def fake_save(**kwargs):
        calls["save"] = kwargs

    def fake_metadata(**kwargs):
        calls["metadata"] = kwargs

    monkeypatch.setattr("src.experiment.save_ensemble_samples", fake_save)
    monkeypatch.setattr("src.experiment.write_samples_metadata", fake_metadata)

    samples_dir = generate_ensemble_samples(
        config,
        checkpoint=tmp_path / "best.ckpt",
        samples_dir=tmp_path / "samples",
    )

    assert samples_dir == tmp_path / "samples"
    assert calls["save"]["sample_seeds"] == [42, 43, 44]
    assert calls["save"]["threshold"] == 0.6
    assert calls["metadata"]["sample_seeds"] == [42, 43, 44]
    assert calls["metadata"]["threshold"] == 0.6
    assert "chunk_size" not in calls["save"]
    assert "chunk_size" not in calls["metadata"]


def test_evaluate_ensemble_samples_uses_explicit_default_names(
    tmp_path: Path,
    monkeypatch,
):
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    monkeypatch.setattr(
        "src.experiment.evaluate_samples_dir",
        lambda **kwargs: pd.DataFrame(
            [{"seq_id": "a", "cons_k1_mean": 1.0, "cons_k1_std": 0.0}]
        ),
    )
    monkeypatch.setattr("src.experiment.write_ensemble_metadata", lambda **kwargs: {})

    metrics_path, stats_path = evaluate_ensemble_samples(
        samples_dir,
        trials=1,
        consensus_sizes=[1],
        seed=42,
        get_best_and_worst=False,
        sample_type="processed",
    )

    assert metrics_path == tmp_path / "processed_ensemble_metrics.csv"
    assert stats_path == tmp_path / "processed_ensemble_stats.csv"
    assert metrics_path.exists()
    assert stats_path.exists()


def test_evaluate_checkpoint_generates_once_evaluates_both_and_exports(
    tmp_path: Path,
    monkeypatch,
):
    output_dir = tmp_path / "evaluation"
    samples_dir = output_dir / "samples"
    calls = {"generate": [], "evaluate": [], "export": []}
    config = {
        "ensemble": {
            "num_samples": 2,
            "base_seed": 42,
            "threshold": 0.5,
            "trials": 1,
            "consensus_sizes": [1],
            "get_best_and_worst": False,
        },
    }

    def fake_generate(**kwargs):
        calls["generate"].append(kwargs)
        samples_dir.mkdir(parents=True, exist_ok=True)
        return samples_dir

    monkeypatch.setattr("src.experiment.generate_ensemble_samples", fake_generate)
    monkeypatch.setattr(
        "src.experiment.evaluate_ensemble_samples",
        lambda **kwargs: calls["evaluate"].append(kwargs),
    )
    monkeypatch.setattr(
        "src.experiment.export_db_ensemble",
        lambda **kwargs: calls["export"].append(kwargs),
    )
    logger = type("Logger", (), {"info": lambda *args, **kwargs: None})()

    evaluate_checkpoint_ensemble(
        config,
        checkpoint=tmp_path / "best.ckpt",
        output_dir=output_dir,
        logger=logger,
        keep_samples=True,
    )

    assert len(calls["generate"]) == 1
    assert [call["sample_type"] for call in calls["evaluate"]] == [
        "raw",
        "processed",
    ]
    assert len(calls["export"]) == 1
    assert calls["export"][0]["samples_dir"] == samples_dir
    assert calls["export"][0]["output_csv"] == output_dir / "generated_ensemble.csv"
    assert [call["output_path"] for call in calls["evaluate"]] == [
        output_dir / "raw_ensemble_metrics.csv",
        output_dir / "processed_ensemble_metrics.csv",
    ]


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

    handle_metrics([make_csv_logger(lightning_dir)], metrics_path, resume=False)

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

    handle_metrics([make_csv_logger(lightning_dir)], metrics_path, resume=True)

    assert metrics_path.read_text(encoding="utf-8") == "\n".join(
        [
            "epoch,step,train_loss,val_f1",
            "0,1,1.0,0.8",
            "1,2,0.7,0.9",
        ]
    ) + "\n"
