import json

from src.utils.reporting import build_run_metric_summaries, build_run_metrics_dataframe, get_run_metadata

from src.utils.reporting import write_experiment_summary_csvs


def test_write_experiment_summary_csvs(tmp_path):
    experiment_dir = tmp_path / "0"
    run_dir = experiment_dir / "run1"
    (run_dir / "params").mkdir(parents=True)
    (run_dir / "tags").mkdir(parents=True)
    (run_dir / "metrics").mkdir(parents=True)
    (run_dir / "artifacts" / "testing").mkdir(parents=True)

    (run_dir / "meta.yaml").write_text("run_name: demo-run\n")
    (run_dir / "params" / "epochs").write_text("3\n")
    (run_dir / "tags" / "partition_file").write_text("sim80_split.csv\n")
    (run_dir / "metrics" / "val_f1").write_text("0 0.3 0\n1 0.7 1\n")
    (run_dir / "artifacts" / "testing" / "test_metrics.json").write_text(json.dumps({"test_f1": 0.9}))

    summary = write_experiment_summary_csvs(experiment_dir, include_ids=True)

    assert summary["params_rows"] == 1
    assert summary["metrics_long_rows"] == 3
    assert summary["metrics_summary_rows"] == 1
    assert summary["params_path"].exists()
    assert summary["metrics_long_path"].exists()
    assert summary["metrics_summary_path"].exists()


def test_build_run_metrics_dataframe_and_summaries(tmp_path):
    run_dir = tmp_path / "0" / "run1"
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    (run_dir / "meta.yaml").write_text("run_name: demo-run\n")
    (metrics_dir / "epoch").write_text("0 0 0\n1 1 1\n")
    (metrics_dir / "train_loss").write_text("0 0.8 0\n1 0.3 1\n")
    (metrics_dir / "val_loss").write_text("0 0.7 0\n1 0.2 1\n")
    (metrics_dir / "val_f1").write_text("0 0.4 0\n1 0.9 1\n")

    run_info = {
        "run_id": "run1",
        "run_dir": run_dir,
        "config": {"network": {"wrapper": {"timesteps": 5}}, "training": {"epochs": 2}},
        "tags": {"exp_name": "exp-demo", "run_name": "demo-run"},
    }

    df = build_run_metrics_dataframe(run_dir)
    metadata = get_run_metadata(run_info)
    last_summary, best_summary = build_run_metric_summaries(run_info, df)

    assert list(df.columns) == ["epoch", "step", "train_loss", "val_loss", "val_f1"]
    assert metadata["timesteps"] == 5
    assert metadata["epochs"] == 2
    assert last_summary["final_val_f1"] == 0.9
    assert best_summary["best_val_loss"] == 0.2
