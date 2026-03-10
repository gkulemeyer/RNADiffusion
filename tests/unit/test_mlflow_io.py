from src.utils.io import save_config
from src.utils.mlflow_io import (
    iter_run_info,
    load_run_info,
    log_summary_metrics,
    prepare_local_tracking_dir,
    resolve_run_dataset,
    resolve_tracking_dir,
)


def test_mlflow_io_paths(tmp_path):
    tracking_dir = resolve_tracking_dir(f"file:{tmp_path}")
    exp_dir = tracking_dir / "0"
    run_dir = exp_dir / "run1"
    artifacts = run_dir / "artifacts"
    ckpt_dir = artifacts / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (artifacts / "config.json").write_text("{}")
    (ckpt_dir / "best_model.pt").write_text("x")

    run_infos = list(iter_run_info(tracking_dir))
    run_infos_with_ckpt = list(iter_run_info(tracking_dir, checkpoints_only=True))
    assert [run_info["run_id"] for run_info in run_infos] == ["run1"]
    assert [run_info["run_id"] for run_info in run_infos_with_ckpt] == ["run1"]
    run_info = load_run_info(tracking_dir, "run1")
    assert run_info["run_id"] == "run1"
    assert run_info["run_dir"] == run_dir
    assert run_info["checkpoint_path"] == ckpt_dir / "best_model.pt"
    assert run_info["config"] is not None
    assert run_info["samples_dir"] is None
    assert run_info["tags"] == {"exp_name": None, "run_name": None}


def test_resolve_run_dataset_from_saved_config(tmp_path):
    tracking_dir = tmp_path / "mlruns"
    run_dir = tracking_dir / "0" / "run1" / "artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(
        {
            "partitioned": True,
            "main_path": "/tmp/main.csv",
            "partition_path": "/tmp/parts.csv",
            "test_partition": "heldout",
            "fold_number": 2,
            "min_len": 8,
            "max_len": 128,
        },
        run_dir,
    )

    dataset_cfg = resolve_run_dataset(
        tracking_dir=tracking_dir,
        run_id="run1",
        default_dataset_cfg={"dataset_path": "/tmp/fallback.csv", "for_prediction": True, "min_len": 0, "max_len": 256},
    )

    assert dataset_cfg == {
        "dataset_path": None,
        "partitioned": True,
        "main_path": "/tmp/main.csv",
        "partition_path": "/tmp/parts.csv",
        "partition_value": "heldout",
        "fold_number": 2,
        "min_len": 8,
        "max_len": 128,
        "for_prediction": True,
    }


def test_resolve_run_dataset_from_data_section(tmp_path):
    tracking_dir = tmp_path / "mlruns"
    run_dir = tracking_dir / "0" / "run1" / "artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(
        {
            "data": {
                "split": "sim80",
                "use_partitions": True,
                "main_path": "/tmp/main.csv",
                "partition_file_template": "/tmp/${split}_parts.csv",
                "test_partition": "heldout",
                "max_len": 128,
            },
            "fold_number": 2,
        },
        run_dir,
    )

    dataset_cfg = resolve_run_dataset(
        tracking_dir=tracking_dir,
        run_id="run1",
        default_dataset_cfg={"dataset_path": "/tmp/fallback.csv", "for_prediction": True, "min_len": 0, "max_len": 256},
    )

    assert dataset_cfg == {
        "dataset_path": None,
        "for_prediction": True,
        "partitioned": True,
        "main_path": "/tmp/main.csv",
        "partition_path": "/tmp/sim80_parts.csv",
        "partition_value": "heldout",
        "fold_number": 2,
        "min_len": 0,
        "max_len": 128,
    }


def test_mlflow_helpers_are_safe_without_run(tmp_path):
    tracking_dir = tmp_path / "mlruns"
    assert load_run_info(tracking_dir, "missing") is None
    assert resolve_run_dataset(tracking_dir, "missing", {"dataset_path": "/tmp/test.csv"}) == {"dataset_path": "/tmp/test.csv"}


def test_prepare_local_tracking_dir_restores_default_experiment_meta(tmp_path):
    tracking_dir = tmp_path / "mlruns"
    (tracking_dir / "0").mkdir(parents=True, exist_ok=True)

    prepare_local_tracking_dir(f"file:{tracking_dir}")

    meta_path = tracking_dir / "0" / "meta.yaml"
    assert meta_path.exists()
    meta_text = meta_path.read_text()
    assert "experiment_id: '0'" in meta_text
    assert "name: Default" in meta_text


class _Summary:
    def __init__(self):
        self.iloc = [self]

    def to_dict(self):
        return {"run_id": "a", "score": 1.0}


def test_log_summary_metrics_noop_on_missing_run(tmp_path):
    log_summary_metrics(tmp_path / "mlruns", "missing", _Summary())
