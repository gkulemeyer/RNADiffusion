from pathlib import Path

from omegaconf import OmegaConf

from src.utils.io import make_file_tracking_uri, resolve_workspace_paths


def test_resolve_workspace_paths_from_app_root(tmp_path):
    app_root = tmp_path / "repo_root"
    app_root.mkdir()

    paths = resolve_workspace_paths(app_root=app_root)

    assert paths["workspace_root"] == app_root
    assert paths["data_root"] == app_root / "data"
    assert paths["logs_root"] == app_root / "logs"
    assert paths["mlruns_root"] == app_root / "mlruns"


def test_resolve_workspace_paths_from_paths_cfg():
    paths_cfg = OmegaConf.create(
        {
            "workspace_root": "/tmp/workspace",
            "data_root": "/tmp/workspace/data",
            "logs_root": "/tmp/workspace/logs",
            "mlruns_root": "/tmp/workspace/mlruns",
        }
    )

    paths = resolve_workspace_paths(paths_cfg=paths_cfg)

    assert paths["workspace_root"] == Path("/tmp/workspace")
    assert paths["data_root"] == Path("/tmp/workspace/data")
    assert paths["logs_root"] == Path("/tmp/workspace/logs")
    assert paths["mlruns_root"] == Path("/tmp/workspace/mlruns")
    assert make_file_tracking_uri(paths["mlruns_root"]) == "file:/tmp/workspace/mlruns"
