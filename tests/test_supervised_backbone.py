from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from src.run_io import RunIO
from supervised.backbone_experiment import (
    BackboneSupervisedModule,
    periodic_eval_complete,
    prepare_backbone_run,
    write_periodic_eval_result,
)


def make_config():
    return {
        "experiment": {"name": "test", "note": "", "seed": 42},
        "data": {"base_path": "dummy.csv", "partition_path": "dummy.csv", "fold": 0},
        "model": {
            "in_channels": 16,
            "out_channels": 2,
            "base_dim": 8,
        },
        "training": {
            "max_epochs": 1,
            "batch_size": 2,
            "lr": 1e-3,
            "num_workers": 0,
            "accelerator": "cpu",
            "devices": 1,
            "precision": 32,
            "accumulate_grad_batches": 1,
        },
        "logging": {
            "save_dir": "supervised/logs/test",
            "log_every_n_steps": 1,
        },
    }


def make_batch():
    contacts = torch.randint(0, 2, (2, 4, 4))
    contact_oh = torch.nn.functional.one_hot(contacts, num_classes=2).permute(0, 3, 1, 2).float()
    return {
        "conditioning": torch.randn(2, 16, 4, 4),
        "contact_oh": contact_oh,
        "mask": torch.ones(2, 1, 4, 4),
        "length": [4, 4],
    }


def test_backbone_train_step():
    module = BackboneSupervisedModule(make_config())
    batch = make_batch()
    loss = module.training_step(batch, 0)
    assert loss.ndim == 0


def test_backbone_validation_step():
    module = BackboneSupervisedModule(make_config())
    batch = make_batch()
    outputs = module.validation_step(batch, 0)
    assert "val_loss" in outputs
    assert "val_f1" in outputs


def test_backbone_test_step():
    module = BackboneSupervisedModule(make_config())
    batch = make_batch()
    outputs = module.test_step(batch, 0)
    assert "test_loss" in outputs
    assert "test_f1" in outputs


def test_prepare_backbone_run_uses_runio_train_layout(tmp_path: Path):
    run = RunIO(tmp_path / "backbone_run")

    prepared, run_root, _logger = prepare_backbone_run(make_config(), run.root)

    assert run_root == run.root
    assert run.config_path.exists()
    assert run.log_path.exists()
    assert prepared["logging"]["experiment_dir"] == str(run.train_dir)
    assert prepared["logging"]["checkpoint_dir"] == str(run.checkpoint_dir)
    assert prepared["logging"]["metrics_path"] == str(run.metrics_path)
    assert prepared["logging"]["train_log_path"] == str(run.log_path)


def test_write_periodic_eval_result_uses_zero_based_runio_eval_dir(tmp_path: Path):
    run = RunIO(tmp_path / "backbone_run")
    checkpoint = run.last_ckpt_path
    checkpoint.parent.mkdir(parents=True)
    torch.save({"epoch": 4}, checkpoint)

    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": 4,
        "trained_epoch_count": 5,
        "test_loss": 0.1,
        "test_f1": 0.2,
        "test_f1_std": 0.0,
    }

    target_dir = write_periodic_eval_result(run.root, checkpoint, summary)

    assert target_dir == run.periodic_eval_dir(4)
    assert (target_dir / "last.ckpt").exists()
    assert (target_dir / "test_summary.csv").exists()
    assert periodic_eval_complete(run.root, trained_epoch_count=5) is True
