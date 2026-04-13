import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from supervised.backbone_experiment import BackboneSupervisedModule


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
            "tensorboard": False,
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
