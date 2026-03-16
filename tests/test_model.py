import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from src.model import build_model
from src.train_module import RNADiffusionModule


def make_config():
    return {
        "experiment": {"name": "test", "note": "", "seed": 42},
        "data": {"base_path": "dummy.csv", "partition_path": "dummy.csv", "fold": 0},
        "model": {
            "timesteps": 2,
            "num_classes": 2,
            "in_channels": 18,
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
        },
        "logging": {
            "save_dir": "logs/test",
            "csv": True,
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


def test_build_model_forward_all_timesteps():
    model = build_model(make_config())
    batch = make_batch()
    loss = model.forward_all_timesteps(
        batch["contact_oh"],
        batch["conditioning"],
        mask=batch["mask"],
    )
    assert loss.ndim == 0
    sample = model.sample(batch["conditioning"])
    assert sample.shape == (2, 4, 4)


def test_train_module_validation_step():
    module = RNADiffusionModule(make_config())
    batch = make_batch()
    outputs = module.validation_step(batch, 0)
    assert "val_loss" in outputs
    assert "val_f1" in outputs


def test_train_module_test_step():
    module = RNADiffusionModule(make_config())
    batch = make_batch()
    outputs = module.test_step(batch, 0)
    assert "test_loss" in outputs
    assert "test_f1" in outputs
