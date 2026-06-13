import lightning as L
import torch as tr
from lightning.pytorch.callbacks import Callback

from src.metrics import contact_f1
from src.model import build_model


class RNADiffusionModule(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = build_model(config)
        self.learning_rate = config["training"]["lr"]
        self.save_hyperparameters(config) 
        
        if config["model"]["load_pretrained"]:
            pretrained_path = config["model"]["pretrained_path"]
            print(f"[INFO] loading pretrained model from {pretrained_path}")
            checkpoint = tr.load(pretrained_path, map_location="cpu")
            if "state_dict" not in checkpoint:
                raise ValueError(f"Expected a Lightning .ckpt file, got: {pretrained_path}")
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)

    def _compute_loss(self, batch):
        return self.model._train_loss(
            batch["contact_oh"],
            batch["conditioning"],
            lengths=batch["length"],
        )

    def _evaluate_batch(self, batch):
        loss = self._compute_loss(batch)
        predictions = self.model.sample(batch["conditioning"], lengths=batch["length"])
        f1_score = contact_f1(
            predictions,
            batch["contact_oh"],
            lengths=batch["length"],
            reduce=True,
        )
        return loss, f1_score

    def training_step(self, batch, _batch_idx):
        loss = self._compute_loss(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _batch_idx):
        loss, f1_score = self._evaluate_batch(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", f1_score, on_step=False, on_epoch=True, prog_bar=True)
        return {"val_loss": loss, "val_f1": f1_score}

    def test_step(self, batch, _batch_idx):
        loss, f1_score = self._evaluate_batch(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_f1", f1_score, on_step=False, on_epoch=True)
        return {"test_loss": loss, "test_f1": f1_score}

    def configure_optimizers(self):
        return tr.optim.Adam(self.model.parameters(), lr=self.learning_rate)


class TestMetricsCollector(Callback):
    def __init__(self):
        self.losses = []
        self.f1_scores = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not outputs:
            return
        self.losses.append(float(outputs["test_loss"].detach().cpu()))
        self.f1_scores.append(float(outputs["test_f1"].detach().cpu()))


def load_rna_module_checkpoint(config, checkpoint_path, eval_mode=True):
    checkpoint = tr.load(checkpoint_path, map_location="cpu")
    if "state_dict" not in checkpoint:
        raise ValueError(f"Expected a Lightning .ckpt file, got: {checkpoint_path}")

    module = RNADiffusionModule(config)
    module.load_state_dict(checkpoint["state_dict"])
    if eval_mode:
        module.eval()
    return module
