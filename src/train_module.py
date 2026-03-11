import lightning as L
import torch as tr

from src.metrics import contact_f1
from src.model import build_model


class RNADiffusionModule(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = build_model(config)
        self.learning_rate = config["training"]["lr"]
        self.save_hyperparameters(config)

    def _compute_loss(self, batch):
        return self.model.forward_all_timesteps(
            batch["contact_one_hot"],
            batch["conditioning"],
            mask=batch["mask"],
        )

    def training_step(self, batch, _batch_idx):
        loss = self._compute_loss(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _batch_idx):
        loss = self._compute_loss(batch)
        predictions = self.model._sample(batch["conditioning"])
        f1_score = contact_f1(
            predictions,
            batch["contact_one_hot"],
            lengths=batch["length"],
            reduce=True,
        )
        f1_tensor = tr.tensor(f1_score, device=self.device)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", f1_tensor, on_step=False, on_epoch=True, prog_bar=True)
        return {"val_loss": loss, "val_f1": f1_tensor}

    def test_step(self, batch, _batch_idx):
        loss = self._compute_loss(batch)
        predictions = self.model._sample(batch["conditioning"])
        f1_score = contact_f1(
            predictions,
            batch["contact_one_hot"],
            lengths=batch["length"],
            reduce=True,
        )
        f1_tensor = tr.tensor(f1_score, device=self.device)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_f1", f1_tensor, on_step=False, on_epoch=True)
        return {"test_loss": loss, "test_f1": f1_tensor}

    def configure_optimizers(self):
        return tr.optim.Adam(self.model.parameters(), lr=self.learning_rate)
