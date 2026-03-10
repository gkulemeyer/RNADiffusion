import torch as tr
import pytorch_lightning as pl

from ..models.factory import create_model
from ..evaluation.metrics import contact_f1



class RNAContactLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.model = create_model(config, eval_mode=False, move_to_device=False)

    def training_step(self, batch, batch_idx):
        cond = batch["outer"]
        target = batch["contact_oh"]
        mask = batch["mask"]

        loss = self.model.forward_all_timesteps(target, cond, mask=mask)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        cond = batch["outer"]
        target = batch["contact_oh"]
        mask = batch["mask"]
        lens = batch["length"]

        val_loss = self.model.forward_all_timesteps(target, cond, mask=mask)

        samples = self.model._sample(cond)
        f1_score = contact_f1(samples, target, lengths=lens, reduce=True)

        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", f1_score, on_step=False, on_epoch=True, prog_bar=True)

        return {"val_loss": val_loss, "val_f1": f1_score}

    def configure_optimizers(self):
        training_cfg = self.config.get("training", {}) or {}
        optimizer_cfg = training_cfg.get("optimizer", self.config.get("optimizer", {})) or {}
        optimizer_type = str(optimizer_cfg.get("type", "adam")).lower()
        weight_decay = float(optimizer_cfg.get("weight_decay", 0.0))
        lr = training_cfg["lr"] if "lr" in training_cfg else self.config["lr"]

        if optimizer_type == "adam":
            return tr.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        if optimizer_type == "adamw":
            return tr.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
