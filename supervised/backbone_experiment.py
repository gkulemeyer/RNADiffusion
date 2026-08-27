from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import lightning as L
import torch as tr
import torch.nn.functional as F
from lightning.pytorch import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from tqdm import tqdm

from src.config import build_experiment_dir, prepare_experiment_config, save_config
from src.data import RNADataModule, build_dataloader
from src.io import build_loggers, configure_logger, handle_metrics
from src.metrics import contact_f1
from src.run_io import RunIO
from supervised.supervised_simpleunet import BackboneSimpleUNet

def build_backbone_model(config):
    model_config = config["model"]
    return BackboneSimpleUNet(
        in_channels=model_config.get("in_channels", 16),
        out_channels=model_config.get("out_channels", 2),
        base_dim=model_config["base_dim"],
    )


def masked_cross_entropy_loss(logits, target_oh, mask):
    targets = target_oh.argmax(dim=1)
    loss_map = F.cross_entropy(logits, targets, reduction="none")
    valid_mask = mask.squeeze(1).bool()
    return loss_map[valid_mask].mean()


class BackboneSupervisedModule(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = build_backbone_model(config)
        self.learning_rate = config["training"]["lr"]
        self.save_hyperparameters(config)

    def _forward_batch(self, batch):
        conditioning = batch["conditioning"]
        return self.model(conditioning)

    def _compute_loss(self, batch):
        logits = self._forward_batch(batch)
        loss = masked_cross_entropy_loss(logits, batch["contact_oh"], batch["mask"])
        return logits, loss

    def _evaluate_batch(self, batch):
        logits, loss = self._compute_loss(batch)
        f1_score = contact_f1(
            logits,
            batch["contact_oh"],
            lengths=batch["length"],
            reduce=True,
        )
        return logits, loss, f1_score

    def training_step(self, batch, _batch_idx):
        _, loss = self._compute_loss(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _batch_idx):
        _, loss, f1_score = self._evaluate_batch(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", f1_score, on_step=False, on_epoch=True, prog_bar=True)
        return {"val_loss": loss, "val_f1": f1_score}

    def test_step(self, batch, _batch_idx):
        _, loss, f1_score = self._evaluate_batch(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_f1", f1_score, on_step=False, on_epoch=True)
        return {"test_loss": loss, "test_f1": f1_score}

    def configure_optimizers(self):
        return tr.optim.Adam(self.model.parameters(), lr=self.learning_rate)


def prepare_backbone_run(config, run_root=None):
    config = prepare_experiment_config(config)

    if run_root is None:
        run_root = build_experiment_dir(config)
    else:
        run_root = Path(run_root)

    run = RunIO(run_root)
    config = prepare_experiment_config(config, run.train_dir)
    run.train_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run.train_dir)


    run_logger = configure_logger(
        config["logging"]["train_log_path"],
        logger_name="backbone_supervised",
    )
    seed_everything(config["experiment"]["seed"], workers=True)
    run_logger.info("Starting supervised backbone run in %s", run.train_dir)
    run_logger.info("Resolved config: %s", json.dumps(config, indent=2))
    return config, run.root, run_logger


def train_backbone(config, run_root, logger, resume=None):
    log_cfg = config["logging"]
    train_cfg = config["training"]

    run = RunIO(run_root)
    data = RNADataModule(config)
    model = BackboneSupervisedModule(config)
    loggers = build_loggers(config, run.train_dir)

    ckpt_cb = ModelCheckpoint(
        dirpath=run.checkpoint_dir,
        filename="best",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        save_last=True,
        every_n_epochs=max(1, int(log_cfg.get("checkpoint_every_n_epochs", 1))),
        auto_insert_metric_name=False,
    )

    trainer = Trainer(
        max_epochs=train_cfg["max_epochs"],
        accelerator=train_cfg["accelerator"],
        devices=train_cfg["devices"],
        precision=train_cfg["precision"],
        accumulate_grad_batches=train_cfg["accumulate_grad_batches"],
        check_val_every_n_epoch=max(1, int(train_cfg.get("check_val_every_n_epoch", 1))),
        logger=loggers,
        callbacks=[ckpt_cb],
        log_every_n_steps=log_cfg["log_every_n_steps"],
    )

    trainer.fit(model, datamodule=data, ckpt_path=resume)
    handle_metrics(loggers, run.metrics_path, resume=resume is not None)

    best_ckpt = ckpt_cb.best_model_path or str(run.best_ckpt_path if run.best_ckpt_path.exists() else "")
    last_ckpt = ckpt_cb.last_model_path or str(run.last_checkpoint() or "")

    if not last_ckpt:
        raise RuntimeError("No checkpoint produced")

    logger.info("Training done. Last checkpoint: %s", last_ckpt)
    return {
        "best_checkpoint": str(best_ckpt) if best_ckpt else "",
        "last_checkpoint": str(last_ckpt),
    }


def run_backbone_experiment(config, run_root=None, resume=None):
    config, root, logger = prepare_backbone_run(config, run_root)
    checkpoints = train_backbone(config, root, logger, resume)
    run = RunIO(root)
    return {
        "experiment_dir": str(run.root),
        "train_dir": str(run.train_dir),
        "checkpoint": checkpoints["last_checkpoint"],
        "last_checkpoint": checkpoints["last_checkpoint"],
        "best_checkpoint": checkpoints["best_checkpoint"],
        "metrics": config["logging"]["metrics_path"],
        "config": config,
    }


def load_backbone_checkpoint(config, checkpoint_path, eval_mode=True):
    checkpoint = tr.load(checkpoint_path, map_location="cpu")
    if "state_dict" not in checkpoint:
        raise ValueError(f"Expected a Lightning .ckpt file, got: {checkpoint_path}")

    cleaned_state_dict = {}
    for key, value in checkpoint["state_dict"].items():
        if key.startswith("model."):
            cleaned_state_dict[key[len("model."):]] = value
        else:
            cleaned_state_dict[key] = value

    model = build_backbone_model(config)
    model.load_state_dict(cleaned_state_dict)

    device = tr.device("cuda" if tr.cuda.is_available() else "cpu")
    model.to(device)
    if eval_mode:
        model.eval()
    return model


@tr.no_grad()
def evaluate_backbone_model(model, loader, device):
    model.eval()
    results = {"loss": [], "f1": []}

    for batch in tqdm(loader, desc="Testing", leave=False):
        conditioning = batch["conditioning"].to(device)
        target = batch["contact_oh"].to(device)
        mask = batch["mask"].to(device)

        logits = model(conditioning)
        loss = masked_cross_entropy_loss(logits, target, mask)
        results["loss"].append(loss.item())

        f1_scores = contact_f1(logits, target, lengths=batch["length"], reduce=False)
        results["f1"].extend(f1_scores.cpu().tolist())

    return results


def build_test_loader(config, batch_size=0):
    return build_dataloader(
        config,
        partition="test",
        batch_size=batch_size or None,
        shuffle=False,
    )


def write_summary_csv(summary, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def evaluate_backbone_checkpoint(
    config,
    checkpoint_path,
    batch_size=0,
    output_path=None,
    run_dir=None,
    periodic_epoch_count=None,
):
    checkpoint_path = Path(checkpoint_path)
    if run_dir is None and checkpoint_path.parent.name == "checkpoints":
        run_dir = RunIO.from_train_dir(checkpoint_path.parent.parent).root

    loader = build_test_loader(config, batch_size=batch_size)
    model = load_backbone_checkpoint(config, checkpoint_path, eval_mode=True)
    device = tr.device("cuda" if tr.cuda.is_available() else "cpu")
    results = evaluate_backbone_model(model, loader, device)

    summary = {
        "run_dir": str(run_dir or checkpoint_path.parent.parent),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": RunIO.checkpoint_epoch(checkpoint_path),
        "trained_epoch_count": RunIO.checkpoint_epoch(checkpoint_path) + 1,
        "epochs": config["training"]["max_epochs"],
        "base_dim": config["model"]["base_dim"],
        "test_loss": float(tr.tensor(results["loss"]).mean().item()),
        "test_f1": float(tr.tensor(results["f1"]).mean().item()),
        "test_f1_std": float(tr.tensor(results["f1"]).std(unbiased=False).item()),
    }
    if periodic_epoch_count is not None:
        summary["periodic_epoch_count"] = int(periodic_epoch_count)

    if output_path is not None:
        write_summary_csv(summary, output_path)
    return summary


def milestone_dir(run_dir, epoch):
    return Path(run_dir) / f"epoch_{int(epoch):03d}"


def milestone_complete(run_dir, epoch):
    target_dir = milestone_dir(run_dir, epoch)
    return (target_dir / "last.ckpt").exists() and (target_dir / "test_summary.csv").exists()


def snapshot_milestone(run_dir, epoch, checkpoint_path, summary):
    target_dir = milestone_dir(run_dir, epoch)
    target_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(Path(checkpoint_path), target_dir / "last.ckpt")
    write_summary_csv(summary, target_dir / "test_summary.csv")
    return target_dir
