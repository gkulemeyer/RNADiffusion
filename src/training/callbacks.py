import json
import logging
import tempfile
from pathlib import Path

import pytorch_lightning as pl
import torch as tr

from ..utils.mlflow_io import load_run_info, resolve_tracking_dir, save_run_config


class BestModelSaverCallback(pl.Callback):
    def __init__(self, config, mlflow_logger=None, logger=None):
        super().__init__()
        self.config = config
        self.best_val_f1 = -1.0
        self.mlflow_logger = mlflow_logger
        self.logger = logger or logging.getLogger(__name__)

    def _build_checkpoint_payload(self, trainer, pl_module, optimizer_state):
        return {
            "format_version": 1,
            "epoch": trainer.current_epoch + 1,
            "model_state": pl_module.model.state_dict(),
            "optimizer_state": optimizer_state,
            "best_val_f1": self.best_val_f1,
            "config": self.config,
        }

    def _write_checkpoint_files(self, checkpoint_path, info_path, payload):
        tr.save(payload, checkpoint_path)
        info = {
            "format_version": payload.get("format_version", 1),
            "epoch": payload["epoch"],
            "best_val_f1": payload.get("best_val_f1"),
            "has_optimizer_state": payload.get("optimizer_state") is not None,
            "model_keys": ["format_version", "epoch", "model_state", "optimizer_state", "best_val_f1", "config"],
        }
        info_path.write_text(json.dumps(info, indent=2))

    def _persist_artifacts(self, filename, payload):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            checkpoint_path = tmpdir / filename
            info_path = tmpdir / f"{checkpoint_path.stem}_info.json"
            self._write_checkpoint_files(checkpoint_path, info_path, payload)

            if self.mlflow_logger is None:
                return

            run_id = getattr(self.mlflow_logger, "run_id", None)
            tracking_uri = self.config.get("mlflow", {}).get("tracking_uri") if isinstance(self.config, dict) else None
            tracking_dir = resolve_tracking_dir(tracking_uri)
            if run_id is not None:
                run_info = load_run_info(tracking_dir, run_id)
                if run_info is not None:
                    destination = run_info["run_dir"] / "artifacts" / "checkpoints"
                    destination.mkdir(parents=True, exist_ok=True)
                    self._write_checkpoint_files(destination / filename, destination / f"{Path(filename).stem}_info.json", payload)
                    save_run_config(tracking_dir, run_id, self.config)
                    self.logger.info("%s saved to MLFlow local artifacts", filename)
                    return

            try:
                self.mlflow_logger.experiment.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
                self.mlflow_logger.experiment.log_artifact(str(info_path), artifact_path="checkpoints")
                self.logger.info("%s saved to MLFlow", filename)
            except Exception as exc:
                self.logger.warning("Failed to log %s to MLFlow: %s", filename, exc)

    def on_validation_epoch_end(self, trainer, pl_module):
        val_f1 = trainer.callback_metrics.get("val_f1")
        if val_f1 is None:
            return
        val_f1_value = val_f1.detach().cpu().item() if isinstance(val_f1, tr.Tensor) else float(val_f1)
        if val_f1_value <= self.best_val_f1:
            return
        self.best_val_f1 = val_f1_value
        optimizer_state = trainer.optimizers[0].state_dict() if trainer.optimizers else None
        payload = self._build_checkpoint_payload(trainer, pl_module, optimizer_state)
        self._persist_artifacts("best_model.pt", payload)

    def on_train_end(self, trainer, pl_module):
        if self.mlflow_logger is None:
            return
        optimizer_state = trainer.optimizers[0].state_dict() if trainer.optimizers else None
        payload = self._build_checkpoint_payload(trainer, pl_module, optimizer_state)
        self._persist_artifacts("last_model.pt", payload)
