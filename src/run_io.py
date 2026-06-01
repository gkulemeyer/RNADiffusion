from __future__ import annotations

import copy 
from pathlib import Path
from dataclasses import dataclass
from ml_collections import ConfigDict
import torch as tr

from src.config import load_config
from src.io import read_yaml


class RunIO:
    def __init__(self, root):
        self.root = Path(root)

    # TRAIN Folders and Files
    @property # method -> calculated atribute
    def train_dir(self): return self.root / "train"

    @property
    def config_path(self): return self.train_dir / "config.yaml"
    @property
    def log_path(self): return self.train_dir / "run.log"
    @property
    def metrics_path(self): return self.train_dir / "metrics.csv"
    ## checkpoints
    @property
    def checkpoint_dir(self): return self.train_dir / "checkpoints"
    @property
    def periodic_ckpt_dir(self): return self.checkpoint_dir / "periodic"
    @property
    def best_ckpt_path(self): return self.checkpoint_dir / "best.ckpt"
    @property
    def last_ckpt_path(self): return self.checkpoint_dir / "last.ckpt"

    # EVAL Folders and Files
    @property
    def best_eval_dir(self): return self.root / "eval" / "best"
    @property
    def periodic_eval_root(self): return self.root / "eval" / "periodic"


    def periodic_checkpoints(self): 
        if self.periodic_ckpt_dir.exists():
            return sorted(self.periodic_ckpt_dir.glob("*.ckpt"), key=self.checkpoint_epoch)
        return []

    def periodic_eval_dir(self, epoch): 
        return self.periodic_eval_root / f"epoch_{epoch:03d}"
    
    def last_checkpoint(self): 
        if self.last_ckpt_path.exists():
            return self.last_ckpt_path
        return None
    
    def last_completed_epoch(self):
        if not self.last_ckpt_path.exists():
            return None

        return self.checkpoint_epoch(self.last_ckpt_path)

    def completed_epoch_count(self):
        last_epoch = self.last_completed_epoch()
        if last_epoch is None:
            return 0
        return last_epoch + 1
    
    def completed_epochs(self):
        return self.completed_epoch_count()
    
    def best_eval_is_current(self): 
        '''
        Checks if the best evaluation results are up to date with the best checkpoint.
         - Compares the checkpoint path and epoch in the metadata with the current best checkpoint.
         - Also checks if the ensemble evaluation files exist.
        '''
        metadata_path = self.best_eval_dir / "ensemble_metadata.yaml"

        if not metadata_path.exists() or not self.best_ckpt_path.exists():
            return False
        
        metadata = read_yaml(metadata_path)

        match_ckpt_path = metadata.get("checkpoint_path") == str(self.best_ckpt_path)
        match_ckpt_epoch = int(metadata.get("checkpoint_epoch", -1)) == self.checkpoint_epoch(self.best_ckpt_path) 

        ensemble = self.best_eval_dir / "ensemble.csv" 
        ensemble_stats = self.best_eval_dir / "ensemble_stats.csv"
        match_ensemble = ensemble.exists() and ensemble_stats.exists()

        return match_ckpt_path and match_ckpt_epoch and match_ensemble

    def run_is_complete(self, total_epochs): 
        if self.completed_epoch_count() < int(total_epochs):
            return False

        required = [
            self.config_path,
            self.metrics_path,
            self.last_ckpt_path,
            self.best_ckpt_path,
        ]

        return (
            all(path.exists() for path in required)
            and self.best_eval_is_current()
        )

    @staticmethod # method without self, can be called on the class itself, does not depend on instance state
    def checkpoint_epoch(path):
        path = Path(path)
        checkpoint = tr.load(path, map_location="cpu")

        if "epoch" not in checkpoint:
            raise ValueError(f"Missing epoch metadata in checkpoint: {path}")

        return int(checkpoint["epoch"])

    def normalize_periodic_checkpoint_names(self):
        for checkpoint_path in sorted(self.periodic_ckpt_dir.glob("*.ckpt")):
            epoch = self.checkpoint_epoch(checkpoint_path)
            target = self.periodic_ckpt_dir / f"epoch{epoch:03d}.ckpt"

            if checkpoint_path == target:
                continue
            if target.exists():
                raise RuntimeError(
                    f"Duplicate periodic checkpoint for epoch {epoch}: "
                    f"{checkpoint_path} and {target}"
                )
            checkpoint_path.rename(target)


    @classmethod # takes the class as first argument instead of instance, allows construce the class from the train dir (cls)
    def from_train_dir(cls, train_dir):
        train_dir = Path(train_dir)
        if train_dir.name != "train":
            raise ValueError(f"Expected train dir named 'train', got: {train_dir}")
        return cls(train_dir.parent)
    
    @classmethod
    def from_eval_dir(cls, eval_dir):
        eval_dir = Path(eval_dir)
        if eval_dir.parts[-2:] == ("eval", "best"):
            return cls(eval_dir.parent.parent)
        if len(eval_dir.parts) >= 3 and eval_dir.parent.name == "periodic" and eval_dir.parent.parent.name == "eval":
            return cls(eval_dir.parent.parent.parent)
        raise ValueError(f"Could not infer run root from eval dir: {eval_dir}")
    
    
# dataclass 

@dataclass
class RunState:
    run: RunIO
    config: ConfigDict
    total_epochs: int
    done_epochs: int
    checkpoint_path: Path | None

    @property
    def should_train(self):
        return self.done_epochs < self.total_epochs


class RunResolver:
    def __init__(self, job_dir, resume=True):
        self.job_dir = Path(job_dir)
        self.resume = bool(resume)

    def latest_attempt_dir(self):
        attempts = sorted(
            p for p in self.job_dir.glob("attempt_*")
            if p.is_dir()
        )
        return attempts[-1] if attempts else None

    def next_attempt_dir(self):
        index = 1
        while True:
            attempt_dir = self.job_dir / f"attempt_{index:03d}"
            if not attempt_dir.exists():
                return attempt_dir
            index += 1

    def resume_run(self):
        if (self.job_dir / "train").exists():
            return self.job_dir

        latest = self.latest_attempt_dir()
        if latest is not None and (latest / "train").exists():
            return latest

        return self.job_dir

    def new_run(self):
        if not self.job_dir.exists():
            return self.job_dir

        latest = self.latest_attempt_dir()
        if (self.job_dir / "train").exists() or latest is not None:
            return self.next_attempt_dir()

        return self.job_dir

    def resolve_root(self, config):
        if self.resume:
            run_root = self.resume_run()
        else:
            run_root = self.new_run()
        config.experiment.name = run_root.name
        return run_root 
    
    def build_state(self, config):
        requested_epochs = int(config.training.max_epochs)
        requested_val_every = int(config.training.check_val_every_n_epoch)
        requested_ckpt_every = int(config.logging.checkpoint_every_n_epochs)

        run_root = self.resolve_root(config)
        run = RunIO(run_root)

        if run.config_path.exists():
            effective_config = ConfigDict(load_config(run.config_path))
        else:
            if hasattr(config, "to_dict"):
                effective_config = ConfigDict(copy.deepcopy(config.to_dict()))
            else:
                effective_config = ConfigDict(copy.deepcopy(config))
        effective_config.training.max_epochs = requested_epochs
        effective_config.training.check_val_every_n_epoch = requested_val_every
        effective_config.logging.checkpoint_every_n_epochs = requested_ckpt_every

        total_epochs = int(effective_config.training.max_epochs)
        done_epochs = run.completed_epoch_count() if run.train_dir.exists() else 0 

        return RunState(
            run=run,
            config=effective_config,
            total_epochs=total_epochs,
            done_epochs=done_epochs,
            checkpoint_path=run.last_checkpoint()
        )
