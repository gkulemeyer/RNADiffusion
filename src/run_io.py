from pathlib import Path

import torch as tr


class RunIO:
    def __init__(self, root):
        self.root = Path(root)

    @property
    def train_dir(self):
        return self.root / "train"

    @property
    def config_path(self):
        return self.train_dir / "config.yaml"

    @property
    def log_path(self):
        return self.train_dir / "run.log"

    @property
    def metrics_path(self):
        return self.train_dir / "metrics.csv"

    @property
    def checkpoint_dir(self):
        return self.train_dir / "checkpoints"

    @property
    def periodic_ckpt_dir(self):
        return self.checkpoint_dir / "periodic"

    @property
    def best_ckpt_path(self):
        return self.checkpoint_dir / "best.ckpt"

    @property
    def last_ckpt_path(self):
        return self.checkpoint_dir / "last.ckpt"

    @property
    def best_eval_dir(self):
        return self.root / "eval" / "best"

    @property
    def periodic_eval_root(self):
        return self.root / "eval" / "periodic"

    def periodic_checkpoints(self):
        return sorted(self.periodic_ckpt_dir.glob("*.ckpt"), key=self.checkpoint_epoch)

    def periodic_eval_dir(self, epoch):
        return self.periodic_eval_root / f"epoch_{epoch:03d}"

    def last_checkpoint(self):
        return self.last_ckpt_path if self.last_ckpt_path.exists() else None

    def last_completed_epoch(self):
        checkpoint = self.last_checkpoint()
        return self.checkpoint_epoch(checkpoint) if checkpoint else None

    def completed_epoch_count(self):
        epoch = self.last_completed_epoch()
        return epoch + 1 if epoch is not None else 0

    @staticmethod
    def checkpoint_epoch(path):
        return int(tr.load(path, map_location="cpu")["epoch"])

    @classmethod
    def from_train_dir(cls, train_dir):
        return cls(Path(train_dir).parent)
