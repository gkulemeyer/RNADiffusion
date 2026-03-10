"""Evaluate the ensemble"""
from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch as tr
from tqdm import tqdm

from .metrics import contact_f1_gpu

logger = logging.getLogger(__name__)

class SeqEnsemble:
    """Wrapper for single sample ensemble (vectorized)."""

    def __init__(self, data_path):
        self.path = Path(data_path)
        self.data = tr.load(self.path, map_location="cpu")

        self.samples = self.data["samples"].float()  # [N, L, L]
        self.seeds = self.data.get("seeds", None)
        self.target = self.data["target"]
        self.length = self.data["length"]
        self.num_samples = len(self.samples)

        # Normalize target if it is one-hot encoded
        if self.target.ndim == 3:
            self.target = self.target.argmax(dim=0)

    def get_consensus(self, indices=None):
        """Calculates consensus using 0.5 threshold. Returns a Float Tensor."""
        subset = self.samples if indices is None else self.samples[indices]
        prob_map = subset.mean(dim=0)
        return (prob_map > 0.5).float()

    def get_uncertainty(self, indices=None):
        """Returns uncertainty map (standard deviation). Returns Numpy array."""
        subset = self.samples if indices is None else self.samples[indices]
        return subset.std(dim=0).numpy()

    def evaluate_single(self, sample_idx):
        """Evaluates F1 score of a single sample by index."""
        pred = self.samples[sample_idx].unsqueeze(0)
        target = self.target.unsqueeze(0)
        return contact_f1_gpu(pred, target, lengths=[self.length], reduce=True)

    def evaluate_consensus(self, indices=None):
        """Evaluates F1 score of the consensus."""
        consensus = self.get_consensus(indices).unsqueeze(0)
        target = self.target.unsqueeze(0)
        return contact_f1_gpu(consensus, target, lengths=[self.length], reduce=True)

    def evaluate_batch_trials(self, indices_matrix):
        """
        Computes F1 for N trials simultaneously.
        Args:
            indices_matrix: Shape [Trials, samples] (e.g., 30 trials x 5 samples)
        Returns:
            List of 30 F1 scores.
        """
        # Convert indices to tensor [Trials, samples]
        # Compute Consensus: Mean over dim 1 (samples) -> [Trials, L, L]
        idx_tensor = tr.tensor(indices_matrix, device=self.samples.device)
        subset = self.samples[idx_tensor]
        consensus = subset.mean(dim=1)
        preds = (consensus > 0.5).float()

        targets = self.target.unsqueeze(0).expand(preds.shape[0], -1, -1)
        return contact_f1_gpu(preds, targets, lengths=[self.length]*len(preds), reduce=False).tolist()


class EnsembleGenerator:
    """Generate Ensemble samples given a model (Optimized)"""

    def __init__(
        self,
        model,
        save_dir: str | Path | None,
        num_samples: int = 50,
        base_seed: int = 42,
        mlflow_tracking_dir: str | Path | None = None,
        mlflow_run_id: str | None = None,
    ):
        self.model = model
        self.num_samples = num_samples
        self.base_seed = base_seed
        self.device = tr.device("cuda" if tr.cuda.is_available() else "cpu")
        self.final_save_dir = None
        self.chunk_size = 25

        if mlflow_tracking_dir is not None and mlflow_run_id is not None:
            track_path = Path(mlflow_tracking_dir)
            found = False
            if track_path.exists():
                for exp_dir in track_path.iterdir():
                    if exp_dir.is_dir():
                        candidate = exp_dir / mlflow_run_id
                        if candidate.exists():
                            self.final_save_dir = candidate / "artifacts" / "ensembles" / "raw_samples"
                            found = True
                            break

            if not found:
                logger.warning(
                    f"MLFlow run dir not found for {mlflow_run_id}. Fallback to local save_dir."
                )
                self.final_save_dir = Path(save_dir) if save_dir else None
        else:
            self.final_save_dir = Path(save_dir) if save_dir else None

        if self.final_save_dir is not None:
            self.final_save_dir.mkdir(parents=True, exist_ok=True)

        self.model.eval()

    def generate(self, loader):
        """Generate vectorized ensemble"""

        if self.final_save_dir is None:
                self.final_save_dir = self.save_dir

        if self.final_save_dir:
            self.final_save_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                f"Generating {self.num_samples} samples per sequence -> {self.final_save_dir}"
            )
        tr.manual_seed(self.base_seed)


        for batch in tqdm(loader, desc="Ensemble generation", leave=False):
            outer_batch = batch["outer"].to(self.device)
            batch_ids = batch["id"]

            for i in range(len(batch_ids)):
                seq_id = batch_ids[i]

                save_path = (self.final_save_dir / f"{seq_id}.pt") if self.final_save_dir else None
                if save_path is not None and save_path.exists():
                    continue
                cond_single = outer_batch[i].unsqueeze(0)

                all_samples = []

                # Instead of making a loop from 1 to 50, we make jumps (chunks)
                for _ in range(0, self.num_samples, self.chunk_size):
                    current_batch = min(self.chunk_size, self.num_samples - len(all_samples))
                    repeat_dims = [1] * cond_single.ndim
                    repeat_dims[0] = current_batch
                    cond_expanded = cond_single.repeat(*repeat_dims)

                    with tr.no_grad():
                        # PyTorch generates different noise in each batch.
                        batch_output = self.model._sample(cond_expanded)
                    all_samples.append(batch_output.detach().cpu())
                final_samples_tensor = tr.cat(all_samples, dim=0).to(tr.int8)

                if save_path is not None:
                    data = {
                        "samples": final_samples_tensor,
                        "base_seed": self.base_seed,
                        "target": batch["contact_oh"][i].cpu().to(tr.int8),
                        "length": batch["length"][i],
                    }
                    tr.save(data, save_path)
        logger.info("Ensemble generation complete")


class EnsembleAnalyzer:
    """Sequential Ensemble Analyzer (Vectorized internally)."""

    def __init__(
        self,
        samples_dir,
        consensus_sizes= [1, 5, 9, 15, 25],
        trials=30,
    ):
        self.samples_dir = Path(samples_dir)
        self.consensus_sizes = consensus_sizes
        self.trials = trials

    def analyze(self):
        """
        Analyze ensemble & return stats DataFrame using a sequential loop.
        """
        # Load file list
        sample_files = sorted(self.samples_dir.glob("*.pt"))
        if not sample_files:
            raise ValueError(f"No samples found in {self.samples_dir}")

        # Determine max_samples from the first file, loading one file to check dimensions
        first_ens = SeqEnsemble(sample_files[0])
        max_samples = first_ens.num_samples
        logger.info(f"Detected {max_samples} samples available per sequence.")
        del first_ens

        chosen_indices = {}
        for k in self.consensus_sizes:
            if k > max_samples:
                continue
            chosen_indices[k] = [
                random.sample(range(max_samples), k) for _ in range(self.trials)
            ]

        results = []

        for f in tqdm(sample_files, desc="Ensemble analysis", leave=False):
            try:
                ens = SeqEnsemble(f)
                row = {"seq_id": f.stem}
                for k in self.consensus_sizes:
                    if k not in chosen_indices:
                        continue
                    trials_indices = chosen_indices[k]
                    # Calculates all 30 trials in one tensor operation
                    scores = ens.evaluate_batch_trials(trials_indices)
                    row[f"F1_cons_mean_{k}"] = np.mean(scores)
                    row[f"F1_cons_std_{k}"]  = np.std(scores)

                results.append(row)

            except Exception as e:
                logger.warning(f"Error processing {f.name}: {e}")
                continue

        return pd.DataFrame(results)
