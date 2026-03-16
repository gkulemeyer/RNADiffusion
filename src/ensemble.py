from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch as tr
from src.metrics import contact_f1_gpu
from tqdm import tqdm

class SequenceEnsemble:
    def __init__(self, data_path):
        self.path = Path(data_path)
        self.data = tr.load(self.path, map_location="cpu")
        self.samples = self.data["samples"].float()
        self.target = self.data["target"]
        self.length = int(self.data["length"])
        self.seeds = self.data.get("seeds", [])
        self.num_samples = len(self.samples)

        if self.target.ndim == 3:
            self.target = self.target.argmax(dim=0)

    def consensus(self, indices=None):
        """Calculates consensus using 0.5 threshold. Returns a Float Tensor."""
        subset = self.samples if indices is None else self.samples[indices]
        return (subset.mean(dim=0) > 0.5).float()

    def uncertainty(self, indices=None):
        """Returns uncertainty map (standard deviation). Returns a Float Tensor."""
        subset = self.samples if indices is None else self.samples[indices]
        return subset.std(dim=0).float()
    
    def consensus_f1(self, indices=None):
        consensus = self.consensus(indices).unsqueeze(0)
        target = self.target.unsqueeze(0)
        return contact_f1_gpu(consensus, target, lengths=[self.length], reduce=True)

    def consensus_f1_trials(self, indices_matrix):
        index_tensor = tr.tensor(indices_matrix, dtype=tr.long)
        subsets = self.samples[index_tensor]
        preds = (subsets.mean(dim=1) > 0.5).float()

        targets = self.target.unsqueeze(0).expand(preds.shape[0], -1, -1)
        return contact_f1_gpu(preds, targets, lengths=[self.length] * len(preds), reduce=False).tolist()

def _sample_batch(model, conditioning, num_samples, base_seed, chunk_size):
    batch_size = conditioning.shape[0]
    chunks = []
    generated = 0

    while generated < num_samples:
        current_chunk = min(chunk_size, num_samples - generated)
        tr.manual_seed(base_seed + generated)
        expanded_conditioning = conditioning.repeat_interleave(current_chunk, dim=0)

        with tr.no_grad():
            sampled = model._sample(expanded_conditioning)

        if sampled.ndim == 4:
            sampled = sampled.argmax(dim=1)
        elif sampled.ndim == 2:
            sampled = sampled.unsqueeze(0)
        if sampled.ndim != 3:
            raise ValueError(f"Unexpected sampled tensor shape: {tuple(sampled.shape)}")

        sampled = sampled.reshape(batch_size, current_chunk, sampled.shape[-2], sampled.shape[-1])
        chunks.append(sampled.cpu().to(tr.int8))
        generated += current_chunk

    return tr.cat(chunks, dim=1)


def generate_raw_samples(model, loader, output_dir, num_samples, base_seed, chunk_size):
    model.eval()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    seeds = [base_seed + index for index in range(num_samples)]

    for batch in tqdm(loader, desc="Generating samples", leave=False):
        batch_ids = batch["id"]
        save_paths = [output_path / f"{seq_id}.pt" for seq_id in batch_ids]
        pending_indices = [index for index, path in enumerate(save_paths) if not path.exists()]
        if not pending_indices:
            continue

        conditioning = batch["conditioning"][pending_indices].to(device)
        sampled = _sample_batch(
            model=model,
            conditioning=conditioning,
            num_samples=num_samples,
            base_seed=base_seed,
            chunk_size=chunk_size,
        )

        for i, batch_index in enumerate(pending_indices):
            tr.save(
                {
                    "samples": sampled[i],
                    "seeds": seeds,
                    "target": batch["contact_one_hot"][batch_index].cpu().to(tr.int8),
                    "length": int(batch["length"][batch_index]),
                },
                save_paths[batch_index],
            )

def evaluate_samples_dir(samples_dir, consensus_sizes, trials, seed=42):
    random.seed(seed)
    samples_path = Path(samples_dir)
    sample_paths = sorted(
        path
        for path in samples_path.iterdir()
        if path.suffix == ".pt"
    )
    if not sample_paths:
        raise ValueError(f"No sample files found in {samples_path}")

    ensembles = [SequenceEnsemble(path) for path in tqdm(sample_paths, desc="Loading", leave=False)]
    max_samples = min(ensemble.num_samples for ensemble in ensembles)
    invalid_sizes = [size for size in consensus_sizes if size > max_samples]
    if invalid_sizes:
        raise ValueError(
            f"Consensus sizes {invalid_sizes} exceed available samples ({max_samples}) in {samples_path}"
        )

    chosen_indices = {
        size: [random.sample(range(max_samples), size) for _ in range(trials)]
        for size in consensus_sizes
    }

    rows = []
    for ensemble in tqdm(ensembles, desc="Evaluating", leave=False):
        row = {"seq_id": ensemble.path.stem}
        for size in consensus_sizes:
            scores = ensemble.consensus_f1_trials(chosen_indices[size])
            row[f"cons_k{size}_mean"] = np.mean(scores)
            row[f"cons_k{size}_std"] = np.std(scores)
        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_samples_stats(samples_csv, consensus_sizes):
    df = Path(samples_csv)
    if not df.exists():
        raise ValueError(f"CSV file {samples_csv} does not exist.")
    df = pd.read_csv(df)
    
    metrics = {
        "consensus": [],
        "mean": [],
        "std": [],
        "std_mean": [],
        "std_std": [],
    }
    for size in consensus_sizes:
        mean_col = f"cons_k{size}_mean"
        std_col = f"cons_k{size}_std"
        mean = df[mean_col].mean()
        std = df[mean_col].std()
        std_mean = df[std_col].mean()
        std_std = df[std_col].std()
        
        metrics["consensus"].append(size)
        metrics["mean"].append(mean)
        metrics["std"].append(std)
        metrics["std_mean"].append(std_mean)
        metrics["std_std"].append(std_std)

    return pd.DataFrame(metrics, columns=["consensus", "mean", "std", "std_mean", "std_std"])
