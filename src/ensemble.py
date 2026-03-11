from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch as tr
from sklearn.metrics import f1_score
from tqdm import tqdm


DEFAULT_NUM_SAMPLES = 50
DEFAULT_BASE_SEED = 42
DEFAULT_TRIALS = 20
DEFAULT_CONSENSUS = [1, 3, 5, 7, 9, 11, 13, 15, 19, 21, 25]


def generate_raw_samples(model, loader, output_dir, num_samples=DEFAULT_NUM_SAMPLES, base_seed=DEFAULT_BASE_SEED):
    model.eval()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device

    for batch in tqdm(loader, desc="Generating samples", leave=False):
        for index, seq_id in enumerate(batch["id"]):
            save_path = output_path / f"{seq_id}.pt"
            if save_path.exists():
                continue

            condition = batch["conditioning"][index : index + 1].to(device)
            samples = []
            seeds = []

            for sample_index in range(num_samples):
                seed = base_seed + sample_index
                tr.manual_seed(seed)
                with tr.no_grad():
                    sample = model._sample(condition).squeeze(0).cpu().to(tr.int8)
                samples.append(sample)
                seeds.append(seed)

            tr.save(
                {
                    "samples": tr.stack(samples),
                    "seeds": seeds,
                    "target": batch["contact_one_hot"][index].cpu().to(tr.int8),
                    "length": batch["length"][index],
                },
                save_path,
            )


class SequenceEnsemble:
    def __init__(self, data_path):
        self.path = Path(data_path)
        self.payload = tr.load(self.path)
        self.samples = self.payload["samples"].float()
        self.target = self.payload["target"]
        self.length = int(self.payload["length"])
        self.num_samples = len(self.samples)

        if self.target.ndim == 3:
            self.target = self.target.argmax(dim=0)
        self.target_array = self.target.numpy()

    def consensus(self, indices=None):
        subset = self.samples if indices is None else self.samples[indices]
        return (subset.mean(dim=0) > 0.5).numpy().astype(int)

    def f1(self, matrix):
        rows, cols = np.triu_indices(self.length, k=1)
        predicted = matrix[: self.length, : self.length][rows, cols]
        expected = self.target_array[: self.length, : self.length][rows, cols]
        return f1_score(expected, predicted, zero_division=0)

    def consensus_f1(self, indices=None):
        return self.f1(self.consensus(indices))


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

    chosen_indices = {
        size: [random.sample(range(max_samples), size) for _ in range(trials)]
        for size in consensus_sizes
    }

    rows = []
    for ensemble in tqdm(ensembles, desc="Evaluating", leave=False):
        row = {"seq_id": ensemble.path.stem}
        for size in consensus_sizes:
            scores = [ensemble.consensus_f1(indices) for indices in chosen_indices[size]]
            row[f"cons_k{size}_mean"] = np.mean(scores)
            row[f"cons_k{size}_std"] = np.std(scores)
        rows.append(row)

    return pd.DataFrame(rows)
