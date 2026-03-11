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
DEFAULT_SAMPLE_CHUNK_SIZE = 25


def _to_contact_map(sample_batch):
    if sample_batch.ndim == 4:
        sample_batch = sample_batch.argmax(dim=1)
    elif sample_batch.ndim == 2:
        sample_batch = sample_batch.unsqueeze(0)

    if sample_batch.ndim != 3:
        raise ValueError(f"Unexpected sampled tensor shape: {tuple(sample_batch.shape)}")
    return sample_batch


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
        sampled = _to_contact_map(sampled)
        sampled = sampled.reshape(batch_size, current_chunk, sampled.shape[-2], sampled.shape[-1])
        chunks.append(sampled.cpu().to(tr.int8))
        generated += current_chunk

    return tr.cat(chunks, dim=1)


def generate_raw_samples(
    model,
    loader,
    output_dir,
    num_samples=DEFAULT_NUM_SAMPLES,
    base_seed=DEFAULT_BASE_SEED,
    chunk_size=DEFAULT_SAMPLE_CHUNK_SIZE,
):
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

        for local_index, batch_index in enumerate(pending_indices):
            tr.save(
                {
                    "samples": sampled[local_index],
                    "seeds": seeds,
                    "target": batch["contact_one_hot"][batch_index].cpu().to(tr.int8),
                    "length": int(batch["length"][batch_index]),
                },
                save_paths[batch_index],
            )


class SequenceEnsemble:
    def __init__(self, data_path):
        self.path = Path(data_path)
        self.payload = tr.load(self.path, map_location="cpu")
        self.samples = self.payload["samples"].float()
        self.target = self.payload["target"]
        self.length = int(self.payload["length"])
        self.seeds = self.payload.get("seeds", [])
        self.num_samples = len(self.samples)

        if self.target.ndim == 3:
            self.target = self.target.argmax(dim=0)
        self.target_array = self.target.numpy()
        self.rows, self.cols = np.triu_indices(self.length, k=1)
        self.target_vector = self.target_array[: self.length, : self.length][self.rows, self.cols]

    def consensus(self, indices=None):
        subset = self.samples if indices is None else self.samples[indices]
        return (subset.mean(dim=0) > 0.5).numpy().astype(int)

    def f1(self, matrix):
        predicted = matrix[: self.length, : self.length][self.rows, self.cols]
        return f1_score(self.target_vector, predicted, zero_division=0)

    def consensus_f1(self, indices=None):
        return self.f1(self.consensus(indices))

    def consensus_f1_trials(self, indices_matrix):
        index_tensor = tr.tensor(indices_matrix, dtype=tr.long)
        subsets = self.samples[index_tensor]
        consensuses = (subsets.mean(dim=1) > 0.5).cpu().numpy().astype(int)

        scores = []
        for consensus in consensuses:
            predicted = consensus[: self.length, : self.length][self.rows, self.cols]
            scores.append(f1_score(self.target_vector, predicted, zero_division=0))
        return scores


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
