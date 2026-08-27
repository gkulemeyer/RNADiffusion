from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch as tr
from tqdm import tqdm

from src.data import mat2db
from src.metrics import contact_f1_gpu


class SequenceEnsemble:
    def __init__(self, data_path):
        self.path = Path(data_path)
        self.data = tr.load(self.path, map_location="cpu")

        self.raw_samples = (
            self.data["raw_samples"]
            if "raw_samples" in self.data
            else self.data["samples"]
        )
        self.processed_samples = self.data.get("processed_samples")
        self.target = self.data["target"]
        self.length = int(self.data["length"])
        self.sample_seeds = self.data.get("sample_seeds", self.data.get("seeds", []))
        self.num_samples = int(self.raw_samples.shape[0])

        if self.target.ndim == 3:
            self.target = self.target.argmax(dim=0)

    def get_samples(self, sample_type):
        return {
            "raw": self.raw_samples,
            "processed": self.processed_samples,
        }[sample_type]

    def consensus(self, indices=None, sample_type="raw"):
        """Calculate a consensus contact map using a 0.5 threshold."""
        samples = self.get_samples(sample_type)
        subset = samples if indices is None else samples[indices]
        return (subset.float().mean(dim=0) > 0.5).float()

    def uncertainty(self, indices=None, sample_type="raw"):
        """Return the contact-map standard deviation across ensemble members."""
        samples = self.get_samples(sample_type)
        subset = samples if indices is None else samples[indices]
        return subset.float().std(dim=0)

    def consensus_f1(self, indices=None, sample_type="raw"):
        consensus = self.consensus(indices, sample_type=sample_type).unsqueeze(0)
        target = self.target.unsqueeze(0)
        return contact_f1_gpu( consensus, target, lengths=[self.length], reduce=True)

    def consensus_f1_trials(self, indices_matrix, sample_type="raw"):
        samples = self.get_samples(sample_type)
        index_tensor = tr.as_tensor(indices_matrix, dtype=tr.long)
        subsets = samples[index_tensor]
        preds = (subsets.float().mean(dim=1) > 0.5).float()

        targets = self.target.unsqueeze(0).expand(preds.shape[0], -1, -1)
        return contact_f1_gpu( preds, targets, lengths=[self.length] * len(preds), reduce=False).tolist()


def process_sample(raw_sample, logits, threshold=0.1):
    """Resolve one sampled contact map into a symmetric, binary matching."""
    length = raw_sample.shape[0]
    scores_map = tr.sigmoid(logits[1])
    i, j = tr.triu_indices(length, length, offset=1, device=raw_sample.device)
    scores = scores_map[i, j]
    valid = (raw_sample[i, j] > 0) & (scores >= threshold)
    i, j, scores = i[valid], j[valid], scores[valid]
    order = scores.argsort(descending=True)

    processed_sample = tr.zeros_like(raw_sample)
    used = set()
    for index in order.tolist():
        left = i[index].item()
        right = j[index].item()
        if left not in used and right not in used:
            processed_sample[left, right] = 1
            processed_sample[right, left] = 1
            used.update((left, right))

    return processed_sample


def _generate_ensemble_batch( model, conditioning, lengths, sample_seeds, threshold):
    lengths = tr.as_tensor(lengths, dtype=tr.long, device=conditioning.device)
    raw_members = []
    processed_members = []

    for sample_seed in sample_seeds:
        tr.manual_seed(sample_seed)
        with tr.inference_mode():
            raw_sample_batch, logits_batch = model.sample(
                conditioning,
                lengths=lengths,
                return_logits=True,
            )

        processed_sample_batch = tr.zeros_like(raw_sample_batch)
        for batch_index, lenght in enumerate(lengths):
            l = int(lenght.item())
            processed_sample_batch[batch_index, :l, :l] = process_sample(
                raw_sample_batch[batch_index, :l, :l],
                logits_batch[batch_index, :, :l, :l],
                threshold=threshold,
            )

        raw_members.append(raw_sample_batch.cpu().to(tr.int8))
        processed_members.append(processed_sample_batch.cpu().to(tr.int8))

    raw_samples = tr.stack(raw_members, dim=1)
    processed_samples = tr.stack(processed_members, dim=1)
    return raw_samples, processed_samples


def save_ensemble_samples(model, loader, output_dir, sample_seeds, threshold):
    model.eval()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device

    for batch in tqdm(loader, desc="Generating ensemble samples", leave=False):
        batch_ids = batch["id"]
        save_paths = [output_path / f"{seq_id}.pt" for seq_id in batch_ids]
        pending_indices = [idx for idx, path in enumerate(save_paths) if not path.exists()]
        if not pending_indices:
            continue

        conditioning = batch["conditioning"][pending_indices].to(device)
        pend_lengths = [batch["length"][index] for index in pending_indices]
        lengths = tr.as_tensor(pend_lengths, dtype=tr.long, device=device)
        raw_samples, processed_samples = _generate_ensemble_batch(
            model=model,
            conditioning=conditioning,
            lengths=lengths,
            sample_seeds=sample_seeds,
            threshold=threshold,
        )

        for pend_idx, batch_idx in enumerate(pending_indices):
            l = int(batch["length"][batch_idx])
            target = batch["contact_oh"][batch_idx].argmax(dim=0)[:l, :l].cpu().to(tr.int8)
            
            tr.save(
                {
                    "id": batch_ids[batch_idx],
                    "raw_samples": raw_samples[pend_idx, :, :l, :l],
                    "processed_samples": processed_samples[pend_idx, :, :l, :l],
                    "target": target,
                    "length": l,
                    "sample_seeds": sample_seeds,
                },
                save_paths[batch_idx],
            )


def evaluate_samples_dir(
    samples_dir,
    consensus_sizes,
    trials,
    seed=42,
    get_best_and_worst=False,
    sample_type="raw",
):
    """Evaluate sequence-level consensus scores for one stored sample type."""
    random.seed(seed)
    samples_path = Path(samples_dir)
    sample_paths = sorted(path for path in samples_path.iterdir() if path.suffix == ".pt")

    ensembles = [
        SequenceEnsemble(path)
        for path in tqdm(sample_paths, desc="Loading", leave=False)
    ]
    max_samples = min(
        ensemble.get_samples(sample_type).shape[0] for ensemble in ensembles
    )
    chosen_indices = {
        size: [random.sample(range(max_samples), size) for _ in range(trials)]
        for size in consensus_sizes
    }

    rows = []
    for ensemble in tqdm(ensembles, desc=f"Evaluating {sample_type}", leave=False):
        row = {"seq_id": ensemble.path.stem}
        for size in consensus_sizes:
            scores = ensemble.consensus_f1_trials(
                chosen_indices[size],
                sample_type=sample_type,
            )
            row[f"cons_k{size}_mean"] = np.mean(scores)
            row[f"cons_k{size}_std"] = np.std(scores)
            if get_best_and_worst:
                row[f"cons_k{size}_best"] = np.max(scores)
                row[f"cons_k{size}_worst"] = np.min(scores)
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_samples_stats(samples_csv, consensus_sizes, get_best_and_worst=False):
    """Aggregate sequence-level ensemble metrics from a CSV file."""
    dataframe = pd.read_csv(samples_csv)

    metrics = {
        "consensus": [],
        "mean": [],
        "std": [],
        "std_mean": [],
        "std_std": [],
    }
    if get_best_and_worst:
        metrics["best"] = []
        metrics["worst"] = []

    for size in consensus_sizes:
        mean_column = f"cons_k{size}_mean"
        std_column = f"cons_k{size}_std"
        metrics["consensus"].append(size)
        metrics["mean"].append(dataframe[mean_column].mean())
        metrics["std"].append(dataframe[mean_column].std())
        metrics["std_mean"].append(dataframe[std_column].mean())
        metrics["std_std"].append(dataframe[std_column].std())
        if get_best_and_worst:
            metrics["best"].append(dataframe[f"cons_k{size}_best"].mean())
            metrics["worst"].append(dataframe[f"cons_k{size}_worst"].mean())

    columns = ["consensus", "mean", "std", "std_mean", "std_std"]
    if get_best_and_worst:
        columns.extend(["best", "worst"])
    return pd.DataFrame(metrics, columns=columns)


def export_db_ensemble(samples_dir, output_csv):
    samples_path = Path(samples_dir)
    sample_paths = sorted(path for path in samples_path.iterdir() if path.suffix == ".pt")

    rows = []
    for sample_path in tqdm(sample_paths, desc="Exporting dot-bracket", leave=False):
        ensemble = SequenceEnsemble(sample_path)
        processed_samples = ensemble.get_samples("processed")
        sequence_id = ensemble.data["id"]
        for sample_id, processed_sample in enumerate(processed_samples):
            rows.append(
                {
                    "id": sequence_id,
                    "sample_id": sample_id,
                    "seed": ensemble.sample_seeds[sample_id],
                    "sampled_structure": mat2db(processed_sample),
                }
            )

    dataframe = pd.DataFrame(
        rows,
        columns=["id", "sample_id", "seed", "sampled_structure"],
    )
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    return dataframe
