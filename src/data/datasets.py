"""PyTorch RNA dataset."""
import json
import logging

import pandas as pd
import torch as tr
from torch.utils.data import Dataset

from ..core.embeddings import OneHotEmbedding
from .transforms import bp2matrix, dot2bp

logger = logging.getLogger(__name__)


class SeqDataset(Dataset):
    """Base dataset for RNA sequences."""

    def __init__(
        self,
        data,
        max_len=512,
        for_prediction=False,
    ):
        self.max_len = max_len
        self.for_prediction = for_prediction
        self.embedding = OneHotEmbedding()

        # Validation
        self._validate_data(data)

        # Processing
        if not for_prediction and "base_pairs" not in data.columns:
            data["base_pairs"] = data.dotbracket.apply(
                lambda x: json.dumps(dot2bp(x))
            )

        # Storage
        self.sequences = data.sequence.tolist()
        self.ids = data.id.tolist()
        self.base_pairs = None
        if "base_pairs" in data.columns:
            self.base_pairs = [json.loads(bp) for bp in data.base_pairs]

    def _validate_data(self, data):
        required = ["id", "sequence"]
        if not self.for_prediction:
            required.append("base_pairs" if "base_pairs" in data.columns else "dotbracket")

        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_id = self.ids[idx]
        sequence = self.sequences[idx]
        L = len(sequence)

        # Embeddings
        seq_emb = self.embedding.seq2emb(sequence)
        outer = self.embedding.outer_emb(seq_emb)

        # Contact matrix
        Mc = None
        Mc_OH = None
        if self.base_pairs is not None:
            Mc = bp2matrix(L, self.base_pairs[idx])
            Mc_OH = tr.nn.functional.one_hot(
                Mc.long(), num_classes=2
            ).float().permute(2, 0, 1)

        return {
            "id": seq_id,
            "length": L,
            "sequence": sequence,
            "embedding": seq_emb,
            "outer": outer,
            "contact": Mc,
            "contact_oh": Mc_OH,
        }


def load_dataset(
    dataset_path=None,
    min_len=0,
    max_len=512,
    for_prediction=False,
    # Partitioned args
    partitioned=False,
    main_path=None,
    partition_path=None,
    partition_value=None,
    fold_number=None,
):
    """
    Factory function. Entry point to create the apropiate dataset.
    """
    if partitioned:
        data = _load_partitioned_data(
            main_path, partition_path, partition_value, fold_number
        )
    else:
        data = pd.read_csv(dataset_path)

    # length filter
    data["len"] = data.sequence.str.len()
    original_len = len(data)
    data = data[(data.len >= min_len) & (data.len <= max_len)]

    if len(data) < original_len:
        logger.info(
            f"Filtered {original_len} -> {len(data)} sequences ({min_len} <= len <= {max_len})"
        )

    return SeqDataset(data, max_len=max_len, for_prediction=for_prediction)


def _load_partitioned_data(
    main_path,
    partition_path,
    partition_value,
    fold_number=None,
):
    """Load data filtered by partition."""
    data = pd.read_csv(main_path)
    partitions = pd.read_csv(partition_path)

    # validate
    required = ["id", "partition"]
    if fold_number is not None:
        required.append("fold_number")

    missing = [col for col in required if col not in partitions.columns]
    if missing:
        raise ValueError(f"Partition file missing columns: {missing}")

    # Filter
    part_df = partitions[partitions["partition"] == partition_value]
    if fold_number is not None:
        part_df = part_df[part_df["fold_number"] == fold_number]

    ids = part_df["id"]
    filtered = data[data["id"].isin(ids)].reset_index(drop=True)

    if len(filtered) == 0:
        raise ValueError(
            f"No samples found for partition={partition_value}, fold={fold_number}"
        )

    return filtered
