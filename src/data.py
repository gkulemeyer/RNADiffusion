from __future__ import annotations

import json
import math

import lightning as L
import pandas as pd
import torch as tr
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.embeddings import OneHotEmbedding

#### Load as a contact map
MATCHING_BRACKETS = [
    ["(", ")"],
    ["[", "]"],
    ["{", "}"],
    ["<", ">"],
    ["A", "a"],
    ["B", "a"],
]


def bp2matrix(length, base_pairs):
    matrix = tr.zeros((length, length))
    for bp in base_pairs:
        matrix[bp[0] - 1, bp[1] - 1] = 1
        matrix[bp[1] - 1, bp[0] - 1] = 1
    return matrix


def fold2bp(structure, open_bracket="(", close_bracket=")"):
    opening_positions = []
    base_pairs = []
    if structure.count(open_bracket) != structure.count(close_bracket):
        return False

    for index, char in enumerate(structure):
        if char == open_bracket:
            opening_positions.append(index)
        elif char == close_bracket:
            if not opening_positions:
                return False
            base_pairs.append([opening_positions.pop() + 1, index + 1])
    return base_pairs


def dot2bp(structure):
    allowed = set(["."] + [char for brackets in MATCHING_BRACKETS for char in brackets])
    if not set(structure).issubset(allowed):
        return False

    base_pairs = []
    for open_bracket, close_bracket in MATCHING_BRACKETS:
        if open_bracket in structure:
            parsed = fold2bp(structure, open_bracket, close_bracket)
            if not parsed:
                return False
            base_pairs.extend(parsed)
    return list(sorted(base_pairs))


### Load as a dataset with partitioning

def _load_partitioned_data(main_path, partition_path, partition_value, fold_number=None):
    """Load the main dataset filtered by partition and optional fold."""
    data = pd.read_csv(main_path)
    partitions = pd.read_csv(partition_path)

    if "id" not in data.columns:
        raise ValueError("Main dataset missing required column: id")

    required_columns = {"id", "partition"}
    if fold_number is not None:
        required_columns.add("fold_number")

    missing_columns = sorted(required_columns.difference(partitions.columns))
    if missing_columns:
        raise ValueError(f"Partition file missing columns: {missing_columns}")

    partition_mask = partitions["partition"] == partition_value
    if fold_number is not None:
        partition_mask &= partitions["fold_number"] == fold_number
    partition_frame = partitions.loc[partition_mask]

    if partition_frame.empty:
        available_partitions = sorted(partitions["partition"].dropna().unique().tolist())
        available_folds = []
        if fold_number is not None:
            available_folds = sorted(
                partitions.loc[
                    partitions["partition"] == partition_value, "fold_number"
                ].dropna().unique().tolist()
            )
        details = f"Available partitions: {available_partitions}"
        if available_folds:
            details += f"; available folds for partition: {available_folds}"
        raise ValueError(
            f"No rows found in partition file for partition={partition_value}, fold={fold_number}. "
            f"{details}"
        )

    filtered = data[data["id"].isin(partition_frame["id"])].reset_index(drop=True)
    if filtered.empty:
        raise ValueError(
            f"No samples from main dataset matched partition ids for "
            f"partition={partition_value}, fold={fold_number}"
        )
    return filtered


load_partitioned_data = _load_partitioned_data

### Dataset and dataloader for training and evaluation
class SeqDataset(Dataset):
    def __init__(self, base_path, partition_path, partition_value, fold_number=None):
        data = _load_partitioned_data(
            main_path=base_path,
            partition_path=partition_path,
            partition_value=partition_value,
            fold_number=fold_number,
        )

        data = _ensure_base_pairs_column(data)

        required_columns = {"id", "sequence", "base_pairs"}
        missing_columns = required_columns.difference(data.columns)
        if missing_columns:
            raise ValueError(f"Dataset missing columns: {sorted(missing_columns)}")

        self.ids = data["id"].tolist()
        self.sequences = data["sequence"].tolist()
        self.base_pairs = [json.loads(value) for value in data["base_pairs"]]
        self.embedding = OneHotEmbedding()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = self.sequences[index]
        seq_id = self.ids[index]
        length = len(sequence)

        contact = bp2matrix(length, self.base_pairs[index])
        contact_oh = F.one_hot(contact.long(), num_classes=2).float().permute(2, 0, 1)
        embedding = self.embedding.seq2emb(sequence)
        conditioning = self.embedding.outer_emb(embedding)

        return {
            "id": seq_id,
            "sequence": sequence,
            "length": length,
            "embedding": embedding,
            "conditioning": conditioning,
            "contact": contact,
            "contact_oh": contact_oh,
        }

##  (REQUIRES base_pairs or dot-bracket column)
def _ensure_base_pairs_column(dataframe):
    if "base_pairs" in dataframe.columns:
        if dataframe["base_pairs"].isna().any():
            raise ValueError("Column base_pairs contains missing values")
        return dataframe

    structure_column = None
    for column_name in ("structure", "dotbracket"):
        if column_name in dataframe.columns:
            structure_column = column_name
            break
    if structure_column is None:
        raise ValueError(
            "Dataset must include base_pairs or a dot-bracket column (structure/dotbracket)"
        )

    base_pairs = []
    for index, structure in enumerate(dataframe[structure_column]):
        if not isinstance(structure, str):
            raise ValueError(
                f"Expected string in {structure_column} at row {index}, got {type(structure).__name__}"
            )
        parsed = dot2bp(structure)
        if parsed is False:
            raise ValueError(
                f"Could not parse dot-bracket string in {structure_column} at row {index}: {structure}"
            )
        base_pairs.append(str(parsed))
    copy_dataframe = dataframe.copy()
    copy_dataframe["base_pairs"] = base_pairs
    return copy_dataframe

def pad_batch(batch):
    """Batch is a dictionary with different variable lists."""

    L = [b["length"] for b in batch]
    raw_max_len = max(L)
    # Ceil max length to a multiple of 4 to keep downsampling valid.
    max_len = math.ceil(raw_max_len / 4) * 4 
    embedding_pad = tr.zeros((len(batch), batch[0]["embedding"].shape[0], max_len)) 
    conditioning_pad = tr.zeros((len(batch), batch[0]["conditioning"].shape[0], max_len, max_len))# mask (B, 1, L, L)
    mask_pad = tr.zeros((len(batch), 1, max_len, max_len), dtype=tr.bool)

    if batch[0]["contact"] is None:
        contact_pad = None
        contact_oh_pad = None
    else: 
        # make 
        contact_pad = tr.zeros((len(batch), max_len, max_len))
        contact_oh_pad = tr.zeros((len(batch), 2, max_len, max_len))
        contact_oh_pad[:, 0, :, :] = 1  # default to no contact

    for k in range(len(batch)):
        embedding_pad[k, :, : L[k]] = batch[k]["embedding"]
        conditioning_pad[k, :, : L[k], : L[k]] = batch[k]["conditioning"]

        mask_pad[k, :, :L[k], :L[k]] = True
        if contact_pad is not None:
            contact_pad[k, : L[k], : L[k]] = batch[k]["contact"]
            contact_oh_pad[k, :, : L[k], : L[k]] = batch[k]["contact_oh"]

    out_batch = {"id": [b["id"] for b in batch],
                 "sequence": [b["sequence"] for b in batch],
                 "length": L,
                 "embedding": embedding_pad,
                 "conditioning": conditioning_pad,
                 "contact": contact_pad,
                 "contact_oh": contact_oh_pad,
                 "mask": mask_pad}

    return out_batch 

def build_dataloader(config, partition, batch_size=None, shuffle=False):
    data_config = config["data"]
    training_config = config["training"]
    dataset = SeqDataset(
        base_path=data_config["base_path"],
        partition_path=data_config["partition_path"],
        partition_value=partition,
        fold_number=data_config["fold"],
    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size or training_config["batch_size"],
        shuffle=shuffle,
        collate_fn=pad_batch,
        num_workers=training_config["num_workers"],
    )


class RNADataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        data_config = self.config["data"]
        dataset_kwargs = {
            "base_path": data_config["base_path"],
            "partition_path": data_config["partition_path"],
            "fold_number": data_config["fold"],
        }

        if stage in (None, "fit"):
            self.train_dataset = SeqDataset(partition_value="train", **dataset_kwargs)
            self.val_dataset = SeqDataset(partition_value="valid", **dataset_kwargs)

        if stage in (None, "test", "predict"):
            self.test_dataset = SeqDataset(partition_value="test", **dataset_kwargs)

    def _build_loader(self, dataset, shuffle):
        training_config = self.config["training"]
        return DataLoader(
            dataset=dataset,
            batch_size=training_config["batch_size"],
            shuffle=shuffle,
            collate_fn=pad_batch,
            num_workers=training_config["num_workers"],
        )

    def train_dataloader(self):
        return self._build_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._build_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._build_loader(self.test_dataset, shuffle=False)
