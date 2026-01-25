import pandas as pd
import math
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch as tr
import os
import json
import pickle
from .embeddings import OneHotEmbedding
from .utils import valid_mask, prob_mat, bp2matrix, dot2bp

class SeqDataset(Dataset):
    def __init__(
        self, dataset_path, min_len=0, max_len=512, verbose=False,  for_prediction=False, training=False, **kargs):
        """
        interaction_prior: none, probmat
        """
        self.max_len = max_len
        self.verbose = verbose   

        # Loading dataset
        data = pd.read_csv(dataset_path)
        self.training = training

        if for_prediction:
            assert (
                "sequence" in data.columns
                and "id" in data.columns
            ), "Dataset should contain 'id' and 'sequence' columns"

        else:
            assert (
                ("base_pairs" in data.columns or "dotbracket" in data.columns)
                and "sequence" in data.columns
                and "id" in data.columns
            ), "Dataset should contain 'id', 'sequence' and 'base_pairs' or 'dotbracket' columns"

            if "base_pairs" not in data.columns and "dotbracket" in data.columns:
                data["base_pairs"] = data.dotbracket.apply(lambda x: str(dot2bp(x)))      

        data["len"] = data.sequence.str.len()

        if max_len is None:
            max_len = max(data.len)
        self.max_len = max_len

        datalen = len(data)
        data = data[(data.len >= min_len) & (data.len <= max_len)]
        if len(data) < datalen:
            print(
                f"From {datalen} sequences, filtering {min_len} < len < {max_len} we have {len(data)} sequences"
            )

        self.sequences = data.sequence.tolist()
        self.ids = data.id.tolist()
        self.embedding = OneHotEmbedding()
        self.embedding_size = self.embedding.emb_size

        self.base_pairs = None
        if "base_pairs" in data.columns:
            self.base_pairs = [
                json.loads(data.base_pairs.iloc[i]) for i in range(len(data))
            ]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seqid = self.ids[idx]
    
        sequence = self.sequences[idx]
        L = len(sequence)
        Mc = None
        if self.base_pairs is not None:
            Mc = bp2matrix(L, self.base_pairs[idx])
        Mc_OH = F.one_hot(Mc.long(), num_classes=2).float().permute(2, 0, 1) if Mc is not None else None

        seq_emb = self.embedding.seq2emb(sequence)
        outer = self.embedding.outer_emb(seq_emb)
      
        item = {"embedding": seq_emb, "contact": Mc, "contact_oh": Mc_OH,"outer" : outer, "length": L,
                "id": seqid, "sequence": sequence} 
                
        return item

def pad_batch(batch):
    """batch is a dictionary with different variables lists"""
    
    L = [b["length"] for b in batch]
    raw_max_len = max(L)
    # ceil the max len multiple of 4, to garantee downsampling works on batchs
    max_len = math.ceil(raw_max_len / 4) * 4
    embedding_pad = tr.zeros((len(batch), batch[0]["embedding"].shape[0], max_len))
    outer_padded = tr.zeros((len(batch), batch[0]["outer"].shape[0], max_len, max_len))# mask (B, 1, L, L)
    mask_pad = tr.zeros((len(batch), 1, max_len, max_len))

    if batch[0]["contact"] is None:
        contact_pad = None
    else:
        contact_pad = -tr.ones((len(batch), max_len, max_len), dtype=tr.long)
        contact_oh_pad = -tr.ones((len(batch), 2, max_len, max_len), dtype=tr.long)
    
    for k in range(len(batch)):
        embedding_pad[k, :, : L[k]] = batch[k]["embedding"]
        outer_padded[k, :, : L[k], : L[k]] = batch[k]["outer"]
        
        mask_pad[k, :, :L[k], :L[k]] = 1.0
        if contact_pad is not None:
            contact_pad[k, : L[k], : L[k]] = batch[k]["contact"]
            contact_oh_pad[k, :, : L[k], : L[k]] = batch[k]["contact_oh"]

    out_batch = {"contact": contact_pad, 
                 "contact_oh": contact_oh_pad,
                 "embedding": embedding_pad, 
                 "outer": outer_padded,
                 "length": L, 
                 "mask": mask_pad,
                 "sequence": [b["sequence"] for b in batch],
                 "id": [b["id"] for b in batch]}
    
    return out_batch
