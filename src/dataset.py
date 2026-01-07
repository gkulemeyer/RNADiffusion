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
        self, dataset_path, min_len=0, max_len=512, verbose=False, cache_path=None, for_prediction=False, 
        interaction_prior="probmat", use_cannonical_mask=False, training=False,
 **kargs):
        """
        interaction_prior: none, probmat
        """
        self.max_len = max_len
        self.verbose = verbose
        if cache_path is not None and not os.path.isdir(cache_path):
            os.mkdir(cache_path)
        self.cache = cache_path

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
        self.interaction_prior = interaction_prior
        self.use_cannonical_mask = use_cannonical_mask

        self.base_pairs = None
        if "base_pairs" in data.columns:
            self.base_pairs = [
                json.loads(data.base_pairs.iloc[i]) for i in range(len(data))
            ]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seqid = self.ids[idx]
        cache = f"{self.cache}/{seqid}.pk"
        if (self.cache is not None) and os.path.isfile(cache):
            item = pickle.load(open(cache, "rb"))
        else:
            sequence = self.sequences[idx]
            L = len(sequence)
            Mc = None
            if self.base_pairs is not None:
                Mc = bp2matrix(L, self.base_pairs[idx])
            Mc_OH = F.one_hot(Mc.long(), num_classes=2).float().permute(2, 0, 1) if Mc is not None else None

            seq_emb = self.embedding.seq2emb(sequence)
            outer = self.embedding.outer_emb(seq_emb)
            mask = None
            if self.use_cannonical_mask:
                mask = valid_mask(sequence)
            interaction_prior = None
            if self.interaction_prior == "probmat":
                interaction_prior = prob_mat(sequence)
            item = {"embedding": seq_emb, "contact": Mc, "contact_oh": Mc_OH,"outer" : outer, "length": L, "canonical_mask": mask,
                    "id": seqid, "sequence": sequence, "interaction_prior": interaction_prior} 

            if self.cache is not None:
                pickle.dump(item, open(cache, "wb"))
                
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
    
    if batch[0]["canonical_mask"] is None:
        canonical_mask_pad = None
    else:
        canonical_mask_pad = tr.zeros((len(batch), max_len, max_len))
    
    interaction_prior_pad = None
    if batch[0]["interaction_prior"] is not None:
        interaction_prior_pad = tr.zeros((len(batch), max_len, max_len))

    for k in range(len(batch)):
        embedding_pad[k, :, : L[k]] = batch[k]["embedding"]
        outer_padded[k, :, : L[k], : L[k]] = batch[k]["outer"]
        # valid mask
        mask_pad[k, :, :L[k], :L[k]] = 1.0
        if contact_pad is not None:
            contact_pad[k, : L[k], : L[k]] = batch[k]["contact"]
            contact_oh_pad[k, :, : L[k], : L[k]] = batch[k]["contact_oh"]
        if canonical_mask_pad is not None:
            canonical_mask_pad[k, : L[k], : L[k]] = batch[k]["canonical_mask"]
        
        if interaction_prior_pad is not None:
            interaction_prior_pad[k, : L[k], : L[k]] = batch[k]["interaction_prior"]

    out_batch = {"contact": contact_pad, 
                 "contact_oh": contact_oh_pad,
                 "embedding": embedding_pad, 
                 "outer": outer_padded,
                 "length": L, 
                 "mask": mask_pad,
                 "canonical_mask": canonical_mask_pad,
                 "interaction_prior": interaction_prior_pad,
                 "sequence": [b["sequence"] for b in batch],
                 "id": [b["id"] for b in batch]}
    
    return out_batch
