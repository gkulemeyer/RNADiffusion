"""Collate functions for DataLoaders."""
import math
import torch as tr

def pad_batch(batch):
    """Batch is a dictionary with different variable lists."""

    L = [b["length"] for b in batch]
    raw_max_len = max(L)
    # Ceil max length to a multiple of 4 to keep downsampling valid.
    max_len = math.ceil(raw_max_len / 4) * 4
    embedding_pad = tr.zeros((len(batch), batch[0]["embedding"].shape[0], max_len))
    outer_padded = tr.zeros((len(batch), batch[0]["outer"].shape[0], max_len, max_len))# mask (B, 1, L, L)
    mask_pad = tr.zeros((len(batch), 1, max_len, max_len))

    if batch[0]["contact"] is None:
        contact_pad = None
        contact_oh_pad = None
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

    out_batch = {"id": [b["id"] for b in batch],
                 "sequence": [b["sequence"] for b in batch],
                 "length": L,
                 "embedding": embedding_pad,
                 "outer": outer_padded,
                 "contact": contact_pad,
                 "contact_oh": contact_oh_pad,
                 "mask": mask_pad}

    return out_batch
