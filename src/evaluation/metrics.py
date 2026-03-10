import torch as tr
from sklearn.metrics import f1_score

def contact_f1(pred_batch, ref_batch, lengths=None, reduce=True):
    """Compute F1 from base pairs. (Triangular matrix)"""
    f1_scores = []
    if pred_batch.ndim == 4:
        pred_batch = pred_batch.argmax(dim=1)

    if ref_batch.ndim == 4:
        ref_batch = ref_batch.argmax(dim=1)

    batch_size = pred_batch.shape[0]
    max_len = pred_batch.shape[-1]

    if lengths is None:
        lengths = [max_len] * batch_size

    ref_batch, pred_batch = ref_batch.cpu(), pred_batch.cpu()

    for ref, pred, length in zip(ref_batch, pred_batch, lengths):
        # ignore padding
        ref_valid = ref[:length, :length]
        pred_valid = pred[:length, :length]
        f1 = f1_triangular(ref_valid, pred_valid)
        f1_scores.append(f1)
    if reduce:
        return tr.tensor(f1_scores).mean().item()
    else:
        return tr.tensor(f1_scores)

def f1_triangular(ref, pred):
    """Compute F1 from the upper triangular connection matrix"""
    # get upper triangular matrix without diagonal
    ind = tr.triu_indices(ref.shape[0], ref.shape[1], offset=1)

    ref = ref[ind[0], ind[1]].numpy().ravel()
    pred = pred[ind[0], ind[1]].numpy().ravel()
    return f1_score(ref, pred, zero_division=0)


def contact_f1_gpu(pred_batch, ref_batch, lengths=None, reduce=True, eps=1e-8):
    """
    Compute F1 from base pairs (Triangular matrix).
    Optimized for GPU: Vectorized, no loops, no CPU transfer.
    """
    if pred_batch.ndim == 4:
        pred_batch = pred_batch.argmax(dim=1)
    if ref_batch.ndim == 4:
        ref_batch = ref_batch.argmax(dim=1)
    pred_batch = pred_batch.float()
    ref_batch = ref_batch.float()

    batch_size, L, _ = pred_batch.shape
    device = pred_batch.device

    # Handle Lengths (Convert to Tensor for masking)
    if lengths is None:
        lengths = tr.full((batch_size,), L, device=device, dtype=tr.long)
    elif isinstance(lengths, list):
        lengths = tr.tensor(lengths, device=device, dtype=tr.long)
    else:
        lengths = lengths.to(device)

    # Upper Triangular Mask
    triu_mask = tr.ones((L, L), device=device).triu(diagonal=1).bool()
    rng = tr.arange(L, device=device) # Mask [B, L]: True if index < length
    len_mask_1d = rng[None, :] < lengths[:, None] # Mask [B, L, L]: True if both i and j are within valid length
    len_mask_2d = len_mask_1d[:, :, None] & len_mask_1d[:, None, :]
    valid_mask = triu_mask & len_mask_2d

    p = pred_batch * valid_mask.float()
    t = ref_batch * valid_mask.float()

    # Sum over dimensions (1, 2) i.e., Height and Width
    tp = (p * t).sum(dim=(1, 2))
    fp = (p * (1 - t)).sum(dim=(1, 2))
    fn = ((1 - p) * t).sum(dim=(1, 2))
    f1_scores = (2 * tp) / (2 * tp + fp + fn + eps)

    if reduce:
        return f1_scores.mean().item()
    else:
        return f1_scores