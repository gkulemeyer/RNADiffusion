import torch as tr 
from sklearn.metrics import f1_score

def contact_f1(pred_batch, ref_batch, lengths=None, reduce=True):
    """Compute F1 from base pairs. (Triangular matrix)"""
    f1_scores = []
    if pred_batch.ndim == 4: pred_batch = pred_batch.argmax(dim=1)
    if ref_batch.ndim == 4:  ref_batch = ref_batch.argmax(dim=1)        
    batch_size, max_len = pred_batch.shape[0], pred_batch.shape[-1]
    if lengths is None: lengths = [max_len] * batch_size
    ref_batch, pred_batch = ref_batch.cpu(), pred_batch.cpu()
    
    for ref, pred, l in zip(ref_batch, pred_batch, lengths): 
        # ignore padding
        ref_valid = ref[:l, :l]
        pred_valid = pred[:l, :l]          
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