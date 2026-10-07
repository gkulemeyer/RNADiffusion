import torch as tr
from src.diffusion import DiffusionModel

class DiffusionModelWithLoss(DiffusionModel):
    """
    Extending DiffusionModel to include a regularization loss term based on the KL divergence 
    between the predicted and true distributions.
    """

    def __init__(
        self,
        balance_bp=None, # class | linar_convex | none
        w=0.02,
        lambda_valid=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.balance_bp = balance_bp  # None | "class" | "linear" | "linear_convex" | "convex"
        self.w = w  # weight for positive class in the loss
        self.lambda_valid = lambda_valid  # weight for invalid pairs in the loss

    def class_weights(self, lengths):         
        if self.balance_bp is None:
            return 1.0, 1.0

        if self.balance_bp == "class":
            return self.w , 1 - self.w
        
        if self.balance_bp == "linear_convex":  # convex: w+ + w- = 1
            # w = params.get("w", 1.84) * lengths 
            lengths = tr.as_tensor(lengths, dtype=tr.float32, device=self.alphas.device)
            L =  self.w * 1.84 * lengths
            return L / (1 + L), 1 / (1 + L)

    def valid_pair_mask(self, conditioning): 
        L = conditioning.shape[-1]
        val_gu = conditioning[:, [11, 14], ...].sum(dim=1).bool()
        val_gc = conditioning[:, [6, 9], ...].sum(dim=1).bool()
        val_au = conditioning[:, [3, 12], ...].sum(dim=1).bool()
        valid  = val_au | val_gc | val_gu        
        canonical = ~ tr.eye(L, device=conditioning.device, dtype=tr.bool)
        for i in range(3):
            canonical[i+ tr.arange(L-i),    tr.arange(L-i)] = False
            canonical[   tr.arange(L-i), i+ tr.arange(L-i)] = False  
        return  valid & canonical 
        
    def kl_prior(self, x0_oh, lengths=None):
        eps = 1e-8

        batch_size = x0_oh.shape[0]
        device     = x0_oh.device
        ones       = tr.ones(batch_size, device=device).long()
        qxT        = self.q_pred(x0_oh, t=(self.time_steps - 1) * ones)
        half_prob  = tr.ones_like(qxT) / self.num_classes 
        p = tr.clamp(qxT, min=eps, max=1.0)
        q = tr.clamp(half_prob, min=eps, max=1.0)

        # Compute KL divergence
        kl = p * (tr.log(p) - tr.log(q))
        kl_pixelwise = tr.sum(kl, dim=1)  # [B, L, L]
        
        if lengths is None:
            raise ValueError("lengths is required to compute the loss.")
        
        len_mask = self._lengths_to_mask(lengths,
                                        kl_pixelwise.shape[-1],
                                        device=kl_pixelwise.device
                                        ).squeeze(1)     # [B, L, L]
    
        kl_masked = kl_pixelwise * len_mask # [B, L, L]
    
        return kl_masked.sum(dim=(-1,-2)) / len_mask.sum(dim=(-1,-2))  # [B] 

    def multinomial_kl(self, p, q, condition, contact_oh, lengths=None):
        eps = 1e-8

        p = tr.clamp(p, min=eps, max=1.0)
        q = tr.clamp(q, min=eps, max=1.0)
        kl_pixelwise = (p * (tr.log(p) - tr.log(q))).sum(dim=1)  # [B, L, L]

        if lengths is None:
            raise ValueError("lengths is required to compute the loss.")

        # region: cropped structures, to normalize
        region = self._lengths_to_mask(lengths, kl_pixelwise.shape[-1], device=kl_pixelwise.device).squeeze(1)

        KL_invalid = 0.0
        if self.lambda_valid > 0:
            valid = self.valid_pair_mask(condition)
            invalid = region & ~valid  # inside the region but not valid
            KL_invalid = (
                self.lambda_valid
                * (kl_pixelwise * invalid).sum(dim=(-1, -2))
                / (invalid.sum(dim=(-1, -2)) + eps)
            )
            region = region & valid  

        target = contact_oh[:, 1].bool()
        contact = region & target
        not_contact = region & ~target
        n_pos = contact.sum(dim=(-1, -2))
        n_neg = not_contact.sum(dim=(-1, -2))

        if self.balance_bp == "class":
            # normalize by the valid class pairs in the region
            pos_den, neg_den = n_pos, n_neg
        else:
            # normalize by the total valid pairs (matrix) in the region
            pos_den = neg_den = region.sum(dim=(-1, -2))

        kl_pos = (kl_pixelwise * contact).sum(dim=(-1, -2)) / (pos_den + eps)
        kl_neg = (kl_pixelwise * not_contact).sum(dim=(-1, -2)) / (neg_den + eps)
        self.kl_pos, self.kl_neg = kl_pos.detach().mean().item(), kl_neg.detach().mean().item()

        w_pos, w_neg = self.class_weights(lengths)
        L_pos = w_pos * kl_pos
        L_neg = w_neg * kl_neg  
        return L_pos + L_neg + KL_invalid

    
    def compute_vlb(self, contact_oh, xt, t, condition, lengths=None, **kwargs):
            """
            Calcula VLB considerando la máscara de padding.
            mask: Tensor [B, 1, L, L] (1=Valido, 0=Padding)
            """
            # 1. Posterior Real
            true_posterior = self.q_posterior(contact_oh, xt, t) 
            
            # 2. Posterior Predicha
            pred_x0_probs = self.predict_start(xt, t, condition, lengths=lengths, return_logits=False)
            pred_posterior = self.q_posterior(pred_x0_probs, xt, t)
            
            return self.multinomial_kl(true_posterior,
                                        pred_posterior,
                                        condition, 
                                        contact_oh, 
                                        lengths=lengths)
        