import torch as tr
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContactModel(nn.Module):
    """
    Supervised wrapper for contact-map prediction.

    It mirrors the diffusion wrapper API used by the training/evaluation code:
    - forward_all_timesteps(target, condition, mask=None) -> scalar loss
    - _sample(condition) -> class indices [B, H, W]
    """

    def __init__(self, model, **kwargs):
        super().__init__()
        self.backbone = model(**kwargs)

    def forward(self, condition):
        return self.backbone(condition)

    def forward_all_timesteps(self, x0_oh, condition, mask=None):
        """
        Compute masked CE loss.

        Args:
            x0_oh: One-hot target tensor [B, C, H, W]
            condition: Model input tensor [B, in_channels, H, W]
            mask: Optional padding mask [B, 1, H, W], where 1 is valid.
        """
        logits = self.backbone(condition)  # [B, C, H, W]
        target_idx = x0_oh.argmax(dim=1).long()  # [B, H, W]

        loss = F.cross_entropy(logits, target_idx, reduction="none")  # [B, H, W]

        if mask is not None:
            mask_2d = mask.squeeze(1).float()  # [B, H, W]
            return (loss * mask_2d).sum() / (mask_2d.sum() + 1e-8)

        return loss.mean()

    @tr.no_grad()
    def _sample(self, condition):
        logits = self.backbone(condition)
        return logits.argmax(dim=1)
