import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import math


def linear_betas(timesteps):
    betas = tr.linspace(0.0001, 0.01, timesteps, dtype=tr.float32)
    return betas

def cosine_betas(timesteps, s=0.02):
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    Better for discrete data to prevent abrupt noise injection.
    """
    steps = timesteps + 1
    x = tr.linspace(0, timesteps, steps, dtype=tr.float32)
    alphas_cumprod = tr.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = tr.clip(betas, 0, 0.999)
    betas[0] = 1e-7
    return betas

def get_schedule(timesteps, get_betas, log=False):
    # beta = 1 - alpha
    betas = get_betas(timesteps)
    alphas = 1 - betas
    alphas_bar = tr.cumprod(alphas, dim=0)
    one_minus_alphas_bar = 1 - alphas_bar
    if log:
        return tr.log(betas), tr.log(alphas) , tr.log(alphas_bar), tr.log(one_minus_alphas_bar)
    else:
        return betas, alphas, alphas_bar, one_minus_alphas_bar

def extract(a, t, x_shape):
    """Extract values from `a` at indices `t` and reshape to match `x_shape`."""
    batch_size = t.shape[0]
    out = a.gather(-1, t)  # Extract values at indices `t`
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))  # Reshape to match `x_shape`

# ref https://github.com/ehoogeboom/multinomial_diffusion/blob/main/diffusion_utils/diffusion_multinomial.py
# ref 2 https://github.com/lucidrains/denoising-diffusion-pytorch/blob/5989f4c77eafcdc6be0fb4739f0f277a6dd7f7d8/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py

class MultinomialDiffusionModel(nn.Module):
    def __init__(self, num_classes, time_steps, model, schedule="cosine", **kwargs):
        super().__init__()
        self.diffuser = model(**kwargs)
        self.num_classes = num_classes
        self.time_steps = time_steps

        # about buffers: https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723
        # Register buffers with shape (T, 1, 1, 1).
        # This enables direct broadcasting over [B, C, H, W] without .view().
        if schedule == "linear":
            beta_fn = linear_betas
        else:
            beta_fn = cosine_betas
        betas, alphas, alphas_bar, one_minus_alphas_bar = get_schedule(time_steps, beta_fn)
        self.register_buffer("one_minus_alphas", 1 - alphas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_bar", alphas_bar)
        self.register_buffer("one_minus_alphas_bar", one_minus_alphas_bar)

        # Optional buffer for stochastic NLL (if needed).

    def q_pred(self, x0, t):
        """Given x0 and time t, compute q(xt|x0).

        x0: tensor (batch_size, channels, height, width) with integer values in [0, num_classes-1]
        t: tensor (batch_size,) with integer values in [0, time_steps-1]
        return: tensor (batch_size, height, width) with integer values in [0, num_classes-1]
        """

        # Add 0/1 class to the contact map
        alphas_bar = extract(self.alphas_bar, t, x0.shape)
        one_minus_alphas_bar = extract(self.one_minus_alphas_bar, t, x0.shape)

        # qt is a mixture of the one-hot distribution and a uniform distribution
        # q(xt|x0)
        probs = alphas_bar * x0 + one_minus_alphas_bar / self.num_classes
        return probs

    def q_step(self, xt_1, t):
        """Given xt_1 and time t, compute q(xt|xt_1).

        xt_1: tensor (batch_size, height, width) with integer values in [0, num_classes-1]
        t: tensor (batch_size,) with integer values in [0, time_steps-1]
        return: tensor (batch_size, height, width) with integer values in [0, num_classes-1]
        """

        # Add 0/1 class to the contact map
        # Convert indices to one-hot and move channels: [B, L, L, 2] -> [B, 2, L, L]

        if xt_1.dim() == 4:
            xt_1_one_hot = xt_1  # Already [B, C, H, W]
        else:
            xt_1_one_hot = F.one_hot(xt_1, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        # alphas_t must be (batch_size, 1, 1, 1) for broadcasting to work
        alphas_t = extract(self.alphas, t, xt_1_one_hot.shape)
        one_minus_alpha_t = extract(self.one_minus_alphas, t, xt_1_one_hot.shape)
        # qt is a mixture of the one-hot distribution and a uniform distribution
        qxt = alphas_t * xt_1_one_hot + one_minus_alpha_t / self.num_classes
        return qxt

    def q_posterior(self, x0, xt, t):
        """Given xt, x0 and time t, compute q(xt-1|xt,x0).

        xt: tensor (batch_size, height, width) with integer values in [0, num_classes-1]
        x0: tensor (batch_size, height, width) with integer values in [0, num_classes-1]
        t: tensor (batch_size,) with integer values in [0, time_steps-1]
        return: tensor (batch_size, height, width, num_classes) with per-class probabilities
        """

        # Case 1: indices [B, L, L] -> needs one-hot
        if x0.dim() == 3:
             x0_vec = F.one_hot(x0.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        # Case 2: already one-hot or probabilities [B, 2, L, L] -> ensure float
        elif x0.dim() == 4:
             x0_vec = x0.float()

        # q(xt-1 | xt, x0) = q(xt | xt-1, x0) * q(xt-1 | x0) / q(xt | x0)
        # where q(xt | xt-1, x0) = q(xt | xt-1).

        t_1 = tr.clamp(t - 1, min=0)  # t-1 but not below 0
        qxt_1_given_x0 = self.q_pred(x0_vec, t_1)  # q(xt-1|x0)

        qxt_1_given_x0 = tr.where(t.view(-1, 1, 1, 1) == 0,
                                  x0_vec,
                                  qxt_1_given_x0
                                  )  # If t=0, then xt-1 = x0
        # q(xt|xt-1)
        # Mathematical note: in Hoogeboom, q(xt|xt-1) is symmetric.
        # We can reuse q_step by passing xt as the base.
        qxt_given_xt_1 = self.q_step(xt, t)
        posterior = qxt_1_given_x0 * qxt_given_xt_1  # p(xt-1|xt,x0)
        # Normalize to a probability distribution over channels (dim 1)
        posterior = posterior / (posterior.sum(dim=1, keepdim=True) + 1e-8)
        return posterior

    def predict_start(self, xt, t, condition, return_logits=False):
            if xt.dim() == 4:  # [B, C, H, W] -> Gumbel
                xt_input = xt
            else:  # [B, H, W] -> one-hot
                xt_input = F.one_hot(xt, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

            unet_input = tr.cat([xt_input, condition], dim=1)
            out = self.diffuser(unet_input, t)

            if return_logits:
                return out
            return F.softmax(out, dim=1)

    def pred_p_xt_1_from_xt(self, xt, t, condition):
        pred = self.predict_start(xt, t, condition)
        return self.q_posterior(pred, xt, t)

    def sample_from_logits(self, probs):

        # Reshape so multinomial works
        batch_size, num_classes, height, width = probs.shape
        probs = probs.permute(0, 2, 3, 1)  # [B, H, W, C]
        probs_flat = probs.reshape(-1, num_classes)
        probs_flat = tr.clamp(probs_flat, min=0.0)
        probs_flat = probs_flat + 1e-6
        probs_flat = probs_flat / probs_flat.sum(dim=-1, keepdim=True)

        # Sample from logits distribution
        sampled = tr.multinomial(probs_flat, num_samples=1).squeeze(-1)
        # Restore original shape
        sampled = sampled.reshape(batch_size, height, width)
        return sampled

    def q_sample(self, x0_oh, t, gumbel=True, temperature=1.0):
        # x0_oh: [B, num_classes, H, W] (must be one-hot float)

        # 1. Get q(xt|x0) probabilities
        qxt_probs = self.q_pred(x0_oh, t)
        qxt_probs = tr.clamp(qxt_probs, min=1e-20, max=1.0)
        if gumbel:
            # Gumbel-Softmax: returns differentiable soft tensor [B, C, H, W]
            # epsilon to avoid log(0)
            eps = 1e-30
            logits = tr.log(qxt_probs + eps)
            # hard=False to keep it differentiable
            return F.gumbel_softmax(logits, tau=temperature, hard=False, dim=1)
        else:
            # Standard sampling (indices) - non-differentiable
            probs_perm = qxt_probs.permute(0, 2, 3, 1) # [B, H, W, C]
            sample_idx = tr.distributions.Categorical(probs_perm).sample()
            return sample_idx

    @tr.no_grad()
    def p_sample(self, xt, t, condition):
        # Get posterior probabilities [B, 2, L, L]
        posterior_probs = self.pred_p_xt_1_from_xt(xt, t, condition)
        # Sample from predicted distribution
        out = self.sample_from_logits(posterior_probs)
        return out

    @tr.no_grad()
    def p_sample_loop(self, shape, condition):
        """Sample an image from pure noise."""
        batch_size = shape[0]
        device = self.alphas.device
        # Start from pure (uniform) noise
        xt = tr.randint(0, self.num_classes, shape, device=device).long()

        for t in reversed(range(0, self.time_steps)):
            t_batch = tr.full((batch_size,), t, device=device, dtype=tr.long)
            xt = self.p_sample(xt, t_batch, condition)
        return xt

    @tr.no_grad()
    def _sample(self, condition):
        shape = (condition.shape[0], condition.shape[2], condition.shape[3])  # batch_size, height, width
        samples = self.p_sample_loop(shape, condition)
        return samples

    def compute_vlb(self, x0_oh, xt, t, condition, mask=None):
            """
            Compute VLB while considering the padding mask.
            mask: Tensor [B, 1, L, L] (1=valid, 0=padding)
            """
            # 1. True posterior
            true_posterior = self.q_posterior(x0_oh, xt, t)

            # 2. Predicted posterior
            pred_x0_probs = self.predict_start(xt, t, condition, return_logits=False)
            pred_posterior = self.q_posterior(pred_x0_probs, xt, t)

            # 3. KL divergence
            eps = 1e-8
            true_posterior = tr.clamp(true_posterior, min=eps, max=1.0)
            pred_posterior = tr.clamp(pred_posterior, min=eps, max=1.0)

            kl = true_posterior * (tr.log(true_posterior) - tr.log(pred_posterior))
            kl_pixelwise = tr.sum(kl, dim=1) # [B, L, L]

            # --- Apply mask ---
            if mask is not None:
                # Ensure mask dimensions: [B, 1, L, L] -> [B, L, L]
                mask_s = mask.squeeze(1)

                # Zero out loss on padding
                kl_pixelwise = kl_pixelwise * mask_s

                # Average only over valid pixels
                # Total sum / number of valid pixels
                return kl_pixelwise.sum() / (mask_s.sum() + 1e-8)

            return kl_pixelwise.mean()

    def forward_all_timesteps(self, x0_oh, condition, mask=None):
        """
        Calculate the total loss by summing VLB over all timesteps.
        """
        batch_size = x0_oh.shape[0]
        device = x0_oh.device
        total_loss = 0

        # Bucle sobre todos los timesteps (0 a T-1)
        for t_step in range(self.time_steps):
            # Crear batch de tiempos constantes para este paso
            t = tr.full((batch_size,), t_step, device=device).long()

            # 1. Sample xt
            xt = self.q_sample(x0_oh, t, gumbel=True, temperature=1.0)

            # 2. Compute VLB for this timestep
            # compute_vlb calcula la KL( Posterior Real || Posterior Predicha )
            loss_t = self.compute_vlb(x0_oh, xt, t, condition, mask=mask)

            total_loss += loss_t

        return total_loss
