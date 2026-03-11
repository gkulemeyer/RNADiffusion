import math

import torch as tr
import torch.nn as nn
import torch.nn.functional as F


def linear_betas(timesteps):
    betas = tr.linspace(0.0001, 0.01, timesteps, dtype=tr.float32)
    return betas


def cosine_betas(timesteps, s=0.02):
    steps = timesteps + 1
    x = tr.linspace(0, timesteps, steps, dtype=tr.float32)
    alphas_cumprod = tr.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = tr.clip(betas, 0, 0.999)
    betas[0] = 1e-7
    return betas


def get_schedule(timesteps, get_betas, log=False):
    betas = get_betas(timesteps)
    alphas = 1 - betas
    alphas_bar = tr.cumprod(alphas, dim=0)
    one_minus_alphas_bar = 1 - alphas_bar
    if log:
        return tr.log(betas), tr.log(alphas), tr.log(alphas_bar), tr.log(one_minus_alphas_bar)
    return betas, alphas, alphas_bar, one_minus_alphas_bar


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class DiffusionModel(nn.Module):
    def __init__(self, num_classes, time_steps, model, **kwargs):
        super().__init__()
        self.diffuser = model(**kwargs)
        self.num_classes = num_classes
        self.time_steps = time_steps

        betas, alphas, alphas_bar, one_minus_alphas_bar = get_schedule(time_steps, cosine_betas)
        self.register_buffer("one_minus_alphas", 1 - alphas )
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_bar", alphas_bar)
        self.register_buffer("one_minus_alphas_bar", one_minus_alphas_bar)

    def q_pred(self, x0, t):
        alphas_bar = extract(self.alphas_bar, t, x0.shape)
        one_minus_alphas_bar = extract(self.one_minus_alphas_bar, t, x0.shape)
        probs = alphas_bar * x0 + one_minus_alphas_bar / self.num_classes
        return probs

    def q_step(self, xt_1, t):
        if xt_1.dim() == 4:
            xt_1_one_hot = xt_1
        else:
            xt_1_one_hot = F.one_hot(xt_1, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        alphas_t = extract(self.alphas, t, xt_1_one_hot.shape)
        one_minus_alpha_t = extract(self.one_minus_alphas, t, xt_1_one_hot.shape)
        qxt = alphas_t * xt_1_one_hot + one_minus_alpha_t / self.num_classes
        return qxt

    def q_posterior(self, x0, xt, t):
        if x0.dim() == 3:
            x0_vec = F.one_hot(x0.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        elif x0.dim() == 4:
            x0_vec = x0.float()
        else:
            raise ValueError(f"Unexpected x0 dimensions: {x0.dim()}")

        t_1 = tr.clamp(t - 1, min=0)
        qxt_1_given_x0 = self.q_pred(x0_vec, t_1)
        qxt_1_given_x0 = tr.where(t.view(-1, 1, 1, 1) == 0, x0_vec, qxt_1_given_x0)
        qxt_given_xt_1 = self.q_step(xt, t)
        posterior = qxt_1_given_x0 * qxt_given_xt_1
        posterior = posterior / (posterior.sum(dim=1, keepdim=True) + 1e-8)
        return posterior

    def predict_start(self, xt, t, condition, return_logits=False):
        if xt.dim() == 4:
            xt_input = xt
        else:
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
        batch_size, num_classes, height, width = probs.shape
        probs = probs.permute(0, 2, 3, 1)
        probs_flat = probs.reshape(-1, num_classes)
        probs_flat = tr.clamp(probs_flat, min=0.0)
        probs_flat = probs_flat + 1e-6
        probs_flat = probs_flat / probs_flat.sum(dim=-1, keepdim=True)
        sampled = tr.multinomial(probs_flat, num_samples=1).squeeze(-1)
        sampled = sampled.reshape(batch_size, height, width)
        return sampled

    def q_sample(self, x0_oh, t, gumbel=True, temperature=1.0):
        qxt_probs = self.q_pred(x0_oh, t)
        qxt_probs = tr.clamp(qxt_probs, min=1e-20, max=1.0)
        if gumbel:
            logits = tr.log(qxt_probs + 1e-30)
            return F.gumbel_softmax(logits, tau=temperature, hard=False, dim=1)

        probs_perm = qxt_probs.permute(0, 2, 3, 1)
        sample_idx = tr.distributions.Categorical(probs_perm).sample()
        return sample_idx

    @tr.no_grad()
    def p_sample(self, xt, t, condition):
        posterior_probs = self.pred_p_xt_1_from_xt(xt, t, condition)
        out = self.sample_from_logits(posterior_probs)
        return out

    @tr.no_grad()
    def p_sample_loop(self, shape, condition):
        batch_size = shape[0]
        device = self.alphas.device
        xt = tr.randint(0, self.num_classes, shape, device=device).long()

        for t in reversed(range(0, self.time_steps)):
            t_batch = tr.full((batch_size,), t, device=device, dtype=tr.long)
            xt = self.p_sample(xt, t_batch, condition)
        return xt

    @tr.no_grad()
    def _sample(self, condition):
        shape = (condition.shape[0], condition.shape[2], condition.shape[3])
        samples = self.p_sample_loop(shape, condition)
        return samples

    def compute_vlb(self, x0_oh, xt, t, condition, mask=None):
        true_posterior = self.q_posterior(x0_oh, xt, t)
        pred_x0_probs = self.predict_start(xt, t, condition, return_logits=False)
        pred_posterior = self.q_posterior(pred_x0_probs, xt, t)

        eps = 1e-8
        true_posterior = tr.clamp(true_posterior, min=eps, max=1.0)
        pred_posterior = tr.clamp(pred_posterior, min=eps, max=1.0)

        kl = true_posterior * (tr.log(true_posterior) - tr.log(pred_posterior))
        kl_pixelwise = tr.sum(kl, dim=1)

        if mask is not None:
            mask_s = mask.squeeze(1)
            kl_pixelwise = kl_pixelwise * mask_s
            return kl_pixelwise.sum() / (mask_s.sum() + 1e-8)

        return kl_pixelwise.mean()

    def forward_all_timesteps(self, x0_oh, condition, mask=None):
        batch_size = x0_oh.shape[0]
        device = x0_oh.device
        total_loss = 0

        for t_step in range(self.time_steps):
            t = tr.full((batch_size,), t_step, device=device).long()
            xt = self.q_sample(x0_oh, t, gumbel=True, temperature=1.0)
            loss_t = self.compute_vlb(x0_oh, xt, t, condition, mask=mask)
            total_loss += loss_t

        return total_loss
