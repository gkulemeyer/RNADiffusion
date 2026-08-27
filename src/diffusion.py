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
    """ Extrae los valores correspondientes de 'a' en los índices 't' y los reformatea para que coincidan con 'x_shape' """
    batch_size = t.shape[0]
    out = a.gather(-1, t)  # Extrae los valores en los índices 't'
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))  # Reformatea para que coincida con 'x_shape'

# ref https://github.com/ehoogeboom/multinomial_diffusion/blob/main/diffusion_utils/diffusion_multinomial.py
# ref 2 https://github.com/lucidrains/denoising-diffusion-pytorch/blob/5989f4c77eafcdc6be0fb4739f0f277a6dd7f7d8/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py

class DiffusionModel(nn.Module):
    def __init__(self, num_classes, time_steps, model, loss_type="vb_all", **kwargs):
        super().__init__() 
        assert loss_type in ('vb_stochastic', 'vb_all')
        self.diffuser = model(**kwargs)
        self.num_classes = num_classes
        self.time_steps = time_steps
        self.loss_type = loss_type
        
        # about buffers: https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723
        # 2. Registramos los buffers con shape (T, 1, 1, 1)
        # Esto permite multiplicar directo por imágenes [B, C, H, W] sin hacer .view() en cada forward
        betas, alphas, alphas_bar, one_minus_alphas_bar = get_schedule(time_steps, cosine_betas)
        self.register_buffer("one_minus_alphas", 1 - alphas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_bar", alphas_bar)
        self.register_buffer("one_minus_alphas_bar", one_minus_alphas_bar)
        
        # buffer de Lt para nll estocastico 
        if loss_type == 'vb_stochastic':
            self.register_buffer('Lt_history', tr.zeros(time_steps))
            self.register_buffer('Lt_count', tr.zeros(time_steps))
        

    def x_to_one_hot(self, x, from_one_hot = True):
        if from_one_hot:
            return x.float()
        else:
            # x is [B, H, W], convert to one-hot [B, C, H, W]
            return F.one_hot(x, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

    def _lengths_to_mask(self, lengths, padded_length, device=None):
        if lengths is None:
            return None
        if device is None:
            device = self.alphas.device
        lengths_t = tr.as_tensor(lengths, device=device, dtype=tr.long)
        rng = tr.arange(padded_length, device=device)
        valid_1d = rng[None, :] < lengths_t[:, None]
        valid_2d = valid_1d[:, :, None] & valid_1d[:, None, :]
        return valid_2d.unsqueeze(1)

    def _prepare_backbone_inputs(self, xt_input, condition, lengths=None):
        xt_input = self.x_to_one_hot(xt_input, False)
        if lengths is None:
            return xt_input, condition

        mask = self._lengths_to_mask(lengths, xt_input.shape[-1], device=xt_input.device)
        xt_fill = tr.zeros_like(xt_input) # [B, C, H, W]
        xt_fill[:, 0, :, :] = 1.0
        valid_xt = mask.expand_as(xt_input)
        valid_cond = mask.expand_as(condition)

        xt_backbone = tr.where(valid_xt, xt_input, xt_fill)
        cond_backbone = tr.where(valid_cond, condition, tr.zeros_like(condition))
        return xt_backbone, cond_backbone
    
    def _gumbel_sample(self, logits, dim=1):
        uniform = tr.rand_like(logits)
        gumbel_noise = -tr.log(-tr.log(uniform + 1e-30) + 1e-30)
        sampled = (gumbel_noise + logits).argmax(dim=dim) # [B, L, L]
        return sampled
    
    def sample_categorical(self, logits, lengths=None):
        if lengths is None:
            return  self._gumbel_sample(logits, dim=1)
             

        B, C, L, _ = logits.shape
        out = tr.zeros(B, L, L, device=logits.device, dtype=tr.long)

        for b, l in enumerate(lengths):
            l = int(l)
            logits_b = logits[b, :, :l, :l]
            out[b, :l, :l] = self._gumbel_sample(logits_b, dim=0)
        return out
    
    
    def sample_from_probs(self, probs, lengths=None):
        # probs: [B, C, H, W]
        probs = tr.clamp(probs, min=1e-30)
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)
        logits = tr.log(probs) 
        return self.sample_categorical(logits, lengths=lengths)  #  [B,H,W]
    
    
    def q_pred(self, x0, t):
        """ Dada una imagen x0 y un tiempo t, obtiene q(xt|x0)
        x0: tensor de shape (batch_size, channels, height, width) con valores enteros entre 0 y num_classes-1
        t: tensor de shape (batch_size,) con valores enteros entre 0 y time_steps-1
        return: tensor de shape (batch_size, height, width) con valores enteros entre 0 y num_classes-1            
        """
    
        # le agrego al mapa de contactos la clase 0/1        
        alphas_bar = extract(self.alphas_bar, t, x0.shape)
        one_minus_alphas_bar = extract(self.one_minus_alphas_bar, t, x0.shape)
        
        # La distribución qt es una mezcla entre la distribución one-hot y la distribución uniforme
        # q(xt|x0) 
        probs = alphas_bar * x0 + one_minus_alphas_bar / self.num_classes
        return probs # [B, C, H, W]
    
    def q_step(self, xt_1, t):
        """ Dada una imagen xt_1 y un tiempo t, obtiene q(xt|xt_1)
        xt_1: tensor de shape (batch_size, height, width) con valores enteros entre 0 y num_classes-1
        t: tensor de shape (batch_size,) con valores enteros entre 0 y time_steps-1
        return: tensor de shape (batch_size, height, width) con valores enteros entre 0 y num_classes-1            
        """
    
        # le agrego al mapa de contactos la clase 0/1
        xt_1_one_hot = self.x_to_one_hot(xt_1, from_one_hot=False)
 
        # el vector alphas_t tiene que tener shape (batch_size, 1, 1, 1) para que la multiplicacion de matrices funcione
        alphas_t = extract(self.alphas, t, xt_1_one_hot.shape)
        one_minus_alpha_t = extract(self.one_minus_alphas, t, xt_1_one_hot.shape)
        # La distribución qt es una mezcla entre la distribución one-hot y la distribución uniforme
        qxt = alphas_t * xt_1_one_hot + one_minus_alpha_t / self.num_classes
        return qxt  # [B, C, H, W]

    def q_posterior(self, x0_oh, xt, t):
        """ Dada una imagen xt, una x0 y un tiempo t, calcula la distribución q(xt-1|xt,x0)
        xt: tensor de shape (batch_size, height, width) con valores enteros entre 0 y num_classes-1
        x0: tensor de shape (batch_size, height, width) con valores enteros entre 0 y num_classes-1
        t: tensor de shape (batch_size,) con valores enteros entre 0 y time_steps-1
        return: tensor de shape (batch_size, height, width, num_classes) con las probabilidades de cada clase en cada pixel           
        """
        
        x0_one_hot = self.x_to_one_hot(x0_oh, from_one_hot=True)
        
        # q(xt-1 | xt, x0) = q(xt | xt-1, x0) * q(xt-1 | x0) / q(xt | x0)
        # where q(xt | xt-1, x0) = q(xt | xt-1).
                
        t_1 = tr.clamp(t-1, min=0) # t-1, pero no menor a 0
        qxt_1_given_x0 = self.q_pred(x0_one_hot, t_1)  # q(xt-1|x0)
        
        qxt_1_given_x0 = tr.where(t.view(-1, 1, 1, 1) == 0,
                                  x0_one_hot,
                                  qxt_1_given_x0
                                  )  # Si t=0, entonces xt-1 = x0
        # q(xt|xt-1)
        # NOTA MATEMÁTICA: En Hoogeboom q(xt|xt-1) es simétrico. 
        # Podemos reusar la logica de q_step pasando xt como "base"
        qxt_given_xt_1 = self.q_step(xt, t)
        posterior = qxt_1_given_x0 * qxt_given_xt_1  # p(xt-1|xt,x0)        
        # Normalizo para que sea una distribución de probabilidad
        # sobre la dimensión de canales (dim 1)
        posterior = posterior / (posterior.sum(dim=1, keepdim=True) + 1e-8)
        return posterior 
        

    def predict_start(self, xt, t, condition, lengths=None, return_logits=False): 
            # takes xt [B,H,W] and condition [B,C,H,W]
            xt_input, condition = self._prepare_backbone_inputs(xt, condition, lengths=lengths)
             # now xt_input is [B,C,H,W] one-hot, with padding handled, and condition is also padded
            unet_input = tr.cat([xt_input, condition], dim=1)
            out = self.diffuser(unet_input, t)

            if return_logits:
                return out
            return F.softmax(out, dim=1)
  
    
    def pred_p_xt_1_from_xt(self, xt, t, condition, lengths=None, return_logits=False):
        if return_logits:
            logits = self.predict_start(xt, t, condition, lengths=lengths, return_logits=True)
            pred = F.softmax(logits, dim=1)
        else:
            pred = self.predict_start(xt, t, condition, lengths=lengths)

        posterior = self.q_posterior(pred, xt, t)
        if return_logits:
            return posterior, logits
        
        return posterior
    
    def q_sample(self, x0_oh, t, lengths=None):
        qxt_probs = self.q_pred(x0_oh, t)
        qxt_probs = tr.clamp(qxt_probs, min=1e-20, max=1.0) 
        return self.sample_from_probs(qxt_probs, lengths=lengths)

    @tr.no_grad()
    def p_sample(self, xt, t, condition, lengths=None, return_logits=False):
        # Get posterior probabilities [B, 2, L, L] considering the mask 
        if return_logits:
            posterior_probs, logits = self.pred_p_xt_1_from_xt(xt, t, condition, lengths=lengths, return_logits=True)
        else:   
            posterior_probs = self.pred_p_xt_1_from_xt(xt, t, condition, lengths=lengths)

        # Sample from the predicted distribution considering the mask
        out = self.sample_from_probs(posterior_probs, lengths=lengths)

        if return_logits:
            return out, logits

        return out


    @tr.no_grad()
    def p_sample_loop(self, shape, condition, lengths=None, return_logits=False):
        """Sample an image from pure noise."""
        batch_size = shape[0]
        device = self.alphas.device
        # start from pure noise (uniform distribution)
        xt = tr.randint(0, self.num_classes, shape, device=device).long()

        for t in reversed(range(0, self.time_steps)):
            t_batch = tr.full((batch_size,), t, device=device, dtype=tr.long)
            if return_logits and t == 0:
                xt, logits = self.p_sample(xt, t_batch, condition, lengths=lengths, return_logits=True)
                return xt, logits
            xt = self.p_sample(xt, t_batch, condition, lengths=lengths)
        return xt

    
    @tr.no_grad()
    def sample(self, condition, lengths=None, return_logits=False):
        # batch_size, height, width, dim=1 -> channel
        shape = (condition.shape[0], condition.shape[2], condition.shape[3])
        samples = self.p_sample_loop(shape, condition, lengths=lengths, return_logits=return_logits)
        return samples
    

    def multinomial_kl(self, p, q, lengths=None):
        eps = 1e-8
        p = tr.clamp(p, min=eps, max=1.0)
        q = tr.clamp(q, min=eps, max=1.0)
        
        kl = p * (tr.log(p) - tr.log(q))
        kl_pixelwise = tr.sum(kl, dim=1)  # [B, L, L]
        
        if lengths is not None:
            valid_means = []
            for b, l in enumerate(lengths):
                l = int(l)
                valid_means.append(kl_pixelwise[b, :l, :l].mean())
            return tr.stack(valid_means)

        return kl_pixelwise

    def compute_vlb(self, x0_oh, xt, t, condition, lengths=None):
            """
            Calcula VLB considerando la máscara de padding.
            mask: Tensor [B, 1, L, L] (1=Valido, 0=Padding)
            """
            # 1. Posterior Real
            true_posterior = self.q_posterior(x0_oh, xt, t) 
            
            # 2. Posterior Predicha
            pred_x0_probs = self.predict_start(xt, t, condition, lengths=lengths, return_logits=False)
            pred_posterior = self.q_posterior(pred_x0_probs, xt, t)
            
            return self.multinomial_kl(true_posterior, pred_posterior, lengths=lengths)
        

    def kl_prior(self, x0_oh, lengths=None):
        batch_size = x0_oh.shape[0]
        device = x0_oh.device
        ones = tr.ones(batch_size, device=device).long()

        qxT = self.q_pred(x0_oh, t= (self.time_steps - 1) * ones)  # q(xT|x0) 
        half_prob = tr.ones_like(qxT) / self.num_classes # p(xT) uniform distribution
        return self.multinomial_kl(qxT, half_prob, lengths=lengths)

    def vb_all(self, x0_oh, condition, lengths=None):
        """
        Calculate the total Loss adding all VLB for each timestep.
        """
        batch_size = x0_oh.shape[0]
        device = x0_oh.device
        total_loss = 0
        
        # Bucle sobre todos los timesteps (0 a T-1)
        for t_step in range(self.time_steps):
            # Crear batch de tiempos constantes para este paso
            t = tr.full((batch_size,), t_step, device=device).long()
            # 1. sample xt ~ q(xt|x0) with gumbel-softmax 
            xt = self.q_sample(x0_oh, t, lengths=lengths) # [B,H,W] 
            # calculate KL on step t ( Posterior GT || Posterior pred )
            loss_t = self.compute_vlb(x0_oh, xt, t, condition, lengths=lengths)
            total_loss += loss_t 
        # add time 0, kl prior between q(xT|x0) and p(xT) (uniform distribution)
        total_loss += self.kl_prior(x0_oh, lengths=lengths)
        return total_loss.mean()
    
    def sample_time(self, b, device, method='uniform'):
        if method == 'uniform':
            t = tr.randint(0, self.time_steps, (b,), device=device).long()
            pt = tr.ones_like(t).float() / self.time_steps
            return t, pt
        
        elif method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = tr.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1] 
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = tr.multinomial(pt_all, num_samples=b, replacement=True)
            pt = pt_all.gather(dim=0, index=t)
            return t, pt

        else:
            raise ValueError

    def vb_stochastic(self, x0_oh, condition, lengths=None):
        
        batch_size = x0_oh.shape[0]
        device = x0_oh.device

        if self.loss_type == 'vb_stochastic': 

            t, pt = self.sample_time(batch_size, device, 'importance') 
            xt = self.q_sample(x0_oh, t, lengths=lengths)
            loss_t = self.compute_vlb(x0_oh, xt, t, condition, lengths=lengths)             
            kl_prior = self.kl_prior(x0_oh, lengths=lengths)
            # Upweigh loss term of the kl
            vb_loss = loss_t / pt + kl_prior

            Lt2 = loss_t.pow(2)
            Lt2_prev = self.Lt_history.gather(dim=0, index=t)
            new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
            self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
            self.Lt_count.scatter_add_(dim=0, index=t, src=tr.ones_like(Lt2))

            return vb_loss.mean()

    def _train_loss(self, x0_oh, condition, lengths=None):
        if self.loss_type == 'vb_all':
            return self.vb_all(x0_oh, condition, lengths=lengths)
        
        elif self.loss_type == 'vb_stochastic':
            return self.vb_stochastic(x0_oh, condition, lengths=lengths)

        else:
            raise ValueError()