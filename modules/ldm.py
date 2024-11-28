import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def linear_beta_schedule(timestep, timescale):
    scale = timescale / timestep
    beta_start = scale * 0.0001
    beta_end = scale * 0.02

    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timestep, dtype=torch.float64) ** 2

def modified_beta_schedule(timestep, timescale):
    scale = timescale / timestep
    beta_start = scale * 0.00005
    beta_end = scale * 0.025
    delta = (1 - (torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timestep, dtype=torch.float64) ** 2 - beta_start) /
             (beta_end - beta_start))
    q_delta = delta ** 0.925
    betas = (1 - q_delta) * (beta_end - beta_start) + beta_start

    return betas

def cosine_beta_schedule(timestep, cosine_s=5e-4):
    timesteps = (
                torch.arange(timestep + 1, dtype=torch.float64) / timestep + cosine_s
        )
    alphas = timesteps / (1 + cosine_s) * np.pi / 2
    alphas = torch.cos(alphas).pow(2)
    alphas = alphas / alphas[0]
    betas = 1 - alphas[1:] / alphas[:-1]
    betas = np.clip(betas, a_min=0, a_max=0.999)

    return betas


class GaussianDiffusion:
    def __init__(self, timestep=250, timescale=250, beta_schedule='cosine'):
        self.timestep = timestep

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timestep, timescale)
        elif beta_schedule == 'modified':
            betas = modified_beta_schedule(timestep, timescale)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timestep)

        self.betas = betas

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)

        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1.)

        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = ((1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) /
                                     (1. - self.alphas_cumprod))

    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
                self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def p_mean_variance(self, model, x_t, context, t):
        noise_pred = model(x_t, context, t)
        x_pred = self.predict_start_from_noise(x_t, t, noise_pred)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(x_pred, x_t, t)

        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, model, x_t, context, t):
        recip_sqrt_alpha_t = self._extract(1. / torch.sqrt(self.alphas), t, x_t.shape)
        beta_t = self._extract(self.betas, t, x_t.shape)
        recip_sqrt_m1_alphas_cumprod_t = self._extract(1 / torch.sqrt(1. - self.alphas_cumprod), t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(torch.sqrt(1. - self.alphas_cumprod), t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t_1 = self._extract(torch.sqrt(1. - self.alphas_cumprod), t - 1,
                                                          x_t.shape) if t > 0 else 0

        mask = torch.randn_like(x_t)
        noise_pred = model(x_t, context, t)

        x_t_1 = (recip_sqrt_alpha_t * (x_t - beta_t * recip_sqrt_m1_alphas_cumprod_t * noise_pred) +
                 sqrt_one_minus_alphas_cumprod_t_1 / sqrt_one_minus_alphas_cumprod_t * beta_t * mask)

        return x_t_1

    @torch.no_grad()
    def p_sample_loop(self, model, x, context, x_scale=1, focus=False, timestep=1000, time_skip=None):
        device = context.device

        if time_skip is not None:
            # time_skip = 200 if time_skip > 200 else time_skip
            total_time = self.timestep - time_skip
            timestep = total_time if timestep > total_time else timestep
            t = torch.full(size=(x.shape[0],), fill_value=self.timestep - 1, device=device)
            t1 = 850 + time_skip // 2
            t2 = 850 - time_skip + time_skip // 2
            step_1 = int(timestep * (self.timestep - t1) / total_time)
            step_2 = timestep - step_1

            interval = (self.timestep - t1 - 1) / (step_1 - 1)
            int_numbers = reversed([t1 + i * interval for i in range(step_1)])
            int_numbers = [int(round(num)) for num in int_numbers]

            x_pred = torch.randn_like(x)
            x_preds = []

            if step_1 != 0:
                for i in tqdm(range(step_1), desc='Sampling', total=step_1):
                    x_pred = self.p_sample(model, x_pred, context, t)
                    t = torch.full(size=(x.shape[0],), fill_value=int_numbers[i], device=device)
                    if focus:
                        x_preds.append(x_pred * x_scale)

            interval = (t2 - 1) / (step_2 - 1)
            int_numbers = reversed([i * interval for i in range(step_2)])
            int_numbers = [int(round(num)) for num in int_numbers]

            t = torch.full(size=(x.shape[0],), fill_value=t2 - 1, device=device)
            u = self.sqrt_one_minus_alphas_cumprod[t2] / self.sqrt_one_minus_alphas_cumprod[t1]
            a = self.sqrt_alphas_cumprod[t2] - self.sqrt_alphas_cumprod[t1] * u
            x_pred = a * x + u * x_pred

            for i in tqdm(range(step_2), desc='Sampling', total=step_2):
                x_pred = self.p_sample(model, x_pred, context, t)
                t = torch.full(size=(x.shape[0],), fill_value=int_numbers[i], device=device)
                if focus:
                    x_preds.append(x_pred * x_scale)

        else:
            t = torch.full(size=(x.shape[0],), fill_value=self.timestep - 1, device=device)
            x_pred = x
            x_preds = []

            interval = (self.timestep - 1) / (timestep - 1)
            int_numbers = reversed([i * interval for i in range(timestep)])
            int_numbers = [int(round(num)) for num in int_numbers]

            for i in tqdm(range(timestep), desc='Sampling', total=timestep):
                x_pred = self.p_sample(model, x_pred, context, t)
                if focus:
                    x_preds.append(x_pred * x_scale)

                t = torch.full(size=(x.shape[0],), fill_value=int_numbers[i], device=device)

        x_pred = x_pred * x_scale

        return x_pred, x_preds

# k = GaussianDiffusion(timestep=1000, timescale=1000)
# a = k.sqrt_one_minus_alphas_cumprod / k.sqrt_alphas_cumprod
# print(a[600] + 1)

