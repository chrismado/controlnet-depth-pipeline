"""
Gaussian diffusion utilities for training and sampling.

Implements the DDPM forward and reverse processes (Ho et al., 2020):

Forward process (q):
    q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t)I)

Reverse process (p):
    p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ²_t I)

where:
    μ_θ = 1/√α_t (x_t - β_t/√(1-ᾱ_t) · ε_θ(x_t, t))
    σ²_t = β̃_t = β_t · (1-ᾱ_{t-1}) / (1-ᾱ_t)

The model predicts noise ε_θ, and training minimises ||ε - ε_θ||².
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def _extract(schedule: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    """Index into a 1-D schedule tensor and reshape for broadcasting.

    Args:
        schedule: (T,) precomputed schedule values.
        t: (B,) integer timestep indices.
        x_shape: Shape of the tensor to broadcast against.

    Returns:
        (B, 1, 1, 1) extracted values ready for element-wise ops with images.
    """
    batch_size = t.shape[0]
    out = schedule.gather(0, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion(nn.Module):
    """DDPM diffusion process with linear beta schedule.

    Precomputes all schedule-derived constants as registered buffers so they
    move with the model across devices and are included in state_dict.

    Args:
        num_timesteps: Total diffusion steps T.
        beta_start: β_1 (smallest noise level).
        beta_end: β_T (largest noise level).
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps

        # Linear beta schedule
        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        # ᾱ_{t-1}: pad with 1.0 at front so index 0 gives ᾱ_0 = 1
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Cast to float32 for use with models
        def _register(name: str, val: torch.Tensor) -> None:
            self.register_buffer(name, val.float())

        _register("betas", betas)
        _register("alphas", alphas)
        _register("alphas_cumprod", alphas_cumprod)
        _register("alphas_cumprod_prev", alphas_cumprod_prev)

        # Precomputed coefficients for q_sample
        _register("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        _register("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())

        # Precomputed coefficients for p_sample (reverse mean)
        # μ = 1/√α_t · (x_t − β_t/√(1−ᾱ_t) · ε_θ)
        _register("recip_sqrt_alphas", (1.0 / alphas.sqrt()))
        _register("beta_over_sqrt_one_minus_alpha_bar", betas / (1.0 - alphas_cumprod).sqrt())

        # Posterior variance: β̃_t = β_t · (1−ᾱ_{t-1}) / (1−ᾱ_t)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        _register("posterior_variance", posterior_variance)
        # Clamp log variance for numerical stability (variance is 0 at t=0)
        _register(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward diffusion: add noise to clean image.

        x_t = √ᾱ_t · x_0  +  √(1−ᾱ_t) · ε

        Args:
            x0: (B, C, H, W) clean images.
            t: (B,) timestep indices in [0, T).
            noise: Optional pre-sampled noise; generated if None.

        Returns:
            (B, C, H, W) noised images x_t.
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_bar = _extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus = _extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def p_losses(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        t: torch.Tensor,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        """Compute simplified diffusion training loss (MSE on predicted noise).

        Args:
            model: Denoising network (UNet). Called as model(x_t, t, **model_kwargs).
            x0: (B, C, H, W) clean images.
            t: (B,) random timestep indices.
            **model_kwargs: Extra keyword args forwarded to model (e.g. controlnet_residuals).

        Returns:
            Scalar MSE loss.
        """
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise=noise)
        predicted_noise = model(x_t, t, **model_kwargs)
        return F.mse_loss(predicted_noise, noise)

    # ------------------------------------------------------------------
    # Reverse process (sampling)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t_index: int,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        """Single reverse diffusion step: x_t → x_{t-1}.

        Uses the DDPM posterior:
            μ_θ = 1/√α_t · (x_t − β_t/√(1−ᾱ_t) · ε_θ(x_t, t))
            x_{t-1} = μ_θ + σ_t · z,  z ~ N(0, I) for t > 0

        Args:
            model: Denoising network.
            x_t: (B, C, H, W) current noisy sample.
            t_index: Scalar timestep index (same for entire batch).
            **model_kwargs: Extra keyword args forwarded to model.

        Returns:
            (B, C, H, W) slightly denoised sample x_{t-1}.
        """
        batch_size = x_t.shape[0]
        t = torch.full((batch_size,), t_index, device=x_t.device, dtype=torch.long)

        # Predict noise
        predicted_noise = model(x_t, t, **model_kwargs)

        # Compute posterior mean
        recip_sqrt_alpha = _extract(self.recip_sqrt_alphas, t, x_t.shape)
        beta_coeff = _extract(self.beta_over_sqrt_one_minus_alpha_bar, t, x_t.shape)
        mean = recip_sqrt_alpha * (x_t - beta_coeff * predicted_noise)

        if t_index > 0:
            variance = _extract(self.posterior_variance, t, x_t.shape)
            noise = torch.randn_like(x_t)
            return mean + variance.sqrt() * noise
        else:
            return mean

    @torch.no_grad()
    def sample_loop(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        device: torch.device | str = "cpu",
        show_progress: bool = True,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        """Full reverse diffusion chain: pure noise → generated image.

        Iterates from t = T-1 down to t = 0, applying p_sample at each step.

        Args:
            model: Denoising network.
            shape: Output tensor shape, e.g. (B, 3, 128, 128).
            device: Device for sampling.
            show_progress: Whether to display a tqdm progress bar.
            **model_kwargs: Extra keyword args forwarded to model.

        Returns:
            (B, C, H, W) generated images (in model's output range).
        """
        x = torch.randn(shape, device=device)
        timesteps = reversed(range(self.num_timesteps))

        if show_progress:
            timesteps = tqdm(timesteps, desc="Sampling", total=self.num_timesteps)

        for t_index in timesteps:
            x = self.p_sample(model, x, t_index, **model_kwargs)

        return x

    # ------------------------------------------------------------------
    # DDIM sampling (Song et al., 2020)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        device: torch.device | str = "cpu",
        ddim_steps: int = 50,
        eta: float = 0.0,
        show_progress: bool = True,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        """DDIM accelerated sampling.

        Uses a sub-sequence of the full diffusion schedule to generate images
        in fewer steps. When eta=0 the process is fully deterministic.

        Args:
            model: Denoising network.
            shape: Output tensor shape, e.g. (B, 3, 128, 128).
            device: Device for sampling.
            ddim_steps: Number of denoising steps (< num_timesteps).
            eta: Stochasticity parameter. 0 = deterministic DDIM, 1 ≈ DDPM.
            show_progress: Whether to display a tqdm progress bar.
            **model_kwargs: Extra keyword args forwarded to model.

        Returns:
            (B, C, H, W) generated images.
        """
        # Build sub-sequence of timesteps evenly spaced across [0, T)
        step_size = self.num_timesteps // ddim_steps
        timesteps = list(range(0, self.num_timesteps, step_size))  # ascending
        timesteps = list(reversed(timesteps))  # descending for sampling

        x = torch.randn(shape, device=device)
        iterator = timesteps
        if show_progress:
            iterator = tqdm(iterator, desc=f"DDIM ({ddim_steps} steps)")

        for i, t_cur in enumerate(iterator):
            batch_size = x.shape[0]
            t = torch.full((batch_size,), t_cur, device=device, dtype=torch.long)

            # Predict noise
            predicted_noise = model(x, t, **model_kwargs)

            # Current and previous alpha_bar
            alpha_bar_t = _extract(self.alphas_cumprod, t, x.shape)
            if i + 1 < len(timesteps):
                t_prev = timesteps[i + 1]
                t_prev_tensor = torch.full((batch_size,), t_prev, device=device, dtype=torch.long)
                alpha_bar_prev = _extract(self.alphas_cumprod, t_prev_tensor, x.shape)
            else:
                # Last step: α̅_0 = 1
                alpha_bar_prev = torch.ones_like(alpha_bar_t)

            # Predict x_0
            sqrt_alpha_bar_t = alpha_bar_t.sqrt()
            sqrt_one_minus_alpha_bar_t = (1.0 - alpha_bar_t).sqrt()
            x0_pred = (x - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t

            # DDIM update
            sigma = (
                eta
                * ((1 - alpha_bar_prev) / (1 - alpha_bar_t)).sqrt()
                * (1 - alpha_bar_t / alpha_bar_prev).sqrt()
            )
            dir_xt = (1.0 - alpha_bar_prev - sigma**2).clamp(min=0).sqrt() * predicted_noise
            x = alpha_bar_prev.sqrt() * x0_pred + dir_xt

            if sigma.sum() > 0:
                x = x + sigma * torch.randn_like(x)

        return x
