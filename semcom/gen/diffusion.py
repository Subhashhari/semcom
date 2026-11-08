"""Gaussian diffusion: the noise schedule, the training loss and a DDIM sampler.

Linear beta schedule over T = 1000 steps, as in DDPM and DiT. The network predicts either the
noise ("eps", DDPM/DiT) or the velocity v = sqrt(a_bar) eps - sqrt(1 - a_bar) x0 ("v", Salimans &
Ho 2022). With eps, at high noise the target is almost the input itself, so the loss barely
depends on the received symbols there, yet those steps set the image's layout; v keeps the clean
latent in the target at every noise level. Sampling uses
DDIM over an evenly spaced subset of the timesteps, so one trained model can be decoded with any
number of steps: eta = 0 is deterministic DDIM, eta = 1 recovers DDPM-like ancestral sampling.
The number of steps is the receiver's main cost, which is why evaluation sweeps it.

Every random draw takes an explicit generator, so evaluation can give every arm the same
sampling noise on the same image (paired comparisons, as in semcom.blind_eval).
"""

from __future__ import annotations

from typing import Callable

import torch


class Diffusion:
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02,
                 prediction: str = "eps"):
        if prediction not in ("eps", "v"):
            raise ValueError(f"prediction must be 'eps' or 'v', got {prediction!r}")
        self.T, self.prediction = num_timesteps, prediction
        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)
        self.alpha_bar = torch.cumprod(1.0 - betas, dim=0).float()

    def _ab(self, t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        return self.alpha_bar.to(like.device)[t].view(-1, *([1] * (like.dim() - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """x_t = sqrt(a_bar_t) x0 + sqrt(1 - a_bar_t) noise."""
        ab = self._ab(t, x0)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def target(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """What the network is trained to output at (x0, noise, t)."""
        if self.prediction == "eps":
            return noise
        ab = self._ab(t, x0)
        return ab.sqrt() * noise - (1 - ab).sqrt() * x0

    def split(self, out: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Network output -> (predicted noise, predicted clean latent)."""
        ab = self._ab(t, x_t)
        if self.prediction == "eps":
            return out, (x_t - (1 - ab).sqrt() * out) / ab.sqrt()
        return (1 - ab).sqrt() * x_t + ab.sqrt() * out, ab.sqrt() * x_t - (1 - ab).sqrt() * out

    def training_targets(self, x0: torch.Tensor, generator: torch.Generator | None = None):
        """(x_t, t, noise) for one training step: t uniform over all T steps."""
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device, generator=generator)
        noise = torch.randn(x0.shape, device=x0.device, generator=generator, dtype=x0.dtype)
        return self.q_sample(x0, t, noise), t, noise

    def timesteps(self, steps: int) -> list[int]:
        """Evenly spaced timesteps from T-1 down to 0, `steps` of them."""
        if not 1 <= steps <= self.T:
            raise ValueError(f"steps must be in [1, {self.T}], got {steps}")
        return torch.linspace(self.T - 1, 0, steps).round().long().unique_consecutive().tolist()

    @torch.no_grad()
    def sample(
        self,
        eps_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        shape: tuple[int, ...],
        steps: int,
        device: torch.device,
        generator: torch.Generator | None = None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """DDIM from pure noise to x0. `eps_fn(x_t, t)` returns the network output (the noise
        for prediction "eps", the velocity for "v")."""
        x = torch.randn(shape, device=device, generator=generator)
        ts = self.timesteps(steps)
        ab_all = self.alpha_bar.to(device)
        for i, t in enumerate(ts):
            ab = ab_all[t]
            ab_prev = ab_all[ts[i + 1]] if i + 1 < len(ts) else torch.ones((), device=device)
            tt = torch.full((shape[0],), t, device=device, dtype=torch.long)
            eps, x0 = self.split(eps_fn(x, tt).float(), x, tt)
            sigma = eta * ((1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev)).clamp_min(0).sqrt()
            x = ab_prev.sqrt() * x0 + (1 - ab_prev - sigma**2).clamp_min(0).sqrt() * eps
            if eta > 0 and i + 1 < len(ts):
                x = x + sigma * torch.randn(shape, device=device, generator=generator)
        return x
