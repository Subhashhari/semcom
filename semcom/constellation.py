"""M-QAM constellation and the DeepJSCC-Q soft-to-hard quantiser.

The quantiser is what makes a finite channel alphabet trainable. Hard nearest-neighbour
assignment in the forward pass (so the network is always trained on the *true* transmitted
signal), a temperature-annealed softmax relaxation in the backward pass (so gradients
exist at all).

Reference: Tung, Kurka, Jankowski, Gunduz, "DeepJSCC-Q: Constellation Constrained Deep
Joint Source-Channel Coding," arXiv:2206.08100.
"""

import math

import torch
import torch.nn as nn


def qam_constellation(
    order: int, avg_power: float = 1.0, dtype=torch.float32, device=None
) -> torch.Tensor:
    """Return a square M-QAM constellation as a real (M, 2) tensor of (I, Q) pairs.

    Points sit on a uniform square lattice, scaled so that the average power under a
    *uniform* distribution over the alphabet equals `avg_power`. This fixed scale is what
    the encoder latent must be power-normalised to before quantisation.

    Args:
        order: constellation order M; must be a perfect square power of two (4, 16, 64, ...).
        avg_power: target average symbol power under uniform usage.
    """
    if order < 4 or (order & (order - 1)) != 0:
        raise ValueError(f"order must be a power of two >= 4, got {order}")

    side = int(round(math.sqrt(order)))
    if side * side != order:
        raise ValueError(f"order must be a perfect square (square QAM), got {order}")

    # Odd-integer lattice: +/-1, +/-3, ... per dimension, then scaled to hit avg_power.
    levels = torch.arange(side, dtype=dtype, device=device) * 2 - (side - 1)
    i_grid, q_grid = torch.meshgrid(levels, levels, indexing="ij")
    points = torch.stack([i_grid.reshape(-1), q_grid.reshape(-1)], dim=1)

    current_power = points.pow(2).sum(dim=1).mean()
    return points * math.sqrt(avg_power) / current_power.sqrt()


class SoftToHardQuantiser(nn.Module):
    """Quantise complex latents to a fixed M-QAM alphabet, differentiably.

    Latents are carried as a real tensor of shape (..., 2) representing (I, Q) pairs.

    Forward pass emits exact constellation points. Backward pass uses the gradient of the
    softmax-weighted convex combination of *all* points, weighted by negative squared
    distance and sharpened by the inverse temperature `sigma_q`.

    `sigma_q` is annealed upward during training via `step_anneal()`: early on the
    relaxation is soft and the encoder can explore; by the end the surrogate closely
    matches the discrete reality the encoder actually faces.
    """

    def __init__(
        self,
        order: int,
        avg_power: float = 1.0,
        sigma_q_init: float = 5.0,
        sigma_q_max: float = 100.0,
        anneal_period: int = 10_000,
        anneal_step: float = 5.0,
    ):
        super().__init__()
        self.order = int(order)
        self.sigma_q_max = float(sigma_q_max)
        self.anneal_period = int(anneal_period)
        self.anneal_step = float(anneal_step)

        self.register_buffer("points", qam_constellation(order, avg_power))
        self.register_buffer("sigma_q", torch.tensor(float(sigma_q_init)))
        self.register_buffer("_iteration", torch.tensor(0, dtype=torch.long))

        # Empirical symbol distribution from the most recent forward pass, for the KL
        # regulariser. Populated by forward(); read by kl_to_uniform().
        self._last_usage: torch.Tensor | None = None

    def step_anneal(self) -> float:
        """Advance the annealing schedule by one training iteration.

        Follows the DeepJSCC-Q schedule literally:
            sigma_q <- min(sigma_q_max, sigma_q + anneal_step * floor(t / anneal_period))

        Note this is an *accumulating* increment, so sigma_q stays flat through the first
        `anneal_period` iterations and then climbs increasingly fast. It saturates at
        `sigma_q_max` quickly once it starts moving; that is the schedule as published.
        """
        self._iteration += 1
        increment = self.anneal_step * (self._iteration.item() // self.anneal_period)
        if increment > 0:
            self.sigma_q.fill_(min(self.sigma_q_max, self.sigma_q.item() + increment))
        return self.sigma_q.item()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Quantise `z` of shape (..., 2) to the constellation.

        Returns a tensor of the same shape whose forward values are exactly constellation
        points, carrying the soft relaxation's gradient.
        """
        if z.shape[-1] != 2:
            raise ValueError(f"expected trailing dim 2 for (I, Q), got {tuple(z.shape)}")

        flat = z.reshape(-1, 2)
        dist = self._squared_distances(flat)  # (N, M)

        weights = torch.softmax(-self.sigma_q * dist, dim=1)
        soft = weights @ self.points  # (N, 2)

        hard = self.points[dist.argmin(dim=1)]

        # Record empirical usage for the KL term. `weights` keeps its graph, so the KL
        # term computed from this is differentiable.
        self._last_usage = weights.mean(dim=0)

        # Straight-through splice: exact constellation points forward, soft gradient back.
        #
        # Written as `hard + (soft - soft.detach())` rather than the more common
        # `soft + (hard - soft).detach()`. The two are algebraically identical and give
        # identical gradients, but only this form is bit-exact in floating point:
        # `soft - soft.detach()` is exactly zero, so the forward value is exactly `hard`.
        # The other form rounds, leaving transmitted values a few ULPs off the
        # constellation - which silently violates the one invariant that makes this
        # scheme standards-legal.
        out = hard + (soft - soft.detach())
        return out.reshape(z.shape)

    def _squared_distances(self, flat: torch.Tensor) -> torch.Tensor:
        """Squared euclidean distance from each latent to each constellation point.

        Uses the quadratic expansion ||z||^2 - 2 z.c + ||c||^2 rather than `torch.cdist`.
        cdist's matmul backend loses enough precision at these magnitudes to blur the
        nearest-neighbour decision, and materialising an (N, M, 2) difference tensor
        costs hundreds of MB at training batch sizes.

        The ||z||^2 term is constant across M, so it cancels in both the softmax and the
        argmin; it is retained only so the returned values are true distances.
        """
        z_sq = flat.pow(2).sum(dim=1, keepdim=True)
        c_sq = self.points.pow(2).sum(dim=1)
        return (z_sq - 2.0 * (flat @ self.points.T) + c_sq).clamp_min(0)

    def kl_to_uniform(self) -> torch.Tensor:
        """KL(P_hat(C) || U(C)) for the most recent forward pass.

        Counters codebook collapse, where the encoder finds a handful of points convenient
        and never uses the rest, wasting the alphabet.
        """
        if self._last_usage is None:
            raise RuntimeError("call forward() before kl_to_uniform()")
        p = self._last_usage.clamp_min(1e-12)
        return (p * (p * self.order).log()).sum()

    @torch.no_grad()
    def is_on_constellation(self, z: torch.Tensor, atol: float = 1e-5) -> bool:
        """True if every value in `z` is an exact constellation point.

        The invariant the whole deployability claim rests on: if the forward pass ever
        transmits an off-constellation value, the scheme is not standards-legal.
        """
        # Computed in float64 and by direct broadcasting: this is the assertion that
        # guards the deployability claim, so it must not inherit the precision
        # characteristics of the fast path it is checking.
        flat = z.reshape(-1, 2).to(torch.float64)
        points = self.points.to(torch.float64)
        gap = (flat.unsqueeze(1) - points.unsqueeze(0)).pow(2).sum(-1).min(dim=1).values
        return bool((gap.sqrt() <= atol).all())

    def extra_repr(self) -> str:
        return f"order={self.order}, sigma_q={self.sigma_q.item():.1f}"
