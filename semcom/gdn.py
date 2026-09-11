"""Generalised Divisive Normalisation (Balle et al., arXiv:1511.06281).

Used throughout the BDJSCC/ADJSCC backbone. GDN normalises each feature channel by a
learned pooled function of all channels at the same spatial location:

    y_i = x_i / sqrt(beta_i + sum_j gamma_ij * x_j^2)

IGDN is the approximate inverse, used in the decoder. Both parameters are kept positive
via a reparameterisation through a lower-bounded square, which is what makes the layer
stable to train.
"""

import torch
import torch.nn as nn


class _LowerBound(torch.autograd.Function):
    """Clamp to a lower bound, but pass gradients through when they push the value back up.

    A plain clamp zeroes the gradient once a parameter hits the bound, so it can never
    recover. This variant only blocks gradients that would push further below the bound.
    """

    @staticmethod
    def forward(ctx, x, bound):
        ctx.save_for_backward(x, torch.as_tensor(bound, dtype=x.dtype, device=x.device))
        return torch.clamp(x, min=bound)

    @staticmethod
    def backward(ctx, grad_output):
        x, bound = ctx.saved_tensors
        pass_through = (x >= bound) | (grad_output < 0)
        return grad_output * pass_through.to(grad_output.dtype), None


class GDN(nn.Module):
    """Generalised divisive normalisation.

    Args:
        num_channels: channel count of the input tensor.
        inverse: if True, behaves as IGDN (multiply rather than divide).
        beta_min: lower bound on beta, for numerical stability.
        gamma_init: diagonal initialiser for gamma.
    """

    # Reparameterisation offset: params are stored as sqrt(value), bounded below by
    # `reparam_offset`, so the squared value stays strictly positive.
    reparam_offset = 2**-18

    def __init__(
        self,
        num_channels: int,
        inverse: bool = False,
        beta_min: float = 1e-6,
        gamma_init: float = 0.1,
    ):
        super().__init__()
        self.inverse = bool(inverse)
        self.num_channels = int(num_channels)

        self.beta_bound = (beta_min + self.reparam_offset**2) ** 0.5
        self.gamma_bound = self.reparam_offset

        beta = torch.ones(num_channels)
        self.beta = nn.Parameter(torch.sqrt(beta + self.reparam_offset**2))

        gamma = gamma_init * torch.eye(num_channels)
        self.gamma = nn.Parameter(torch.sqrt(gamma + self.reparam_offset**2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        beta = _LowerBound.apply(self.beta, self.beta_bound) ** 2
        gamma = _LowerBound.apply(self.gamma, self.gamma_bound) ** 2
        gamma = gamma.view(self.num_channels, self.num_channels, 1, 1)

        norm = nn.functional.conv2d(x * x, gamma, beta)
        norm = torch.sqrt(norm)

        return x * norm if self.inverse else x / norm
