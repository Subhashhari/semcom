"""Channel models and power normalisation.

The channel is a fixed, non-trainable, differentiable layer sitting in the bottleneck of
the autoencoder. That differentiability is precisely why DeepJSCC works at all: you can
backpropagate through additive Gaussian noise.

Latents are carried as real tensors of shape (batch, k, 2), where the trailing dimension
holds the (I, Q) pair of one complex channel symbol.
"""

import torch


def snr_to_noise_power(snr_db: torch.Tensor, signal_power: float = 1.0) -> torch.Tensor:
    """Total complex noise power sigma^2 for a given SNR in dB."""
    return signal_power * torch.pow(10.0, -snr_db / 10.0)


def power_normalise(z: torch.Tensor, avg_power: float = 1.0) -> torch.Tensor:
    """Scale each latent in the batch to have average symbol power `avg_power`.

    Applied per example, after AF gating and immediately before quantisation.

    Ordering matters and is the one design decision the ADJSCC/DeepJSCC-Q composition
    forces. AF gates are sigmoid-bounded, so at low SNR they attenuate, shrinking the
    latent's overall magnitude. If normalisation ran *before* gating, the gates would end
    up controlling transmit power rather than resource allocation, and quantisation
    against a fixed-power constellation would silently change meaning with SNR.

    Args:
        z: (batch, k, 2) real tensor of (I, Q) pairs.
        avg_power: target average power per complex symbol.
    """
    if z.dim() != 3 or z.shape[-1] != 2:
        raise ValueError(f"expected (batch, k, 2), got {tuple(z.shape)}")

    k = z.shape[1]
    # ||z||^2 summed over both k and the I/Q pair; divide by k for per-symbol power.
    power = z.pow(2).sum(dim=(1, 2), keepdim=True) / k
    return z * (avg_power**0.5) / power.clamp_min(1e-12).sqrt()


def awgn(z: torch.Tensor, snr_db: torch.Tensor, signal_power: float = 1.0) -> torch.Tensor:
    """Additive white Gaussian noise channel: y = z + n, n ~ CN(0, sigma^2).

    Args:
        z: (batch, k, 2) transmitted symbols.
        snr_db: (batch,) or scalar SNR in dB.
    """
    sigma_sq = snr_to_noise_power(snr_db.to(z.dtype), signal_power).view(-1, 1, 1)
    # Complex noise power sigma^2 splits evenly across the I and Q components.
    noise = torch.randn_like(z) * (sigma_sq / 2).sqrt()
    return z + noise


def rayleigh_slow_fading(
    z: torch.Tensor, snr_db: torch.Tensor, signal_power: float = 1.0
) -> torch.Tensor:
    """Slow Rayleigh fading with receiver-side equalisation (DeepJSCC-Q Scenario 2).

    One realisation h ~ CN(0, 1) per image, held constant across the whole latent. The
    receiver knows h and equalises; the transmitter knows only the noise power.

    Returns the equalised observation, so the decoder network sees the same kind of input
    it would under AWGN, but with noise amplified by 1/|h|^2.
    """
    batch = z.shape[0]
    # h ~ CN(0, 1): each component variance 1/2 so E|h|^2 = 1.
    h = torch.randn(batch, 2, device=z.device, dtype=z.dtype) * (0.5**0.5)
    h_real, h_imag = h[:, 0:1], h[:, 1:2]

    z_real, z_imag = z[..., 0], z[..., 1]
    # Complex multiply h * z.
    faded = torch.stack(
        [h_real * z_real - h_imag * z_imag, h_real * z_imag + h_imag * z_real], dim=-1
    )

    y = awgn(faded, snr_db, signal_power)

    # Equalise: y <- conj(h) / |h|^2 * y.
    h_sq = (h_real.pow(2) + h_imag.pow(2)).clamp_min(1e-12)
    y_real, y_imag = y[..., 0], y[..., 1]
    equalised = torch.stack(
        [h_real * y_real + h_imag * y_imag, h_real * y_imag - h_imag * y_real], dim=-1
    )
    return equalised / h_sq.unsqueeze(-1)


CHANNELS = {"awgn": awgn, "rayleigh": rayleigh_slow_fading}


def get_channel(name: str):
    if name not in CHANNELS:
        raise ValueError(f"unknown channel {name!r}; expected one of {sorted(CHANNELS)}")
    return CHANNELS[name]
