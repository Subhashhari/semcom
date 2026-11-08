"""The prototype generative JSCC (a small DiT-JSCC) and the CNN baseline, behind one interface.

    transmitter                                   channel            receiver
    DINOv2 features (cached) -> semantic encoder -> k_s symbols -+
                                                                  +-> conditional DiT -> VAE latent
    image -> detail encoder (pixels, /16)        -> k_d symbols -+     (DDIM, `steps` steps)

Each branch is power-normalised on its own to unit average symbol power, so the semantic/detail
split moves bandwidth between the branches but never power. The received symbols are laid back
onto their (image_size/16)^2 grid for the receiver.

Both models expose:
    loss(batch, snr_db)                          training loss (dict with "loss")
    reconstruct(batch, snr_db, vae, ...)         images in [0, 1], with optional explicit
                                                 channel and sampling generators (paired eval)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..channel import power_normalise, snr_to_noise_power
from .config import DIT_SIZES, GenConfig
from .diffusion import Diffusion
from .dit import ConditionalDiT

#: The SNR enters the DiT's SNR slot as snr_db / SNR_SCALE_DB.
SNR_SCALE_DB = 10.0


def channel(
    z: torch.Tensor, snr_db: torch.Tensor, kind: str = "awgn", avg_power: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """AWGN or equalised slow Rayleigh fading on (B, k, 2) symbols, with an optional generator.

    Same models as semcom.channel, re-implemented so evaluation can draw the noise (and the
    fading) from a seeded generator and give every arm identical channel realisations.
    """
    b = z.shape[0]
    sigma_sq = snr_to_noise_power(snr_db.to(z.dtype), avg_power).view(-1, 1, 1)
    noise = torch.randn(z.shape, device=z.device, dtype=z.dtype, generator=generator) * (sigma_sq / 2).sqrt()
    if kind == "awgn":
        return z + noise
    if kind != "rayleigh":
        raise ValueError(f"unknown channel {kind!r}")
    h = torch.randn(b, 2, device=z.device, dtype=z.dtype, generator=generator) * 0.5**0.5
    hr, hi = h[:, :1], h[:, 1:]
    zr, zi = z[..., 0], z[..., 1]
    y = torch.stack([hr * zr - hi * zi, hr * zi + hi * zr], dim=-1) + noise
    yr, yi = y[..., 0], y[..., 1]
    eq = torch.stack([hr * yr + hi * yi, hr * yi - hi * yr], dim=-1)
    return eq / (hr.pow(2) + hi.pow(2)).clamp_min(1e-12).unsqueeze(-1)


def to_symbols(feat: torch.Tensor) -> torch.Tensor:
    """(B, C, g, g) -> (B, C*g*g/2, 2), pairing values into (I, Q)."""
    return feat.reshape(feat.shape[0], -1, 2)


def from_symbols(z: torch.Tensor, c: int, g: int) -> torch.Tensor:
    return z.reshape(z.shape[0], c, g, g)


class SemanticEncoder(nn.Module):
    """DINOv2 features on the symbol grid (B, D, g, g) -> (B, c_sem, g, g)."""

    def __init__(self, dino_dim: int, c_sem: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dino_dim, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, c_sem, 1),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.net(f)


class DetailEncoder(nn.Module):
    """Pixels (B, 3, S, S) -> (B, c_det, S/16, S/16): four stride-2 stages."""

    def __init__(self, c_det: int, hidden: int = 128):
        super().__init__()
        layers, c = [], 3
        for _ in range(4):
            layers += [nn.Conv2d(c, hidden, 4, stride=2, padding=1), nn.GELU()]
            c = hidden
        layers += [nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(), nn.Conv2d(hidden, c_det, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def snr_input(cfg: GenConfig, snr_db: torch.Tensor) -> torch.Tensor | None:
    """What fills the receiver's SNR slot: nothing (specialist), the SNR, or a constant (blind)."""
    if cfg.snr_mode == "fixed":
        return None
    if cfg.snr_mode == "snr":
        return snr_db / SNR_SCALE_DB
    return torch.zeros_like(snr_db)


class GenJSCC(nn.Module):
    def __init__(self, cfg: GenConfig, dino_dim: int):
        super().__init__()
        self.cfg = cfg
        depth, hidden, heads = DIT_SIZES[cfg.dit_size]
        self.sem_enc = SemanticEncoder(dino_dim, cfg.c_sem)
        self.det_enc = DetailEncoder(cfg.c_det, cfg.det_hidden)
        self.dit = ConditionalDiT(
            latent_size=cfg.image_size // 8, c_sem=cfg.c_sem, c_det=cfg.c_det,
            depth=depth, hidden=hidden, heads=heads, patch=cfg.patch_size,
            cond_blocks_sem=cfg.cond_blocks_sem, cond_blocks_det=cfg.cond_blocks_det,
            snr_slot=cfg.snr_mode != "fixed",
        )
        self.diffusion = Diffusion(cfg.num_timesteps, cfg.beta_start, cfg.beta_end, cfg.prediction)

    def transmit(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Channel inputs (z_sem, z_det), each (B, k, 2) at unit average power."""
        z_s = power_normalise(to_symbols(self.sem_enc(batch["dino"])), self.cfg.avg_power)
        z_d = power_normalise(to_symbols(self.det_enc(batch["image"])), self.cfg.avg_power)
        return z_s, z_d

    def received(self, batch, snr_db, generator=None):
        """Received symbol grids (y_sem, y_det) after the channel."""
        cfg, g = self.cfg, self.cfg.grid
        z_s, z_d = self.transmit(batch)
        y_s = channel(z_s, snr_db, cfg.channel, cfg.avg_power, generator)
        y_d = channel(z_d, snr_db, cfg.channel, cfg.avg_power, generator)
        return from_symbols(y_s, cfg.c_sem, g), from_symbols(y_d, cfg.c_det, g)

    def loss(self, batch: dict, snr_db: torch.Tensor, drop: str = "train") -> dict:
        """Diffusion loss (noise or velocity target, cfg.prediction). `drop`: "train" drops the received symbols with probability
        cfg_dropout (for classifier-free guidance); "none" keeps them; "all" drops them for every
        image, which measures how much the receiver relies on the channel at all."""
        y_s, y_d = self.received(batch, snr_db)
        x0 = batch["latent"]
        x_t, t, noise = self.diffusion.training_targets(x0)
        b = x0.shape[0]
        if drop == "train":
            mask = torch.rand(b, device=x0.device) < self.cfg.cfg_dropout
        elif drop in ("none", "all"):
            mask = torch.full((b,), drop == "all", device=x0.device)
        else:
            raise ValueError(f"drop must be 'train', 'none' or 'all', got {drop!r}")
        out = self.dit(x_t, t, y_s, y_d, snr_input(self.cfg, snr_db), mask)
        mse = F.mse_loss(out.float(), self.diffusion.target(x0, noise, t).float())
        return {"loss": mse}

    @torch.no_grad()
    def sample_latent(self, y_s, y_d, snr_db, steps, generator=None, eta=0.0, cfg_scale=1.0):
        s_in = snr_input(self.cfg, snr_db)
        b = y_s.shape[0]

        def eps_fn(x, t):
            if cfg_scale == 1.0:
                return self.dit(x, t, y_s, y_d, s_in)
            drop = torch.cat([torch.zeros(b, dtype=torch.bool), torch.ones(b, dtype=torch.bool)]).to(x.device)
            two = lambda a: None if a is None else torch.cat([a, a])  # noqa: E731
            e = self.dit(two(x), two(t), two(y_s), two(y_d), two(s_in), drop)
            cond, uncond = e.chunk(2)
            return uncond + cfg_scale * (cond - uncond)

        lat = self.cfg.image_size // 8
        return self.diffusion.sample(eps_fn, (b, 4, lat, lat), steps, y_s.device, generator, eta)

    @torch.no_grad()
    def reconstruct(self, batch, snr_db, vae, steps=50, channel_gen=None, sample_gen=None,
                    eta=0.0, cfg_scale=1.0) -> torch.Tensor:
        y_s, y_d = self.received(batch, snr_db, channel_gen)
        z = self.sample_latent(y_s, y_d, snr_db, steps, sample_gen, eta, cfg_scale)
        return vae.decode(z.float())


class CNNBaseline(nn.Module):
    """The repo's CNN deep JSCC (semcom.models.JSCC) at the same bandwidth, MSE-trained on pixels.

    snr_mode maps onto the blind-study decoder inputs: fixed -> no AF modules (specialist),
    snr -> decoder told the SNR (B), blank -> same modules, constant input (C-att).
    """

    def __init__(self, cfg: GenConfig):
        super().__init__()
        from ..models import JSCC

        self.cfg = cfg
        dec = {"fixed": "none", "snr": "snr", "blank": "blank"}[cfg.snr_mode]
        self.jscc = JSCC(c_out=cfg.cnn_c_out, image_size=cfg.image_size, hidden=cfg.cnn_hidden,
                         channel=cfg.channel, avg_power=cfg.avg_power, encoder_snr=False,
                         decoder_input=dec)

    def loss(self, batch: dict, snr_db: torch.Tensor, drop: str = "train") -> dict:
        x = batch["image"]
        x_hat = self.jscc(x, snr_db)["x_hat"]
        mse = F.mse_loss(x_hat.float(), x)
        return {"loss": mse, "pixel_mse": mse.detach()}

    @torch.no_grad()
    def reconstruct(self, batch, snr_db, vae=None, steps=None, channel_gen=None, sample_gen=None,
                    eta=0.0, cfg_scale=1.0) -> torch.Tensor:
        x = batch["image"]
        z = self.jscc.transmit(x, snr_db)
        y = channel(z, snr_db, self.cfg.channel, self.cfg.avg_power, channel_gen)
        return self.jscc.decode(y, snr_db, z).float().clamp(0, 1)


def build_model(cfg: GenConfig, dino_dim: int) -> nn.Module:
    return GenJSCC(cfg, dino_dim) if cfg.model == "dit" else CNNBaseline(cfg)


def num_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
