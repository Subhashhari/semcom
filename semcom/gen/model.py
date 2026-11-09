"""The prototype generative JSCC (a small DiT-JSCC) and the CNN baseline, behind one interface.

    transmitter                                   channel            receiver
    DINOv2 features (cached) -> semantic encoder -> k_s symbols -+
                                                                  +-> conditional DiT -> VAE latent
    image -> detail encoder (pixels, /16)        -> k_d symbols -+     (DDIM, `steps` steps)

Each branch is power-normalised on its own to unit average symbol power, so the semantic/detail
split moves bandwidth between the branches but never power. The received symbols are laid back
onto their (image_size/16)^2 grid for the receiver.

With cfg.digital, both branches are then quantised to M-QAM points (one shared constellation,
semcom.constellation.SoftToHardQuantiser): what goes over the air is a stream of standard QAM
symbols. Quantisation runs in float32 outside autocast, so half precision cannot move a symbol's
nearest-point decision.

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
from ..constellation import SoftToHardQuantiser
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


class SNRFiLM(nn.Module):
    """Per-channel scale and shift from the SNR: h * (1 + gamma) + beta. The output layer starts
    at zero, so the layer starts as the identity."""

    def __init__(self, channels: int, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, 2 * channels))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, h: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.mlp((snr_db / SNR_SCALE_DB).to(h.dtype).view(-1, 1)).chunk(2, dim=1)
        return h * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]


def _film_forward(net: nn.Sequential, films, h: torch.Tensor, snr_db) -> torch.Tensor:
    """Run `net`, applying films[str(i)] after layer i when the encoder is SNR-aware."""
    if films is not None and snr_db is None:
        raise ValueError("this transmitter is SNR-aware: give it the SNR")
    for i, layer in enumerate(net):
        h = layer(h)
        if films is not None and str(i) in films:
            h = films[str(i)](h, snr_db)
    return h


class SemanticEncoder(nn.Module):
    """DINOv2 features on the symbol grid (B, D, g, g) -> (B, c_sem, g, g)."""

    def __init__(self, dino_dim: int, c_sem: int, hidden: int = 256, snr_aware: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dino_dim, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, c_sem, 1),
        )
        # After each hidden activation (layers 1 and 3).
        self.film = nn.ModuleDict({"1": SNRFiLM(hidden), "3": SNRFiLM(hidden)}) if snr_aware else None

    def forward(self, f: torch.Tensor, snr_db: torch.Tensor | None = None) -> torch.Tensor:
        return _film_forward(self.net, self.film, f, snr_db)


class DetailEncoder(nn.Module):
    """Pixels (B, 3, S, S) -> (B, c_det, S/16, S/16): four stride-2 stages."""

    def __init__(self, c_det: int, hidden: int = 128, snr_aware: bool = False):
        super().__init__()
        layers, c = [], 3
        for _ in range(4):
            layers += [nn.Conv2d(c, hidden, 4, stride=2, padding=1), nn.GELU()]
            c = hidden
        layers += [nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(), nn.Conv2d(hidden, c_det, 1)]
        self.net = nn.Sequential(*layers)
        # After the second and fourth stride-2 stages and the last hidden activation (layers 3, 7, 9).
        self.film = nn.ModuleDict({str(i): SNRFiLM(hidden) for i in (3, 7, 9)}) if snr_aware else None

    def forward(self, x: torch.Tensor, snr_db: torch.Tensor | None = None) -> torch.Tensor:
        return _film_forward(self.net, self.film, x, snr_db)


def make_quantiser(cfg: GenConfig) -> SoftToHardQuantiser | None:
    if not cfg.digital:
        return None
    return SoftToHardQuantiser(cfg.modulation_order, avg_power=cfg.avg_power, sigma_q_init=cfg.sigma_q,
                               anneal_period=anneal_period(cfg))


def anneal_period(cfg: GenConfig) -> int:
    return cfg.anneal_period if cfg.anneal_period is not None else 2**62  # never: fixed temperature


def quantise(q: SoftToHardQuantiser, *zs: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Quantise (B, k_i, 2) blocks to one constellation in float32; returns them and the KL term."""
    sizes = [z.shape[1] for z in zs]
    with torch.autocast(zs[0].device.type, enabled=False):
        out = q(torch.cat([z.float() for z in zs], dim=1))
        kl = q.kl_to_uniform()
    return list(out.split(sizes, dim=1)), kl


#: Momentum of the running transmitted-energy estimate (es_ref) kept during training.
ES_MOMENTUM = 0.99


class EnergyReference:
    """Mixin for digital models with power_ref="actual": the channel noise is scaled to the energy
    the symbols really carry, so the SNR is Es/N0 for the transmitted signal.

    Training uses the batch's mean symbol energy (detached: a fixed function of the signal, not
    something the encoder can trade against noise; scaling both together changes nothing anyway)
    and keeps a running estimate in the `es_ref` buffer. Evaluation uses `es_ref`, one constant,
    which semcom.gen.evaluate re-measures on the test images before scoring."""

    def _init_energy(self, cfg: GenConfig) -> None:
        if cfg.digital and cfg.power_ref == "actual":
            self.register_buffer("es_ref", torch.tensor(1.0))

    def signal_energy(self, z: torch.Tensor):
        """Energy the SNR refers to, for transmitted symbols z (B, k, 2)."""
        if not hasattr(self, "es_ref"):
            return self.cfg.avg_power
        if self.training:
            es = z.detach().float().pow(2).sum(-1).mean()
            self.es_ref.lerp_(es, 1 - ES_MOMENTUM)
            return es
        return self.es_ref


def snr_input(cfg: GenConfig, snr_db: torch.Tensor) -> torch.Tensor | None:
    """What fills the receiver's SNR slot: nothing (specialist), the SNR, or a constant (blind)."""
    if cfg.snr_mode == "fixed":
        return None
    if cfg.snr_mode == "snr":
        return snr_db / SNR_SCALE_DB
    return torch.zeros_like(snr_db)


class GenJSCC(EnergyReference, nn.Module):
    def __init__(self, cfg: GenConfig, dino_dim: int):
        super().__init__()
        self.cfg = cfg
        depth, hidden, heads = DIT_SIZES[cfg.dit_size]
        self.sem_enc = SemanticEncoder(dino_dim, cfg.c_sem, snr_aware=cfg.encoder_snr)
        self.det_enc = DetailEncoder(cfg.c_det, cfg.det_hidden, snr_aware=cfg.encoder_snr)
        self.dit = ConditionalDiT(
            latent_size=cfg.image_size // 8, c_sem=cfg.c_sem, c_det=cfg.c_det,
            depth=depth, hidden=hidden, heads=heads, patch=cfg.patch_size,
            cond_blocks_sem=cfg.cond_blocks_sem, cond_blocks_det=cfg.cond_blocks_det,
            snr_slot=cfg.snr_mode != "fixed",
        )
        self.diffusion = Diffusion(cfg.num_timesteps, cfg.beta_start, cfg.beta_end, cfg.prediction)
        self.quantiser = make_quantiser(cfg)
        self._kl = None
        self._init_energy(cfg)

    def transmit(self, batch: dict, snr_db: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Channel inputs (z_sem, z_det), each (B, k, 2) at unit average power; exact M-QAM points
        when digital (average power then varies a little per image with the points used).
        `snr_db` reaches the encoders only when cfg.encoder_snr (an SNR-aware transmitter)."""
        s = snr_db if self.cfg.encoder_snr else None
        z_s = power_normalise(to_symbols(self.sem_enc(batch["dino"], s)), self.cfg.avg_power)
        z_d = power_normalise(to_symbols(self.det_enc(batch["image"], s)), self.cfg.avg_power)
        if self.quantiser is not None:
            (z_s, z_d), self._kl = quantise(self.quantiser, z_s, z_d)
        return z_s, z_d

    def channel_input(self, batch: dict) -> torch.Tensor:
        """Every transmitted symbol of each image, (B, k, 2)."""
        return torch.cat(self.transmit(batch), dim=1)

    def received(self, batch, snr_db, generator=None):
        """Received symbol grids (y_sem, y_det) after the channel."""
        cfg, g = self.cfg, self.cfg.grid
        z_s, z_d = self.transmit(batch, snr_db)
        es = self.signal_energy(torch.cat([z_s, z_d], dim=1))
        y_s = channel(z_s, snr_db, cfg.channel, es, generator)
        y_d = channel(z_d, snr_db, cfg.channel, es, generator)
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
        return with_kl(self, mse)

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


class CNNBaseline(EnergyReference, nn.Module):
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
                         decoder_input=dec, digital=cfg.digital, modulation_order=cfg.modulation_order,
                         sigma_q_init=cfg.sigma_q, anneal_period=anneal_period(cfg))
        self._kl = None
        self._init_energy(cfg)

    @property
    def quantiser(self) -> SoftToHardQuantiser | None:
        return self.jscc.quantiser

    def channel_input(self, batch: dict) -> torch.Tensor:
        x = batch["image"]
        return self.jscc.transmit(x, torch.zeros(x.shape[0], device=x.device))

    def loss(self, batch: dict, snr_db: torch.Tensor, drop: str = "train") -> dict:
        x = batch["image"]
        if snr_db.dim() == 0:
            snr_db = snr_db.expand(x.shape[0])
        # The JSCC's own forward pass, with this module's channel so the SNR reference applies.
        z = self.jscc.transmit(x, snr_db)
        self._kl = self.quantiser.kl_to_uniform() if self.quantiser is not None else None
        y = channel(z, snr_db, self.cfg.channel, self.signal_energy(z))
        x_hat = self.jscc.decode(y, snr_db, z).float()
        mse = F.mse_loss(x_hat, x)
        out = {**with_kl(self, mse), "pixel_mse": mse.detach()}
        if self.cfg.cnn_lpips_weight:
            lp = self._lpips(x.device)(x_hat * 2 - 1, x * 2 - 1).mean()
            out["loss"] = out["loss"] + self.cfg.cnn_lpips_weight * lp
            out["lpips_vgg"] = lp.detach()
        return out

    def _lpips(self, device):
        """Frozen LPIPS(VGG), created on first use and kept out of the state dict and parameters."""
        net = self.__dict__.get("_lpips_net")
        if net is None:
            import lpips

            net = lpips.LPIPS(net="vgg", verbose=False).to(device).eval().requires_grad_(False)
            self.__dict__["_lpips_net"] = net
        return net

    @torch.no_grad()
    def reconstruct(self, batch, snr_db, vae=None, steps=None, channel_gen=None, sample_gen=None,
                    eta=0.0, cfg_scale=1.0) -> torch.Tensor:
        x = batch["image"]
        z = self.jscc.transmit(x, snr_db)
        y = channel(z, snr_db, self.cfg.channel, self.signal_energy(z), channel_gen)
        return self.jscc.decode(y, snr_db, z).float().clamp(0, 1)


def with_kl(model: nn.Module, mse: torch.Tensor) -> dict:
    """The training loss: the reconstruction/diffusion MSE, plus the codebook-collapse penalty
    while training a digital model. Validation (eval mode) reports the MSE alone, so analog and
    digital validation losses compare directly."""
    if model.quantiser is None or not model.training or model._kl is None:
        return {"loss": mse}
    kl = model._kl.float()
    return {"loss": mse + model.cfg.kl_weight * kl, "mse": mse.detach(), "kl": kl.detach()}


def build_model(cfg: GenConfig, dino_dim: int) -> nn.Module:
    return GenJSCC(cfg, dino_dim) if cfg.model == "dit" else CNNBaseline(cfg)


def num_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
