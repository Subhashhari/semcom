"""Frozen pretrained parts: the Stable Diffusion VAE and DINOv2.

Both stay frozen for the whole study, so their outputs are computed once per image and cached
(semcom.gen.data). Training then never runs them; evaluation runs the VAE decoder to turn
sampled latents into images, and DINOv2 once more for the semantic-similarity metric.

The real models download on first use (diffusers / torch.hub). `cfg.fake_frozen` gives tiny
fixed random stand-ins with the same interfaces and shapes, so the pipeline can be tested
offline on a CPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DINO_DIMS = {"dinov2_vits14": 384, "dinov2_vitb14": 768, "dinov2_vitl14": 1024}
FAKE_DINO_DIM = 32


class FrozenVAE(nn.Module):
    """Images in [0, 1] <-> scaled latents (4 channels, 1/8 resolution)."""

    latent_channels = 4
    factor = 8

    def __init__(self, model_id: str, scale: float):
        super().__init__()
        from diffusers import AutoencoderKL

        self.vae = AutoencoderKL.from_pretrained(model_id).eval().requires_grad_(False)
        self.scale = scale

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # The posterior mean, not a sample: deterministic, so the cache is reproducible.
        return self.vae.encode(x * 2 - 1).latent_dist.mean * self.scale

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return ((self.vae.decode(z / self.scale).sample + 1) / 2).clamp(0, 1)


class FrozenDINO(nn.Module):
    """Images in [0, 1] -> DINOv2 patch features pooled to a grid, and the CLS embedding."""

    patch = 14

    def __init__(self, model_name: str, input_size: int):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", model_name).eval().requires_grad_(False)
        self.dim = DINO_DIMS[model_name]
        self.input_size = input_size
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x: torch.Tensor, grid: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (patch features (B, dim, grid, grid), CLS embedding (B, dim))."""
        x = F.interpolate(x, size=self.input_size, mode="bicubic", align_corners=False)
        out = self.model.forward_features((x.clamp(0, 1) - self.mean) / self.std)
        side = self.input_size // self.patch
        tokens = out["x_norm_patchtokens"].transpose(1, 2).reshape(x.shape[0], self.dim, side, side)
        return F.adaptive_avg_pool2d(tokens, grid), out["x_norm_clstoken"]


class FakeVAE(nn.Module):
    """Fixed random stand-in for FrozenVAE: same shapes, no download."""

    latent_channels = 4
    factor = 8

    def __init__(self, scale: float = 0.18215):
        super().__init__()
        gen = torch.Generator().manual_seed(1)
        self.register_buffer("w_enc", torch.randn(4, 3, generator=gen) * 0.5)
        self.register_buffer("w_dec", torch.linalg.pinv(self.w_enc))
        self.scale = scale

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = F.avg_pool2d(x * 2 - 1, self.factor)
        return torch.einsum("oc,bchw->bohw", self.w_enc, z)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = torch.einsum("co,bohw->bchw", self.w_dec, z)
        x = F.interpolate(x, scale_factor=self.factor, mode="nearest")
        return ((x + 1) / 2).clamp(0, 1)


class FakeDINO(nn.Module):
    """Fixed random stand-in for FrozenDINO."""

    def __init__(self):
        super().__init__()
        gen = torch.Generator().manual_seed(2)
        self.register_buffer("w", torch.randn(FAKE_DINO_DIM, 3, 4, 4, generator=gen) * 0.2)
        self.dim = FAKE_DINO_DIM

    @torch.no_grad()
    def forward(self, x: torch.Tensor, grid: int) -> tuple[torch.Tensor, torch.Tensor]:
        feat = torch.tanh(F.conv2d(x * 2 - 1, self.w, stride=4))
        return F.adaptive_avg_pool2d(feat, grid), feat.mean(dim=(2, 3))


def load_frozen(cfg, device: torch.device) -> tuple[nn.Module, nn.Module]:
    """(vae, dino) for this config, moved to `device`."""
    if cfg.fake_frozen:
        return FakeVAE(cfg.vae_scale).to(device), FakeDINO().to(device)
    return (
        FrozenVAE(cfg.vae_model, cfg.vae_scale).to(device),
        FrozenDINO(cfg.dino_model, cfg.dino_input).to(device),
    )

