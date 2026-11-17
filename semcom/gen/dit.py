"""A small conditional DiT: the receiver of the generative-JSCC prototype.

Follows the DiT reference design (Peebles & Xie, 2023): patchify the VAE latent, a stack of
transformer blocks modulated by adaLN-Zero from the conditioning vector, a modulated final
layer that predicts the diffusion target (noise or velocity, semcom.gen.diffusion). Two additions from DiT-JSCC (Tan et al., arXiv:2601.03112):

* **Channel fusion.** The received symbols, laid out on the token grid, pass through a small
  adapter and are fused into the token stream by concatenation and projection, rather than
  cross-attention. Each conditional block adds  W [x ; c]  to its input, with W zero-initialised
  so training starts from the unconditional DiT.
* **Coarse-to-fine schedule.** The first `cond_blocks_sem` blocks take the semantic symbols,
  the next `cond_blocks_det` blocks the detail symbols; the rest see only the tokens.

Deliberate simplifications against the paper, for a prototype that trains on one consumer GPU:
no U-Net-style long skips, no learned variance (the output is the noise or velocity target only), trained from scratch
rather than from a pretrained class-conditional DiT.

The conditioning vector is the timestep embedding, plus an SNR embedding when the model has an
SNR slot (snr_mode "snr" or "blank"; see semcom.gen.config).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def sincos_2d(dim: int, side: int) -> torch.Tensor:
    """Fixed 2-D sine-cosine position embedding, (side*side, dim)."""
    assert dim % 4 == 0
    coords = np.stack(np.meshgrid(np.arange(side), np.arange(side), indexing="xy"), 0).reshape(2, -1)
    omega = 1.0 / 10000 ** (np.arange(dim // 4) / (dim // 4))
    parts = []
    for c in coords:
        out = np.outer(c, omega)
        parts += [np.sin(out), np.cos(out)]
    return torch.from_numpy(np.concatenate(parts, axis=1)).float()


class ScalarEmbedder(nn.Module):
    """Scalar (timestep, or normalised SNR) -> sinusoidal features -> MLP -> hidden."""

    def __init__(self, hidden: int, freq_dim: int = 256, max_period: float = 10000.0):
        super().__init__()
        self.freq_dim, self.max_period = freq_dim, max_period
        self.mlp = nn.Sequential(nn.Linear(freq_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(half, device=s.device) / half)
        args = s.float().unsqueeze(1) * freqs.unsqueeze(0)
        return self.mlp(torch.cat([torch.cos(args), torch.sin(args)], dim=1))


class Block(nn.Module):
    """Transformer block with adaLN-Zero, and an optional zero-initialised fusion input."""

    def __init__(self, hidden: int, heads: int, mlp_ratio: float = 4.0, fuse: bool = False):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        h = int(hidden * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(hidden, h), nn.GELU(approximate="tanh"), nn.Linear(h, hidden))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden))
        self.fuse = nn.Linear(2 * hidden, hidden) if fuse else None

    def forward(self, x: torch.Tensor, c: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        if self.fuse is not None and cond is not None:
            x = x + self.fuse(torch.cat([x, cond], dim=-1))
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(c).chunk(6, dim=1)
        b, n, d = x.shape
        qkv = self.qkv(modulate(self.norm1(x), sh1, sc1)).reshape(b, n, 3, self.heads, d // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d)
        x = x + g1.unsqueeze(1) * self.proj(attn)
        return x + g2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), sh2, sc2))


class CondAdapter(nn.Module):
    """Received symbols on their grid (B, c, g, g) -> token features (B, T, hidden).

    A 3x3 conv lets each token see its neighbours' symbols; the grid is resized to the token
    grid when the two differ. With `in_side` a multiple r of the token side (the refiner's
    16x16 latent condition on an 8x8 token grid), an r x r patchify replaces the resize, so
    no condition value is thrown away.
    """

    def __init__(self, c_in: int, hidden: int, token_side: int, in_side: int | None = None):
        super().__init__()
        self.token_side = token_side
        r = 1 if in_side is None else in_side // token_side
        if in_side is not None and r * token_side != in_side:
            raise ValueError(f"in_side {in_side} must be a multiple of the token side {token_side}")
        first = nn.Conv2d(c_in, hidden, 3, padding=1) if r == 1 else nn.Conv2d(c_in, hidden, r, stride=r)
        self.net = nn.Sequential(first, nn.GELU(), nn.Conv2d(hidden, hidden, 1))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if self.net[0].stride == (1, 1) and y.shape[-1] != self.token_side:
            y = F.interpolate(y, size=self.token_side, mode="nearest")
        return self.net(y).flatten(2).transpose(1, 2)


class ConditionalDiT(nn.Module):
    """Predicts the diffusion target (noise or velocity) for a VAE latent, guided by received semantic and detail symbols.

    Args:
        latent_size: side of the VAE latent (image_size / 8).
        c_sem, c_det: channels of the received semantic / detail symbol grids.
        snr_slot: build an SNR embedder (snr_mode "snr" or "blank").
        cond_side: side of the condition grids when it differs from the symbol grid (refiner).
    """

    def __init__(
        self,
        latent_size: int,
        c_sem: int,
        c_det: int,
        depth: int = 12,
        hidden: int = 384,
        heads: int = 6,
        patch: int = 2,
        latent_channels: int = 4,
        cond_blocks_sem: int = 3,
        cond_blocks_det: int = 3,
        snr_slot: bool = False,
        cond_side: int | None = None,
    ):
        super().__init__()
        self.patch, self.c_lat = patch, latent_channels
        self.side = latent_size // patch
        self.n_sem, self.n_det = cond_blocks_sem, cond_blocks_det

        self.x_embed = nn.Conv2d(latent_channels, hidden, patch, stride=patch)
        self.register_buffer("pos", sincos_2d(hidden, self.side).unsqueeze(0), persistent=False)
        self.t_embed = ScalarEmbedder(hidden)
        # SNR in dB is divided by 10 and fed through the same sinusoidal map as the timestep,
        # scaled up so a 1 dB step moves the features as much as ~10 diffusion steps.
        self.snr_embed = ScalarEmbedder(hidden, max_period=10000.0) if snr_slot else None
        self.hidden = hidden
        self.sem_adapter = CondAdapter(c_sem, hidden, self.side, cond_side)
        self.det_adapter = CondAdapter(c_det, hidden, self.side, cond_side)
        # Learned stand-ins for "nothing received", used for classifier-free guidance.
        self.null_sem = nn.Parameter(torch.zeros(1, 1, hidden))
        self.null_det = nn.Parameter(torch.zeros(1, 1, hidden))

        n_cond = cond_blocks_sem + cond_blocks_det
        self.blocks = nn.ModuleList([Block(hidden, heads, fuse=i < n_cond) for i in range(depth)])
        self.final_norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))
        self.final = nn.Linear(hidden, patch * patch * latent_channels)
        self._init()

    def _init(self):
        def basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(basic)
        w = self.x_embed.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.x_embed.bias)
        for emb in (self.t_embed, self.snr_embed):
            if emb is not None:
                nn.init.normal_(emb.mlp[0].weight, std=0.02)
                nn.init.normal_(emb.mlp[2].weight, std=0.02)
        # adaLN-Zero: every block starts as the identity; fusion starts switched off.
        for blk in self.blocks:
            nn.init.zeros_(blk.ada[-1].weight)
            nn.init.zeros_(blk.ada[-1].bias)
            if blk.fuse is not None:
                nn.init.zeros_(blk.fuse.weight)
                nn.init.zeros_(blk.fuse.bias)
        nn.init.zeros_(self.final_ada[-1].weight)
        nn.init.zeros_(self.final_ada[-1].bias)
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        b, p, c, s = x.shape[0], self.patch, self.c_lat, self.side
        x = x.reshape(b, s, s, p, p, c).permute(0, 5, 1, 3, 2, 4)
        return x.reshape(b, c, s * p, s * p)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y_sem: torch.Tensor,
        y_det: torch.Tensor,
        snr: torch.Tensor | None = None,
        drop: torch.Tensor | None = None,
        hidden_at: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """The diffusion target (noise or velocity), same shape as x_t; with `hidden_at` = n,
        also the tokens (B, T, hidden) after the first n blocks (for REPA).

        Args:
            y_sem, y_det: received symbol grids (B, c_sem, g, g) and (B, c_det, g, g).
            snr: (B,) conditioning value for the SNR slot (already normalised; zeros when blind).
            drop: (B,) bool, replace both received conditions by the null embeddings (CFG).
        """
        x = self.x_embed(x_t).flatten(2).transpose(1, 2) + self.pos
        c = self.t_embed(t)
        if self.snr_embed is not None:
            if snr is None:
                raise ValueError("this model has an SNR slot; pass snr (zeros when blind)")
            c = c + self.snr_embed(snr * 100.0)
        sem, det = self.sem_adapter(y_sem), self.det_adapter(y_det)
        if drop is not None:
            m = drop.view(-1, 1, 1)
            sem = torch.where(m, self.null_sem.to(sem.dtype), sem)
            det = torch.where(m, self.null_det.to(det.dtype), det)
        h = None
        for i, blk in enumerate(self.blocks):
            cond = sem if i < self.n_sem else det if i < self.n_sem + self.n_det else None
            x = blk(x, c, cond)
            if hidden_at is not None and i + 1 == hidden_at:
                h = x
        sh, sc = self.final_ada(c).chunk(2, dim=1)
        out = self.unpatchify(self.final(modulate(self.final_norm(x), sh, sc)))
        return out if hidden_at is None else (out, h)
