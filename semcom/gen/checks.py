"""Correctness checks for generative-JSCC training: run before any GPU hours are spent, and
during training, so a broken model shows up in the numbers early rather than in the results.

Pre-flight (step 0 only), each printed as a `SANITY ...` line and raising `SanityError` on failure:

* data    the cache is finite; images lie in [0, 1]; latents have roughly unit spread (the SD
          scaling factor was applied); DINOv2 features are not constant
* ceiling the VAE alone (decode the true cached latent): the best quality any DiT receiver can
          reach, reported next to every result
* overfit a copy of the model trained on one fixed batch, with fixed timesteps and noise, must
          drive its loss well down, and gradient must reach both encoders through the channel.
          This catches a detached path, a wrong target, or a sign error in minutes.

During training, validation reports the loss with the received symbols and with them dropped;
the difference (`cond_gain`) is how much the receiver uses the channel at all. A receiver that
ignores the channel still produces realistic images, so this is the check that matters most.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from .config import GenConfig
from ..data import psnr


class SanityError(RuntimeError):
    pass


def kv(**items) -> str:
    """Machine-readable `key=value` fields for the job-log watcher."""
    def fmt(v):
        return f"{v:.6g}" if isinstance(v, float) else str(v)
    return " ".join(f"{k}={fmt(v)}" for k, v in items.items())


@torch.no_grad()
def check_data(batch: dict, cfg: GenConfig) -> dict:
    img, lat, dino = batch["image"], batch["latent"], batch["dino"]
    stats = dict(img_min=img.min().item(), img_max=img.max().item(), lat_mean=lat.mean().item(),
                 lat_std=lat.std().item(), dino_std=dino.std().item())
    print(f"SANITY data run={cfg.run_name} {kv(**stats)}", flush=True)
    if not all(torch.isfinite(t).all() for t in (img, lat, dino)):
        raise SanityError("non-finite values in the cached data")
    if stats["img_min"] < 0 or stats["img_max"] > 1:
        raise SanityError("images outside [0, 1]")
    if not cfg.fake_frozen and not 0.3 < stats["lat_std"] < 3.0:
        raise SanityError(f"latent std {stats['lat_std']:.3f}: is the VAE scale factor applied?")
    if stats["dino_std"] < 1e-4:
        raise SanityError("DINOv2 features are constant")
    return stats


@torch.no_grad()
def vae_ceiling(batch: dict, vae, dino, cfg: GenConfig) -> dict:
    x = batch["image"]
    x_hat = vae.decode(batch["latent"]).float()
    _, cls = dino(x, cfg.grid)
    _, cls_hat = dino(x_hat, cfg.grid)
    out = dict(psnr=psnr(x, x_hat).mean().item(),
               dino_sim=F.cosine_similarity(cls.float(), cls_hat.float(), dim=1).mean().item())
    print(f"SANITY vae_ceiling run={cfg.run_name} {kv(**out)}", flush=True)
    return out


#: Fraction by which the one-batch loss must fall. The DiT's noise-prediction loss on fixed
#: targets starts near 1 and collapses quickly; the CNN baseline's pixel MSE starts low (a sigmoid
#: at 0.5 is already close) and falls slowly, and the CNN is the repo's long-tested model, so its
#: bar only asks for a clear decrease.
MIN_DROP = {"dit": 0.5, "cnn": 0.1}


def overfit_check(model, batch: dict, cfg: GenConfig, device, steps: int = 150,
                  min_drop: float | None = None) -> dict:
    """Train a throwaway copy on one batch with frozen randomness; the loss must fall by
    `min_drop` (fraction; default per model), and the encoders must receive gradient."""
    min_drop = MIN_DROP[cfg.model] if min_drop is None else min_drop
    probe = copy.deepcopy(model).train()
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    snr = torch.full((batch["image"].shape[0],), float(cfg.snr_train_fixed or 0.0), device=device)
    losses = []
    for _ in range(steps):
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(0)  # same timesteps, noise and channel every step
            loss = probe.loss(batch, snr, drop="none")["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    enc = (probe.sem_enc, probe.det_enc) if cfg.model == "dit" else (probe.jscc.encoder,)
    enc_grad = min(sum(p.grad.norm().item() for p in m.parameters() if p.grad is not None) for m in enc)
    out = dict(first=losses[0], last=losses[-1], ratio=losses[-1] / max(losses[0], 1e-12),
               encoder_grad=enc_grad)
    print(f"SANITY overfit run={cfg.run_name} steps={steps} {kv(**out)}", flush=True)
    if not all(map(lambda v: v == v and abs(v) != float("inf"), losses)):
        raise SanityError("non-finite loss while overfitting one batch")
    if out["ratio"] > 1 - min_drop:
        raise SanityError(f"loss fell only to {out['ratio']:.2f} of its start on one fixed batch")
    if enc_grad <= 0:
        raise SanityError("no gradient reaches the encoders: the channel path is detached")
    del probe
    return out


@torch.no_grad()
def val_quality(model, cfg: GenConfig, batch: dict, vae, dino, snrs, steps: int = 20) -> dict:
    """PSNR and DINOv2 similarity of real reconstructions of a fixed batch, per SNR."""
    out = {}
    was = model.training
    model.eval()
    x = batch["image"]
    _, cls = dino(x, cfg.grid)
    for i, snr in enumerate(snrs):
        g = torch.Generator(device=x.device).manual_seed(1000 + i)
        s = torch.full((x.shape[0],), float(snr), device=x.device)
        x_hat = model.reconstruct(batch, s, vae, steps=steps, channel_gen=g, sample_gen=g,
                                  cfg_scale=cfg.cfg_scale).float()
        _, cls_hat = dino(x_hat, cfg.grid)
        out[f"psnr@{snr:g}"] = psnr(x, x_hat).mean().item()
        out[f"dino@{snr:g}"] = F.cosine_similarity(cls.float(), cls_hat.float(), dim=1).mean().item()
    model.train(was)
    return out
