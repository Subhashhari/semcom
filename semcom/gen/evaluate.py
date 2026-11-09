"""Evaluate trained generative-JSCC runs on the test split.

    python -m semcom.gen.evaluate results/gen/p128/dit-S_cbr48_spec0_s0 [more runs...]
    python -m semcom.gen.evaluate RUN --steps 1 2 4 --suffix fewstep   # keeps eval.json

For each SNR in eval_snrs and (DiT only) each step count in sample_steps: PSNR, LPIPS,
DINOv2 similarity and FID over the first `eval_images` test images, plus the receiver's
decoding time per image.

**Paired randomness.** Channel noise is drawn from a generator seeded by (SNR, batch) and the
diffusion sampling noise from one seeded by (SNR, steps, batch), the test images are read in a
fixed order, and the noise does not depend on the arm. So every arm, specialist or not, sees the
same channel realisations on the same images, and per-image differences between arms are paired.

Writes into each run folder:
    eval.json           means per (SNR, steps), timing, settings
    eval_images.npz     per-image metrics, for paired statistics across arms
    eval_samples.png    first 8 test images, then one row per (SNR, steps)
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .checks import kv, symbol_stats
from .config import GenConfig
from .data import CachedSplit, prepare
from .frozen import load_frozen
from .metrics import Metrics
from .model import build_model


def seed_of(*key) -> int:
    """Deterministic 31-bit seed from a key; stable across processes and machines."""
    return zlib.crc32("|".join(f"{k}" for k in key).encode()) & 0x7FFFFFFF


def load_run(run_dir: Path, device: torch.device) -> tuple[torch.nn.Module, GenConfig]:
    cfg = GenConfig.from_yaml(run_dir / "config.yaml")
    path = run_dir / ("best.pt" if cfg.eval_checkpoint == "best" and (run_dir / "best.pt").exists() else "final.pt")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    print(f"evaluating {path.name} (step {ckpt.get('step')})", flush=True)
    model = build_model(cfg, ckpt["dino_dim"]).to(device)
    model.load_state_dict(ckpt["ema"])
    return model.eval(), cfg


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate_run(run_dir: Path, device: torch.device | None = None, eval_images: int | None = None,
                 snrs=None, steps_list=None, external: bool = True, suffix: str = "") -> dict:
    """`external=False` skips LPIPS and FID, whose networks download on first use. A `suffix`
    writes eval-<suffix>.json etc. next to the run's main evaluation instead of replacing it."""
    if suffix and not all(c.isalnum() or c == "-" for c in suffix):
        raise ValueError(f"suffix must be letters, digits and '-', got {suffix!r}")
    tail = f"-{suffix}" if suffix else ""
    run_dir = Path(run_dir)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_run(run_dir, device)
    prepare(cfg, device)
    vae, dino = load_frozen(cfg, device)
    metrics = Metrics(dino, device, fid=external, lpips_net=external)

    n = eval_images if eval_images is not None else cfg.eval_images
    test = CachedSplit(cfg.cache_dir, "test")
    if n:
        test = Subset(test, range(min(n, len(test))))
    loader = DataLoader(test, batch_size=cfg.eval_batch, shuffle=False, num_workers=0)
    snrs = list(cfg.eval_snrs if snrs is None else snrs)
    steps_list = list(cfg.sample_steps if steps_list is None else steps_list) if cfg.model == "dit" else [None]

    results, images, sample_rows = {}, {}, []
    first = None
    symbols = None
    if getattr(model, "quantiser", None) is not None:
        # Digital runs: every transmitted symbol of the evaluated images must be an exact
        # constellation point (symbol_stats raises otherwise); usage statistics go in eval.json.
        stats = []
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            stats.append(symbol_stats(model, batch))
        symbols = {k: float(np.mean([st[k] for st in stats])) for k in stats[0]}
        symbols.update(on_constellation=True, modulation_order=cfg.modulation_order)
        symbols["power_ref"] = cfg.power_ref
        if hasattr(model, "es_ref"):
            # SNR against the energy actually sent: one constant for the whole evaluation,
            # measured on these test images (equal-size batches, so the batch mean is the mean).
            symbols["es_ref_trained"] = model.es_ref.item()
            model.es_ref.fill_(symbols["qam_power"])
        print(f"SYMBOLS run={cfg.run_name} " + kv(**{k: v for k, v in symbols.items()
                                                    if k not in ("on_constellation", "power_ref")}),
              flush=True)
    for snr in snrs:
        for steps in steps_list:
            key = f"{snr:g}" + ("" if steps is None else f"_{steps}")
            per = {}
            fid_acc = metrics.new_fid()
            seconds, count = 0.0, 0
            for b, batch in enumerate(loader):
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                x = batch["image"]
                if first is None:
                    first = x[:8]
                s = torch.full((x.shape[0],), float(snr), device=device)
                ch = torch.Generator(device=device).manual_seed(seed_of("channel", snr, b))
                sg = torch.Generator(device=device).manual_seed(seed_of("sample", snr, steps, b))
                _sync(device)
                t0 = time.time()
                x_hat = model.reconstruct(batch, s, vae, steps=steps, channel_gen=ch, sample_gen=sg,
                                          cfg_scale=cfg.cfg_scale).float()
                _sync(device)
                seconds += time.time() - t0
                count += x.shape[0]
                for k, v in metrics.per_image(x, x_hat, cfg.grid).items():
                    per.setdefault(k, []).append(v.cpu())
                if fid_acc is not None:
                    fid_acc.update(x, real=True)
                    fid_acc.update(x_hat, real=False)
                if b == 0:
                    sample_rows.append(x_hat[:8].cpu())
            entry = {k: float(torch.cat(v).mean()) for k, v in per.items()}
            if fid_acc is not None:
                entry["fid"] = float(fid_acc.compute())
            entry["seconds_per_image"] = seconds / max(count, 1)
            entry["n_images"] = count
            results[key] = entry
            for k, v in per.items():
                images[f"{k}_{key}"] = torch.cat(v).numpy()
            print(f"EVAL run={cfg.run_name} " + kv(snr=float(snr), steps=steps or 0, **entry), flush=True)

    out = {"run_name": cfg.run_name, "model": cfg.model, "arm": cfg.arm, "snr_mode": cfg.snr_mode,
           "seed": cfg.seed, "checkpoint": cfg.eval_checkpoint, "k": cfg.k, "bandwidth_ratio": cfg.bandwidth_ratio,
           "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
           "digital": cfg.digital, "modulation_order": cfg.modulation_order if cfg.digital else None,
           "symbols": symbols,
           "snrs": snrs, "steps": steps_list, "cfg_scale": cfg.cfg_scale, "results": results}
    (run_dir / f"eval{tail}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    np.savez_compressed(run_dir / f"eval_images{tail}.npz", **images)
    if first is not None:
        from torchvision.utils import save_image

        save_image(torch.cat([first.cpu()] + sample_rows), run_dir / f"eval_samples{tail}.png", nrow=first.shape[0])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", type=Path, nargs="+")
    p.add_argument("--eval-images", type=int, default=None)
    p.add_argument("--no-external", action="store_true", help="skip LPIPS and FID")
    p.add_argument("--snrs", nargs="+", type=float, default=None)
    p.add_argument("--steps", nargs="+", type=int, default=None, help="DiT sampling step counts")
    p.add_argument("--suffix", default="", help="write eval-<suffix>.json etc. (keeps eval.json)")
    args = p.parse_args()
    for run_dir in args.run_dirs:
        evaluate_run(run_dir, eval_images=args.eval_images, external=not args.no_external,
                     snrs=args.snrs, steps_list=args.steps, suffix=args.suffix)


if __name__ == "__main__":
    main()
