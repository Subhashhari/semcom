"""Evaluate one trained blind-JSCC run: everything the week-1 gate and RQ2 need.

    python -m semcom.blind_eval results/blind/r12/b_r0.0833_m16_snr0-20_s0 [more runs...]

Writes into each run directory:

  blind_eval.json         curve, sensitivity curve, plug-in estimates, power-mismatch
                          summary, symbol usage
  blind_eval_images.npz   per-image PSNR at the primary SNRs and under power mismatch,
                          for the paired statistics and the RQ2 bootstrap

**Paired noise.** Channel noise is generated here, not by the model, from a generator
seeded by (SNR, power scale, repeat, batch). The test set is read in a fixed order, so every
run - every arm and every seed - sees exactly the same noise on exactly the same images, and
every decoding of one received block (true SNR, each cue's implied SNR) shares that block.
That is what lets per-image differences between arms be paired.
"""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import torch

from .channel import snr_to_noise_power
from .data import cifar10_loaders, psnr
from .evaluate import load_run, verify_transmitted_symbols
from .snr_estimation import implied_snr_db

PRIMARY_SNRS = (18.0, 20.0)
MISMATCH_SNRS = (10.0, 18.0, 20.0)
ALPHAS = (0.85, 1.0, 1.1, 1.2, 1.3, 1.4)
DELTAS = tuple(float(d) for d in range(-10, 11, 2))
SENSITIVITY_SNRS = (0.0, 4.0, 8.0, 12.0, 16.0, 18.0, 20.0)


def channel_seed(*key) -> int:
    """Deterministic 31-bit seed from a key; stable across processes and machines."""
    return zlib.crc32("|".join(f"{k}" for k in key).encode()) & 0x7FFFFFFF


def paired_awgn(
    z: torch.Tensor, snr_db: float, alpha: float, gen: torch.Generator, avg_power: float = 1.0
) -> torch.Tensor:
    """y = alpha z + n with the noise scaled so the true SNR stays `snr_db`.

    Transmitting alpha z raises the signal power to alpha^2 P, so the noise power is set to
    alpha^2 P 10^(-SNR/10): the SNR is unchanged, but every cue that assumes unit transmit
    power is misled (proposal section 4.3). alpha = 1 is the ordinary AWGN channel.
    """
    sigma_sq = alpha**2 * float(snr_to_noise_power(torch.tensor(float(snr_db)), avg_power))
    noise = torch.randn(z.shape, generator=gen, device=z.device, dtype=z.dtype)
    return alpha * z + noise * (sigma_sq / 2) ** 0.5


def _generator(device: torch.device, *key) -> torch.Generator:
    gen = torch.Generator(device=device)
    gen.manual_seed(channel_seed(*key))
    return gen


def _signal_kurtosis(model, loader, device) -> tuple[float, dict | None]:
    """Kurtosis of the transmitted symbols, and usage statistics when digital."""
    zs = []
    for i, (x, _) in enumerate(loader):
        if i >= 8:
            break
        x = x.to(device)
        zs.append(model.transmit(x, torch.full((x.shape[0],), 10.0, device=device)).float())
    z = torch.cat(zs)
    if model.quantiser is not None:
        from .constellation import usage_statistics

        usage = usage_statistics(z, model.quantiser.points.float())
        return usage["kurtosis"], usage
    r2 = z.pow(2).sum(-1).flatten().double()
    return float((r2.pow(2).mean() / r2.mean().pow(2)).item()), None


@torch.no_grad()
def evaluate_blind(
    run_dir: Path,
    max_images: int | None = None,
    repeats: int = 4,
    snrs=None,
    mismatch_snrs=MISMATCH_SNRS,
    alphas=ALPHAS,
    deltas=DELTAS,
    sensitivity_snrs=SENSITIVITY_SNRS,
    primary_snrs=PRIMARY_SNRS,
    batch_size: int = 250,
    device: torch.device | None = None,
) -> dict:
    run_dir = Path(run_dir)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_run(run_dir, device)
    model.eval()
    enc_snr, dec = cfg.conditioning
    snrs = list(cfg.eval_snrs if snrs is None else snrs)

    _, _, loader = cifar10_loaders(cfg.data_root, batch_size, cfg.num_workers, seed=cfg.seed)
    batches = []
    seen = 0
    for x, _ in loader:
        if max_images is not None and seen >= max_images:
            break
        if max_images is not None:
            x = x[: max_images - seen]
        batches.append(x)
        seen += x.shape[0]

    if model.quantiser is not None and not verify_transmitted_symbols(model, loader, device):
        raise RuntimeError(f"{run_dir}: off-constellation symbols; refusing to report")

    points = model.quantiser.points if model.quantiser is not None else None
    kurtosis, usage = _signal_kurtosis(model, batches_as_loader(batches), device)
    cues = ["energy"] + (["dd", "hybrid"] if points is not None else [])
    m2m4_ok = kurtosis < 1.95
    mismatch_cues = cues + (["m2m4"] if m2m4_ok else [])
    is_b = dec == "snr"

    def transmit(x, snr):
        return model.transmit(x, torch.full((x.shape[0],), float(snr), device=device))

    def decode(y, snr, z, decoder_snr=None):
        s = torch.full((y.shape[0],), float(snr), device=device)
        return model.decode(y, s, z, decoder_snr)

    images: dict[str, np.ndarray] = {}

    # 1) PSNR vs SNR, repeats averaged per image; plug-in estimates for arm B.
    curve, plugin = {}, {c: {} for c in cues} if is_b else {}
    for snr in snrs:
        per_image = []
        plug = {c: [] for c in cues} if is_b else {}
        fails = {c: 0 for c in cues}
        for b, x in enumerate(batches):
            x = x.to(device)
            z = transmit(x, snr)
            acc = torch.zeros(x.shape[0], device=device)
            pacc = {c: torch.zeros(x.shape[0], device=device) for c in plug}
            for rep in range(repeats):
                y = paired_awgn(z, snr, 1.0, _generator(device, "curve", snr, rep, b), model.avg_power)
                acc += psnr(x, decode(y, snr, z))
                for c in plug:
                    s_hat, failed = implied_snr_db(c, y, points, model.avg_power, kurtosis)
                    fails[c] += int(failed.sum())
                    pacc[c] += psnr(x, decode(y, snr, z, s_hat))
            per_image.append((acc / repeats).cpu())
            for c in plug:
                plug[c].append((pacc[c] / repeats).cpu())
        per_image = torch.cat(per_image).numpy()
        curve[float(snr)] = float(per_image.mean())
        if float(snr) in primary_snrs:
            images[f"curve_{snr:g}"] = per_image
        for c in plug:
            arr = torch.cat(plug[c]).numpy()
            plugin[c][float(snr)] = {
                "psnr": float(arr.mean()),
                "penalty": float(per_image.mean() - arr.mean()),
                "failure_rate": fails[c] / (len(per_image) * repeats),
            }
            if float(snr) in primary_snrs:
                images[f"plugin_{c}_{snr:g}"] = arr

    # 2) Sensitivity curve for arm B: told SNR_true + delta, clipped and unclipped.
    sensitivity = {}
    if is_b:
        lo, hi = cfg.snr_train_min, cfg.snr_train_max
        for snr in sensitivity_snrs:
            sums = {(d, lab): 0.0 for d in deltas for lab in ("clipped", "unclipped")}
            n = 0
            for b, x in enumerate(batches):
                x = x.to(device)
                z = transmit(x, snr)
                y = paired_awgn(z, snr, 1.0, _generator(device, "curve", snr, 0, b), model.avg_power)
                for delta in deltas:
                    told = float(snr) + float(delta)
                    for lab, value in (("clipped", min(max(told, lo), hi)), ("unclipped", told)):
                        s = torch.full((x.shape[0],), value, device=device)
                        sums[(delta, lab)] += float(psnr(x, decode(y, snr, z, s)).sum())
                n += x.shape[0]
            sensitivity[float(snr)] = {
                float(d): {lab: sums[(d, lab)] / n for lab in ("clipped", "unclipped")}
                for d in deltas
            }

    # 3) Power mismatch: scale the transmitted block, keep the true SNR.
    mismatch = {}
    for snr in mismatch_snrs:
        mismatch[float(snr)] = {}
        for alpha in alphas:
            own, implied = [], {c: [] for c in mismatch_cues}
            by_cue = {c: [] for c in mismatch_cues} if is_b else {}
            for b, x in enumerate(batches):
                x = x.to(device)
                z = alpha * transmit(x, snr)
                y = paired_awgn(z / alpha, snr, alpha, _generator(device, "pm", snr, alpha, b), model.avg_power)
                own.append(psnr(x, decode(y, snr, z)).cpu())
                for c in mismatch_cues:
                    s_hat, _ = implied_snr_db(c, y, points, model.avg_power, kurtosis)
                    implied[c].append(s_hat.cpu())
                    if is_b:
                        by_cue[c].append(psnr(x, decode(y, snr, z, s_hat)).cpu())
            own_arr = torch.cat(own).numpy()
            images[f"pm_{snr:g}_{alpha:g}_own"] = own_arr
            entry = {
                "psnr_own": float(own_arr.mean()),
                "implied_snr_median": {
                    c: float(torch.cat(v).median()) for c, v in implied.items()
                },
            }
            if is_b:
                entry["psnr_at_implied"] = {}
                for c, v in by_cue.items():
                    arr = torch.cat(v).numpy()
                    images[f"pm_{snr:g}_{alpha:g}_{c}"] = arr
                    entry["psnr_at_implied"][c] = float(arr.mean())
            mismatch[float(snr)][float(alpha)] = entry

    result = {
        "run_name": cfg.run_name,
        "arm": cfg.arm,
        "seed": cfg.seed,
        "digital": cfg.digital,
        "modulation_order": cfg.modulation_order if cfg.digital else None,
        "encoder_snr": enc_snr,
        "decoder_input": dec,
        "num_workers": cfg.num_workers,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "n_images": int(sum(x.shape[0] for x in batches)),
        "repeats": repeats,
        "signal_kurtosis": kurtosis,
        "m2m4_identifiable": m2m4_ok,
        "usage": usage,
        "curve": curve,
        "plugin": plugin,
        "sensitivity": sensitivity,
        "mismatch": mismatch,
    }
    (run_dir / "blind_eval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(run_dir / "blind_eval_images.npz", **images)
    print(
        f"{cfg.arm:9s} seed {cfg.seed}  PSNR@18 {curve.get(18.0, float('nan')):.3f} dB"
        + (f"  plug-in energy penalty@18 {plugin['energy'][18.0]['penalty']:.3f} dB"
           if is_b and 18.0 in plugin.get("energy", {}) else "")
    )
    return result


def batches_as_loader(batches):
    """Wrap a list of image batches so helpers written for loaders can iterate it."""
    return [(x, None) for x in batches]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", type=Path, nargs="+")
    p.add_argument("--max-images", type=int, default=None, help="cap the test set (smoke tests)")
    p.add_argument("--repeats", type=int, default=4, help="channel realisations per image")
    args = p.parse_args()
    for run_dir in args.run_dirs:
        evaluate_blind(run_dir, max_images=args.max_images, repeats=args.repeats)


if __name__ == "__main__":
    main()
