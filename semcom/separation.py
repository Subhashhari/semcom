"""Separation-based baseline: classical source coding + channel coding.

This is the external reference point the 2x2 ablation lacks on its own. Without it the
learned arms can only be compared against each other, and the cliff effect - the property
the whole line of work exists to demonstrate - cannot be shown at all.

Two modes, because they exhibit the two different failure shapes:

  * `ideal` - the source codec plus a *capacity-achieving* channel code, with the rate
    re-chosen at every SNR. This is the standard construction in the DeepJSCC literature
    ("BPG with a capacity-achieving channel code represents the best performance
    obtainable by a separation-based scheme"), and it is an **upper bound**: no real
    separation scheme can beat it, because no real code beats capacity and no real system
    gets to re-optimise per-SNR with an oracle. If a learned scheme beats this curve, it
    beats every separation scheme, not just the one we happened to implement.

  * `fixed_mcs` - a fixed (modulation, code rate) pair, as a real link with a chosen MCS
    would use. The bit budget no longer tracks the channel, so quality is flat above the
    decoding threshold and the transport block is lost entirely below it. This is what
    manufactures the cliff.

Success is adjudicated by comparing the scheme's spectral efficiency against Shannon
capacity. That treats the channel code as ideal, which is deliberately generous to the
baseline: a real 5G LDPC code sits roughly 1 dB worse than this.

A note on resolution, which matters more than it looks. Classical codecs carry a fixed
header, and at 32x32 that header dominates: the same JPEG2000 encoder that reaches
0.13 bpp on a 128x128 image cannot go below ~2 bpp on a CIFAR tile. So on CIFAR-10 the
separation baseline is genuinely infeasible at low rates - not because the comparison is
rigged, but because classical codecs were never designed for 32x32 payloads. That is a
real, reportable result, and `feasible=False` records it rather than hiding it behind an
interpolated number.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

# Codecs in preference order, each with the knob that trades size against quality and the
# direction that knob moves. WebP has by far the lowest floor at 32x32, which is why it
# leads; JPEG2000 is the closest available stand-in for the BPG used in the literature.
CODECS = {
    "webp": dict(fmt="WEBP", lo=0, hi=100, better_is_higher=True),
    "jpeg2000": dict(fmt="JPEG2000", lo=1, hi=1000, better_is_higher=False),
    "jpeg": dict(fmt="JPEG", lo=1, hi=95, better_is_higher=True),
}


def capacity_bits_per_symbol(snr_db: float) -> float:
    """Shannon capacity of the complex AWGN channel, bits per channel use."""
    return math.log2(1.0 + 10.0 ** (snr_db / 10.0))


def shannon_threshold_db(spectral_efficiency: float) -> float:
    """Lowest SNR at which capacity supports the given bits/symbol.

    Inverts C = log2(1+SNR). Below this the transport block cannot be decoded by *any*
    code, so it is the most generous possible location for the cliff edge.
    """
    return 10.0 * math.log10(2.0**spectral_efficiency - 1.0)


def _encode_at(img: Image.Image, codec: str, knob: int) -> bytes:
    spec = CODECS[codec]
    buf = io.BytesIO()
    if spec["fmt"] == "JPEG2000":
        img.save(buf, format="JPEG2000", quality_mode="rates",
                 quality_layers=[knob], irreversible=True)
    elif spec["fmt"] == "WEBP":
        img.save(buf, format="WEBP", quality=knob, method=6)
    else:
        img.save(buf, format="JPEG", quality=knob, optimize=True)
    return buf.getvalue()


@dataclass
class CompressionResult:
    feasible: bool
    bytes_used: int = 0
    floor_bytes: int = 0
    decoded: np.ndarray | None = None
    codec: str = ""
    quality: float = -math.inf  # PSNR of this encoding, used to pick between codecs


def compress_to_budget(
    img: Image.Image, budget_bytes: float, codecs=("webp", "jpeg2000", "jpeg")
) -> CompressionResult:
    """Compress `img` into at most `budget_bytes`, at the best quality that fits.

    Tries each codec by binary search over its quality knob and keeps the encoding with
    the highest quality setting that still fits. Returns `feasible=False` when no codec
    can produce a file that small - the header-floor case described in the module
    docstring - along with the smallest floor seen, so the caller can report how far
    short the codec fell.
    """
    budget = int(math.floor(budget_bytes))
    best: CompressionResult | None = None
    best_floor = math.inf

    for codec in codecs:
        spec = CODECS[codec]
        lo, hi = spec["lo"], spec["hi"]

        # The knob value giving the smallest file tells us this codec's floor.
        floor_knob = lo if spec["better_is_higher"] else hi
        try:
            floor = len(_encode_at(img, codec, floor_knob))
        except Exception:
            continue
        best_floor = min(best_floor, floor)
        if floor > budget:
            continue  # this codec cannot reach the budget at all

        # Binary search for the best quality that still fits.
        good_knob, good_data = floor_knob, _encode_at(img, codec, floor_knob)
        for _ in range(12):
            mid = (lo + hi) // 2
            if mid in (lo, hi):
                break
            try:
                data = _encode_at(img, codec, mid)
            except Exception:
                break
            fits = len(data) <= budget
            # "Better" means larger knob for quality-style codecs, smaller for rate-style.
            if fits:
                good_knob, good_data = mid, data
                if spec["better_is_higher"]:
                    lo = mid
                else:
                    hi = mid
            else:
                if spec["better_is_higher"]:
                    hi = mid
                else:
                    lo = mid

        decoded = np.asarray(Image.open(io.BytesIO(good_data)).convert("RGB"))
        quality = _psnr(np.asarray(img.convert("RGB")), decoded)

        # Select across codecs by reconstruction quality, never by file size. Within one
        # codec more bytes do mean better quality, but across codecs they do not: at
        # 32x32, JPEG2000 becomes feasible around 270 bytes and produces a larger file
        # than WebP while reconstructing *worse*. Choosing by size therefore made the
        # rate-distortion curve non-monotonic - quality collapsed at exactly the SNR
        # where the budget crossed the JPEG2000 floor.
        if best is None or quality > best.quality:
            best = CompressionResult(True, len(good_data), floor, decoded, codec, quality)

    if best is not None:
        return best
    return CompressionResult(False, floor_bytes=int(best_floor) if best_floor < math.inf else 0)


def _psnr(ref: np.ndarray, rec: np.ndarray) -> float:
    mse = ((ref.astype(np.float64) - rec.astype(np.float64)) ** 2).mean()
    return 99.0 if mse <= 0 else 10.0 * math.log10(255.0**2 / mse)


@torch.no_grad()
def evaluate_separation(
    images: np.ndarray,
    k: int,
    snrs,
    mode: str = "ideal",
    bits_per_symbol: int = 4,
    code_rate: float = 0.5,
    codecs=("webp", "jpeg2000", "jpeg"),
) -> dict:
    """Evaluate a separation pipeline over an SNR sweep.

    Args:
        images: (N, H, W, 3) uint8 array.
        k: complex channel symbols available per image - the same budget the learned
           arms get, which is what makes the comparison bandwidth-matched.
        mode: "ideal" (capacity-achieving code, rate re-chosen per SNR) or "fixed_mcs".
        bits_per_symbol / code_rate: the MCS, used only in "fixed_mcs" mode.

    Returns a dict with, per SNR, the mean PSNR over the images that could be encoded,
    the fraction that were feasible, and whether the block decoded at all.
    """
    if mode not in ("ideal", "fixed_mcs"):
        raise ValueError(f"mode must be 'ideal' or 'fixed_mcs', got {mode!r}")

    results, threshold = {}, None
    if mode == "fixed_mcs":
        spectral_efficiency = bits_per_symbol * code_rate
        threshold = shannon_threshold_db(spectral_efficiency)

    for snr in snrs:
        snr = float(snr)

        if mode == "ideal":
            # Rate tracks the channel: an oracle picking the best MCS at every SNR.
            budget_bytes = k * capacity_bits_per_symbol(snr) / 8.0
            decodable = True
        else:
            # Rate is fixed; the only question is whether the channel supports it.
            budget_bytes = k * bits_per_symbol * code_rate / 8.0
            decodable = snr >= threshold

        if not decodable:
            # Below threshold the transport block is lost. Not a degraded image - none.
            results[snr] = {
                "psnr": None, "decoded": False, "feasible_fraction": 0.0,
                "budget_bytes": budget_bytes, "mean_bytes_used": 0.0,
                "status": "channel_outage",
            }
            continue

        psnrs, used, n_feasible, floors = [], [], 0, []
        for arr in images:
            img = Image.fromarray(arr)
            out = compress_to_budget(img, budget_bytes, codecs)
            if out.feasible:
                n_feasible += 1
                psnrs.append(_psnr(arr, out.decoded))
                used.append(out.bytes_used)
            else:
                floors.append(out.floor_bytes)

        # Two failure modes that look identical in a PSNR column but are not the same
        # thing at all, so they are labelled separately. "channel_outage" is the cliff -
        # the link could not carry the block. "codec_infeasible" means the channel was
        # fine and the *source coder* could not produce a file small enough, which on
        # 32x32 images is a header-floor artefact rather than a statement about wireless.
        frac = n_feasible / len(images)
        results[snr] = {
            "psnr": float(np.mean(psnrs)) if psnrs else None,
            "decoded": True,
            "feasible_fraction": frac,
            "budget_bytes": budget_bytes,
            "mean_bytes_used": float(np.mean(used)) if used else 0.0,
            "codec_floor_bytes": float(np.mean(floors)) if floors else None,
            "status": "ok" if frac == 1.0 else ("codec_infeasible" if frac == 0.0 else "partial"),
        }

    return {
        "mode": mode,
        "k": k,
        "bits_per_symbol": bits_per_symbol if mode == "fixed_mcs" else None,
        "code_rate": code_rate if mode == "fixed_mcs" else None,
        "threshold_db": threshold,
        "by_snr": results,
    }


def cifar_test_images(n: int = 64, root: str = "data", seed: int = 0) -> np.ndarray:
    """A deterministic subset of the CIFAR-10 test set as a uint8 array."""
    from torchvision import datasets

    ds = datasets.CIFAR10(root, train=False, download=False)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    return np.stack([np.asarray(ds[int(i)][0]) for i in idx])


# A small 5G-NR-like MCS table: (bits per symbol, code rate, label).
MCS_TABLE = [
    (2, 0.5, "QPSK r1/2"),
    (4, 0.5, "16-QAM r1/2"),
    (4, 0.75, "16-QAM r3/4"),
    (6, 0.75, "64-QAM r3/4"),
    (8, 0.75, "256-QAM r3/4"),
]


def choose_fixed_mcs(k: int, snrs, codec_floor_bytes: float = 96.0):
    """Pick an MCS whose cliff is visible in the swept range and whose budget is usable.

    Two constraints, and on CIFAR-10 they genuinely conflict:

      * the Shannon threshold must fall *inside* the SNR sweep, or there is no cliff to
        see - the curve is either flat everywhere or absent everywhere;
      * the resulting byte budget must clear the source codec's floor, or every point
        reports `codec_infeasible` and the plot says nothing about wireless at all.

    Returns the lowest-order MCS satisfying both, or None if no entry does - which is
    itself the honest answer at very small `k`, and the caller should say so rather than
    quietly plotting an MCS whose cliff is off-screen.
    """
    lo, hi = min(snrs), max(snrs)
    for bits, rate, label in MCS_TABLE:
        threshold = shannon_threshold_db(bits * rate)
        budget = k * bits * rate / 8.0
        if lo < threshold < hi and budget >= codec_floor_bytes:
            return {"bits_per_symbol": bits, "code_rate": rate,
                    "label": label, "threshold_db": threshold, "budget_bytes": budget}
    return None


def measure_codec_floor(images: np.ndarray, codecs=("webp", "jpeg2000", "jpeg")) -> float:
    """Mean smallest file any codec can produce for these images, in bytes.

    Used to decide whether an MCS is viable before plotting it, and worth reporting in
    its own right: on 32x32 inputs it is the binding constraint, not the channel.
    """
    floors = []
    for arr in images:
        out = compress_to_budget(Image.fromarray(arr), 0, codecs)
        floors.append(out.floor_bytes)
    return float(np.mean(floors))
