"""Is the latent's importance actually graded, and can it be predicted cheaply?

Unequal error protection - spending more of the transmission budget on the parts of the
latent that matter most - rests on two assumptions that are usually asserted rather than
measured:

  1. **Importance is graded.** If every latent symbol contributes about equally to the
     reconstruction, there is nothing to allocate and UEP cannot help, however clever the
     coding is.
  2. **Importance is predictable without measuring it.** Ground truth requires erasing
     each group and re-decoding, which is far too slow to do per image at transmit time.
     A real system needs a cheap proxy computed from the latent itself.

This module measures both. Ground truth comes from *erasure*: knock out one group of
symbols at the receiver, re-decode, and record how far PSNR falls. That is the direct
operational definition of "how much did this matter", and it is exactly the damage a lost
packet does.

Against that it scores cheap predictors - latent energy and gradient saliency - by rank
correlation. A predictor that ranks groups well is a usable UEP signal; one that does not
means attention-free proxies are insufficient and a dedicated importance head is needed.

The approach is adapted from the importance-ranking idea in
https://github.com/Pranavv0307/rs-uep-jscc, which reads per-channel attention gates at
the bottleneck to drive Reed-Solomon tiering. That mapping needs an AF module *on* the
bottleneck so gate channels index transmitted symbols directly. This repo follows ADJSCC
as published - "each FL module is followed by an AF module except the last" - so its final
gates sit on a hidden layer that the last conv then mixes into `c_out`, and there is no
direct channel correspondence to exploit. Measuring importance by erasure sidesteps that
entirely and works on the unmodified architecture.
"""

from __future__ import annotations

import torch

from .data import psnr


def symbol_groups(model, n_groups: int | None = None) -> list[torch.Tensor]:
    """Partition the k transmitted symbols into contiguous groups.

    `_to_symbols` reshapes (B, C, H, W) with `reshape(B, -1, 2)`, so the layout is
    channel-major: latent channel c owns a contiguous run of H*W/2 symbols. Grouping by
    that run therefore means "one latent channel", which is the natural unit both for
    interpreting the result and for any later per-tier protection scheme.

    Args:
        n_groups: number of groups; defaults to one per latent channel (`c_out`).

    Returns:
        A list of index tensors partitioning range(k).
    """
    k = model.k
    # `n_groups or model.c_out` would silently turn an invalid 0 into c_out, so an
    # out-of-range request would return a valid-looking partition instead of failing.
    n_groups = model.c_out if n_groups is None else n_groups
    if n_groups <= 0 or n_groups > k:
        raise ValueError(f"n_groups must be in 1..{k}, got {n_groups}")
    if k % n_groups != 0:
        raise ValueError(f"k={k} is not divisible into {n_groups} equal groups")

    size = k // n_groups
    return [torch.arange(i * size, (i + 1) * size) for i in range(n_groups)]


@torch.no_grad()
def erasure_importance(
    model,
    x: torch.Tensor,
    snr_db: float,
    n_groups: int | None = None,
    repeats: int = 4,
) -> torch.Tensor:
    """Ground-truth importance: PSNR lost when each symbol group is erased.

    Returns a (n_groups,) tensor of mean PSNR drop in dB. Larger means more important.

    Averaged over `repeats` channel realisations, and measured against the model's own
    unerased reconstruction rather than against the source image, so the number isolates
    the damage done by the erasure instead of folding in the model's baseline distortion.
    """
    model.eval()
    groups = symbol_groups(model, n_groups)
    snr = torch.full((x.shape[0],), float(snr_db), device=x.device)

    drops = torch.zeros(len(groups), device=x.device)
    for _ in range(repeats):
        # Fix the channel draw across the intact and erased passes so the comparison
        # isolates the erasure rather than a different noise realisation.
        seed = torch.randint(0, 2**31 - 1, (1,)).item()

        torch.manual_seed(seed)
        intact = psnr(x, model(x, snr)["x_hat"]).mean()

        for i, idx in enumerate(groups):
            mask = torch.zeros(model.k, dtype=torch.bool, device=x.device)
            mask[idx] = True
            torch.manual_seed(seed)
            erased = psnr(x, model(x, snr, erase_mask=mask)["x_hat"]).mean()
            drops[i] += intact - erased

    return drops / repeats


@torch.no_grad()
def energy_importance(model, x: torch.Tensor, snr_db: float,
                      n_groups: int | None = None) -> torch.Tensor:
    """Cheap predictor: mean transmitted power per symbol group.

    Free to compute - the encoder already produced these symbols. The hypothesis is that
    the encoder spends amplitude on what matters. Note the latent is power-normalised as a
    whole, not per group, so groups are free to differ in energy.
    """
    model.eval()
    snr = torch.full((x.shape[0],), float(snr_db), device=x.device)
    z = model.transmit(x, snr)                      # (B, k, 2)
    per_symbol = z.pow(2).sum(-1).mean(0)           # (k,)
    return torch.stack([per_symbol[idx].mean() for idx in symbol_groups(model, n_groups)])


def gradient_importance(model, x: torch.Tensor, snr_db: float,
                        n_groups: int | None = None) -> torch.Tensor:
    """Cheap predictor: gradient saliency of the reconstruction loss w.r.t. each symbol.

    One backward pass, no re-decoding per group. Measures the *local* sensitivity of the
    output to each symbol, which is a first-order approximation of erasing it - so it
    should track the erasure ranking if the response is roughly linear, and diverge where
    the decoder compensates for a missing group non-linearly.
    """
    model.eval()
    snr = torch.full((x.shape[0],), float(snr_db), device=x.device)

    feat, _ = model.encoder(x, snr)
    z = model._to_symbols(feat)
    from .channel import power_normalise

    z = power_normalise(z, model.avg_power)
    if model.quantiser is not None:
        z = model.quantiser(z)
    z.retain_grad()

    y = model.channel_fn(z, snr, model.avg_power)
    x_hat = model.decode(y, snr, z)
    torch.nn.functional.mse_loss(x_hat, x).backward()

    saliency = z.grad.abs().sum(-1).mean(0)         # (k,)
    return torch.stack([saliency[idx].mean() for idx in symbol_groups(model, n_groups)])


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Rank correlation. Rank, not value, is what a tiering scheme actually consumes."""
    if a.numel() < 2:
        return float("nan")

    def ranks(v: torch.Tensor) -> torch.Tensor:
        order = v.argsort()
        r = torch.empty_like(v, dtype=torch.float64)
        r[order] = torch.arange(v.numel(), dtype=torch.float64, device=v.device)
        return r

    ra, rb = ranks(a.double().flatten()), ranks(b.double().flatten())
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = ra.norm() * rb.norm()
    return float((ra @ rb / denom).item()) if denom > 0 else float("nan")


def concentration(importance: torch.Tensor) -> dict:
    """How unevenly importance is distributed - i.e. whether UEP has anything to work with.

    `top_half_share` is the headline: the fraction of total importance carried by the more
    important half of the groups. 0.5 means perfectly flat and UEP is pointless; the
    further above 0.5, the more there is to gain from protecting selectively.
    """
    v = importance.double().flatten()
    v = v - v.min() if v.min() < 0 else v          # drops can go slightly negative
    total = v.sum()
    if total <= 0:
        return {"top_half_share": float("nan"), "max_over_median": float("nan"),
                "ratio_max_min": float("nan")}

    ordered = v.sort(descending=True).values
    half = max(1, v.numel() // 2)
    median = ordered.median()
    return {
        "top_half_share": float((ordered[:half].sum() / total).item()),
        "max_over_median": float((ordered[0] / median).item()) if median > 0 else float("inf"),
        "ratio_max_min": float((ordered[0] / ordered[-1]).item())
        if ordered[-1] > 0 else float("inf"),
    }


def analyse_importance(model, x: torch.Tensor, snrs=(1.0, 10.0, 19.0),
                       n_groups: int | None = None, repeats: int = 4) -> dict:
    """Full study: is importance graded, does that change with SNR, and is it predictable?"""
    out = {
        "n_groups": model.c_out if n_groups is None else n_groups,
        "k": model.k,
        "by_snr": {},
    }

    for snr in snrs:
        truth = erasure_importance(model, x, snr, n_groups, repeats)
        energy = energy_importance(model, x, snr, n_groups)
        grad = gradient_importance(model, x, snr, n_groups)

        out["by_snr"][float(snr)] = {
            "erasure_psnr_drop_db": truth.tolist(),
            "concentration": concentration(truth),
            "predictors": {
                "energy": {"values": energy.tolist(), "spearman": spearman(energy, truth)},
                "gradient": {"values": grad.tolist(), "spearman": spearman(grad, truth)},
            },
        }

    # ADJSCC reports gates becoming more selective as SNR rises. If that mechanism has any
    # bearing on the transmitted latent, importance should concentrate at high SNR too.
    shares = {s: v["concentration"]["top_half_share"] for s, v in out["by_snr"].items()}
    lo, hi = min(shares), max(shares)
    out["concentration_rises_with_snr"] = shares[hi] > shares[lo]
    out["top_half_share_by_snr"] = shares
    return out
