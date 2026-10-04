"""Receiver-side noise and SNR estimators for the blind-JSCC study.

Every estimator here sees only the received block `y` (and, where stated, the known
constellation), never the true SNR. They serve three purposes:

  * as the candidate cues a blind decoder might be implementing (energy, decision-directed,
    hybrid, moment-based), whose implied SNRs the cue-conflict tests compare against;
  * as plug-in estimates fed to the SNR-conditioned arm B, to predict what an
    energy-reading decoder would lose;
  * as the statistics vector of the self-conditioned arm D.

Latents are (batch, k, 2) real tensors of (I, Q) pairs, as everywhere else in the repo.
All estimates are per image: one value per row of the batch.

Reference for the estimator families: N. Pauluzzi and N. C. Beaulieu, "A comparison of SNR
estimation techniques for the AWGN channel," IEEE Trans. Commun., 48(10), 2000.
"""

from __future__ import annotations

import torch

#: Kurtosis E|c|^4 / (E|c|^2)^2 of square M-QAM under uniform usage. The M2M4 estimator
#: needs it; a complex Gaussian has 2, which is why M2M4 cannot separate a Gaussian-like
#: signal from Gaussian noise.
QAM_KURTOSIS = {4: 1.0, 16: 1.32, 64: 1.381, 256: 1.395}


def _check(y: torch.Tensor) -> None:
    if y.dim() != 3 or y.shape[-1] != 2:
        raise ValueError(f"expected (batch, k, 2), got {tuple(y.shape)}")


def mean_power(y: torch.Tensor) -> torch.Tensor:
    """Mean per-symbol power of each block, shape (batch,)."""
    _check(y)
    return y.pow(2).sum(dim=-1).mean(dim=1)


def hard_decisions(y: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Nearest constellation point for every received symbol, same shape as `y`.

    Uses the quadratic expansion, like the quantiser, for the same precision reasons.
    """
    _check(y)
    flat = y.reshape(-1, 2).to(points.dtype)
    dist = flat.pow(2).sum(1, keepdim=True) - 2.0 * flat @ points.T + points.pow(2).sum(1)
    return points[dist.argmin(dim=1)].reshape(y.shape).to(y.dtype)


def energy_noise(y: torch.Tensor, signal_power: float = 1.0) -> torch.Tensor:
    """Noise variance from received energy, assuming known transmit power.

    sigma^2 = (||y||^2 - k * P) / k. Unbiased when every block is transmitted at exactly
    power P, which per-image power normalisation guarantees in analog JSCC but not after
    quantisation. Can be negative at high SNR (the signal-noise cross term dominates);
    that is a failure, handled by `noise_to_snr_db`.
    """
    return mean_power(y) - signal_power


def dd_noise(y: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Decision-directed noise variance: mean squared residual to the nearest point.

    Exact above the symbol-error threshold; under-estimates the noise (over-estimates the
    SNR) below it, because wrong decisions snap to closer points.
    """
    return (y - hard_decisions(y, points)).pow(2).sum(dim=-1).mean(dim=1)


def hybrid_noise(y: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Received energy minus the energy of the decisions: mean|y|^2 - mean|Q(y)|^2.

    Repairs the energy cue for QAM, whose transmitted energy varies per image, but only up
    to analog quality: with correct decisions it still carries the cross term.
    """
    return mean_power(y) - mean_power(hard_decisions(y, points))


def m2m4(y: torch.Tensor, kurtosis: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Moment-based (M2M4) signal and noise power estimates, each shape (batch,).

    S = sqrt((2 M2^2 - M4) / (2 - k_a)), N = M2 - S, for complex Gaussian noise and a
    signal of kurtosis k_a. Unlike the energy estimator it does not assume the transmit
    power, so it is unaffected by a transmit-power mismatch.
    """
    if not kurtosis < 2.0:
        raise ValueError(f"M2M4 needs signal kurtosis < 2, got {kurtosis}")
    _check(y)
    r2 = y.pow(2).sum(dim=-1)
    m2 = r2.mean(dim=1)
    m4 = r2.pow(2).mean(dim=1)
    s = ((2.0 * m2.pow(2) - m4).clamp_min(0.0) / (2.0 - kurtosis)).sqrt()
    return s, m2 - s


def noise_to_snr_db(
    noise: torch.Tensor,
    signal_power: float | torch.Tensor = 1.0,
    clip: tuple[float, float] | None = (0.0, 20.0),
    failure_value: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a noise-variance estimate to an SNR in dB.

    Returns (snr_db, failed). A non-positive noise estimate is a failure: its SNR is set to
    `failure_value`, which defaults to the top of `clip` (the proposal floors failures at
    the top of B's input range). With `clip=None` nothing is clipped and failures default
    to +inf.
    """
    failed = noise <= 0
    safe = noise.clamp_min(1e-12)
    snr = 10.0 * torch.log10(torch.as_tensor(signal_power, dtype=noise.dtype) / safe)
    if failure_value is None:
        failure_value = clip[1] if clip is not None else float("inf")
    snr = torch.where(failed, torch.full_like(snr, failure_value), snr)
    if clip is not None:
        snr = snr.clamp(clip[0], clip[1])
    return snr, failed


def implied_snr_db(
    cue: str,
    y: torch.Tensor,
    points: torch.Tensor | None = None,
    signal_power: float = 1.0,
    kurtosis: float | None = None,
    clip: tuple[float, float] | None = (0.0, 20.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-image SNR implied by one named cue. Returns (snr_db, failed).

    `cue` is one of "energy", "dd", "hybrid", "m2m4". "dd" and "hybrid" need the
    constellation; "m2m4" needs the signal kurtosis.
    """
    if cue == "energy":
        return noise_to_snr_db(energy_noise(y, signal_power), signal_power, clip)
    if cue in ("dd", "hybrid"):
        if points is None:
            raise ValueError(f"cue {cue!r} needs the constellation points")
        noise = dd_noise(y, points) if cue == "dd" else hybrid_noise(y, points)
        return noise_to_snr_db(noise, signal_power, clip)
    if cue == "m2m4":
        if kurtosis is None:
            raise ValueError("cue 'm2m4' needs the signal kurtosis")
        s, n = m2m4(y, kurtosis)
        # M2M4 estimates the signal power too, so the SNR is S/N, not P/N.
        failed = (n <= 0) | (s <= 0)
        snr, _ = noise_to_snr_db(n.clamp_min(1e-12), s.clamp_min(1e-12), clip)
        top = clip[1] if clip is not None else float("inf")
        snr = torch.where(failed, torch.full_like(snr, top), snr)
        return snr, failed
    raise ValueError(f"unknown cue {cue!r}")


def statistics_dim(digital: bool) -> int:
    """Width of the statistics vector `receiver_statistics` returns."""
    return 5 if digital else 3


def receiver_statistics(
    y: torch.Tensor, points: torch.Tensor | None = None, signal_power: float = 1.0
) -> torch.Tensor:
    """The self-conditioned arm D's input: pilot-free statistics of the whole block.

    Shape (batch, 5) with a constellation, (batch, 3) without:

        energy noise estimate     (||y||^2 - kP) / k             low and mid SNR
        log10 DD noise estimate   log10 mean|y - Q(y)|^2          high SNR      [digital]
        hybrid noise estimate     mean|y|^2 - mean|Q(y)|^2        mid SNR       [digital]
        M2                        mean|y|^2                       scale
        M4 / M2^2                 normalised fourth moment        scale-invariant

    All are O(1) over 0-20 dB, so they can enter the attention MLP without rescaling.
    """
    _check(y)
    r2 = y.pow(2).sum(dim=-1)
    m2 = r2.mean(dim=1)
    m4 = r2.pow(2).mean(dim=1)
    feats = [m2 - signal_power]
    if points is not None:
        decisions = hard_decisions(y, points)
        feats.append(torch.log10((y - decisions).pow(2).sum(-1).mean(1).clamp_min(1e-6)))
        feats.append(m2 - mean_power(decisions))
    feats += [m2, m4 / m2.pow(2).clamp_min(1e-12)]
    return torch.stack(feats, dim=1)
