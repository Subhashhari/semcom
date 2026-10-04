"""Pre-registered statistics for the blind-JSCC study.

Implements the decision machinery of docs/proposal/blind-jscc-proposal.md, so that the
rules fixed in advance are applied by code rather than by eye:

  * paired t-intervals across seeds (Student's t on n - 1 degrees of freedom, not normal
    quantiles - with 3-5 seeds the difference is large);
  * equivalence by two one-sided tests at 5%: the 90% t-interval must lie inside the
    margin (default +/-0.15 dB);
  * one-sided superiority tests;
  * Holm correction;
  * the week-1 gate (section 5.1);
  * the RQ2 cue-fit decision rule with a hierarchical bootstrap (section 4.3).

Inputs are per-seed paired differences: for a gap "B - C-att at 18 dB" pass one number per
seed, B's mean PSNR minus C-att's mean PSNR, both from the same seed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats

EQUIVALENCE_MARGIN_DB = 0.15


@dataclass
class Interval:
    n: int
    mean: float
    lo: float
    hi: float
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


def paired_interval(diffs, confidence: float = 0.95) -> Interval:
    """Two-sided t-interval for the mean of per-seed paired differences."""
    d = np.asarray(diffs, dtype=float)
    n = d.size
    mean = float(d.mean()) if n else float("nan")
    if n < 2:
        return Interval(n, mean, float("nan"), float("nan"), confidence)
    se = d.std(ddof=1) / np.sqrt(n)
    half = stats.t.ppf(0.5 + confidence / 2, n - 1) * se
    return Interval(n, mean, mean - half, mean + half, confidence)


def clearly_nonzero(diffs, alpha: float = 0.05) -> bool:
    """True if the two-sided (1 - alpha) t-interval excludes zero."""
    ci = paired_interval(diffs, 1 - alpha)
    return bool(ci.lo > 0 or ci.hi < 0)


def equivalent(diffs, margin: float = EQUIVALENCE_MARGIN_DB, alpha: float = 0.05) -> bool:
    """Two one-sided tests at level alpha: the (1 - 2 alpha) t-interval lies in +/-margin."""
    ci = paired_interval(diffs, 1 - 2 * alpha)
    return bool(ci.lo > -margin and ci.hi < margin)


def superiority_p(diffs) -> float:
    """One-sided p-value for H1: mean difference > 0 (paired t-test)."""
    d = np.asarray(diffs, dtype=float)
    if d.size < 2:
        return float("nan")
    sd = d.std(ddof=1)
    if sd == 0:
        return 0.0 if d.mean() > 0 else 1.0
    t = d.mean() / (sd / np.sqrt(d.size))
    return float(stats.t.sf(t, d.size - 1))


def holm(pvalues, alpha: float = 0.05) -> list[bool]:
    """Holm step-down: which hypotheses are rejected at family-wise level alpha."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    reject = np.zeros(p.size, dtype=bool)
    for rank, i in enumerate(order):
        if p[i] <= alpha / (p.size - rank):
            reject[i] = True
        else:
            break
    return reject.tolist()


def classify(diffs, margin: float = EQUIVALENCE_MARGIN_DB, alpha: float = 0.05) -> dict:
    """Summary of one paired contrast: both intervals and both verdicts.

    "verdict" is "nonzero" if the 95% interval excludes zero, "equivalent" if the 90%
    interval lies inside the margin, both joined if both hold (a small but real
    difference), and "inconclusive" if neither.
    """
    nz = clearly_nonzero(diffs, alpha)
    eq = equivalent(diffs, margin, alpha)
    verdict = (
        "nonzero+equivalent" if nz and eq else "nonzero" if nz else "equivalent" if eq
        else "inconclusive"
    )
    return {
        "ci95": paired_interval(diffs, 1 - alpha).to_dict(),
        "ci90": paired_interval(diffs, 1 - 2 * alpha).to_dict(),
        "clearly_nonzero": nz,
        "equivalent": eq,
        "verdict": verdict,
    }


def gate(g_primary, p_primary, p_all_snrs: dict, margin: float = EQUIVALENCE_MARGIN_DB) -> dict:
    """The week-1 gate (proposal section 5.1).

    Args:
        g_primary: per-seed G = B - C-att at the primary SNR (18 dB), 16-QAM.
        p_primary: per-seed plug-in energy penalty P = B(true SNR) - B(energy estimate) at
            the primary SNR.
        p_all_snrs: {snr: per-seed P} over the evaluation grid, for the insensitivity row.
    """
    p_zero_everywhere = all(equivalent(v, margin) for v in p_all_snrs.values())
    p_nonzero_somewhere = any(
        clearly_nonzero(v) and np.mean(v) > 0 for v in p_all_snrs.values()
    )
    g_eq, g_nz = equivalent(g_primary, margin), clearly_nonzero(g_primary)

    if p_zero_everywhere:
        row, action = (
            "decoder insensitive",
            "Drop the mechanism study; ship plain SI-JSCC-Q and the capacity map",
        )
    elif g_eq and p_nonzero_somewhere:
        row, action = (
            "beats plug-in energy",
            "Continue; mechanism claims come only from the cue-conflict tests",
        )
    elif g_nz:
        diff = np.asarray(g_primary, float) - np.asarray(p_primary, float)
        if clearly_nonzero(diff) and diff.mean() > 0:
            row, action = (
                "G larger than P",
                "Unexpected (P is an upper bound); run the decoder-width sweep (RQ3) first",
            )
        else:
            row, action = (
                "G consistent with P",
                "Continue: the energy story, the digital contrast and capacity",
            )
    else:
        row, action = (
            "inconclusive",
            "Add contingency seeds to the primary contrast, then re-apply the gate",
        )
    return {
        "outcome": row,
        "action": action,
        "G": classify(g_primary, margin),
        "P": classify(p_primary, margin),
        "P_equivalent_to_zero_everywhere": p_zero_everywhere,
        "P_clearly_positive_somewhere": p_nonzero_somewhere,
    }


def cue_fit(
    r: np.ndarray,
    predictions: dict[str, np.ndarray],
    n_boot: int = 10_000,
    win_fraction: float = 0.95,
    seed: int = 0,
) -> dict:
    """The RQ2 cue-fit decision rule (proposal section 4.3).

    Args:
        r: blind arm's double-difference readout, shape (seeds, images, alphas): per image,
            [C-att(alpha) - C-att(1)] - [B_true(alpha) - B_true(1)].
        predictions: {cue: array of the same shape}, per image,
            [B(alpha; s_cue) - B(alpha; true)] - [B(1; s_cue) - B(1; true)].
        n_boot: hierarchical bootstrap resamples (seeds, then images within each seed).

    Returns the winner (or "not resolved" / "tracks none"), the discriminability check,
    and the fit statistics.
    """
    r = np.asarray(r, float)
    cues = list(predictions)
    if r.ndim != 3:
        raise ValueError(f"r must be (seeds, images, alphas), got {r.shape}")
    n_seeds, n_img, n_alpha = r.shape
    resid = {c: r - np.asarray(predictions[c], float) for c in cues}

    # Discriminability margin m: largest 95% half-width of r(alpha) across seeds.
    seed_means = r.mean(axis=1)  # (seeds, alphas)
    halves = [
        (paired_interval(seed_means[:, a]).hi - paired_interval(seed_means[:, a]).lo) / 2
        for a in range(n_alpha)
    ]
    m = float(np.nanmax(halves)) if n_seeds >= 2 else float("nan")

    pred_means = {c: np.asarray(predictions[c], float).mean(axis=(0, 1)) for c in cues}
    pairs_ok = all(
        np.max(np.abs(pred_means[a] - pred_means[b])) > 2 * m
        for i, a in enumerate(cues)
        for b in cues[i + 1 :]
    )

    s_point = {c: float((resid[c].mean(axis=(0, 1)) ** 2).sum()) for c in cues}
    ranked = sorted(cues, key=lambda c: s_point[c])

    # All cues share each resample's indices, so the comparison between cues is paired.
    stacked = np.stack([resid[c] for c in cues])  # (cues, seeds, images, alphas)
    rng = np.random.default_rng(seed)
    wins = np.zeros(len(cues), dtype=int)
    for _ in range(n_boot):
        s_idx = rng.integers(0, n_seeds, n_seeds)
        img_idx = rng.integers(0, n_img, (n_seeds, n_img))
        sample = stacked[:, s_idx[:, None], img_idx]  # (cues, seeds, images, alphas)
        fits = (sample.mean(axis=(1, 2)) ** 2).sum(axis=1)
        wins[int(np.argmin(fits))] += 1

    best = ranked[0]
    win_rate = float(wins[cues.index(best)] / n_boot)
    rms_best = float(np.sqrt(s_point[best] / n_alpha))
    if not pairs_ok:
        outcome = "not discriminable"
    elif rms_best > m:
        outcome = "tracks none"
    elif win_rate >= win_fraction:
        outcome = best
    else:
        outcome = "not resolved"
    return {
        "outcome": outcome,
        "best_cue": best,
        "best_win_rate": win_rate,
        "fit": s_point,
        "rms_residual_best": rms_best,
        "margin_m": m,
        "discriminable": pairs_ok,
        "r_mean": r.mean(axis=(0, 1)).tolist(),
        "prediction_means": {c: v.tolist() for c, v in pred_means.items()},
    }
