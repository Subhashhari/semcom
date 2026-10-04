"""Receiver-side estimators: do they reproduce the proposal's simulations (section 3.2)?

The reference numbers came from an independent numpy simulation. Tolerances are set from
the Monte Carlo spread at the trial counts used here, not chosen to make the tests pass.
"""

import math

import pytest
import torch

from semcom.constellation import qam_constellation, usage_statistics
from semcom.snr_estimation import (
    QAM_KURTOSIS,
    dd_noise,
    energy_noise,
    implied_snr_db,
    m2m4,
    noise_to_snr_db,
    receiver_statistics,
    statistics_dim,
)

QAM16 = qam_constellation(16).double()


def received(snr_db, k=256, trials=2000, alpha=1.0, digital=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    if digital:
        idx = torch.randint(0, 16, (trials, k), generator=g)
        z = QAM16[idx]
    else:
        z = torch.randn(trials, k, 2, generator=g, dtype=torch.float64)
        z = z / z.pow(2).sum(-1).mean(1, keepdim=True).sqrt().unsqueeze(-1)
    sigma_sq = alpha**2 * 10 ** (-snr_db / 10)
    n = torch.randn(trials, k, 2, generator=g, dtype=torch.float64) * math.sqrt(sigma_sq / 2)
    return alpha * z + n


def median_snr_error(est_snr_db, true_snr_db):
    return float((est_snr_db - true_snr_db).median())


def test_energy_estimator_is_unbiased_at_low_snr_and_fails_at_high_snr_small_k():
    y = received(0.0, digital=False)
    snr, failed = noise_to_snr_db(energy_noise(y), clip=None)
    assert abs(median_snr_error(snr, 0.0)) < 0.1
    assert failed.float().mean() == 0

    # Proposal 3.2: 13% failures for analog energy at k = 256, 20 dB.
    _, failed = noise_to_snr_db(energy_noise(received(20.0, digital=False)), clip=None)
    assert 0.09 < failed.float().mean() < 0.17


def test_energy_fails_far_more_often_for_16qam_at_high_snr():
    # Proposal 3.2: 41% for 16-QAM vs 13% analog at k = 256, 20 dB.
    _, failed = noise_to_snr_db(energy_noise(received(20.0)), clip=None)
    assert 0.35 < failed.float().mean() < 0.47


def test_decision_directed_is_exact_at_high_snr_and_biased_at_low_snr():
    snr_hi, _ = noise_to_snr_db(dd_noise(received(20.0), QAM16), clip=None)
    assert abs(median_snr_error(snr_hi, 20.0)) < 0.1
    # Over-estimates the SNR by about 4.6 dB at 0 dB.
    snr_lo, _ = noise_to_snr_db(dd_noise(received(0.0), QAM16), clip=None)
    assert 4.2 < median_snr_error(snr_lo, 0.0) < 5.0


def test_m2m4_is_unbiased_with_the_right_kurtosis():
    s, n = m2m4(received(10.0, k=4096, trials=200), QAM_KURTOSIS[16])
    est = 10 * torch.log10(s / n)
    assert abs(median_snr_error(est, 10.0)) < 0.2


def test_power_mismatch_misleads_energy_and_dd_but_not_m2m4():
    # Proposal 4.3 table, true 18 dB, alpha = 1.1: energy ~6.4, DD ~15.4, M2M4 ~18.2.
    y = received(18.0, k=4096, trials=200, alpha=1.1)
    energy, _ = implied_snr_db("energy", y, QAM16, clip=None)
    dd, _ = implied_snr_db("dd", y, QAM16, clip=None)
    mm, _ = implied_snr_db("m2m4", y, QAM16, kurtosis=QAM_KURTOSIS[16], clip=None)
    assert abs(float(energy.median()) - 6.4) < 0.5
    assert abs(float(dd.median()) - 15.4) < 0.5
    assert abs(float(mm.median()) - 18.0) < 0.5


def test_failures_are_floored_at_the_top_of_the_clip_range():
    snr, failed = noise_to_snr_db(torch.tensor([-0.1, 0.01]), clip=(0.0, 20.0))
    assert failed.tolist() == [True, False]
    assert snr[0].item() == 20.0 and abs(snr[1].item() - 20.0) < 1e-6


def test_receiver_statistics_shapes_and_scale():
    y = received(10.0, trials=8).float()
    assert receiver_statistics(y, QAM16.float()).shape == (8, statistics_dim(True))
    assert receiver_statistics(y).shape == (8, statistics_dim(False))
    assert receiver_statistics(y, QAM16.float()).abs().max() < 10


def test_usage_statistics_for_uniform_16qam():
    z = QAM16[torch.randint(0, 16, (200_000,))]
    stats = usage_statistics(z, QAM16)
    assert stats["entropy_bits"] == pytest.approx(4.0, abs=0.01)
    assert stats["kurtosis"] == pytest.approx(1.32, abs=0.01)
