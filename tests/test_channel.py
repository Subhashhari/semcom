"""Tests for power normalisation and the channel models.

Power normalisation is the quiet correctness risk in this codebase: if the transmitted
power does not actually match the constellation's fixed scale, every SNR on every plot is
wrong by an unknown offset and nothing downstream is comparable.
"""

import pytest
import torch

from semcom.channel import (
    awgn,
    get_channel,
    power_normalise,
    rayleigh_slow_fading,
    snr_to_noise_power,
)


def measured_power(z: torch.Tensor) -> torch.Tensor:
    """Average power per complex symbol, per example."""
    return z.pow(2).sum(dim=(1, 2)) / z.shape[1]


@pytest.mark.parametrize("avg_power", [0.5, 1.0, 2.0])
def test_power_normalise_hits_target_exactly(avg_power):
    z = torch.randn(16, 256, 2) * torch.rand(16, 1, 1) * 10
    out = power_normalise(z, avg_power)
    torch.testing.assert_close(
        measured_power(out), torch.full((16,), avg_power), rtol=1e-5, atol=1e-5
    )


def test_power_normalise_is_per_example():
    """Examples with wildly different scales must each land on unit power."""
    z = torch.randn(4, 128, 2)
    z[0] *= 1e-3
    z[3] *= 1e3
    out = power_normalise(z)
    torch.testing.assert_close(measured_power(out), torch.ones(4), rtol=1e-4, atol=1e-4)


def test_power_normalise_preserves_direction():
    z = torch.randn(2, 64, 2)
    out = power_normalise(z)
    cos = torch.nn.functional.cosine_similarity(
        z.reshape(2, -1), out.reshape(2, -1), dim=1
    )
    torch.testing.assert_close(cos, torch.ones(2), rtol=1e-5, atol=1e-5)


def test_power_normalise_survives_all_zero_input():
    """Must not produce NaN; the clamp_min guard is what prevents a silent training death."""
    out = power_normalise(torch.zeros(2, 32, 2))
    assert torch.isfinite(out).all()


def test_power_normalise_rejects_wrong_shape():
    with pytest.raises(ValueError):
        power_normalise(torch.randn(8, 32))
    with pytest.raises(ValueError):
        power_normalise(torch.randn(8, 32, 3))


def test_snr_to_noise_power_matches_definition():
    snr = torch.tensor([0.0, 10.0, 20.0])
    torch.testing.assert_close(
        snr_to_noise_power(snr), torch.tensor([1.0, 0.1, 0.01]), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("snr_db", [0.0, 5.0, 10.0, 20.0])
def test_awgn_noise_power_matches_requested_snr(snr_db):
    """Empirical SNR of the channel output must match the requested SNR."""
    torch.manual_seed(0)
    z = power_normalise(torch.randn(256, 512, 2))
    y = awgn(z, torch.full((256,), snr_db))

    noise_power = (y - z).pow(2).sum(dim=(1, 2)).mean() / z.shape[1]
    measured_snr = 10 * torch.log10(1.0 / noise_power)
    assert measured_snr.item() == pytest.approx(snr_db, abs=0.2)


def test_awgn_splits_noise_evenly_across_i_and_q():
    torch.manual_seed(0)
    z = torch.zeros(512, 256, 2)
    y = awgn(z, torch.zeros(512))
    # Total complex noise power 1.0 at 0 dB => 0.5 per component.
    assert y[..., 0].pow(2).mean().item() == pytest.approx(0.5, abs=0.02)
    assert y[..., 1].pow(2).mean().item() == pytest.approx(0.5, abs=0.02)


def test_awgn_accepts_per_example_snr():
    """A batch with mixed SNRs must apply each example's own noise level."""
    torch.manual_seed(0)
    z = power_normalise(torch.randn(2, 4096, 2))
    snr = torch.tensor([0.0, 20.0])
    y = awgn(z, snr)
    per_example_noise = (y - z).pow(2).sum(dim=(1, 2)) / z.shape[1]
    assert per_example_noise[0] > per_example_noise[1] * 50


def test_awgn_is_differentiable():
    z = power_normalise(torch.randn(4, 64, 2)).requires_grad_(True)
    awgn(z, torch.full((4,), 10.0)).pow(2).sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_rayleigh_equalisation_recovers_signal_at_high_snr():
    """With negligible noise, equalisation must invert the fade and return z."""
    torch.manual_seed(0)
    z = power_normalise(torch.randn(64, 128, 2))
    y = rayleigh_slow_fading(z, torch.full((64,), 100.0))
    torch.testing.assert_close(y, z, rtol=1e-2, atol=1e-2)


def test_rayleigh_amplifies_noise_relative_to_awgn():
    """Equalising a deep fade divides by |h|^2, so noise is amplified on average."""
    torch.manual_seed(0)
    z = power_normalise(torch.randn(512, 256, 2))
    snr = torch.full((512,), 10.0)
    awgn_err = (awgn(z, snr) - z).pow(2).mean()
    fade_err = (rayleigh_slow_fading(z, snr) - z).pow(2).mean()
    assert fade_err > awgn_err


def test_rayleigh_is_differentiable():
    z = power_normalise(torch.randn(4, 64, 2)).requires_grad_(True)
    rayleigh_slow_fading(z, torch.full((4,), 10.0)).pow(2).sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_get_channel_dispatch():
    assert get_channel("awgn") is awgn
    assert get_channel("rayleigh") is rayleigh_slow_fading
    with pytest.raises(ValueError):
        get_channel("nonexistent")
