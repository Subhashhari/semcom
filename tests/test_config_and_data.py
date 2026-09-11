"""Tests for configuration, SNR sampling, and metrics."""

from pathlib import Path

import pytest
import torch
import yaml

from semcom.config import Config
from semcom.data import psnr, sample_snr

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


# ------------------------------------------------------------------------------ config


def test_rate_arithmetic_matches_the_plan():
    assert Config(c_out=8).k == 256
    assert Config(c_out=8).bandwidth_ratio == pytest.approx(1 / 12)
    assert Config(c_out=16).k == 512
    assert Config(c_out=16).bandwidth_ratio == pytest.approx(1 / 6)


@pytest.mark.parametrize(
    "kw,expected",
    [
        (dict(), "BDJSCC"),
        (dict(snr_adaptive=True), "ADJSCC"),
        (dict(digital=True), "DeepJSCC-Q"),
        (dict(snr_adaptive=True, digital=True), "ADJSCC-Q"),
    ],
)
def test_arm_property(kw, expected):
    assert Config(**kw).arm == expected


def test_run_name_distinguishes_every_run_in_the_ablation():
    """Run names are directory names; a collision would silently overwrite a result."""
    names = {
        Config(snr_adaptive=True, digital=False).run_name,
        Config(snr_adaptive=True, digital=True).run_name,
    }
    for snr in (1.0, 4.0, 7.0, 13.0, 19.0):
        names.add(Config(snr_adaptive=False, digital=False, snr_train_fixed=snr).run_name)
        names.add(Config(snr_adaptive=False, digital=True, snr_train_fixed=snr).run_name)
    assert len(names) == 12


def test_run_name_encodes_modulation_order_only_when_digital():
    assert "m16" in Config(digital=True, modulation_order=16).run_name
    assert "m16" not in Config(digital=False, modulation_order=16).run_name
    assert Config(digital=True, modulation_order=64).run_name != (
        Config(digital=True, modulation_order=16).run_name
    )


def test_run_name_encodes_channel_only_when_not_awgn():
    assert "rayleigh" in Config(channel="rayleigh").run_name
    assert "awgn" not in Config(channel="awgn").run_name


def test_invalid_configs_are_rejected_at_construction():
    with pytest.raises(ValueError):
        Config(c_out=7)  # cannot pair into (I, Q)
    with pytest.raises(ValueError):
        Config(image_size=30)  # backbone downsamples by 4
    with pytest.raises(ValueError):
        Config(snr_train_min=20.0, snr_train_max=0.0)
    with pytest.raises(ValueError):
        # DeepJSCC-Q found a favoured subset beats uniform usage at very large M.
        Config(digital=True, modulation_order=4096, kl_weight=0.05)


def test_replace_returns_an_independent_config():
    base = Config()
    other = base.replace(snr_adaptive=True, digital=True)
    assert other.arm == "ADJSCC-Q"
    assert base.arm == "BDJSCC", "replace() must not mutate the original"


def test_yaml_roundtrip(tmp_path):
    cfg = Config(snr_adaptive=True, digital=True, modulation_order=64, c_out=16)
    cfg.save(tmp_path / "c.yaml")
    # to_dict() adds derived keys; from_yaml must reject them rather than silently drop.
    raw = yaml.safe_load((tmp_path / "c.yaml").read_text())
    for derived in ("arm", "k", "bandwidth_ratio", "run_name"):
        raw.pop(derived)
    (tmp_path / "clean.yaml").write_text(yaml.safe_dump(raw))

    loaded = Config.from_yaml(tmp_path / "clean.yaml")
    assert loaded.run_name == cfg.run_name
    assert loaded.arm == "ADJSCC-Q"


def test_from_yaml_rejects_unknown_keys(tmp_path):
    """A typo'd key must fail loudly, not be silently ignored for a whole training run."""
    (tmp_path / "bad.yaml").write_text("c_out: 8\nlearning_rate: 0.1\n")
    with pytest.raises(ValueError, match="unknown config keys"):
        Config.from_yaml(tmp_path / "bad.yaml")


def test_from_yaml_overrides_ignore_none():
    cfg = Config.from_yaml(CONFIG_DIR / "cifar_r12.yaml", digital=True, epochs=None)
    assert cfg.digital is True
    assert cfg.epochs == 1280, "None overrides must not clobber the file value"


@pytest.mark.parametrize("name,c_out,ratio", [("cifar_r12", 8, 1 / 12), ("cifar_r6", 16, 1 / 6)])
def test_shipped_configs_load_and_have_the_advertised_rate(name, c_out, ratio):
    cfg = Config.from_yaml(CONFIG_DIR / f"{name}.yaml")
    assert cfg.c_out == c_out
    assert cfg.bandwidth_ratio == pytest.approx(ratio)


# ------------------------------------------------------------------------- SNR sampling


def test_fixed_snr_specialists_get_a_constant():
    cfg = Config(snr_train_fixed=7.0)
    snr = sample_snr(64, cfg, torch.device("cpu"))
    assert snr.shape == (64,)
    assert (snr == 7.0).all()


def test_adaptive_snr_is_sampled_per_example_and_in_range():
    """Per-example resampling is what stops the model collapsing to one operating point."""
    torch.manual_seed(0)
    cfg = Config(snr_train_min=0.0, snr_train_max=20.0)
    snr = sample_snr(4096, cfg, torch.device("cpu"))

    assert snr.min() >= 0.0 and snr.max() <= 20.0
    assert snr.unique().numel() > 4000, "SNR must vary within the batch, not per batch"
    assert snr.mean().item() == pytest.approx(10.0, abs=0.5)


def test_snr_range_is_respected_for_a_narrow_band():
    cfg = Config(snr_train_min=5.0, snr_train_max=6.0)
    snr = sample_snr(512, cfg, torch.device("cpu"))
    assert snr.min() >= 5.0 and snr.max() <= 6.0


# ------------------------------------------------------------------------------ metrics


def test_psnr_of_a_perfect_reconstruction_is_large():
    x = torch.rand(4, 3, 32, 32)
    assert psnr(x, x.clone()).min().item() > 100


def test_psnr_matches_the_closed_form():
    x = torch.zeros(1, 3, 8, 8)
    x_hat = torch.full_like(x, 0.1)  # MSE = 0.01 -> PSNR = 20 dB
    assert psnr(x, x_hat).item() == pytest.approx(20.0, abs=1e-4)


def test_psnr_is_per_image_not_per_batch():
    """Averaging per-image PSNR is not the same as the PSNR of the batch mean MSE.

    The literature reports the former, so the metric must return one value per image.
    """
    x = torch.zeros(2, 3, 8, 8)
    x_hat = torch.stack([torch.full((3, 8, 8), 0.1), torch.full((3, 8, 8), 0.01)])
    values = psnr(x, x_hat)
    assert values.shape == (2,)
    assert values[0].item() == pytest.approx(20.0, abs=1e-4)
    assert values[1].item() == pytest.approx(40.0, abs=1e-4)


def test_psnr_decreases_monotonically_with_error():
    x = torch.rand(1, 3, 16, 16)
    errors = [0.01, 0.05, 0.2]
    values = [psnr(x, (x + e).clamp(0, 1)).item() for e in errors]
    assert values == sorted(values, reverse=True)
