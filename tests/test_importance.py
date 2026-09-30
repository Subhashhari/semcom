"""Tests for erasure-based importance measurement.

The measurement is only meaningful if erasure genuinely damages the reconstruction and
the damage is attributable to the erased group. Most of these tests pin that down, since
a mask that silently does nothing would produce a clean-looking all-zeros importance
profile and the wrong conclusion ("importance is flat, UEP is pointless").
"""

import pytest
import torch

from semcom.importance import (
    analyse_importance,
    concentration,
    energy_importance,
    erasure_importance,
    gradient_importance,
    spearman,
    symbol_groups,
)
from semcom.models import JSCC


def make_model(**kw):
    kw.setdefault("hidden", 16)
    kw.setdefault("c_out", 8)
    return JSCC(**kw)


@pytest.fixture
def batch():
    torch.manual_seed(0)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, 32), torch.linspace(0, 1, 32), indexing="ij")
    phases = torch.arange(6).view(6, 1, 1, 1) * 0.5
    return (torch.sin(torch.stack([xx, yy, xx * yy])[None] * 6.28 + phases) * 0.5 + 0.5).clamp(0, 1)


# ------------------------------------------------------------------------------ grouping


def test_groups_partition_every_symbol_exactly_once():
    model = make_model()
    groups = symbol_groups(model)
    assert len(groups) == model.c_out
    allidx = torch.cat(groups).sort().values
    assert torch.equal(allidx, torch.arange(model.k))


def test_groups_are_contiguous_channel_major_runs():
    """Grouping must match the (B, C, H, W) -> (B, k, 2) layout, or groups mix channels."""
    model = make_model()
    for g in symbol_groups(model):
        assert torch.equal(g, torch.arange(g[0], g[-1] + 1))
    assert symbol_groups(model)[0].numel() == model.k // model.c_out


def test_group_count_is_validated():
    model = make_model()
    with pytest.raises(ValueError):
        symbol_groups(model, 0)
    with pytest.raises(ValueError):
        symbol_groups(model, model.k + 1)
    with pytest.raises(ValueError):
        symbol_groups(model, 7)  # k=256 is not divisible by 7


# ------------------------------------------------------------------------- the erasure hook


@pytest.mark.parametrize("digital", [False, True])
def test_erase_mask_changes_the_reconstruction(digital, batch):
    """A mask that did nothing would make every importance number zero."""
    model = make_model(digital=digital).eval()
    snr = torch.full((batch.shape[0],), 10.0)

    mask = torch.zeros(model.k, dtype=torch.bool)
    mask[: model.k // 2] = True

    torch.manual_seed(1)
    intact = model(batch, snr)["x_hat"]
    torch.manual_seed(1)
    erased = model(batch, snr, erase_mask=mask)["x_hat"]
    assert not torch.allclose(intact, erased)


def test_empty_mask_is_a_no_op(batch):
    model = make_model().eval()
    snr = torch.full((batch.shape[0],), 10.0)
    mask = torch.zeros(model.k, dtype=torch.bool)

    torch.manual_seed(2)
    a = model(batch, snr)["x_hat"]
    torch.manual_seed(2)
    b = model(batch, snr, erase_mask=mask)["x_hat"]
    torch.testing.assert_close(a, b)


def test_erasure_does_not_touch_the_transmitted_signal(batch):
    """Erasure happens at the receiver, so the constellation invariant must survive."""
    model = make_model(digital=True, modulation_order=16).eval()
    snr = torch.full((batch.shape[0],), 10.0)
    mask = torch.zeros(model.k, dtype=torch.bool)
    mask[::2] = True

    out = model(batch, snr, erase_mask=mask)
    assert model.quantiser.is_on_constellation(out["z"])


def test_per_example_mask_is_supported(batch):
    model = make_model().eval()
    mask = torch.zeros(batch.shape[0], model.k, dtype=torch.bool)
    mask[0, :10] = True
    out = model(batch, torch.full((batch.shape[0],), 10.0), erase_mask=mask)
    assert out["x_hat"].shape == batch.shape


def briefly_train(model, x, steps: int = 200, snr_db: float = 15.0):
    """Overfit the model to `x` so the decoder actually depends on the latent.

    Importance measurement is meaningless on an untrained model: the decoder ignores its
    input, so erasing any part of the latent moves PSNR by ~5e-4 dB and any ordering is
    numerical noise. Tests that need erasure to *mean* something must train first.
    """
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    snr = torch.full((x.shape[0],), snr_db)
    model.train()
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.mse_loss(model(x, snr)["x_hat"], x).backward()
        opt.step()
    return model.eval()


def test_erasing_more_hurts_more(batch):
    """Monotonicity: a larger erasure must not do less damage."""
    from semcom.data import psnr

    torch.manual_seed(0)
    model = briefly_train(make_model(), batch)
    snr = torch.full((batch.shape[0],), 15.0)

    values = []
    for frac in (0.0, 0.25, 0.5, 1.0):
        mask = torch.zeros(model.k, dtype=torch.bool)
        mask[: int(model.k * frac)] = True
        torch.manual_seed(3)
        values.append(psnr(batch, model(batch, snr, erase_mask=mask)["x_hat"]).mean().item())

    # Sanity: erasure must actually bite, or the ordering below is noise again.
    assert values[0] - values[-1] > 1.0, f"erasure had almost no effect: {values}"
    assert values == sorted(values, reverse=True), values


def test_importance_is_measurable_on_a_trained_model(batch):
    """The precondition for any UEP work: importance must be graded, not flat.

    On an untrained model every group looks equally (un)important. Once the decoder
    depends on its latent, erasure damage should vary across groups - otherwise there is
    nothing for importance-aware protection to allocate.
    """
    torch.manual_seed(0)
    model = briefly_train(make_model(), batch)

    imp = erasure_importance(model, batch, 15.0, repeats=3)
    assert imp.max() > imp.min(), "importance is perfectly flat"
    assert concentration(imp)["top_half_share"] > 0.5


# ------------------------------------------------------------------------------- measures


def test_erasure_importance_shape_and_sign(batch):
    model = make_model().eval()
    imp = erasure_importance(model, batch, 10.0, repeats=2)
    assert imp.shape == (model.c_out,)
    assert torch.isfinite(imp).all()
    # Erasing information should on average cost quality, not add it.
    assert imp.mean() > 0


@pytest.mark.parametrize("fn", [energy_importance, gradient_importance])
def test_predictors_have_the_right_shape_and_are_finite(fn, batch):
    model = make_model()
    v = fn(model, batch, 10.0)
    assert v.shape == (model.c_out,)
    assert torch.isfinite(v).all()


def test_gradient_importance_is_nonzero(batch):
    """A detached graph would silently give an all-zero, useless predictor."""
    v = gradient_importance(make_model(), batch, 10.0)
    assert v.abs().sum() > 0


def test_energy_importance_works_on_digital_arms(batch):
    v = energy_importance(make_model(digital=True), batch, 10.0)
    assert torch.isfinite(v).all() and v.sum() > 0


# ----------------------------------------------------------------------------- statistics


def test_spearman_detects_perfect_and_inverse_rankings():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert spearman(a, a * 10) == pytest.approx(1.0, abs=1e-9)
    assert spearman(a, -a) == pytest.approx(-1.0, abs=1e-9)
    assert abs(spearman(a, torch.tensor([2.0, 1.0, 4.0, 3.0]))) < 1.0


def test_spearman_is_rank_based_not_value_based():
    """Monotone rescaling must not change the score - tiering consumes ranks only."""
    a = torch.tensor([1.0, 2.0, 3.0, 9.0])
    b = torch.tensor([0.1, 0.2, 0.3, 0.4])
    assert spearman(a, b) == pytest.approx(1.0, abs=1e-9)


def test_concentration_flags_flat_versus_peaked():
    flat = torch.ones(8)
    peaked = torch.tensor([10.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    assert concentration(flat)["top_half_share"] == pytest.approx(0.5, abs=1e-6)
    assert concentration(peaked)["top_half_share"] > 0.8
    assert concentration(peaked)["max_over_median"] > concentration(flat)["max_over_median"]


def test_concentration_survives_degenerate_input():
    out = concentration(torch.zeros(4))
    assert out["top_half_share"] != out["top_half_share"]  # NaN, not a crash


# ------------------------------------------------------------------------------ the study


def test_analyse_importance_reports_every_snr_and_predictor(batch):
    model = make_model().eval()
    out = analyse_importance(model, batch, snrs=(1.0, 19.0), repeats=1)

    assert set(out["by_snr"]) == {1.0, 19.0}
    for entry in out["by_snr"].values():
        assert len(entry["erasure_psnr_drop_db"]) == model.c_out
        assert set(entry["predictors"]) == {"energy", "gradient"}
        for p in entry["predictors"].values():
            assert len(p["values"]) == model.c_out
    assert isinstance(out["concentration_rises_with_snr"], bool)
