"""Tests for the 2x2 model class.

Two things here matter more than the rest:

  * `test_transmitted_symbols_are_always_on_constellation` - the end-to-end version of the
    quantiser invariant, run through the real encoder and the real power normalisation.
  * `test_af_gating_does_not_control_transmit_power` - guards the ordering decision that
    the ADJSCC/DeepJSCC-Q composition forces. AF gates are sigmoid-bounded and attenuate
    at low SNR; if power normalisation ran before gating, the gates would end up
    controlling transmit power rather than resource allocation.
"""

import pytest
import torch

from semcom.models import ARM_NAMES, JSCC, arm_name

ARMS = [(False, False), (True, False), (False, True), (True, True)]
ARM_IDS = [ARM_NAMES[a] for a in ARMS]


def make_model(snr_adaptive=False, digital=False, **kw):
    """Small-but-real model: same structure, narrower, so tests stay fast."""
    kw.setdefault("hidden", 16)
    kw.setdefault("c_out", 8)
    return JSCC(snr_adaptive=snr_adaptive, digital=digital, **kw)


@pytest.fixture
def batch():
    torch.manual_seed(0)
    return torch.rand(4, 3, 32, 32)


# --------------------------------------------------------------------------- naming


def test_arm_names_cover_the_full_2x2():
    assert {arm_name(a, d) for a, d in ARMS} == {
        "BDJSCC",
        "ADJSCC",
        "DeepJSCC-Q",
        "ADJSCC-Q",
    }
    assert arm_name(True, True) == "ADJSCC-Q"


@pytest.mark.parametrize("adaptive,digital", ARMS, ids=ARM_IDS)
def test_model_name_matches_its_flags(adaptive, digital):
    assert make_model(adaptive, digital).name == ARM_NAMES[(adaptive, digital)]


# ---------------------------------------------------------------------- shapes / rate


@pytest.mark.parametrize("adaptive,digital", ARMS, ids=ARM_IDS)
def test_forward_shapes_and_range(adaptive, digital, batch):
    model = make_model(adaptive, digital)
    out = model(batch, torch.full((4,), 10.0))

    assert out["x_hat"].shape == batch.shape
    assert out["z"].shape == (4, model.k, 2)
    # The decoder ends in a sigmoid, so reconstructions are valid images.
    assert out["x_hat"].min() >= 0.0 and out["x_hat"].max() <= 1.0
    assert torch.isfinite(out["x_hat"]).all()


@pytest.mark.parametrize("c_out,expected_k,expected_r", [(8, 256, 1 / 12), (16, 512, 1 / 6)])
def test_bandwidth_ratio_arithmetic(c_out, expected_k, expected_r):
    """c_out is the knob that sets the rate; the plan's two configs must come out right."""
    model = make_model(c_out=c_out)
    assert model.k == expected_k
    assert model.bandwidth_ratio == pytest.approx(expected_r)
    assert model.source_dim == 3072


def test_odd_c_out_is_rejected():
    """Channels pair into (I, Q), so an odd count cannot form complex symbols."""
    with pytest.raises(ValueError):
        make_model(c_out=7)


def test_symbol_reshape_roundtrips():
    model = make_model()
    feat = torch.randn(4, model.c_out, model.latent_spatial, model.latent_spatial)
    torch.testing.assert_close(model._from_symbols(model._to_symbols(feat)), feat)


def test_scalar_snr_is_broadcast_over_the_batch(batch):
    out = make_model(True, True)(batch, torch.tensor(10.0))
    assert out["x_hat"].shape == batch.shape


# ------------------------------------------------------------------ the core invariants


@pytest.mark.parametrize("order", [4, 16, 64, 256])
def test_transmitted_symbols_are_always_on_constellation(order, batch):
    """End-to-end: what actually goes over the air is a legal QAM symbol.

    Run for both digital arms and across SNRs, because the AF gates change the latent's
    geometry with SNR and must not be able to push it off the alphabet.
    """
    for adaptive in (False, True):
        model = make_model(adaptive, True, modulation_order=order)
        for snr in (0.0, 10.0, 20.0):
            z = model.transmit(batch, torch.full((4,), snr))
            assert model.quantiser.is_on_constellation(z), (
                f"off-constellation symbol: order={order} adaptive={adaptive} snr={snr}"
            )


@pytest.mark.parametrize("adaptive,digital", ARMS, ids=ARM_IDS)
def test_transmit_power_matches_constellation_scale(adaptive, digital, batch):
    """Average transmit power must be ~1 for every arm, or SNR means different things.

    Analog arms are normalised exactly. Digital arms are normalised and then snapped to
    the lattice, so their power is close to but not exactly 1 - that displacement is the
    quantisation error itself.
    """
    model = make_model(adaptive, digital)
    z = model.transmit(batch, torch.full((4,), 10.0))
    power = z.pow(2).sum(dim=(1, 2)) / model.k

    tol = 0.25 if digital else 1e-4
    assert (power - 1.0).abs().max().item() < tol


def test_af_gating_does_not_control_transmit_power(batch):
    """The ordering guard: gating -> power normalise -> quantise.

    AF gates attenuate at low SNR. Because normalisation runs *after* gating, transmit
    power must be identical across SNRs even though the latent's direction changes. If
    someone reorders these two steps, this test fails and the SNR axis silently stops
    meaning what the plots claim it means.
    """
    model = make_model(snr_adaptive=True, digital=False)

    powers, directions = [], []
    for snr in (0.0, 10.0, 20.0):
        z = model.transmit(batch, torch.full((4,), snr))
        powers.append((z.pow(2).sum(dim=(1, 2)) / model.k).mean().item())
        directions.append(torch.nn.functional.normalize(z.reshape(4, -1), dim=1))

    # Power is invariant to SNR...
    assert max(powers) - min(powers) < 1e-4
    for p in powers:
        assert p == pytest.approx(1.0, abs=1e-4)

    # ...but the conditioning is live: the gates themselves respond to SNR.
    #
    # Checked at the gates rather than at the latent. An untrained AF MLP outputs roughly
    # 0.5 everywhere - the SNR is 1 input among c+1 - so its effect on the final latent
    # direction is ~1e-5 at init and only becomes large once trained. Asserting on the
    # gate vectors tests the mechanism directly instead of a washed-out downstream proxy.
    with torch.no_grad():
        feat = model.encoder.fl[0](batch)
        g_lo = model.encoder.af[0].compute_gates(feat, torch.full((4,), 0.0))
        g_hi = model.encoder.af[0].compute_gates(feat, torch.full((4,), 20.0))
    assert not torch.allclose(g_lo, g_hi), "gates ignore SNR; conditioning is inert"

    # And the latent does move, even if only slightly at initialisation.
    cos = (directions[0] * directions[-1]).sum(dim=1).mean().item()
    assert cos < 1.0


def test_snr_conditioning_changes_output_only_for_adaptive_arms(batch):
    """Non-adaptive arms must ignore SNR at the encoder; adaptive arms must not."""
    for adaptive in (False, True):
        model = make_model(adaptive, False).eval()
        with torch.no_grad():
            lo, _ = model.encoder(batch, torch.full((4,), 0.0))
            hi, _ = model.encoder(batch, torch.full((4,), 20.0))
        differs = not torch.allclose(lo, hi, atol=1e-6)
        assert differs is adaptive


# ---------------------------------------------------------------------------- training


@pytest.mark.parametrize("adaptive,digital", ARMS, ids=ARM_IDS)
def test_gradients_reach_every_parameter(adaptive, digital, batch):
    """Nothing in any arm is accidentally detached from the loss.

    Particularly worth checking on the digital arms, where the straight-through splice is
    the only path by which the encoder receives gradient at all.
    """
    model = make_model(adaptive, digital)
    out = model(batch, torch.full((4,), 10.0))
    (torch.nn.functional.mse_loss(out["x_hat"], batch) + out["kl"]).backward()

    missing = [
        name
        for name, p in model.named_parameters()
        if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())
    ]
    assert not missing, f"no finite gradient for: {missing}"


def test_af_modules_are_a_small_parameter_overhead():
    """ADJSCC reports ~+0.6% parameters; the AF module must stay near-free.

    A regression here means someone widened the AF MLP, which would confound the
    ablation: the adaptive arm would win partly on capacity rather than on conditioning.
    """
    base = make_model(False, False, hidden=256).num_parameters()
    adaptive = make_model(True, False, hidden=256).num_parameters()
    overhead = (adaptive - base) / base
    assert 0 < overhead < 0.05, f"AF overhead {overhead:.2%} is too large"


def test_digital_arm_adds_no_trainable_parameters():
    """The constellation is fixed, so digital and analog arms have equal capacity.

    This is what makes the 2x2 a controlled comparison.
    """
    assert make_model(True, False).num_parameters() == make_model(True, True).num_parameters()


@pytest.mark.parametrize("channel", ["awgn", "rayleigh"])
def test_both_channels_run_end_to_end(channel, batch):
    out = make_model(True, True, channel=channel)(batch, torch.full((4,), 10.0))
    assert torch.isfinite(out["x_hat"]).all()


def test_unknown_channel_is_rejected():
    with pytest.raises(ValueError):
        make_model(channel="gaussian-ish")


def test_kl_is_zero_for_analog_arms(batch):
    out = make_model(True, False)(batch, torch.full((4,), 10.0))
    assert out["kl"].item() == 0.0


def test_collect_gates_returns_one_entry_per_encoder_af_module(batch):
    model = make_model(True, True)
    gates = model(batch, torch.full((4,), 10.0), collect_gates=True)["gates"]

    assert len(gates) == 4
    for g in gates:
        assert g.shape == (4, 16)
        # The sigmoid makes this a gate: it can attenuate, never amplify.
        assert g.min() > 0.0 and g.max() < 1.0

    assert make_model(False, False)(batch, torch.full((4,), 10.0), collect_gates=True)[
        "gates"
    ] == []


def smooth_batch(n: int = 4) -> torch.Tensor:
    """Low-frequency synthetic images.

    Deliberately *not* the uniform-random `batch` fixture. At R=1/12 the model compresses
    12:1, and white noise is incompressible by construction - asking the model to
    reconstruct it through that bottleneck tests nothing but the bottleneck's existence.
    Natural images are compressible because they are spatially correlated, so the smoke
    test uses a signal that has the structure the architecture is designed to exploit.
    """
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, 32), torch.linspace(0, 1, 32), indexing="ij"
    )
    phases = torch.arange(n).view(n, 1, 1, 1) * 0.7
    base = torch.stack([xx, yy, xx * yy])[None] * 6.28
    return (torch.sin(base + phases) * 0.5 + 0.5).clamp(0, 1)


@pytest.mark.parametrize("adaptive,digital", ARMS, ids=ARM_IDS)
def test_model_can_overfit_a_single_batch(adaptive, digital):
    """Smoke test that the whole chain actually learns, digital path included.

    Catches the class of bug where everything runs, shapes are right, gradients are
    finite - and the loss never moves because the signal path is broken somewhere. Run
    for all four arms so a break in the straight-through splice cannot hide behind the
    analog arms passing.
    """
    torch.manual_seed(0)
    x = smooth_batch()
    model = make_model(adaptive, digital, modulation_order=64)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    snr = torch.full((x.shape[0],), 20.0)

    losses = []
    for _ in range(150):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x, snr)["x_hat"], x)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.5, (
        f"{model.name} did not learn: {losses[0]:.4f} -> {losses[-1]:.4f}"
    )
