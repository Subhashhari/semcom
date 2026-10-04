"""The blind-JSCC arms: configuration, wiring and the invariants the design relies on."""

import pytest
import torch

from semcom.config import Config
from semcom.models import JSCC, shuffle_across_batch
from semcom.train import build_model, build_scheduler

ARM_KW = {
    "B": dict(decoder_input="snr", encoder_snr=False),
    "C-att": dict(decoder_input="blank", encoder_snr=False),
    "C": dict(decoder_input="none", encoder_snr=False),
    "C+E": dict(decoder_input="energy", encoder_snr=False),
    "C+E-shuf": dict(decoder_input="energy_shuf", encoder_snr=False),
    "D": dict(decoder_input="stats", encoder_snr=False),
    "D-shuf": dict(decoder_input="stats_shuf", encoder_snr=False),
}


def model(arm, digital=True, **kw):
    return JSCC(c_out=8, hidden=16, digital=digital, **ARM_KW[arm], **kw)


@pytest.mark.parametrize("arm", list(ARM_KW))
def test_arm_labels(arm):
    if arm == "C":
        assert model(arm).name == "DeepJSCC-Q" and model(arm, digital=False).name == "BDJSCC"
    else:
        assert model(arm).name == arm


def test_b_and_catt_are_identical_apart_from_what_fills_the_slot():
    b, c = model("B"), model("C-att")
    assert b.num_parameters() == c.num_parameters()
    assert [n for n, _ in b.named_parameters()] == [n for n, _ in c.named_parameters()]


@pytest.mark.parametrize("arm", list(ARM_KW))
def test_encoder_blind_arms_have_no_encoder_af(arm):
    assert model(arm).encoder.af is None


def test_c_has_no_decoder_attention_and_catt_does():
    assert model("C").decoder.af is None
    assert model("C-att").decoder.af is not None


def test_legacy_arms_are_unchanged():
    adj = JSCC(c_out=8, hidden=16, snr_adaptive=True, digital=True)
    assert adj.name == "ADJSCC-Q"
    assert adj.encoder.af is not None and adj.decoder.af is not None
    assert JSCC(c_out=8, hidden=16).name == "BDJSCC"


def test_contradictory_or_meaningless_arms_are_rejected():
    with pytest.raises(ValueError, match="both encoder and decoder"):
        Config(snr_adaptive=True, decoder_input="blank")
    with pytest.raises(ValueError, match="digital=True"):
        Config(decoder_input="energy", digital=False)
    with pytest.raises(ValueError):
        Config(decoder_input="bogus")


def test_replace_still_works_from_a_default_config():
    # run_ablation.py builds adaptive arms by replace(snr_adaptive=True) on defaults.
    assert Config().replace(snr_adaptive=True).arm == "ADJSCC"


def test_run_names_are_unique_across_arms_mods_and_seeds():
    names = set()
    for arm, kw in ARM_KW.items():
        for digital in (False, True):
            if arm.startswith("C+E") and not digital:
                continue
            for seed in range(3):
                names.add(Config(digital=digital, seed=seed, **kw).run_name)
    expected = sum(3 * (1 if a.startswith("C+E") else 2) for a in ARM_KW)
    assert len(names) == expected


def test_legacy_run_names_keep_their_old_form_at_seed_zero():
    assert Config(snr_adaptive=True, digital=True).run_name == "adjsccq_r0.0833_m16_snr0-20"
    assert Config(digital=True, decoder_input="blank", encoder_snr=False).run_name.endswith("_s0")


def test_config_round_trips_with_the_new_fields(tmp_path):
    cfg = Config(digital=True, decoder_input="stats", encoder_snr=False, seed=3,
                 hidden_dec=64, lr_schedule="cosine", patience=None)
    cfg.save(tmp_path / "c.yaml")
    loaded = Config.from_yaml(tmp_path / "c.yaml")
    assert loaded.run_name == cfg.run_name and loaded.arm == "D" and loaded.patience is None


def test_decoder_snr_override_changes_reconstruction_but_not_transmission():
    torch.manual_seed(0)
    m = model("B").eval()
    x = torch.rand(4, 3, 32, 32)
    snr = torch.full((4,), 10.0)
    torch.manual_seed(1)
    a = m(x, snr)
    torch.manual_seed(1)
    b = m(x, snr, decoder_snr=torch.full((4,), 0.0))
    assert torch.equal(a["z"], b["z"]) and torch.equal(a["y"], b["y"])
    assert not torch.allclose(a["x_hat"], b["x_hat"])


def test_decoder_snr_override_is_rejected_for_arms_without_an_snr_slot():
    m = model("C-att")
    with pytest.raises(ValueError, match="decoder_snr"):
        m(torch.rand(2, 3, 32, 32), torch.full((2,), 10.0), decoder_snr=torch.full((2,), 5.0))


def test_shuffled_inputs_never_return_an_images_own_row():
    cond = torch.arange(6, dtype=torch.float32).unsqueeze(1)
    shuffled = shuffle_across_batch(cond)
    assert (shuffled != cond).all()
    assert sorted(shuffled.squeeze(1).tolist()) == cond.squeeze(1).tolist()
    assert torch.equal(shuffle_across_batch(cond[:1]), torch.zeros(1, 1))


def test_energy_condition_is_the_transmitted_blocks_power():
    m = model("C+E").eval()
    x = torch.rand(3, 3, 32, 32)
    z = m.transmit(x, torch.full((3,), 10.0))
    cond = m.decoder_condition(torch.full((3,), 10.0), z, z)
    p = z.pow(2).sum(-1).mean(1)
    assert torch.allclose(cond.squeeze(1), (p - 1.0) * m.k**0.5, atol=1e-5)


@pytest.mark.parametrize("arm", list(ARM_KW))
def test_every_arm_backpropagates_end_to_end(arm):
    m = model(arm)
    out = m(torch.rand(4, 3, 32, 32), torch.rand(4) * 20)
    (out["x_hat"].mean() + 0.05 * out["kl"]).backward()
    enc_grads = [p.grad for p in m.encoder.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in enc_grads)


def test_per_side_widths():
    m = JSCC(c_out=8, hidden=32, hidden_enc=16, hidden_dec=64)
    assert m.encoder.fl[0].conv.out_channels == 16
    assert m.decoder.fl[0].conv.out_channels == 64
    assert "w16x64" in Config(hidden=32, hidden_enc=16, hidden_dec=64).run_name


def test_cosine_schedule_decays_to_the_floor():
    cfg = Config(lr=1e-3, lr_schedule="cosine", lr_min_factor=0.01, epochs=10)
    opt = torch.optim.Adam(build_model(cfg.replace(hidden=8)).parameters(), lr=cfg.lr)
    sched = build_scheduler(cfg, opt)
    lrs = []
    for _ in range(10):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(1e-3)
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-5)
    assert build_scheduler(Config(), opt) is None
