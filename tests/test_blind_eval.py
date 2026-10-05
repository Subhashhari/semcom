"""Blind evaluation and the week-1 pipeline, end to end on a tiny synthetic dataset."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from semcom.blind_eval import channel_seed, evaluate_blind, paired_awgn
from semcom.blind_report import report
from semcom.config import Config
from semcom.train import train

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.run_blind import STAGES, build_configs  # noqa: E402


def test_power_mismatch_preserves_the_true_snr():
    z = torch.randn(64, 4096, 2)
    z = z / z.pow(2).sum(-1).mean(1, keepdim=True).sqrt().unsqueeze(-1)
    gen = torch.Generator().manual_seed(0)
    y = paired_awgn(z, 18.0, 1.3, gen)
    signal = (1.3 * z).pow(2).sum(-1).mean()
    noise = (y - 1.3 * z).pow(2).sum(-1).mean()
    assert 10 * math.log10(signal / noise) == pytest.approx(18.0, abs=0.05)


def test_channel_noise_is_identical_across_runs_and_processes():
    assert channel_seed("curve", 18.0, 0, 3) == channel_seed("curve", 18.0, 0, 3)
    assert channel_seed("curve", 18.0, 0, 3) != channel_seed("curve", 18.0, 1, 3)
    z = torch.zeros(2, 8, 2)
    a = paired_awgn(z, 10.0, 1.0, torch.Generator().manual_seed(channel_seed("x")))
    b = paired_awgn(z, 10.0, 1.0, torch.Generator().manual_seed(channel_seed("x")))
    assert torch.equal(a, b)


def tiny(tmp_path, **kw):
    base = dict(
        hidden=8, c_out=8, batch_size=8, epochs=2, eval_every=1, eval_repeats=1,
        eval_snrs=[10.0, 18.0, 20.0], num_workers=0, amp=False, wandb=False,
        out_dir=str(tmp_path), lr_schedule="cosine", patience=None, digital=True,
    )
    base.update(kw)
    return Config(**base)


def test_week1_plan_matches_the_seed_plan():
    cfgs = build_configs(Config(), STAGES["week1"])
    count = {}
    for c in cfgs:
        key = (c.arm, c.digital)
        count[key] = count.get(key, 0) + 1
    assert count[("B", True)] == 5 and count[("C-att", True)] == 5
    assert count[("B", False)] == 3 and count[("C+E", True)] == 3
    assert len({c.run_name for c in cfgs}) == len(cfgs) == 25


def test_eval_and_report_end_to_end(tmp_path, fake_cifar):
    small = dict(max_images=16, repeats=1, snrs=[10.0, 18.0, 20.0], mismatch_snrs=(18.0,),
                 alphas=(1.0, 1.1, 1.2, 1.3), deltas=(-4.0, 0.0, 4.0), sensitivity_snrs=(18.0,),
                 batch_size=8)
    for seed in (0, 1):
        for arm in ("snr", "blank"):
            cfg = tiny(tmp_path, decoder_input=arm, encoder_snr=False, seed=seed)
            train(cfg)
            res = evaluate_blind(cfg.run_dir, **small)
            assert res["n_images"] == 16
            if arm == "snr":
                assert "energy" in res["plugin"] and res["sensitivity"]
            else:
                assert res["plugin"] == {} and res["sensitivity"] == {}

    out = report(tmp_path, n_boot=50)
    assert out["gate_qam16"] is not None and out["gate_qam16"]["seeds"] == [0, 1]
    assert out["rq2_qam16"] is not None
    assert (tmp_path / "blind_report.json").exists()
    saved = json.loads((tmp_path / "blind_report.json").read_text())
    assert "outcome" in saved["gate_qam16"]


def test_evaluation_is_deterministic_so_runs_can_be_paired(tmp_path, fake_cifar):
    # Channel noise comes from generators keyed by (SNR, alpha, repeat, batch), so the same
    # run evaluated twice - or two arms evaluated anywhere - see identical noise.
    cfg = tiny(tmp_path, decoder_input="stats", encoder_snr=False)
    train(cfg)
    kw = dict(max_images=8, repeats=1, snrs=[18.0], mismatch_snrs=(18.0,), alphas=(1.0, 1.2),
              sensitivity_snrs=(), batch_size=8)
    first = evaluate_blind(cfg.run_dir, **kw)
    a = dict(np.load(cfg.run_dir / "blind_eval_images.npz"))
    second = evaluate_blind(cfg.run_dir, **kw)
    b = dict(np.load(cfg.run_dir / "blind_eval_images.npz"))
    assert first["curve"] == second["curve"] and first["arm"] == "D"
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_eval_and_report_analog(tmp_path, fake_cifar):
    """Analog has no lattice: no DD/M2M4 cues, but the curve, plug-in energy and mismatch run."""
    small = dict(max_images=16, repeats=1, snrs=[10.0, 18.0, 20.0], mismatch_snrs=(18.0,),
                 alphas=(1.0, 1.1, 1.2, 1.3), deltas=(-4.0, 0.0, 4.0), sensitivity_snrs=(18.0,),
                 batch_size=8)
    for seed in (0, 1):
        for arm in ("snr", "blank"):
            cfg = tiny(tmp_path, decoder_input=arm, encoder_snr=False, seed=seed, digital=False)
            train(cfg)
            res = evaluate_blind(cfg.run_dir, **small)
            assert not res["digital"] and 18.0 in res["curve"]
            assert res["mismatch"][18.0][1.2]["psnr_own"] > 0
            if arm == "snr":
                assert "energy" in res["plugin"]
    out = report(tmp_path, n_boot=50)
    assert out["gaps_to_B"]["analog"]["C-att"]["seeds"] == [0, 1]
