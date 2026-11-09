"""Generative-JSCC prototype: bandwidth accounting, channel pairing, DiT and diffusion invariants,
the data cache, and an end-to-end train/evaluate/resume run on stand-in frozen models."""

import json

import pytest
import torch

from semcom.gen.config import GenConfig
from semcom.gen.diffusion import Diffusion
from semcom.gen.dit import ConditionalDiT
from semcom.gen.model import GenJSCC, CNNBaseline, channel, snr_input


def tiny(tmp_path, **kw) -> GenConfig:
    base = dict(
        dataset="synthetic", fake_frozen=True, data_root=str(tmp_path / "data"),
        out_dir=str(tmp_path / "runs"), steps=4, batch_size=4, warmup_steps=1, val_every=2,
        sample_every=4, ckpt_every=2, log_every=2, num_workers=0, eval_images=8, eval_batch=4,
        eval_snrs=[0.0], sample_steps=[2], cnn_hidden=8, det_hidden=8, wandb=False,
        sanity_steps=25, val_quality_images=4,
    )
    base.update(kw)
    return GenConfig(**base)


# ------------------------------------------------------------------ config and bandwidth


def test_default_bandwidth_is_one_in_48():
    cfg = GenConfig()
    assert (cfg.grid, cfg.k_sem, cfg.k_det, cfg.k) == (8, 256, 768, 1024)
    assert cfg.bandwidth_ratio == pytest.approx(1 / 48)
    # The CNN baseline matches it: (128/4)^2 * 2 / 2 = 1024.
    assert cfg.replace(model="cnn").k == 1024


def test_run_names_and_yaml_round_trip(tmp_path):
    cfg = GenConfig(snr_mode="blank")
    assert cfg.run_name == "dit-S_cbr48_blind-5to5_s0"
    assert GenConfig(snr_mode="fixed", snr_train_fixed=-5).run_name == "dit-S_cbr48_spec-5_s0"
    cfg.save(tmp_path / "c.yaml")
    assert GenConfig.from_yaml(tmp_path / "c.yaml") == cfg


@pytest.mark.parametrize("kw", [dict(snr_mode="bogus"), dict(model="vit"), dict(c_sem=7),
                                dict(cond_blocks_sem=10, cond_blocks_det=10), dict(image_size=100)])
def test_invalid_configs_raise(kw):
    with pytest.raises(ValueError):
        GenConfig(**kw)


# ------------------------------------------------------------------ channel


def test_channel_noise_is_paired_and_has_the_right_power():
    z1, z2 = torch.randn(64, 512, 2), torch.randn(64, 512, 2)
    snr = torch.full((64,), 3.0)
    g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    n1 = channel(z1, snr, generator=g1) - z1
    n2 = channel(z2, snr, generator=g2) - z2
    # Same generator state -> identical noise, whatever the transmitted symbols.
    assert torch.allclose(n1, n2, atol=1e-6)
    assert n1.pow(2).sum(-1).mean().item() == pytest.approx(10 ** (-0.3), rel=0.03)


def test_rayleigh_is_equalised_and_shapes_hold():
    z = torch.randn(8, 100, 2)
    y = channel(z, torch.full((8,), 40.0), kind="rayleigh", generator=torch.Generator().manual_seed(0))
    assert y.shape == z.shape
    # At 40 dB the equalised output is close to the input for typical fades.
    assert (y - z).pow(2).mean() < 0.05


# ------------------------------------------------------------------ DiT


def make_dit(snr_slot=False, **kw):
    return ConditionalDiT(latent_size=16, c_sem=8, c_det=24, depth=4, hidden=64, heads=4,
                          cond_blocks_sem=1, cond_blocks_det=1, snr_slot=snr_slot, **kw)


def dit_inputs(b=2):
    return torch.randn(b, 4, 16, 16), torch.randint(0, 1000, (b,)), torch.randn(b, 8, 8, 8), torch.randn(b, 24, 8, 8)


def test_dit_starts_at_zero_and_ignores_the_condition_until_trained():
    torch.manual_seed(0)
    dit = make_dit()
    x, t, ys, yd = dit_inputs()
    out = dit(x, t, ys, yd)
    assert out.shape == x.shape and torch.count_nonzero(out) == 0
    for blk in dit.blocks:  # switch the zero-initialised layers on
        torch.nn.init.normal_(blk.ada[-1].weight, std=0.02)
        if blk.fuse is not None:
            torch.nn.init.normal_(blk.fuse.weight, std=0.02)
    torch.nn.init.normal_(dit.final.weight, std=0.02)
    torch.nn.init.normal_(dit.final_ada[-1].weight, std=0.02)
    a = dit(x, t, ys, yd)
    assert not torch.allclose(a, dit(x, t, ys + 1, yd)), "semantic symbols must reach the output"
    assert not torch.allclose(a, dit(x, t, ys, yd + 1)), "detail symbols must reach the output"
    dropped = torch.ones(2, dtype=torch.bool)
    assert torch.allclose(dit(x, t, ys, yd, drop=dropped), dit(x, t, ys + 1, yd + 1, drop=dropped)), \
        "with the condition dropped, the received symbols must not matter"


def test_snr_slot_is_required_and_told_and_blind_share_an_architecture(tmp_path):
    with pytest.raises(ValueError):
        make_dit(snr_slot=True)(*dit_inputs())
    told, blind = (GenJSCC(tiny(tmp_path, snr_mode=m), 32) for m in ("snr", "blank"))
    spec = GenJSCC(tiny(tmp_path), 32)
    n = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert n(told) == n(blind) > n(spec)
    s = torch.tensor([-5.0, 5.0])
    assert snr_input(told.cfg, s).tolist() == [-0.5, 0.5]
    assert snr_input(blind.cfg, s).tolist() == [0.0, 0.0]
    assert snr_input(spec.cfg, s) is None


def test_each_branch_is_power_normalised(tmp_path):
    cfg = tiny(tmp_path)
    model = GenJSCC(cfg, 32)
    batch = {"dino": torch.randn(3, 32, 8, 8), "image": torch.rand(3, 3, 128, 128)}
    z_s, z_d = model.transmit(batch)
    assert z_s.shape == (3, cfg.k_sem, 2) and z_d.shape == (3, cfg.k_det, 2)
    for z in (z_s, z_d):
        assert torch.allclose(z.pow(2).sum(-1).mean(1), torch.ones(3), atol=1e-5)


# ------------------------------------------------------------------ diffusion


def test_timesteps_are_spaced_and_end_at_zero():
    d = Diffusion()
    for steps in (1, 10, 50, 1000):
        ts = d.timesteps(steps)
        assert ts[0] == 999 and ts[-1] == (0 if steps > 1 else 999) and len(ts) == steps
        assert all(a > b for a, b in zip(ts, ts[1:]))


def test_ddim_with_an_oracle_denoiser_recovers_x0():
    d = Diffusion()
    x0 = torch.randn(2, 4, 8, 8)

    def oracle(x, t):  # the exact noise, given x0
        ab = d.alpha_bar[t].view(-1, 1, 1, 1)
        return (x - ab.sqrt() * x0) / (1 - ab).sqrt()

    for steps in (5, 50):
        out = d.sample(oracle, x0.shape, steps, torch.device("cpu"), torch.Generator().manual_seed(0))
        assert torch.allclose(out, x0, atol=1e-3)


def test_velocity_target_round_trips_and_its_oracle_recovers_x0():
    d = Diffusion(prediction="v")
    x0, noise = torch.randn(3, 4, 8, 8), torch.randn(3, 4, 8, 8)
    t = torch.tensor([0, 500, 999])
    x_t = d.q_sample(x0, t, noise)
    eps, x0_hat = d.split(d.target(x0, noise, t), x_t, t)
    assert torch.allclose(eps, noise, atol=1e-4) and torch.allclose(x0_hat, x0, atol=1e-4)

    def oracle(x, t):  # the exact velocity, given x0
        ab = d.alpha_bar[t].view(-1, 1, 1, 1)
        n = (x - ab.sqrt() * x0[:1]) / (1 - ab).sqrt()
        return ab.sqrt() * n - (1 - ab).sqrt() * x0[:1]

    out = d.sample(oracle, (1, 4, 8, 8), 20, torch.device("cpu"), torch.Generator().manual_seed(0))
    assert torch.allclose(out, x0[:1], atol=1e-3)
    with pytest.raises(ValueError):
        Diffusion(prediction="x0")


def test_sampling_is_reproducible_with_a_generator():
    d = Diffusion()
    eps = lambda x, t: 0.1 * x  # noqa: E731
    run = lambda: d.sample(eps, (1, 4, 4, 4), 5, torch.device("cpu"), torch.Generator().manual_seed(3), eta=1.0)  # noqa: E731
    assert torch.equal(run(), run())


# ------------------------------------------------------------------ data cache


def test_cache_shapes_and_flip_copies(tmp_path):
    from semcom.gen.data import CachedSplit, cache_meta, prepare

    cfg = tiny(tmp_path)
    prepare(cfg)
    meta = cache_meta(cfg)
    assert meta["counts"] == {"train": 64, "val": 32, "test": 32} and meta["fake"]
    train = CachedSplit(cfg.cache_dir, "train", augment=True)
    a = train._open()
    # The stand-in VAE is flip-equivariant, so the cached flipped latent is the flipped latent.
    assert torch.allclose(torch.from_numpy(a["latents_flip"][0].astype("float32")),
                          torch.from_numpy(a["latents"][0].astype("float32")).flip(-1), atol=1e-2)
    item = CachedSplit(cfg.cache_dir, "test")[0]
    assert item["image"].shape == (3, 128, 128) and item["latent"].shape == (4, 16, 16)
    assert item["dino"].shape == (32, 8, 8)
    assert not (cfg.cache_dir / "test_latents_flip.npy").exists()


# ------------------------------------------------------------------ end to end


@pytest.mark.parametrize("model", ["dit", "cnn"])
def test_train_evaluate_and_resume(tmp_path, model):
    from semcom.gen.evaluate import evaluate_run
    from semcom.gen.train import train

    cfg = tiny(tmp_path, model=model)
    # A zero-hour budget stops before the first step, leaving a resumable checkpoint.
    assert train(cfg.replace(max_hours=0.0)) is None
    assert (cfg.run_dir / "checkpoint.pt").exists() and not (cfg.run_dir / "history.json").exists()
    summary = train(cfg)
    assert summary["steps"] == cfg.steps
    hist = json.loads((cfg.run_dir / "history.json").read_text())
    assert [h["step"] for h in hist["history"]] == [2, 4]
    out = evaluate_run(cfg.run_dir, external=False)
    key = "0" if model == "cnn" else "0_2"
    assert set(out["results"]) == {key}
    assert out["results"][key]["n_images"] == 8 and "dino_sim" in out["results"][key]
    assert (cfg.run_dir / "eval_samples.png").exists()


def test_evaluation_is_deterministic(tmp_path):
    from semcom.gen.evaluate import evaluate_run
    from semcom.gen.train import train

    cfg = tiny(tmp_path)
    train(cfg)
    a = evaluate_run(cfg.run_dir, external=False)["results"]
    b = evaluate_run(cfg.run_dir, external=False)["results"]
    for k in a:
        assert a[k]["psnr"] == pytest.approx(b[k]["psnr"], abs=1e-6)


def test_cnn_baseline_maps_snr_modes_to_blind_study_arms(tmp_path):
    assert CNNBaseline(tiny(tmp_path, model="cnn")).jscc.decoder_input == "none"
    assert CNNBaseline(tiny(tmp_path, model="cnn", snr_mode="snr")).jscc.decoder_input == "snr"
    assert CNNBaseline(tiny(tmp_path, model="cnn", snr_mode="blank")).jscc.decoder_input == "blank"


# ------------------------------------------------------------------ correctness checks


def test_overfit_check_passes_a_working_model_and_catches_a_detached_one(tmp_path):
    from semcom.gen.checks import SanityError, overfit_check
    from semcom.gen.data import CachedSplit, prepare

    cfg = tiny(tmp_path)
    prepare(cfg)
    items = [CachedSplit(cfg.cache_dir, "val")[i] for i in range(4)]
    batch = {k: torch.stack([it[k] for it in items]) for k in ("image", "latent", "dino")}
    model = GenJSCC(cfg, 32)
    out = overfit_check(model, batch, cfg, torch.device("cpu"), steps=25)
    assert out["ratio"] < 0.5 and out["encoder_grad"] > 0

    class Detached(GenJSCC):  # the channel path cut: encoders get no gradient
        def received(self, batch, snr_db, generator=None):
            return tuple(y.detach() for y in super().received(batch, snr_db, generator))

    with pytest.raises(SanityError, match="encoders"):
        overfit_check(Detached(cfg, 32), batch, cfg, torch.device("cpu"), steps=25)


def test_validation_reports_how_much_the_receiver_uses_the_channel(tmp_path):
    from semcom.gen.train import train

    cfg = tiny(tmp_path)
    train(cfg)
    hist = json.loads((cfg.run_dir / "history.json").read_text())
    last = hist["history"][-1]
    entry = last["val_by_snr"]["0.0"]
    assert entry["cond_gain"] == pytest.approx(entry["uncond"] - entry["loss"])
    # EMA and raw weights both, with the sampled latents' spread, so an EMA problem shows up
    assert set(last["quality"]) == {"psnr@0", "dino@0", "zstd@0", "raw_psnr@0", "raw_dino@0",
                                    "raw_zstd@0", "raw_val_loss", "ema_decay"}
    assert (cfg.run_dir / "best.pt").exists() and hist["summary"]["best_step"] in (2, 4)
    sanity = json.loads((cfg.run_dir / "sanity.json").read_text())
    assert set(sanity) == {"data", "vae_ceiling", "overfit"}


def _watch():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "watch_job.py"
    spec = importlib.util.spec_from_file_location("watch_job", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_watcher_parses_metric_lines():
    w = _watch()
    rec = w.parse_line("VAL run=dit-S_cbr48_spec0_s0 step=2000 val_loss=0.31 cond_gain=0.02 psnr@0=18.5")
    assert rec == {"kind": "VAL", "run": "dit-S_cbr48_spec0_s0", "step": 2000.0, "val_loss": 0.31,
                   "cond_gain": 0.02, "psnr@0": 18.5}
    assert w.parse_line("SANITY overfit run=x ratio=0.1")["kind"] == "SANITY overfit"
    assert w.parse_line("step 10 loss 0.3") is None


def test_watcher_rules_fire_on_the_failures_they_name():
    w = _watch()
    steps = [{"kind": "STEP", "run": "r", "step": 1000.0 * i, "loss": 0.2, "it_s": 8.0, "skipped": 0.0}
             for i in range(1, 13)]
    assert w.check(steps, "") == []
    spiked = steps + [{"kind": "STEP", "run": "r", "step": 13000.0, "loss": 0.5, "it_s": 8.0, "skipped": 0.0}]
    assert any(k.startswith("spike") for k, _ in w.check(spiked, ""))
    slow1 = steps + [{"kind": "STEP", "run": "r", "step": 13000.0, "loss": 0.2, "it_s": 1.0, "skipped": 0.0}]
    assert not any(k.startswith("slow") for k, _ in w.check(slow1, ""))  # one window: a validation pass
    slow = slow1 + [{"kind": "STEP", "run": "r", "step": 13100.0, "loss": 0.2, "it_s": 1.0, "skipped": 0.0}]
    assert any(k.startswith("slow") for k, _ in w.check(slow, ""))
    vals = [{"kind": "VAL", "run": "r", "step": 2000.0 * i, "val_loss": v, "cond_gain": 0.05}
            for i, v in enumerate([0.3, 0.31, 0.32, 0.33], 1)]
    assert any(k.startswith("val_rising") for k, _ in w.check(vals, ""))
    # g-w3 spec5: a 0.4% creep from 30k is mild overfitting that best.pt handles, not an alert.
    creep = [{**v, "val_loss": x} for v, x in zip(vals, [0.249855, 0.250048, 0.250368, 0.250847])]
    assert not any(k.startswith("val_rising") for k, _ in w.check(creep, ""))
    ignored = [{"kind": "VAL", "run": "r", "step": 12000.0, "val_loss": 0.3, "cond_gain": 0.0}]
    assert any(k.startswith("no_cond") for k, _ in w.check(ignored, ""))
    assert any(k.startswith("nonfinite") for k, _ in w.check([{"kind": "STEP", "run": "r", "step": 5.0,
                                                               "loss": float("nan")}], ""))
    assert any(k.startswith("error") for k, _ in w.check([], "Traceback (most recent call last):"))


def test_watcher_quality_rules_catch_g_w1():
    """The reconstruction-quality rules fire on g-w1's real validation numbers (EMA weights that
    sampled meaningless images while the losses looked healthy) and stay quiet on a good run."""
    w = _watch()
    g_w1 = [(2000, 11.281, 0.022), (4000, 11.087, 0.020), (6000, 9.460, 0.005), (8000, 7.818, -0.009),
            (10000, 7.028, -0.016)]
    vals = lambda rows, **extra: [{"kind": "VAL", "run": "dit-r", "step": float(s), "val_loss": 0.18,  # noqa: E731
                                   "cond_gain": 0.03, "psnr@0": p, "dino@0": d, **extra} for s, p, d in rows]
    keys = lambda rows, **extra: {k.split(":")[0] for k, _ in w.check(vals(rows, **extra), "")}  # noqa: E731
    assert "quality_drop" in keys(g_w1[:4])
    assert "unrelated" in keys(g_w1)
    good = [(2000 * i, 14.0 + i, 0.3 + 0.05 * i) for i in range(1, 7)]
    assert keys(good, **{"zstd@0": 1.0, "raw_psnr@0": 20.0}) == set()
    assert "zscale" in keys(good, **{"zstd@0": 3.1})
    assert "ema_gap" in keys(good, **{"raw_psnr@0": 30.0})
    # g-w3's healthy -5 dB specialist: DINO ~0.04 is normal at -5 dB, so "unrelated" stays quiet there.
    spec_m5 = [{"kind": "VAL", "run": "dit-S-v_cbr48_spec-5_s0", "step": float(s), "val_loss": 0.42 - 0.003 * i,
                "cond_gain": 0.2, "psnr@-5": 15.0 + 0.05 * i, "dino@-5": 0.035 + 0.002 * i}
               for i, s in enumerate(range(2000, 12_001, 2000))]
    assert {k.split(":")[0] for k, _ in w.check(spec_m5, "")} == set()
    # g-w4's 16-QAM specialist: behind the analog one but still improving, so not "unrelated".
    qam = [(2000, 15.2, 0.03), (4000, 15.6, 0.05), (6000, 15.89, 0.064), (8000, 16.0, 0.073),
           (10000, 16.16, 0.082)]
    assert "unrelated" not in keys(qam)
    assert "unrelated" in keys(qam + [(12000, 16.2, 0.083)])  # stalled below 0.1


def test_ema_decay_ramps_up_to_its_cap():
    from semcom.gen.train import ema_decay

    cfg = GenConfig()
    assert ema_decay(cfg, 0) == pytest.approx(0.1)
    assert ema_decay(cfg, 10_000) == pytest.approx(10_001 / 10_010)
    assert ema_decay(cfg, 10**6) == cfg.ema_decay
    old = cfg.replace(ema_warmup=False)
    assert (ema_decay(old, 999), ema_decay(old, 1000)) == (0.0, cfg.ema_decay)


def test_velocity_dit_trains_evaluates_and_is_named_apart(tmp_path):
    from semcom.gen.evaluate import evaluate_run
    from semcom.gen.train import train

    cfg = tiny(tmp_path, prediction="v")
    assert cfg.run_name == "dit-S-v_cbr48_spec0_s0"
    train(cfg)
    out = evaluate_run(cfg.run_dir, external=False)
    assert out["checkpoint"] == "best" and all(v["psnr"] == v["psnr"] for v in out["results"].values())


# ------------------------------------------------------------------ digital (M-QAM) transmission


def test_digital_configs_are_named_apart_and_need_square_qam():
    assert GenConfig(digital=True, prediction="v").run_name == "dit-S-v_cbr48_qam16_spec0_s0"
    assert GenConfig(digital=True, modulation_order=4, model="cnn").run_name == "cnn_cbr48_qam4_spec0_s0"
    assert GenConfig(modulation_order=8).run_name == "dit-S_cbr48_spec0_s0"  # analog ignores M
    for m in (8, 2, 12):
        with pytest.raises(ValueError):
            GenConfig(digital=True, modulation_order=m)


@pytest.mark.parametrize("model", ["dit", "cnn"])
@pytest.mark.parametrize("order", [4, 16])
def test_digital_models_send_exact_qam_points_and_still_train_the_encoders(tmp_path, model, order):
    from semcom.gen.data import CachedSplit, prepare

    cfg = tiny(tmp_path, model=model, digital=True, modulation_order=order)
    prepare(cfg)
    items = [CachedSplit(cfg.cache_dir, "val")[i] for i in range(4)]
    batch = {k: torch.stack([it[k] for it in items]) for k in ("image", "latent", "dino")}
    m = GenJSCC(cfg, 32) if model == "dit" else CNNBaseline(cfg)
    z = m.channel_input(batch)
    assert z.shape == (4, cfg.k, 2) and m.quantiser.is_on_constellation(z)
    assert len(torch.unique(z.reshape(-1, 2), dim=0)) <= order

    m.train()
    out = m.loss(batch, torch.zeros(4))
    assert out["loss"].item() == pytest.approx((out["mse"] + cfg.kl_weight * out["kl"]).item(), rel=1e-5)
    out["loss"].backward()
    enc = (m.sem_enc, m.det_enc) if model == "dit" else (m.jscc.encoder,)
    for e in enc:  # the relaxation carries gradient back through the quantiser
        assert sum(p.grad.abs().sum().item() for p in e.parameters() if p.grad is not None) > 0
    m.eval()
    assert "kl" not in m.loss(batch, torch.zeros(4))  # validation: the MSE alone


def test_default_temperature_is_fixed_and_scaled_to_the_constellation_spacing(tmp_path):
    from semcom.constellation import qam_constellation
    from semcom.gen.model import make_quantiser

    for m in (4, 16, 64):
        cfg = GenConfig(digital=True, modulation_order=m)
        p = qam_constellation(m)
        d = torch.cdist(p, p)
        assert cfg.sigma_q == pytest.approx(2 / d[d > 0].min().item() ** 2, rel=1e-4)
    assert GenConfig(digital=True).sigma_q == pytest.approx(5.0)  # DeepJSCC-Q's starting value
    q = make_quantiser(GenConfig(digital=True))
    q._iteration.fill_(10**6)
    assert q.step_anneal() == pytest.approx(5.0)  # no schedule by default


def test_quantisation_ignores_autocast(tmp_path):
    from semcom.gen.model import make_quantiser, quantise

    q = make_quantiser(tiny(tmp_path, digital=True))
    z = torch.randn(2, 64, 2)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        (a,), _ = quantise(q, z.bfloat16())
    assert a.dtype == torch.float32 and q.is_on_constellation(a)


def test_symbol_stats_rejects_off_constellation_symbols(tmp_path):
    from semcom.gen.checks import SanityError, symbol_stats
    from semcom.gen.data import CachedSplit, prepare

    cfg = tiny(tmp_path, digital=True)
    prepare(cfg)
    items = [CachedSplit(cfg.cache_dir, "val")[i] for i in range(4)]
    batch = {k: torch.stack([it[k] for it in items]) for k in ("image", "latent", "dino")}
    m = GenJSCC(cfg, 32)
    st = symbol_stats(m, batch)
    assert 0 < st["qam_usage"] <= 1 and st["qam_power"] > 0
    m.quantiser = None
    assert symbol_stats(m, batch) == {}  # analog: nothing to check

    class Leaky(GenJSCC):
        def channel_input(self, batch):
            return super().channel_input(batch) + 1e-3

    with pytest.raises(SanityError, match="constellation"):
        symbol_stats(Leaky(cfg, 32), batch)


@pytest.mark.parametrize("model", ["dit", "cnn"])
def test_digital_run_trains_resumes_and_evaluates(tmp_path, model):
    from semcom.gen.evaluate import evaluate_run
    from semcom.gen.train import train

    cfg = tiny(tmp_path, model=model, digital=True, prediction="v", anneal_period=2, steps=6, val_every=3,
               ckpt_every=3)
    train(cfg.replace(steps=3))  # stop mid-run: the anneal state must survive the resume
    ck = torch.load(cfg.run_dir / "checkpoint.pt", weights_only=False)
    key = "quantiser.sigma_q" if model == "dit" else "jscc.quantiser.sigma_q"
    sigma3 = ck["model"][key].item()
    assert sigma3 > cfg.sigma_q  # annealed by step 3 (period 2)
    (cfg.run_dir / "history.json").unlink()
    train(cfg)
    ck = torch.load(cfg.run_dir / "checkpoint.pt", weights_only=False)
    assert ck["model"][key].item() > sigma3 and ck["model"][key.replace("sigma_q", "_iteration")].item() == 6
    hist = json.loads((cfg.run_dir / "history.json").read_text())
    assert {"qam_usage", "qam_kurtosis", "qam_power"} <= set(hist["history"][-1]["quality"])
    out = evaluate_run(cfg.run_dir, external=False)
    assert out["digital"] and out["modulation_order"] == 16 and out["symbols"]["on_constellation"]


def test_watcher_flags_codebook_collapse():
    w = _watch()
    vals = [{"kind": "VAL", "run": "dit-S-v_cbr48_qam16_spec0_s0", "step": s, "val_loss": 0.3,
             "qam_usage": u} for s, u in ((2000, 0.3), (4000, 0.9))]
    assert not [k for k, _ in w.check(vals, "") if k.startswith("collapse")]
    vals.append({**vals[-1], "step": 6000, "qam_usage": 0.4})
    assert [k for k, _ in w.check(vals, "") if k.startswith("collapse")]
    assert w.parse_line("SYMBOLS run=x qam_usage=0.9")["kind"] == "SYMBOLS"


def test_downloads_are_retried_then_give_up():
    from semcom.gen.frozen import retry

    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise OSError("HTTP Error 504: Gateway Time-out")
        return "ok"

    assert retry(flaky, "x", wait=0) == "ok" and len(calls) == 3
    with pytest.raises(OSError):
        retry(lambda: (_ for _ in ()).throw(OSError("down")), "x", tries=2, wait=0)


def _batch(cfg, n=8):
    from semcom.gen.data import CachedSplit, prepare

    prepare(cfg)
    items = [CachedSplit(cfg.cache_dir, "val")[i] for i in range(n)]
    return {k: torch.stack([it[k] for it in items]) for k in ("image", "latent", "dino")}


@pytest.mark.parametrize("power_ref", ["nominal", "actual"])
def test_digital_snr_refers_to_the_chosen_energy(tmp_path, power_ref):
    from semcom.gen.model import from_symbols

    cfg = tiny(tmp_path, digital=True, power_ref=power_ref, c_sem=64, c_det=64)  # many symbols
    batch = _batch(cfg)
    m = GenJSCC(cfg, 32)
    snr = torch.full((8,), 3.0)
    for training in (True, False):
        m.train(training)
        z_s, z_d = m.transmit(batch)
        es = torch.cat([z_s, z_d], 1).pow(2).sum(-1).mean().item()
        y_s, _ = m.received(batch, snr)
        noise = (y_s - from_symbols(z_s, cfg.c_sem, cfg.grid)).pow(2).sum(1).mean().item() * 2 / cfg.c_sem
        ref = 1.0 if power_ref == "nominal" else (es if training else m.es_ref.item())
        assert noise == pytest.approx(ref * 10 ** -0.3, rel=0.05)
    if power_ref == "actual":
        assert m.es_ref.item() != 1.0  # the running estimate moved toward the energy sent
    else:
        assert not hasattr(m, "es_ref")


def test_analog_models_keep_their_state_dict(tmp_path):
    cfg = tiny(tmp_path, power_ref="actual")  # no effect without digital
    assert "es_ref" not in GenJSCC(cfg, 32).state_dict()
    assert not [k for k in CNNBaseline(cfg.replace(model="cnn")).state_dict() if "es_ref" in k]


def test_ema_copies_buffers(tmp_path):
    import copy

    from semcom.gen.train import update_ema

    m = GenJSCC(tiny(tmp_path, digital=True, power_ref="actual"), 32)
    ema = copy.deepcopy(m)
    m.es_ref.fill_(0.8)
    update_ema(ema, m, 0.999)
    assert ema.es_ref.item() == pytest.approx(0.8)


@pytest.mark.parametrize("model", ["dit", "cnn"])
def test_actual_energy_run_trains_and_evaluates_against_test_set_energy(tmp_path, model):
    from semcom.gen.evaluate import evaluate_run
    from semcom.gen.train import train

    cfg = tiny(tmp_path, model=model, digital=True, power_ref="actual", prediction="v")
    train(cfg)
    out = evaluate_run(cfg.run_dir, external=False)
    sym = out["symbols"]
    assert sym["power_ref"] == "actual" and "es_ref_trained" in sym and 0 < sym["qam_power"] < 2


def test_kaggle_runner_restores_uploaded_run_folders(tmp_path):
    """The job runner's restore step, run on a fake /kaggle/input of flat dataset files."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "kaggle_jobs.py"
    spec = importlib.util.spec_from_file_location("kaggle_jobs", path)
    kj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kj)
    block = kj.RUNNER.split("# Local run folders")[1].split("try:")[0].split("\n", 1)[1]
    inp, src = tmp_path / "input" / "ds", tmp_path / "src"
    inp.mkdir(parents=True)
    run = ("results", "gen", "p128", "dit-S-v_cbr48_blind-5to5_s0")
    (inp / "__".join(run + ("best.pt",))).write_bytes(b"weights")
    (inp / "__".join(run + ("config.yaml",))).write_text("from upload")
    (src / Path(*run)).mkdir(parents=True)
    (src / Path(*run) / "config.yaml").write_text("from git")  # tracked files are kept
    (inp / "unrelated.txt").write_text("x")
    import os
    import shutil

    exec(block.replace('"/kaggle/input"', repr(str(tmp_path / "input"))),
         {"os": os, "shutil": shutil, "src": str(src)})
    assert (src / Path(*run) / "best.pt").read_bytes() == b"weights"
    assert (src / Path(*run) / "config.yaml").read_text() == "from git"
    assert not (src / "unrelated.txt").exists()


def test_report_envelope_is_per_seed(tmp_path):
    """A replicate seed's specialists must not replace seed 0's in the envelope, and each told arm
    is compared with the specialists of its own seed."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "gen_report.py"
    spec = importlib.util.spec_from_file_location("gen_report", path)
    rep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rep)
    psnr = {("spec-5", 0): 15.0, ("spec5", 0): 19.0, ("spec-5", 1): 14.0, ("spec5", 1): 20.0,
            ("told-5to5", 0): 18.0, ("told-5to5", 1): 18.0}
    for (arm, seed), v in psnr.items():
        name = f"dit-S-v_cbr48_{arm}_s{seed}"
        (tmp_path / name).mkdir()
        res = {f"{s}_20": {"psnr": v, "lpips": 0.3, "dino_sim": 0.2, "fid": 100.0} for s in (-5, 5)}
        (tmp_path / name / "eval.json").write_text(json.dumps(
            {"run_name": name, "model": "dit", "arm": arm, "results": res}))
    runs = rep.load(tmp_path, 20)
    assert rep.envelope(runs, "dit", seed=0) == {-5.0: runs["dit-S-v_cbr48_spec-5_s0"]["by_snr"][-5.0],
                                                 5.0: runs["dit-S-v_cbr48_spec5_s0"]["by_snr"][5.0]}
    text = rep.table(runs, 20)
    assert "| dit-S-v_cbr48_told-5to5_s0 vs envelope | +3.000 | -1.000 |" in text
    assert "| dit-S-v_cbr48_told-5to5_s1 vs envelope | +4.000 | -2.000 |" in text
    assert set(rep.first_seed(runs)) == {n for n in runs if n.endswith("_s0")}


@pytest.mark.parametrize("parts", ["receiver", "transmitter"])
def test_fine_tune_starts_from_a_run_and_trains_only_its_part(tmp_path, parts):
    from semcom.gen.train import train

    src = tiny(tmp_path, prediction="v", snr_mode="snr")
    train(src)
    start = torch.load(src.run_dir / "best.pt", weights_only=False)["ema"]
    ft = src.replace(init_from=str(src.run_dir), train_parts=parts, tag=f"ft-{parts[:2]}",
                     snr_train_min=-5.0, snr_train_max=-5.0, steps=2, val_every=2)
    assert ft.run_name == f"dit-S-v_cbr48_told-5to-5_ft-{parts[:2]}_s0" and ft.run_dir != src.run_dir
    train(ft)
    end = torch.load(ft.run_dir / "final.pt", weights_only=False)["ema"]
    frozen = ("dit.",) if parts == "transmitter" else ("sem_enc.", "det_enc.")
    moved = {n for n in start if start[n].is_floating_point() and not torch.equal(start[n], end[n])}
    assert moved and all(not n.startswith(frozen) for n in moved)
    assert any(n.startswith("dit." if parts == "receiver" else "det_enc.") for n in moved)


def test_fine_tune_options_are_validated():
    with pytest.raises(ValueError):
        GenConfig(train_parts="decoder")
    with pytest.raises(ValueError):
        GenConfig(model="cnn", train_parts="receiver")
    with pytest.raises(ValueError):
        GenConfig(tag="a_b")
    assert GenConfig().run_name == "dit-S_cbr48_spec0_s0"


def test_snr_aware_transmitter_starts_from_a_blind_run_and_learns_to_use_the_snr(tmp_path):
    from semcom.gen.model import GenJSCC
    from semcom.gen.train import train

    src = tiny(tmp_path, prediction="v", snr_mode="blank")
    train(src)
    start = torch.load(src.run_dir / "best.pt", weights_only=False)["ema"]
    ft = src.replace(init_from=str(src.run_dir), encoder_snr=True, train_parts="transmitter",
                     tag="ft-tx", steps=4, lr=1e-2)
    assert ft.run_name == "dit-S-v_cbr48_blind-5to5_txsnr_ft-tx_s0"
    # FiLM layers start at the identity: before any step the SNR-aware transmitter sends exactly
    # what the blind run's transmitter sends, at every SNR.
    m = GenJSCC(ft, 32).eval()
    from semcom.gen.train import load_init

    load_init(m, start)
    ref = GenJSCC(src, 32).eval()
    ref.load_state_dict(start)
    batch = _batch(src, 4)
    for snr in (-5.0, 5.0):
        s = torch.full((4,), snr)
        for a, b in zip(m.transmit(batch, s), ref.transmit(batch)):
            assert torch.allclose(a, b, atol=1e-6)
    with pytest.raises(ValueError):
        m.transmit(batch)  # an SNR-aware transmitter must be given the SNR
    train(ft)
    end = torch.load(ft.run_dir / "final.pt", weights_only=False)["ema"]
    moved = {n for n in end if end[n].is_floating_point() and (n not in start or not torch.equal(start[n], end[n]))}
    assert any(".film." in n for n in moved) and not any(n.startswith("dit.") for n in moved)
    m.load_state_dict(end)
    lo, hi = (m.transmit(batch, torch.full((4,), v))[1] for v in (-5.0, 5.0))
    assert not torch.allclose(lo, hi)  # after training, what is sent depends on the SNR


def test_snr_aware_transmitter_options_are_validated():
    with pytest.raises(ValueError):
        GenConfig(model="cnn", encoder_snr=True)
    with pytest.raises(ValueError):
        GenConfig(digital=True, modulation_order=16, encoder_snr=True)


def test_suffixed_evaluation_keeps_the_main_one(tmp_path):
    from semcom.gen.evaluate import evaluate_run
    from semcom.gen.train import train

    cfg = tiny(tmp_path, prediction="v", snr_mode="blank")
    train(cfg)
    evaluate_run(cfg.run_dir, external=False)
    main = (cfg.run_dir / "eval.json").read_text()
    out = evaluate_run(cfg.run_dir, external=False, snrs=[-5.0], steps_list=[1], suffix="fewstep")
    assert set(out["results"]) == {"-5_1"}
    assert (cfg.run_dir / "eval.json").read_text() == main
    assert (cfg.run_dir / "eval-fewstep.json").exists() and (cfg.run_dir / "eval_images-fewstep.npz").exists()
    with pytest.raises(ValueError):
        evaluate_run(cfg.run_dir, external=False, suffix="a_b")
