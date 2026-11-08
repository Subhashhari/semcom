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
    slow = steps + [{"kind": "STEP", "run": "r", "step": 13000.0, "loss": 0.2, "it_s": 1.0, "skipped": 0.0}]
    assert any(k.startswith("slow") for k, _ in w.check(slow, ""))
    vals = [{"kind": "VAL", "run": "r", "step": 2000.0 * i, "val_loss": v, "cond_gain": 0.05}
            for i, v in enumerate([0.3, 0.31, 0.32, 0.33], 1)]
    assert any(k.startswith("val_rising") for k, _ in w.check(vals, ""))
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
