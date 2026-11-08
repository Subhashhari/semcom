"""Train one arm of the generative-JSCC prototype (or its CNN baseline) from the cached data.

    python -m semcom.gen.train --config configs/gen/p128.yaml --snr-mode fixed --snr-train-fixed 0

Step-budgeted, with an exponential moving average (EMA) of the weights: the EMA model is what
gets evaluated, as is standard for diffusion models. Resumable from checkpoint.pt; `max_hours`
stops a run cleanly so it can continue in a later session.

Run folder:
    config.yaml        the full config, derived fields included
    checkpoint.pt      model, EMA, optimiser, scaler, step (for resuming)
    final.pt           EMA weights + config, written when the step budget is reached
    best.pt            the same at the validation with the lowest validation loss (500 held-out
                       val images, never the test split); evaluation uses it by default
    history.json       summary + validation history; its presence means "finished"
    samples/           reconstruction grids during training
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from ..train import init_wandb, log
from .checks import SanityError, check_data, kv, overfit_check, symbol_stats, val_quality, vae_ceiling
from .config import GenConfig
from .data import cache_meta, loaders, prepare
from .model import build_model, num_parameters


def sample_snr(cfg: GenConfig, b: int, device: torch.device) -> torch.Tensor:
    if cfg.snr_mode == "fixed":
        return torch.full((b,), float(cfg.snr_train_fixed), device=device)
    return torch.rand(b, device=device) * (cfg.snr_train_max - cfg.snr_train_min) + cfg.snr_train_min


def val_snrs(cfg: GenConfig) -> list[float]:
    if cfg.snr_mode == "fixed":
        return [float(cfg.snr_train_fixed)]
    return [cfg.snr_train_min, (cfg.snr_train_min + cfg.snr_train_max) / 2, cfg.snr_train_max]


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def ema_decay(cfg: GenConfig, step: int) -> float:
    if cfg.ema_warmup:
        return min(cfg.ema_decay, (1 + step) / (10 + step))
    return cfg.ema_decay if step >= cfg.warmup_steps else 0.0  # the g-w1 rule


@torch.no_grad()
def update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    for e, p in zip(ema.parameters(), model.parameters()):
        e.lerp_(p.detach(), 1 - decay)
    # Buffers are copied, not averaged: the transmitted-energy reference (es_ref) and the
    # quantiser's temperature state follow the trained model.
    for e, b in zip(ema.buffers(), model.buffers()):
        e.copy_(b)


@torch.no_grad()
def validate(model, loader, cfg: GenConfig, device, use_amp: bool) -> dict[float, dict]:
    """Validation loss at each validation SNR, with fixed randomness across calls.

    For the DiT also the loss with every received symbol dropped ("uncond"): the gap between the
    two (cond_gain) is how much the receiver uses the channel.
    """
    model.eval()
    out = {}
    modes = ("none", "all") if cfg.model == "dit" else ("none",)
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        for snr in val_snrs(cfg):
            entry = {}
            for mode in modes:
                torch.manual_seed(1234)  # same timesteps, noise and channel for both modes
                total, n = 0.0, 0
                for batch in loader:
                    batch = to_device(batch, device)
                    s = torch.full((batch["image"].shape[0],), snr, device=device)
                    with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                        total += model.loss(batch, s, drop=mode)["loss"].item() * s.shape[0]
                    n += s.shape[0]
                entry["loss" if mode == "none" else "uncond"] = total / max(n, 1)
            if "uncond" in entry:
                entry["cond_gain"] = entry["uncond"] - entry["loss"]
            out[snr] = entry
    model.train()
    return out


@torch.no_grad()
def save_samples(model, cfg: GenConfig, batch: dict, vae, path: Path, steps: int = 20) -> None:
    """Grid: originals on top, then one row per validation SNR."""
    from torchvision.utils import save_image

    model.eval()
    rows = [batch["image"]]
    gen = torch.Generator(device=batch["image"].device)
    for i, snr in enumerate(val_snrs(cfg)):
        gen.manual_seed(i)
        s = torch.full((batch["image"].shape[0],), snr, device=batch["image"].device)
        rows.append(model.reconstruct(batch, s, vae, steps=steps, channel_gen=gen, sample_gen=gen,
                                      cfg_scale=cfg.cfg_scale).float())
    model.train()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(torch.cat(rows), path, nrow=batch["image"].shape[0])


def train(cfg: GenConfig) -> dict | None:
    """Train to cfg.steps. Returns the summary, or None if stopped early by max_hours."""
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    prepare(cfg, device)
    meta = cache_meta(cfg)

    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.yaml")

    model = build_model(cfg, meta["dino_dim"]).to(device)
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    train_loader, val_loader, _ = loaders(cfg)

    step, history = 0, []
    ckpt_path = run_dir / "checkpoint.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        scaler.load_state_dict(ckpt["scaler"])
        step, history = ckpt["step"], ckpt["history"]
        print(f"resumed at step {step}")

    print(f"{cfg.run_name}: {cfg.model} k={cfg.k} (CBR 1/{1 / cfg.bandwidth_ratio:.0f}) "
          f"params={num_parameters(model):,} device={device} train={meta['counts']['train']}")
    wandb_run = init_wandb(cfg)

    def save_ckpt():
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "step": step, "history": history}, ckpt_path)

    def batches():
        while True:
            yield from train_loader

    from .frozen import load_frozen

    vae, dino = load_frozen(cfg, device)
    first_val = to_device(next(iter(val_loader)), device)
    quality_batch = {k: v[: cfg.val_quality_images] if torch.is_tensor(v) else v for k, v in first_val.items()}
    sample_batch = {k: v[:8] if torch.is_tensor(v) else v for k, v in first_val.items()}

    sanity = {}
    if cfg.sanity and step == 0:
        sanity["data"] = check_data(to_device(next(iter(train_loader)), device), cfg)
        sanity["vae_ceiling"] = vae_ceiling(quality_batch, vae, dino, cfg)
        sanity["overfit"] = overfit_check(model, quality_batch, cfg, device, steps=cfg.sanity_steps)
        (run_dir / "sanity.json").write_text(json.dumps(sanity, indent=2), encoding="utf-8")

    t0, t_log, running, gnorm_sum, skipped, bad_streak = time.time(), time.time(), 0.0, 0.0, 0, 0
    model.train()
    for batch in batches():
        if step >= cfg.steps:
            break
        if cfg.max_hours is not None and time.time() - t0 > cfg.max_hours * 3600:
            save_ckpt()
            print(f"max_hours reached at step {step}; checkpoint saved, resume later")
            if wandb_run is not None:
                wandb_run.finish()
            return None
        batch = to_device(batch, device)
        lr = cfg.lr * min(1.0, (step + 1) / max(cfg.warmup_steps, 1))
        for g in opt.param_groups:
            g["lr"] = lr
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            out = model.loss(batch, sample_snr(cfg, batch["image"].shape[0], device))
        if not torch.isfinite(out["loss"]):
            bad_streak += 1
            if bad_streak >= cfg.max_bad_steps:
                save_ckpt()
                raise SanityError(f"{bad_streak} consecutive non-finite losses at step {step}")
        else:
            bad_streak = 0
        opt.zero_grad(set_to_none=True)
        scaler.scale(out["loss"]).backward()
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scale_before = scaler.get_scale() if use_amp else 1.0
        scaler.step(opt)
        scaler.update()
        if use_amp and scaler.get_scale() < scale_before:
            skipped += 1  # inf/nan gradient: the scaler skipped this step
        update_ema(ema, model, ema_decay(cfg, step))
        if model.quantiser is not None:
            # The EMA copy's quantiser keeps its initial temperature, which only shapes gradients;
            # its forward pass (the nearest point) is the same at any temperature.
            model.quantiser.step_anneal()
        step += 1
        running += out["loss"].detach()
        gnorm_sum += torch.nan_to_num(gnorm.detach().float(), nan=0.0, posinf=0.0)

        if step % cfg.log_every == 0:
            loss = (running / cfg.log_every).item()
            gn = (gnorm_sum / cfg.log_every).item()
            rate = cfg.log_every / (time.time() - t_log)
            digital = {} if model.quantiser is None else {
                "kl": out["kl"].item() if "kl" in out else 0.0, "sigma_q": model.quantiser.sigma_q.item()}
            print(f"STEP run={cfg.run_name} " + kv(step=step, loss=loss, gnorm=gn, lr=lr, it_s=rate,
                                                   skipped=skipped, hours=(time.time() - t0) / 3600,
                                                   **digital),
                  flush=True)
            log(wandb_run, {"train/loss": loss, "train/grad_norm": gn, "train/lr": lr,
                            "train/it_per_s": rate, "train/skipped_steps": skipped}, step=step)
            running, gnorm_sum, t_log = 0.0, 0.0, time.time()

        if step % cfg.val_every == 0 or step == cfg.steps:
            val = validate(ema, val_loader, cfg, device, use_amp)
            quality = val_quality(ema, cfg, quality_batch, vae, dino, val_snrs(cfg))
            # The raw (non-EMA) weights too, so an EMA problem cannot pass for a model problem.
            raw_val = validate(model, val_loader, cfg, device, use_amp)
            raw_quality = val_quality(model, cfg, quality_batch, vae, dino, val_snrs(cfg))
            quality.update({f"raw_{k}": v for k, v in raw_quality.items()})
            quality["raw_val_loss"] = sum(v["loss"] for v in raw_val.values()) / len(raw_val)
            quality["ema_decay"] = ema_decay(cfg, step)
            quality.update(symbol_stats(ema, quality_batch))
            entry = {"step": step, "val_loss": sum(v["loss"] for v in val.values()) / len(val),
                     "val_by_snr": val, "quality": quality, "skipped": skipped,
                     "hours": (time.time() - t0) / 3600}
            fields = {"val_loss": entry["val_loss"]}
            if cfg.model == "dit":
                entry["cond_gain"] = sum(v["cond_gain"] for v in val.values()) / len(val)
                fields["uncond"] = sum(v["uncond"] for v in val.values()) / len(val)
                fields["cond_gain"] = entry["cond_gain"]
            history.append(entry)
            if entry["val_loss"] < min((h["val_loss"] for h in history[:-1]), default=float("inf")):
                torch.save({"ema": ema.state_dict(), "config": cfg.to_dict(), "step": step,
                            "dino_dim": meta["dino_dim"]}, run_dir / "best.pt")
            print(f"VAL run={cfg.run_name} " + kv(step=step, **fields, **quality), flush=True)
            log(wandb_run, {f"val/{k}": v for k, v in {**fields, **quality}.items()}, step=step)

        if cfg.model == "dit" and (step % cfg.sample_every == 0 or step == cfg.steps):
            save_samples(ema, cfg, sample_batch, vae, run_dir / "samples" / f"step{step:07d}.png")

        if step % cfg.ckpt_every == 0:
            save_ckpt()

    save_ckpt()
    torch.save({"ema": ema.state_dict(), "config": cfg.to_dict(), "step": step,
                "dino_dim": meta["dino_dim"]}, run_dir / "final.pt")
    summary = {
        "run_name": cfg.run_name, "model": cfg.model, "arm": cfg.arm, "steps": step,
        "parameters": num_parameters(model), "k": cfg.k, "bandwidth_ratio": cfg.bandwidth_ratio,
        "final_val_loss": history[-1]["val_loss"] if history else None,
        "best_step": min(history, key=lambda h: h["val_loss"])["step"] if history else None,
        "best_val_loss": min(h["val_loss"] for h in history) if history else None,
        "final_quality": history[-1]["quality"] if history else None,
        "skipped_steps": skipped,
        "sanity": json.loads((run_dir / "sanity.json").read_text()) if (run_dir / "sanity.json").exists() else None,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "wall_clock_s": time.time() - t0,
    }
    (run_dir / "history.json").write_text(json.dumps({"summary": summary, "history": history}, indent=2),
                                          encoding="utf-8")
    if wandb_run is not None:
        wandb_run.summary.update(summary)
        for name in ("final.pt", "best.pt", "config.yaml", "history.json"):
            try:
                wandb_run.save(str(run_dir / name), base_path=str(run_dir), policy="now")
            except Exception as exc:  # noqa: BLE001 - logging must never kill training
                print(f"[wandb] could not upload {name} ({exc})")
        wandb_run.finish()
    print(f"done: {cfg.run_name} at step {step} -> {run_dir}")
    return summary


def add_arguments(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--config", type=Path, default=Path("configs/gen/p128.yaml"))
    p.add_argument("--model", choices=["dit", "cnn"], default=None)
    p.add_argument("--snr-mode", choices=["fixed", "snr", "blank"], default=None)
    p.add_argument("--snr-train-fixed", type=float, default=None)
    p.add_argument("--dit-size", choices=["S", "B", "L"], default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-hours", type=float, default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    return p


def config_from_args(args: argparse.Namespace) -> GenConfig:
    overrides = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
    return GenConfig.from_yaml(args.config, **overrides)


def main() -> None:
    train(config_from_args(add_arguments(argparse.ArgumentParser(description=__doc__)).parse_args()))


if __name__ == "__main__":
    main()
