"""Train one arm of the 2x2.

    python -m semcom.train --config configs/cifar_r12.yaml --snr-adaptive --digital -M 16

Logs to Weights & Biases by default; pass --no-wandb (or set wandb_mode: disabled) to run
without it. Runs are resumable from the checkpoint in the run directory.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import Config
from .data import cifar10_loaders, psnr, sample_snr
from .models import JSCC


def build_model(cfg: Config) -> JSCC:
    return JSCC(
        c_out=cfg.c_out,
        snr_adaptive=cfg.snr_adaptive,
        digital=cfg.digital,
        modulation_order=cfg.modulation_order,
        channel=cfg.channel,
        image_size=cfg.image_size,
        in_channels=cfg.in_channels,
        hidden=cfg.hidden,
        avg_power=cfg.avg_power,
        sigma_q_init=cfg.sigma_q_init,
        anneal_period=cfg.anneal_period,
    )


def init_wandb(cfg: Config):
    """Start a wandb run, or return None if logging is off or wandb is unavailable.

    Never fatal: a missing or unconfigured wandb must not cost you a training run.
    """
    if not cfg.wandb or cfg.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        print("[wandb] not installed; continuing without logging")
        return None

    try:
        run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.run_name,
            mode=cfg.wandb_mode,
            config=cfg.to_dict(),
            tags=[cfg.arm, cfg.channel, f"R={cfg.bandwidth_ratio:.4f}"],
            reinit=True,
        )
        # Group the panels the way they are actually read: quality against SNR.
        wandb.define_metric("eval/snr_db")
        wandb.define_metric("eval/*", step_metric="eval/snr_db")
        return run
    except Exception as exc:  # noqa: BLE001 - logging must never kill training
        print(f"[wandb] init failed ({exc}); continuing without logging")
        return None


def log(run, data: dict, step: int | None = None) -> None:
    if run is not None:
        run.log(data, step=step)


@torch.no_grad()
def evaluate(
    model: JSCC,
    loader,
    device: torch.device,
    snrs,
    repeats: int = 1,
) -> dict[float, float]:
    """Mean PSNR at each SNR, averaging `repeats` channel realisations per image."""
    model.eval()
    results = {}
    for snr in snrs:
        total, count = 0.0, 0
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            snr_t = torch.full((x.shape[0],), float(snr), device=device)
            for _ in range(repeats):
                x_hat = model(x, snr_t)["x_hat"]
                total += psnr(x, x_hat).sum().item()
                count += x.shape[0]
        results[float(snr)] = total / max(count, 1)
    model.train()
    return results


def train(cfg: Config) -> dict:
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.yaml")

    model = build_model(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    train_loader, val_loader, _ = cifar10_loaders(
        cfg.data_root, cfg.batch_size, cfg.num_workers, seed=cfg.seed
    )

    print(
        f"{cfg.arm}  R={cfg.bandwidth_ratio:.4f}  k={model.k}  "
        f"params={model.num_parameters():,}  device={device}"
        + (f"  M={cfg.modulation_order}" if cfg.digital else "")
    )

    wandb_run = init_wandb(cfg)
    ckpt_path = run_dir / "checkpoint.pt"

    start_epoch, best_psnr, best_epoch, step = 0, -float("inf"), 0, 0
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimiser"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr, best_epoch, step = ckpt["best_psnr"], ckpt["best_epoch"], ckpt["step"]
        print(f"resumed from epoch {start_epoch} (best {best_psnr:.3f} dB)")

    # The SNRs used for the mid-training validation signal. Deliberately coarse: the full
    # sweep belongs in evaluate.py, and running it every few epochs would dominate cost.
    val_snrs = (
        [cfg.snr_train_fixed]
        if cfg.snr_train_fixed is not None
        else [cfg.snr_train_min, (cfg.snr_train_min + cfg.snr_train_max) / 2, cfg.snr_train_max]
    )

    history = []
    t0 = time.time()
    for epoch in range(start_epoch, cfg.epochs):
        running = {"loss": 0.0, "mse": 0.0, "kl": 0.0, "psnr": 0.0, "n": 0}

        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)
            snr = sample_snr(x.shape[0], cfg, device)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(x, snr)
                mse = F.mse_loss(out["x_hat"], x)
                loss = mse + cfg.kl_weight * out["kl"]

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            # Advance the quantiser's temperature schedule once per optimiser step.
            if model.quantiser is not None:
                model.quantiser.step_anneal()

            step += 1
            bs = x.shape[0]
            running["loss"] += loss.item() * bs
            running["mse"] += mse.item() * bs
            running["kl"] += out["kl"].detach().item() * bs
            running["psnr"] += psnr(x, out["x_hat"].float()).sum().item()
            running["n"] += bs

            if step % cfg.log_every == 0:
                entry = {
                    "train/loss": loss.item(),
                    "train/mse": mse.item(),
                    "train/psnr": psnr(x, out["x_hat"].float()).mean().item(),
                    "train/epoch": epoch,
                }
                if model.quantiser is not None:
                    entry["train/kl"] = out["kl"].detach().item()
                    entry["train/sigma_q"] = model.quantiser.sigma_q.item()
                log(wandb_run, entry, step=step)

        n = max(running["n"], 1)
        epoch_stats = {
            "epoch": epoch,
            "train_loss": running["loss"] / n,
            "train_psnr": running["psnr"] / n,
        }

        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
            val = evaluate(model, val_loader, device, val_snrs, repeats=1)
            mean_psnr = sum(val.values()) / len(val)
            epoch_stats["val_psnr"] = mean_psnr
            epoch_stats["val_by_snr"] = val

            log(
                wandb_run,
                {
                    "val/psnr_mean": mean_psnr,
                    **{f"val/psnr_at_{s:g}dB": v for s, v in val.items()},
                    "val/epoch": epoch,
                },
                step=step,
            )

            improved = mean_psnr > best_psnr
            if improved:
                best_psnr, best_epoch = mean_psnr, epoch
                torch.save(
                    {"model": model.state_dict(), "config": cfg.to_dict(), "epoch": epoch},
                    run_dir / "best.pt",
                )

            print(
                f"epoch {epoch:4d}  train {epoch_stats['train_psnr']:6.3f} dB  "
                f"val {mean_psnr:6.3f} dB{'  *' if improved else ''}  "
                f"({time.time() - t0:.0f}s)"
            )

            torch.save(
                {
                    "model": model.state_dict(),
                    "optimiser": opt.state_dict(),
                    "epoch": epoch,
                    "best_psnr": best_psnr,
                    "best_epoch": best_epoch,
                    "step": step,
                },
                ckpt_path,
            )

            if epoch - best_epoch >= cfg.patience:
                print(f"early stopping at epoch {epoch} (best {best_psnr:.3f} dB @ {best_epoch})")
                history.append(epoch_stats)
                break

        history.append(epoch_stats)

    summary = {
        "run_name": cfg.run_name,
        "arm": cfg.arm,
        "best_val_psnr": best_psnr,
        "best_epoch": best_epoch,
        # Reported alongside the numbers because the ADJSCC results being reproduced were
        # trained for 1280 epochs; early stopping usually fires far sooner.
        "epochs_trained": history[-1]["epoch"] + 1 if history else 0,
        "epochs_configured": cfg.epochs,
        "parameters": model.num_parameters(),
        "bandwidth_ratio": cfg.bandwidth_ratio,
        "k": model.k,
        "wall_clock_s": time.time() - t0,
    }
    (run_dir / "history.json").write_text(
        json.dumps({"summary": summary, "history": history}, indent=2), encoding="utf-8"
    )

    if wandb_run is not None:
        wandb_run.summary.update(summary)
        wandb_run.finish()

    print(f"done: best {best_psnr:.3f} dB @ epoch {best_epoch} -> {run_dir}")
    return summary


def add_arguments(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--config", type=Path, default=None, help="YAML config path")
    p.add_argument("--snr-adaptive", action="store_true", default=None)
    p.add_argument("--digital", action="store_true", default=None)
    p.add_argument("-M", "--modulation-order", type=int, default=None)
    p.add_argument("--snr-train-fixed", type=float, default=None)
    p.add_argument("--channel", choices=["awgn", "rayleigh"], default=None)
    p.add_argument("--c-out", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--no-wandb", dest="wandb", action="store_false", default=None)
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    overrides = {
        k: v
        for k, v in vars(args).items()
        if k != "config" and v is not None
    }
    if args.config:
        return Config.from_yaml(args.config, **overrides)
    return Config(**overrides)


def main() -> None:
    parser = add_arguments(argparse.ArgumentParser(description=__doc__))
    cfg = config_from_args(parser.parse_args())
    train(cfg)


if __name__ == "__main__":
    main()
