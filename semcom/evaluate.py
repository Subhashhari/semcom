"""Evaluate a trained run: PSNR against test SNR over the full sweep.

    python -m semcom.evaluate results/r12/adjsccq_r0.0833_m16_snr0-20

Writes `evaluation.json` into the run directory and, unless --no-wandb, logs the curve to
the run's wandb project.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import Config
from .data import cifar10_loaders
from .models import JSCC
from .train import build_model, evaluate, init_wandb, log


def load_run(run_dir: Path, device: torch.device) -> tuple[JSCC, Config]:
    """Rebuild a model from a run directory and load its best checkpoint."""
    cfg = Config.from_yaml(run_dir / "config.yaml")
    model = build_model(cfg).to(device)

    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"no best.pt in {run_dir}; has it finished training?")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


@torch.no_grad()
def verify_transmitted_symbols(model: JSCC, loader, device: torch.device) -> bool:
    """Re-assert the constellation invariant on the trained model, on real test data.

    The unit tests check an untrained quantiser. This checks that training - which moves
    the encoder's output distribution a long way - did not break the property that makes
    the scheme standards-legal. Cheap, and it runs before every reported number.
    """
    x, _ = next(iter(loader))
    x = x.to(device)
    for snr in (0.0, 10.0, 20.0):
        z = model.transmit(x, torch.full((x.shape[0],), snr, device=device))
        if not model.quantiser.is_on_constellation(z):
            return False
    return True


def evaluate_run(run_dir: Path, use_wandb: bool = True, repeats: int | None = None) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_run(run_dir, device)
    _, _, test_loader = cifar10_loaders(
        cfg.data_root, cfg.batch_size, cfg.num_workers, seed=cfg.seed
    )

    on_constellation = None
    if model.quantiser is not None:
        on_constellation = verify_transmitted_symbols(model, test_loader, device)
        if not on_constellation:
            raise RuntimeError(
                f"{run_dir}: trained model transmits off-constellation symbols. "
                "Results are not standards-legal and must not be reported."
            )

    curve = evaluate(
        model,
        test_loader,
        device,
        cfg.eval_snrs,
        repeats=repeats if repeats is not None else cfg.eval_repeats,
    )

    result = {
        "run_name": cfg.run_name,
        "arm": cfg.arm,
        "bandwidth_ratio": cfg.bandwidth_ratio,
        "k": model.k,
        "parameters": model.num_parameters(),
        "modulation_order": cfg.modulation_order if cfg.digital else None,
        "snr_train_fixed": cfg.snr_train_fixed,
        "channel": cfg.channel,
        "on_constellation": on_constellation,
        "psnr_by_snr": curve,
        "mean_psnr": sum(curve.values()) / len(curve),
    }
    (run_dir / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    if use_wandb:
        run = init_wandb(cfg.replace(wandb=True))
        for snr, value in sorted(curve.items()):
            log(run, {"eval/snr_db": snr, "eval/psnr": value})
        if run is not None:
            run.summary.update(
                {"eval/mean_psnr": result["mean_psnr"], "eval/on_constellation": on_constellation}
            )
            run.finish()

    print(f"{cfg.arm:12s}  mean {result['mean_psnr']:6.3f} dB over {len(curve)} SNRs")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dirs", type=Path, nargs="+", help="run directories to evaluate")
    p.add_argument("--repeats", type=int, default=None, help="channel realisations per image")
    p.add_argument("--no-wandb", dest="wandb", action="store_false")
    args = p.parse_args()

    for run_dir in args.run_dirs:
        evaluate_run(run_dir, use_wandb=args.wandb, repeats=args.repeats)


if __name__ == "__main__":
    main()
