"""Run generative-JSCC prototype plans: build the data cache, train, evaluate. Resumable.

    python scripts/run_gen.py --stage smoke                      # offline, CPU, ~minutes
    python scripts/run_gen.py --stage week1 --evaluate           # CNN + DiT-S specialists at 0 dB
    python scripts/run_gen.py --stage week2 --evaluate           # 3 specialists + told + blind
    python scripts/run_gen.py --models dit --arms told blind --seeds 0 --steps 40000 --evaluate
    python scripts/run_gen.py --models dit --arms spec0 told --qam 16 --out-dir results/gen/p128-qam --evaluate
    python scripts/run_gen.py --models dit --arms blind --evaluate --sample-steps 1 2 4 --eval-suffix fewstep

Arms: spec<SNR> (a specialist trained at that SNR, e.g. spec0, spec-5), told (one model for
U[min, max] with the SNR in its slot), blind (the same model with a constant in the slot).

Finished runs (history.json present) are skipped; unfinished ones resume from checkpoint.pt.
Runs within one comparison should use the same GPU type, as in the blind study.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semcom.gen.config import GenConfig  # noqa: E402

STAGES = {
    "smoke": [("dit", "spec0"), ("cnn", "spec0"), ("dit", "blind")],
    "week1": [("dit", "spec0"), ("cnn", "spec0")],
    "week2": [("dit", "spec-5"), ("dit", "spec5"), ("dit", "told"), ("dit", "blind")],
    "baselines": [("cnn", "spec-5"), ("cnn", "spec5"), ("cnn", "told"), ("cnn", "blind")],
}

#: Overrides for the smoke stage: synthetic data, stand-in frozen models, tiny everything.
SMOKE = dict(
    dataset="synthetic", fake_frozen=True, out_dir="results/gen/smoke", steps=30, batch_size=8,
    warmup_steps=5, val_every=15, sample_every=30, ckpt_every=15, log_every=10, num_workers=0,
    eval_images=16, eval_batch=8, eval_snrs=[0.0], sample_steps=[5, 2], cnn_hidden=16,
    det_hidden=16, wandb=False,
)


def arm_config(base: GenConfig, model: str, arm: str, seed: int) -> GenConfig:
    if arm.startswith("spec"):
        return base.replace(model=model, snr_mode="fixed", snr_train_fixed=float(arm[4:]), seed=seed)
    if arm in ("told", "blind"):
        return base.replace(model=model, snr_mode="snr" if arm == "told" else "blank", seed=seed)
    raise ValueError(f"unknown arm {arm!r}: spec<SNR>, told or blind")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=Path("configs/gen/p128.yaml"))
    p.add_argument("--stage", choices=sorted(STAGES), default=None)
    p.add_argument("--models", nargs="+", choices=["dit", "cnn"], default=None)
    p.add_argument("--arms", nargs="+", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--steps", type=int, default=None, help="training steps per run")
    p.add_argument("--dit-size", choices=["S", "B", "L"], default=None)
    p.add_argument("--prediction", choices=["eps", "v"], default=None, help="DiT target: noise or velocity")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--qam", type=int, default=None, metavar="M",
                   help="digital: transmit M-QAM symbols (4, 16, 64, ...); default analog")
    p.add_argument("--anneal-period", type=int, default=None, help="digital: quantiser temperature schedule")
    p.add_argument("--max-hours", type=float, default=None, help="per-run wall-clock budget")
    p.add_argument("--eval-images", type=int, default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    p.add_argument("--evaluate", action="store_true", help="evaluate each run after training")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--snr-range", nargs=2, type=float, default=None, metavar=("LO", "HI"),
                   help="training SNR range (dB) of the told / blind arms")
    p.add_argument("--init-from", default=None, help="fine-tune: start from this run folder's weights")
    p.add_argument("--train-parts", choices=["all", "transmitter", "receiver"], default=None)
    p.add_argument("--tag", default=None, help="appended to the run name")
    p.add_argument("--eval-snrs", nargs="+", type=float, default=None)
    p.add_argument("--sample-steps", nargs="+", type=int, default=None)
    p.add_argument("--eval-cfg-scale", type=float, default=None,
                   help="with --evaluate: classifier-free guidance scale for DiT sampling (use with --eval-suffix)")
    p.add_argument("--eval-subset", choices=["first", "spread"], default="first",
                   help="with --evaluate: first n test images, or n evenly spaced over all classes")
    p.add_argument("--cnn-lpips-weight", type=float, default=None,
                   help="CNN baseline: add this weight x LPIPS(VGG) to the pixel MSE loss")
    p.add_argument("--channel", choices=["awgn", "rayleigh"], default=None,
                   help="channel for training and evaluation (rayleigh: equalised slow fading per image)")
    p.add_argument("--c-sem", type=int, default=None,
                   help="DiT: semantic channels per grid cell (bandwidth split; with --c-det, keep c_sem + c_det "
                        "to keep k; name the run with --tag)")
    p.add_argument("--c-det", type=int, default=None, help="DiT: detail channels per grid cell")
    p.add_argument("--encoder-snr", action="store_true",
                   help="DiT: SNR-aware transmitter (FiLM on the SNR in both encoders)")
    p.add_argument("--eval-suffix", default="",
                   help="with --evaluate: write eval-<suffix>.json (at --eval-snrs / --sample-steps), "
                        "keeping a run's main eval.json; for re-evaluating finished runs")
    args = p.parse_args()
    # Checked here as well as in evaluate_run, so a --dry-run (kaggle_jobs push) catches it (g-g1).
    if args.eval_suffix and not all(c.isalnum() or c == "-" for c in args.eval_suffix):
        p.error(f"--eval-suffix must be letters, digits and '-', got {args.eval_suffix!r}")

    overrides = {"steps": args.steps, "dit_size": args.dit_size, "prediction": args.prediction, "batch_size": args.batch_size,
                 "max_hours": args.max_hours, "eval_images": args.eval_images,
                 "out_dir": args.out_dir, "data_root": args.data_root, "wandb_mode": args.wandb_mode,
                 "anneal_period": args.anneal_period, "init_from": args.init_from,
                 "train_parts": args.train_parts, "tag": args.tag, "eval_snrs": args.eval_snrs,
                 "sample_steps": args.sample_steps, "encoder_snr": args.encoder_snr or None,
                 "channel": args.channel, "c_sem": args.c_sem, "c_det": args.c_det}
    if args.snr_range:
        overrides.update(snr_train_min=args.snr_range[0], snr_train_max=args.snr_range[1])
    if args.qam:
        overrides.update(digital=True, modulation_order=args.qam)
    if args.stage == "smoke":
        overrides = {**SMOKE, **{k: v for k, v in overrides.items() if v is not None}}
    base = GenConfig.from_yaml(args.config, **overrides)

    if args.stage:
        pairs = STAGES[args.stage]
    elif args.models and args.arms:
        pairs = [(m, a) for m in args.models for a in args.arms]
    else:
        p.error("give --stage, or both --models and --arms")
    cfgs = [arm_config(base, m, a, s) for s in args.seeds for m, a in pairs]
    if args.cnn_lpips_weight is not None:
        if not any(c.model == "cnn" for c in cfgs):
            p.error("--cnn-lpips-weight needs a CNN run in the plan")
        cfgs = [c.replace(cnn_lpips_weight=args.cnn_lpips_weight) if c.model == "cnn" else c for c in cfgs]

    names = [c.run_name for c in cfgs]
    if len(set(names)) != len(names):
        raise SystemExit("run-name collision in the plan; refusing to run")
    print(f"{len(cfgs)} runs (steps={base.steps}, dataset={base.dataset}, cache={base.cache_dir}):")
    for c in cfgs:
        state = "done " if (c.run_dir / "history.json").exists() else (
            "part " if (c.run_dir / "checkpoint.pt").exists() else "     ")
        print(f"  {state}{c.run_name:34s} k={c.k} -> {c.run_dir}")
    if args.dry_run:
        return

    from semcom.gen.evaluate import evaluate_run
    from semcom.gen.train import train

    for i, cfg in enumerate(cfgs, 1):
        print(f"\n--- [{i}/{len(cfgs)}] {cfg.run_name}", flush=True)
        if not (cfg.run_dir / "history.json").exists():
            if train(cfg) is None:
                print("stopped by max_hours; rerun the same plan to resume")
                continue
        else:
            print("already trained; skipping")
        tail = f"-{args.eval_suffix}" if args.eval_suffix else ""
        if args.evaluate and not (cfg.run_dir / f"eval{tail}.json").exists():
            evaluate_run(cfg.run_dir, external=cfg.dataset != "synthetic", snrs=args.eval_snrs,
                         steps_list=args.sample_steps, suffix=args.eval_suffix, subset=args.eval_subset,
                         cfg_scale=args.eval_cfg_scale)


if __name__ == "__main__":
    main()
