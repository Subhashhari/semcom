"""Train (and optionally evaluate) the blind-JSCC arms.

    # what would run, without running it
    python scripts/run_blind.py --stage calibration --dry-run

    # the two calibration runs (B and C-att, 16-QAM, seed 0), into results/blind/r12-calibration;
    # then read them with scripts/convergence_report.py to set the shared epoch budget
    python scripts/run_blind.py --stage calibration --epochs 300
    python scripts/convergence_report.py results/blind/r12-calibration

    # week 1, once the shared epoch budget is known
    python scripts/run_blind.py --stage week1 --epochs 250 --evaluate

    # split week 1 across two machines BY SEED, so every arm of a seed shares a machine
    python scripts/run_blind.py --stage week1 --epochs 250 --shard 0/2   # machine 1
    python scripts/run_blind.py --stage week1 --epochs 250 --shard 1/2   # machine 2

    # any custom subset
    python scripts/run_blind.py --arms B C-att --mods qam16 --seeds 0 1 2

Completed runs (history.json present) are skipped, and interrupted runs resume from their
checkpoint, so the same command can simply be re-run after a crash.

Sharding is by seed, not by run: the paired comparisons are between arms at the same seed,
and pairing needs identical data order and channel noise, so all arms of one seed must run
on the same platform with the same num_workers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semcom.config import Config  # noqa: E402

#: Arm -> config overrides (proposal section 4.1).
ARMS = {
    "A": dict(snr_adaptive=True),
    "B": dict(decoder_input="snr", encoder_snr=False),
    "C-att": dict(decoder_input="blank", encoder_snr=False),
    "C": dict(decoder_input="none", encoder_snr=False),
    "C+E": dict(decoder_input="energy", encoder_snr=False),
    "C+E-shuf": dict(decoder_input="energy_shuf", encoder_snr=False),
    "D": dict(decoder_input="stats", encoder_snr=False),
    "D-shuf": dict(decoder_input="stats_shuf", encoder_snr=False),
}

MODS = {
    "analog": dict(digital=False),
    "qam16": dict(digital=True, modulation_order=16),
}

#: Stage -> list of (arm, modulation, seeds). Seed counts follow the proposal's seed plan.
STAGES = {
    "calibration": [("B", "qam16", [0]), ("C-att", "qam16", [0])],
    "week1": [
        ("B", "qam16", range(5)),
        ("C-att", "qam16", range(5)),
        ("B", "analog", range(3)),
        ("C-att", "analog", range(3)),
        ("C", "qam16", range(3)),
        ("C", "analog", range(3)),
        ("C+E", "qam16", range(3)),
    ],
    "sijsccq-cifar": [
        ("D", "qam16", range(5)),
        ("D-shuf", "qam16", range(5)),
        ("A", "qam16", [0]),
    ],
    "cpe-shuf": [("C+E-shuf", "qam16", range(3))],
}

#: Endpoint DeepJSCC-Q specialists for the envelope and storage comparison.
SPECIALIST_SNRS = (1.0, 19.0)


def build_configs(base: Config, plan) -> list[Config]:
    cfgs = []
    for arm, mod, seeds in plan:
        for seed in seeds:
            cfgs.append(base.replace(**ARMS[arm], **MODS[mod], seed=int(seed)))
    return cfgs


def specialists(base: Config) -> list[Config]:
    return [
        base.replace(decoder_input="none", encoder_snr=False, snr_train_fixed=s, **MODS["qam16"])
        for s in SPECIALIST_SNRS
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=Path("configs/blind/cifar_r12.yaml"))
    p.add_argument("--stage", choices=sorted(STAGES) + ["specialists"], default=None)
    p.add_argument("--arms", nargs="+", choices=sorted(ARMS), default=None)
    p.add_argument("--mods", nargs="+", choices=sorted(MODS), default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None, help="shared epoch budget")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    p.add_argument("--out-dir", default=None, help="override the config's results folder")
    p.add_argument("--shard", default=None, help="i/n: run only seeds with seed %% n == i")
    p.add_argument("--evaluate", action="store_true", help="run semcom.blind_eval after each run")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    overrides = {
        "epochs": args.epochs,
        "num_workers": args.num_workers,
        "wandb_mode": args.wandb_mode,
        "out_dir": args.out_dir,
    }
    if args.no_wandb:
        overrides["wandb"] = False
    base = Config.from_yaml(args.config, **overrides)
    if args.stage == "calibration":
        # Separate folder: calibration runs share names with week 1's seed-0 runs but are
        # trained to a different (exploratory) epoch budget, so they must not be reused.
        base = base.replace(out_dir=f"{base.out_dir}-calibration")

    if args.stage == "specialists":
        cfgs = specialists(base)
    elif args.stage:
        cfgs = build_configs(base, STAGES[args.stage])
    else:
        if not (args.arms and args.mods and args.seeds is not None):
            p.error("give --stage, or all of --arms, --mods and --seeds")
        cfgs = build_configs(base, [(a, m, args.seeds) for a in args.arms for m in args.mods])

    if args.shard:
        i, n = (int(v) for v in args.shard.split("/"))
        cfgs = [c for c in cfgs if c.seed % n == i]

    names = [c.run_name for c in cfgs]
    if len(set(names)) != len(names):
        raise SystemExit("run-name collision in the plan; refusing to run")

    print(f"{len(cfgs)} runs (epochs={base.epochs}, num_workers={base.num_workers}):")
    for c in cfgs:
        done = (c.run_dir / "history.json").exists()
        print(f"  {'done ' if done else '     '}{c.arm:9s} {'qam16' if c.digital else 'analog':6s} "
              f"seed {c.seed}  -> {c.run_dir}")
    if args.dry_run:
        return

    from semcom.blind_eval import evaluate_blind
    from semcom.train import train

    # Refuse to mix epoch budgets: the cosine schedule depends on `epochs`, and the design
    # compares arms trained for the same shared budget.
    for cfg in cfgs:
        hist = cfg.run_dir / "history.json"
        if hist.exists():
            configured = json.loads(hist.read_text(encoding="utf-8"))["summary"]["epochs_configured"]
            if configured != cfg.epochs:
                raise SystemExit(
                    f"{cfg.run_dir} was trained with epochs={configured}, this plan uses "
                    f"{cfg.epochs}; move it aside or pass --epochs {configured}"
                )

    for i, cfg in enumerate(cfgs, 1):
        print(f"\n--- [{i}/{len(cfgs)}] {cfg.run_name}")
        if not (cfg.run_dir / "history.json").exists():
            train(cfg)
        else:
            print("already trained; skipping")
        if args.evaluate and not (cfg.run_dir / "blind_eval.json").exists():
            evaluate_blind(cfg.run_dir)


if __name__ == "__main__":
    main()
