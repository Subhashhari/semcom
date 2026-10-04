"""Rebuild results/<run>/history.json from the local wandb run directories.

A run's records can be reconstructed without the run directory, because wandb keeps its
own copy of everything that was printed and summarised:

  files/output.log         every per-epoch line, so the full history
  files/wandb-summary.json the summary dict, which train() writes wholesale via
                           wandb_run.summary.update(summary)
  files/config.yaml        the run config, in wandb's {desc, value} wrapper

What cannot be recovered is best.pt / checkpoint.pt - the weights were never sent to
wandb, so a recovered run is readable but not re-evaluatable or resumable.

Usage:
    python scripts/recover_from_wandb.py                      # dry run, prints what it found
    python scripts/recover_from_wandb.py --write               # writes into results/recovered/
    python scripts/recover_from_wandb.py --write --out results/adjscc-q/r12
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# "ADJSCC-Q  R=0.0833  k=256  params=10,567,527  device=cuda  M=16"
HEADER = re.compile(
    r"^(?P<arm>[A-Za-z0-9-]+)\s+R=(?P<ratio>[\d.]+)\s+k=(?P<k>\d+)\s+"
    r"params=(?P<params>[\d,]+)\s+device=(?P<device>\S+)(?:\s+M=(?P<m>\d+))?"
)

# "epoch  149  train 27.229 dB  val 26.761 dB  *  [28.5s train + 5.0s val, 4443s total]"
# "epoch    1  train 19.401 dB  [30.6s]"
EPOCH = re.compile(
    r"^epoch\s+(?P<epoch>\d+)\s+train\s+(?P<train>[-\d.]+)\s*dB"
    r"(?:\s+val\s+(?P<val>[-\d.]+)\s*dB)?"
)
TIMES = re.compile(r"\[(?P<train_s>[\d.]+)s(?:\s+train\s+\+\s+(?P<val_s>[\d.]+)s\s+val)?")

# "done: best 26.761 dB @ epoch 149 -> results\r12\adjscc_..."
DONE = re.compile(r"^done:\s+best\s+(?P<psnr>[-\d.]+)\s+dB\s+@\s+epoch\s+(?P<epoch>\d+)")


def parse_log(text: str) -> dict:
    """Pull the header, per-epoch history and final line out of a captured stdout log."""
    header, history, done = None, [], None
    for line in text.splitlines():
        line = line.rstrip()
        if header is None:
            m = HEADER.match(line)
            if m:
                header = m.groupdict()
                continue
        m = EPOCH.match(line)
        if m:
            entry = {"epoch": int(m["epoch"]), "train_psnr": float(m["train"])}
            if m["val"] is not None:
                entry["val_psnr"] = float(m["val"])
            t = TIMES.search(line)
            if t:
                entry["train_seconds"] = float(t["train_s"])
                if t["val_s"]:
                    entry["val_seconds"] = float(t["val_s"])
            history.append(entry)
            continue
        m = DONE.match(line)
        if m:
            done = {"best_psnr": float(m["psnr"]), "best_epoch": int(m["epoch"])}
    return {"header": header, "history": history, "done": done}


def wandb_config_values(path: Path) -> dict:
    """Flatten wandb's config.yaml ({key: {desc, value}}) back to {key: value}.

    Parsed by hand rather than with pyyaml: the file is machine-written with a fixed
    two-space shape, and the only thing wanted is the scalar under each "value:".
    """
    if not path.exists():
        return {}
    out, key = {}, None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith((" ", "\t")) and line.rstrip().endswith(":"):
            key = line.rstrip()[:-1]
        elif key and line.strip().startswith("value:"):
            raw = line.split("value:", 1)[1].strip()
            if raw in ("null", ""):
                out[key] = None
            elif raw in ("true", "false"):
                out[key] = raw == "true"
            else:
                try:
                    out[key] = int(raw) if re.fullmatch(r"-?\d+", raw) else float(raw)
                except ValueError:
                    out[key] = raw.strip("'\"")
            key = None
    return out


def recover_one(run_dir: Path) -> dict | None:
    files = run_dir / "files"
    log = files / "output.log"
    if not log.exists():
        return None

    parsed = parse_log(log.read_text(encoding="utf-8", errors="replace"))
    if not parsed["history"]:
        return None

    summary = {}
    sj = files / "wandb-summary.json"
    if sj.exists():
        try:
            raw = json.loads(sj.read_text(encoding="utf-8"))
            # Drop wandb's own bookkeeping keys; keep what train() put there.
            summary = {k: v for k, v in raw.items() if not k.startswith("_")}
        except json.JSONDecodeError:
            pass

    cfg = wandb_config_values(files / "config.yaml")
    h = parsed["header"] or {}

    # Prefer the values train() recorded; fall back to the printed header, then the config.
    summary.setdefault("arm", h.get("arm"))
    summary.setdefault("run_name", cfg.get("run_name") or run_dir.name)
    if "snr_train_fixed" not in summary:
        summary["snr_train_fixed"] = cfg.get("snr_train_fixed")
    if parsed["done"]:
        summary.setdefault("best_val_psnr", parsed["done"]["best_psnr"])
        summary.setdefault("best_epoch", parsed["done"]["best_epoch"])
    summary.setdefault("epochs_trained", parsed["history"][-1]["epoch"] + 1)
    if h.get("params"):
        summary.setdefault("parameters", int(h["params"].replace(",", "")))
    if h.get("k"):
        summary.setdefault("k", int(h["k"]))

    # Honest provenance: these records were reconstructed, and the weights are gone.
    summary["recovered_from"] = str(run_dir)
    summary["recovered_note"] = (
        "Rebuilt from the wandb run log after the run directory was lost. Metrics are the "
        "values printed during training; model weights were not recoverable."
    )
    return {"summary": summary, "history": parsed["history"], "config": cfg}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wandb-dir", type=Path, default=Path("wandb"))
    p.add_argument("--out", type=Path, default=Path("results/recovered"))
    p.add_argument("--write", action="store_true", help="actually write; default is a dry run")
    args = p.parse_args()

    if not args.wandb_dir.exists():
        print(f"no wandb directory at {args.wandb_dir}")
        return

    found = 0
    for run_dir in sorted(args.wandb_dir.glob("run-*")):
        rec = recover_one(run_dir)
        if rec is None:
            continue
        found += 1
        s, hist = rec["summary"], rec["history"]
        evals = [e for e in hist if "val_psnr" in e]
        snr = s.get("snr_train_fixed")
        print(
            f"{s.get('arm', '?'):<12} snr={'adaptive' if snr is None else f'{snr:g} dB':>9}  "
            f"epochs={s.get('epochs_trained', 0):>4}  evals={len(evals):>3}  "
            f"best={s.get('best_val_psnr', float('nan')):.3f} dB  <- {run_dir.name}"
        )
        if args.write:
            d = args.out / s["run_name"]
            d.mkdir(parents=True, exist_ok=True)
            (d / "history.json").write_text(
                json.dumps({"summary": s, "history": hist}, indent=2), encoding="utf-8"
            )
            if rec["config"]:
                (d / "wandb_config.json").write_text(
                    json.dumps(rec["config"], indent=2), encoding="utf-8"
                )

    if not found:
        print("found no usable wandb run logs")
    elif args.write:
        print(f"\nwrote {found} run records to {args.out}")
    else:
        print(f"\n{found} runs recoverable. Re-run with --write to save them.")


if __name__ == "__main__":
    main()
