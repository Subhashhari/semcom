"""Watch a running Kaggle job: record every metric line, apply fixed alert rules, exit on trouble.

    python scripts/watch_job.py g-w1                  # poll every 10 minutes
    python scripts/watch_job.py g-w1 --once           # one poll, print the summary, exit

Each poll pulls the job log (scripts/kaggle_jobs.py logs works while a job runs), keeps the
machine-readable lines the training and evaluation code print

    SANITY <check> run=... key=value ...     pre-flight checks (semcom.gen.checks)
    STEP run=... step= loss= gnorm= it_s= skipped=
    VAL  run=... step= val_loss= uncond= cond_gain= psnr@<snr>= dino@<snr>=
    EVAL run=... snr= steps= psnr= lpips= dino_sim= fid=

appends new ones to .kaggle_jobs/<job>/metrics.jsonl, and checks the rules below. The watcher
exits with a non-zero code on a new alert (so whoever launched it in the background is woken),
0 when the job completes, 1 when it fails. Alerts already raised are remembered in
.kaggle_jobs/<job>/watch_state.json and not raised twice.

Rules (thresholds fixed here, before any run):
    error       a traceback, SanityError, CUDA out-of-memory, or a non-finite logged value
    spike       training loss more than 1.5x the minimum of the previous 10 logged losses
    val_rising  validation loss up at three consecutive validations (overfitting or divergence)
    no_cond     DiT cond_gain <= 0.001 at step >= 10,000: the receiver ignores the channel
    skipped     the AMP scaler skipped more than 1% of steps
    slow        throughput below half its median for the run
    stall       no new STEP line for --stall-minutes while the job is running

Reconstruction-quality rules, added after g-w1 (whose EMA weights sampled meaningless images
while every loss looked healthy). They read the fixed validation batch's real reconstructions:
    quality_drop  PSNR (by > 0.5 dB) or DINOv2 similarity (by > 0.02) fell at two consecutive
                  validations, at step >= 4,000
    unrelated     DiT only: DINOv2 similarity below 0.1 at step >= 10,000, outputs unrelated to the
                  inputs (the VAE alone scores 0.77, an unrelated real photo 0.34, flat grey 0.0)
    zscale        sampled latents' spread outside 0.5-2x the true latents' at step >= 4,000
    ema_gap       the raw weights reconstruct > 2 dB better than the EMA at step >= 10,000
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINE = re.compile(r"^(SANITY \w+|STEP|VAL|EVAL) (.*)$")
ERRORS = ("Traceback (most recent call last)", "SanityError", "CUDA out of memory", "RuntimeError:")


def parse_line(line: str) -> dict | None:
    m = LINE.match(line.strip())
    if not m:
        return None
    rec = {"kind": m.group(1)}
    for field in m.group(2).split():
        if "=" not in field:
            continue
        k, v = field.split("=", 1)
        try:
            rec[k] = float(v)
        except ValueError:
            rec[k] = v
    return rec


def check(records: list[dict], log_text: str, stall: bool = False) -> list[tuple[str, str]]:
    """Alerts as (key, message); the key makes an alert raised once per run and step."""
    alerts = []
    for marker in ERRORS:
        if marker in log_text:
            alerts.append((f"error:{marker}", f"job log contains {marker!r}"))
    for r in records:
        bad = [k for k, v in r.items() if isinstance(v, float) and not math.isfinite(v)]
        if bad:
            alerts.append((f"nonfinite:{r.get('run')}:{r.get('step')}", f"{r['kind']} {r.get('run')} "
                           f"step {r.get('step')}: non-finite {bad}"))
    runs = {r.get("run") for r in records if r["kind"] in ("STEP", "VAL")}
    for run in runs:
        steps = [r for r in records if r["kind"] == "STEP" and r.get("run") == run]
        vals = [r for r in records if r["kind"] == "VAL" and r.get("run") == run]
        losses = [r["loss"] for r in steps if math.isfinite(r.get("loss", math.nan))]
        if len(losses) > 10 and steps[-1]["step"] >= 2000 and losses[-1] > 1.5 * min(losses[-11:-1]):
            alerts.append((f"spike:{run}:{steps[-1]['step']}",
                           f"{run} step {steps[-1]['step']:.0f}: loss {losses[-1]:.4f} vs recent min "
                           f"{min(losses[-11:-1]):.4f}"))
        vl = [r["val_loss"] for r in vals]
        if len(vl) >= 4 and vl[-1] > vl[-2] > vl[-3] > vl[-4]:
            alerts.append((f"val_rising:{run}:{vals[-1]['step']}",
                           f"{run}: validation loss rose three times in a row: {vl[-4:]}"))
        if vals and "cond_gain" in vals[-1] and vals[-1]["step"] >= 10_000 and vals[-1]["cond_gain"] <= 1e-3:
            alerts.append((f"no_cond:{run}:{vals[-1]['step']}",
                           f"{run} step {vals[-1]['step']:.0f}: cond_gain {vals[-1]['cond_gain']:.5f}, "
                           "the receiver is not using the channel"))
        if steps and steps[-1]["step"] > 0 and steps[-1].get("skipped", 0) > 0.01 * steps[-1]["step"]:
            alerts.append((f"skipped:{run}", f"{run}: {steps[-1]['skipped']:.0f} AMP-skipped steps "
                           f"of {steps[-1]['step']:.0f}"))
        rates = [r["it_s"] for r in steps if "it_s" in r]
        if len(rates) >= 10 and rates[-1] < 0.5 * statistics.median(rates):
            alerts.append((f"slow:{run}:{steps[-1]['step']}",
                           f"{run}: {rates[-1]:.2f} it/s, median {statistics.median(rates):.2f}"))
        alerts += quality_alerts(run, vals)
    if stall:
        alerts.append((f"stall:{len(records)}", "no new STEP line within the stall window" + (
            "; no metric line read at all, so check the log reader" if not records else "")))
    return alerts


QUALITY_TOL = {"psnr": 0.5, "dino": 0.02}


def quality_alerts(run, vals: list[dict]) -> list[tuple[str, str]]:
    if not vals:
        return []
    out, last, step = [], vals[-1], vals[-1]["step"]
    for key in [k for k in last if re.match(r"^(psnr|dino)@", k)]:
        seq = [v[key] for v in vals if key in v]
        tol = QUALITY_TOL[key.split("@")[0]]
        if step >= 4000 and len(seq) >= 3 and seq[-1] < seq[-2] - tol and seq[-2] < seq[-3] - tol:
            out.append((f"quality_drop:{run}:{key}:{step}",
                        f"{run} step {step:.0f}: {key} fell twice in a row: {[round(x, 3) for x in seq[-3:]]}"))
        # DiT only: the CNN's blurry, noisy outputs score low DINOv2 similarity while clearly
        # matching their inputs (g-w1 CNN at 18-19 dB: 0.05-0.10).
        if key.startswith("dino@") and str(run).startswith("dit") and step >= 10_000 and last[key] < 0.1:
            out.append((f"unrelated:{run}:{key}:{step}",
                        f"{run} step {step:.0f}: {key}={last[key]:.3f}, outputs unrelated to the inputs"))
        raw = last.get("raw_" + key)
        if key.startswith("psnr@") and raw is not None and step >= 10_000 and raw - last[key] > 2:
            out.append((f"ema_gap:{run}:{key}:{step}",
                        f"{run} step {step:.0f}: raw {key}={raw:.2f} vs EMA {last[key]:.2f}"))
    for key in [k for k in last if re.match(r"^(raw_)?zstd@", k)]:
        if step >= 4000 and not 0.5 <= last[key] <= 2.0:
            out.append((f"zscale:{run}:{key}:{step}",
                        f"{run} step {step:.0f}: {key}={last[key]:.2f}, sampled latents have the wrong scale"))
    return out


def summary(records: list[dict]) -> str:
    out = []
    for kind in ("SANITY", "STEP", "VAL", "EVAL"):
        latest = {}
        for r in records:
            if r["kind"].startswith(kind):
                latest[(r["kind"], r.get("run"))] = r
        for (k, run), r in latest.items():
            fields = " ".join(f"{a}={b:.4g}" if isinstance(b, float) else f"{a}={b}"
                              for a, b in r.items() if a not in ("kind", "run"))
            out.append(f"  {k:16s} {run}: {fields}")
    return "\n".join(out)


def kaggle_jobs(*args: str, timeout: int = 180) -> str:
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "kaggle_jobs.py"), *args],
                          cwd=ROOT, text=True, capture_output=True, encoding="utf-8",
                          errors="replace", timeout=timeout)
    return proc.stdout


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("job")
    p.add_argument("--interval", type=int, default=600, help="seconds between polls")
    p.add_argument("--stall-minutes", type=int, default=60)
    p.add_argument("--stream-seconds", type=int, default=45)
    p.add_argument("--once", action="store_true")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    folder = ROOT / ".kaggle_jobs" / args.job
    store, state_path = folder / "metrics.jsonl", folder / "watch_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"alerted": [], "seen": []}
    seen = set(state["seen"])
    records = [json.loads(line) for line in store.read_text(encoding="utf-8").splitlines()] if store.exists() else []
    last_new = time.time()

    while True:
        status = kaggle_jobs("status", args.job).strip()
        log_text = kaggle_jobs("logs", args.job, "--tail", "1000000", "--seconds", str(args.stream_seconds),
                               timeout=args.stream_seconds + 180)
        new = 0
        with store.open("a", encoding="utf-8") as f:
            for line in log_text.splitlines():
                rec = parse_line(line)
                if rec is None or line.strip() in seen:
                    continue
                seen.add(line.strip())
                records.append(rec)
                f.write(json.dumps(rec) + "\n")
                new += 1 if rec["kind"] == "STEP" else 0
        if new:
            last_new = time.time()
        running = "running" in status.lower()
        stalled = running and time.time() - last_new > args.stall_minutes * 60
        fresh = [(k, m) for k, m in check(records, log_text, stalled) if k not in state["alerted"]]
        state["alerted"] += [k for k, _ in fresh]
        state["seen"] = sorted(seen)
        state_path.write_text(json.dumps(state), encoding="utf-8")

        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {status}  ({len(records)} records, {new} new STEP)")
        print(summary(records), flush=True)
        if fresh:
            print("ALERTS:\n" + "\n".join(f"  - {m}" for _, m in fresh), flush=True)
            sys.exit(3)
        if "complete" in status.lower():
            print("job complete", flush=True)
            sys.exit(0)
        if any(w in status.lower() for w in ("error", "cancel")):
            print("job failed", flush=True)
            sys.exit(1)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
