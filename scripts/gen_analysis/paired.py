"""Paired per-image comparison of evaluated runs against a reference run on the same test images
(same channel and sampling seeds when evaluated on the same device type): mean difference
(run - reference) with a 95% bootstrap interval, for PSNR, LPIPS and retrieval top-1/top-5.

usage: paired.py EVAL_NAME REF_RUN RUN [RUN ...] [--json OUT.json]
  EVAL_NAME: the per-image file suffix, e.g. "spread" for eval_images-spread.npz, "" for eval_images.npz
  RUN may be RUN_DIR:EVAL_NAME to read another eval of that run (e.g. a guidance re-evaluation)."""
import json
import sys
from pathlib import Path

import numpy as np

args = sys.argv[1:]
out = None
if "--json" in args:
    i = args.index("--json")
    out = Path(args[i + 1])
    del args[i:i + 2]
name, ref, runs = args[0], args[1], args[2:]
rng = np.random.default_rng(0)
B = 2000


def load(spec: str) -> tuple[str, dict]:
    run, _, ev = spec.partition(":")
    ev = ev or name
    f = Path(run) / (f"eval_images-{ev}.npz" if ev else "eval_images.npz")
    d = np.load(f)
    per = {k: d[k].astype(np.float64) for k in d.files}
    for k in [k for k in per if k.startswith("retrieval_rank_")]:
        tail = k[len("retrieval_rank_"):]
        per[f"top1_{tail}"] = (per[k] < 1).astype(np.float64)
        per[f"top5_{tail}"] = (per[k] < 5).astype(np.float64)
    return f"{Path(run).name}{':' + ev if ev != name else ''}", per


def match(k: str, other: dict) -> str | None:
    """The key of `other` for the same metric and SNR: DiT keys are METRIC_SNR_STEPS, CNN keys
    METRIC_SNR, so a DiT run compares with a CNN at every sampling-step count."""
    if k in other:
        return k
    m, snr = k.split("_")[:2]
    if len(k.split("_")) == 3:  # DiT key against a CNN reference
        return f"{m}_{snr}" if f"{m}_{snr}" in other else None
    hits = [o for o in other if o.startswith(f"{m}_{snr}_")]  # CNN key against a DiT reference
    return hits[0] if len(hits) == 1 else None


def ci(diff: np.ndarray) -> tuple[float, float, float]:
    idx = rng.integers(0, len(diff), (B, len(diff)))
    boots = diff[idx].mean(1)
    return float(diff.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


ref_name, a = load(ref)
res = {"reference": ref_name, "eval": name, "runs": {}}
for spec in runs:
    run_name, b = load(spec)
    rows = {}
    print(f"\n{run_name}  minus  {ref_name}")
    for m in ["psnr", "lpips", "top1", "top5"]:
        for k in sorted((k for k in b if k.startswith(m + "_") and match(k, a)),
                        key=lambda k: float(k.split("_")[1])):
            mean, lo, hi = ci(b[k] - a[match(k, a)])
            scale = 100 if m.startswith("top") else 1
            rows[k] = {"diff": mean * scale, "lo": lo * scale, "hi": hi * scale, "n": len(b[k])}
            sig = "*" if lo > 0 or hi < 0 else " "
            print(f"  {k:<16} {mean * scale:+8.3f}  [{lo * scale:+.3f}, {hi * scale:+.3f}] {sig}")
    res["runs"][run_name] = rows
if out:
    out.write_text(json.dumps(res, indent=1))
