"""Quality and decoding time against the number of DDIM sampling steps, for one run, at each test SNR.

Reads <run>/eval.json and any <run>/eval-*.json scored on the same images (e.g. eval-fewstep.json),
merging their step counts, and writes a figure with one panel per metric (no shared axes) plus a
JSON table of the numbers drawn.

usage: steps.py RUN_DIR OUT.png [EVAL_NAME[,EVAL_NAME...]]
  e.g. steps.py results/gen/p128/dit-S-v_cbr48_blind-5to5_s0 results/gen/analysis/steps.png eval,eval-fewstep"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

run, out = Path(sys.argv[1]), Path(sys.argv[2])
names = sys.argv[3].split(",") if len(sys.argv) > 3 else ["eval", "eval-fewstep"]

pts: dict[float, dict[int, dict]] = {}
device = subsets = None
for name in names:
    e = json.loads((run / f"{name}.json").read_text())
    subsets = subsets or e.get("subset", "first")
    if e.get("subset", "first") != subsets:
        raise SystemExit(f"{name} is on another test subset ({e.get('subset')} vs {subsets})")
    device = device or e.get("device")
    for key, v in e["results"].items():
        snr, steps = key.split("_")
        pts.setdefault(float(snr), {})[int(steps)] = v
pts = {s: dict(sorted(d.items())) for s, d in sorted(pts.items())}

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]  # reference categorical palette, fixed order
METRICS = [("psnr", "PSNR (dB)", "higher is better", 1), ("lpips", "LPIPS", "lower is better", 1),
           ("fid", "FID", "lower is better", 1), ("seconds_per_image", "Time (ms / image)",
                                                   f"{device}", 1000)]
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), constrained_layout=True)
for ax, (m, label, hint, scale) in zip(axes, METRICS):
    for (snr, d), c in zip(pts.items(), COLOURS):
        xs = list(d)
        ys = [d[s][m] * scale for s in xs]
        ax.plot(xs, ys, color=c, lw=2, marker="o", ms=4, label=f"{snr:+g} dB")
        if m != "seconds_per_image":
            ax.annotate(f"{snr:+g} dB", (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points",
                        color=MUTED, fontsize=7, va="center")
    ax.set_xscale("log")
    ax.set_xticks(sorted({s for d in pts.values() for s in d}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.minorticks_off()
    ax.set_xlabel("DDIM sampling steps")
    ax.set_ylabel(label)
    ax.set_title(f"{label.split(' (')[0]} ({hint})", fontsize=9, color=INK, loc="left")
    ax.grid(color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, [f"test SNR {lab}" for lab in labels], loc="outside lower center", ncol=3,
           frameon=False, fontsize=8)
fig.suptitle(f"{run.name}: one trained model, sampling steps chosen at the receiver "
             f"({subsets} 1000 test images)", fontsize=10, color=INK, x=0.01, ha="left")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=160)
table = {f"{snr:g}": {str(s): {m: v[m] for m, *_ in METRICS if m in v} for s, v in d.items()}
         for snr, d in pts.items()}
out.with_suffix(".json").write_text(json.dumps({"run": run.name, "subset": subsets, "device": device,
                                                "results": table}, indent=1))
print("wrote", out, "and", out.with_suffix(".json"))
