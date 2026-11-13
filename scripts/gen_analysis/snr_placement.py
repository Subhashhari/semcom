"""Where the SNR helps: PSNR gain over the blind receiver (one model, no SNR anywhere) of
  - telling the receiver the SNR (told),
  - telling the transmitter the SNR (SNR-aware transmitter, blind receiver),
  - training a separate specialist for the test SNR (the per-SNR model a fixed design needs),
with 95% paired bootstrap intervals, read from paired.py outputs whose reference is the blind run.

usage: snr_placement.py OUT.png TITLE=PAIRED.json [TITLE=PAIRED.json ...] [--steps 10] [--metric psnr]
  e.g. snr_placement.py results/gen/analysis/snr_placement.png \
         "Analog=results/gen/analysis/paired_spread_vs_blind_s0.json" \
         "16-QAM=results/gen/analysis/paired_spread_qam16_vs_blind_s0.json\""""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

args = sys.argv[1:]


def opt(name: str, default: str) -> str:
    if name in args:
        i = args.index(name)
        v = args[i + 1]
        del args[i:i + 2]
        return v
    return default


steps, metric = opt("--steps", "10"), opt("--metric", "psnr")
out, panels = Path(args[0]), [a.split("=", 1) for a in args[1:]]
SNRS = [-5, 0, 5]
ARMS = [  # (label, run-name test, colour): reference categorical palette, fixed order
    ("Receiver told the SNR", lambda r, s: "_told-" in r, "#2a78d6"),
    ("SNR-aware transmitter, blind receiver", lambda r, s: "_txsnr_" in r, "#eb6834"),
    ("Specialist trained at the test SNR", lambda r, s: f"_spec{s}_" in r, "#1baf7a"),
]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
unit = {"psnr": "dB", "lpips": "", "top1": "pts"}.get(metric, "")
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.6), constrained_layout=True,
                         squeeze=False)
table = {}
w = 0.26
for ax, (title, path) in zip(axes[0], panels):
    d = json.loads(Path(path).read_text())
    runs = {r: v for r, v in d["runs"].items() if not r.startswith("cnn")}
    table[title] = {"reference": d["reference"], "file": path, "rows": {}}
    for j, (label, pick, colour) in enumerate(ARMS):
        xs, ys, lo, hi = [], [], [], []
        for i, s in enumerate(SNRS):
            hits = [r for r in runs if pick(r, s)]
            key = next((k for k in (f"{metric}_{s}_{steps}", f"{metric}_{s}") if hits and k in runs[hits[0]]), None)
            if len(hits) != 1 or key is None:
                continue
            v = runs[hits[0]][key]
            xs.append(i + (j - 1) * w)
            ys.append(v["diff"])
            lo.append(v["diff"] - v["lo"])
            hi.append(v["hi"] - v["diff"])
            table[title]["rows"].setdefault(label, {})[f"{s:g}"] = {**v, "run": hits[0]}
        ax.bar(xs, ys, width=w - 0.03, color=colour, label=label, zorder=2)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=1, capsize=2, zorder=3)
        for x, y, l, h in zip(xs, ys, lo, hi):
            if abs(y) >= 0.1:  # selective labels: only the bars that matter, past the interval's end
                ax.annotate(f"{y:+.2f}", (x, y + h if y > 0 else y - l), xytext=(0, 2 if y > 0 else -2),
                            textcoords="offset points",
                            ha="center", va="bottom" if y > 0 else "top", fontsize=7, color=INK)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.margins(y=0.12)
    ax.set_xticks(range(len(SNRS)), [f"{s:+g} dB" for s in SNRS])
    ax.set_xlabel("Test SNR")
    ax.set_ylabel(f"{metric.upper()} minus blind receiver{f' ({unit})' if unit else ''}")
    ax.set_title(f"{title} (vs {d['reference']})", fontsize=8, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
ymin = min(a.get_ylim()[0] for a in axes[0])
ymax = max(a.get_ylim()[1] for a in axes[0])
for a in axes[0]:
    a.set_ylim(ymin, ymax)  # same scale across panels, so analog and digital compare by eye
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=8)
fig.suptitle("Where knowing the SNR helps: gain over one blind model trained on -5..5 dB "
             "(95% paired bootstrap intervals)", fontsize=10, color=INK, x=0.01, ha="left")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=160)
out.with_suffix(".json").write_text(json.dumps(table, indent=1))
print("wrote", out, "and", out.with_suffix(".json"))
