"""Quality against the actual channel SNR: separate designs (AVIF + a finite-blocklength channel
code, designed for SNR d) against generative JSCC and the CNN, on the class-spread test images.

A separate design fixed at d sends its d-budget payload: at an actual SNR >= d it decodes (and
gains nothing from the extra SNR); below d it fails and the receiver shows the mean colour (the
failure floor). JSCC curves come from the runs' eval files, at whatever SNRs they were scored.

usage: cliff.py OUT.png SEPARATION_CURVE.json [STEPS] [EVAL_NAME[,EVAL_NAME...]] [--digital]
  e.g. cliff.py results/gen/analysis/cliff.png results/gen/analysis/separation_curve_fbl.json 10 eval-spread
Several eval names are merged per run (e.g. eval-spread,eval-spread-curve).
--digital draws the 16-QAM runs (every symbol a constellation point) instead of the analog ones."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path(__file__).resolve().parents[2]  # repo root
digital = "--digital" in sys.argv
if digital:
    sys.argv.remove("--digital")
out, curve_path = Path(sys.argv[1]), Path(sys.argv[2])
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 10
names = sys.argv[4].split(",") if len(sys.argv) > 4 else ["eval-spread"]
curve = json.loads(curve_path.read_text())
fail = json.loads((R / "results/gen/analysis/separation_spread.json").read_text())["failure"]
sep = {float(k.split("_")[1]): v for k, v in curve.items() if k.startswith("avif_")}
DESIGNS = [-5.0, 0.0, 5.0]

# Reference categorical palette (dataviz skill), slots in fixed order; ink for text.
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
RUNS = [  # (run dir, label, colour)
    ("results/gen/p128/dit-S-v_cbr48_blind-5to5_s0", "Generative JSCC, blind receiver", "#2a78d6"),
    ("results/gen/p128/dit-S-v_cbr48_blind-5to5_txsnr_s0", "Generative JSCC, SNR-aware transmitter", "#eb6834"),
    ("results/gen/p128/cnn_cbr48_spec0_s0", "CNN JSCC (MSE), trained at 0 dB", "#1baf7a"),
]
if digital:
    RUNS = [
        ("results/gen/p128-qam-es/dit-S-v_cbr48_qam16_blind-5to5_s0", "Generative JSCC 16-QAM, blind receiver", "#2a78d6"),
        ("results/gen/p128-qam-es/dit-S-v_cbr48_qam16_blind-5to5_txsnr_s0",
         "Generative JSCC 16-QAM, SNR-aware transmitter", "#eb6834"),
        ("results/gen/p128-qam-es/cnn_cbr48_qam16_spec0_s0", "CNN JSCC 16-QAM (MSE), trained at 0 dB", "#1baf7a"),
    ]
SEP_COLOUR = "#6250d6"
METRICS = [("psnr", "PSNR (dB)", "higher is better"), ("lpips", "LPIPS", "lower is better"),
           ("retrieval_top1", "Retrieval top-1 (%)", "higher is better")]


def run_points(run_dir: Path) -> dict[float, dict]:
    pts = {}
    for name in names:
        f = run_dir / f"{name}.json"
        if not f.exists():
            continue
        for key, v in json.loads(f.read_text())["results"].items():
            snr, _, st = key.partition("_")  # DiT: "SNR_STEPS"; CNN: "SNR"
            if not st or int(st) == steps:
                pts[float(snr)] = v
    return dict(sorted(pts.items()))


def val(m: str, v: dict) -> float:
    return 100 * v[m] if m.startswith("retrieval") else v[m]


plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
xs = sorted(sep)
lo, hi = min(xs), max(xs)
fine = [lo + i * 0.05 for i in range(int((hi - lo) / 0.05) + 1)]
for ax, (m, ylabel, hint) in zip(axes, METRICS):
    # separation: one step curve per design SNR
    for j, d in enumerate(DESIGNS):
        y = [val(m, sep[d]) if s >= d else val(m, fail) for s in fine]
        ax.plot(fine, y, color=SEP_COLOUR, lw=1.5, ls=["-", "--", ":"][j],
                label=f"Separate (AVIF + code), designed at {d:g} dB")
    ax.plot(xs, [val(m, sep[s]) for s in xs], color=MUTED, lw=1, ls=(0, (4, 2)), marker=".", ms=4,
            label="Separate, re-designed for every SNR (needs the SNR at the sender)")
    for run, label, colour in RUNS:
        pts = run_points(R / run)
        if pts:
            ax.plot(list(pts), [val(m, v) for v in pts.values()], color=colour, lw=2,
                    marker="o", ms=4, label=label)
    ax.axhline(val(m, fail), color=MUTED, lw=0.8, ls=(0, (1, 2)))
    ax.text(lo, val(m, fail), "decoding failure", color=MUTED, fontsize=7,
            va="top" if m == "lpips" else "bottom", ha="left")
    ax.set_xlabel("Actual channel SNR (dB)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel.split(' (')[0]} ({hint})", fontsize=9, color=INK, loc="left")
    ax.grid(color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=8)
fig.suptitle(f"{'Digital (16-QAM) JSCC: q' if digital else 'Q'}uality vs actual SNR, 1000 test images from all 10 classes, k = 1024 symbols "
             f"(DiT: {steps} sampling steps)", fontsize=10, color=INK, x=0.01, ha="left")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=160)
print("wrote", out)
