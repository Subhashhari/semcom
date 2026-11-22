"""The diffusion JSCC against separate source-channel coding on the same 1000 test images and the
same 4096 16-QAM channel uses: PSNR, LPIPS, DINOv2 similarity and retrieval top-1 against SNR,
for AWGN and block Rayleigh fading.

Series (absolute values from each run's eval-spread*.json):
  - DiT JSCC, blind, trained on 0..20 dB (study A; its Rayleigh twin on the Rayleigh panel)
  - DiT JSCC, blind, trained on -5..5 dB, and the perceptual CNN JSCC of test B (AWGN only)
  - separation, practical: AVIF + LDPC (n = 2048) + QPSK/16/64-QAM, best scheme per SNR by PSNR
  - separation, ideal: AVIF at the 16-QAM capacity of the channel uses (error-free code)
The SD-VAE reconstruction ceiling of the JSCC is drawn on the PSNR row.

usage: separation_curves.py OUT.png"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
G = ROOT / "results/gen"
out = Path(sys.argv[1])

# (label, colour, line style, marker, {channel: [(run dir, eval suffix), ...]}): reference
# categorical palette in fixed slot order; separation dashed, so identity is not colour alone.
SERIES = [
    ("DiT JSCC, trained 0..20 dB", "#2a78d6", "-", "o", {
        "awgn": [(G / "p256-a/dit-S-v_cbr48_qam16_blind0to20_s0", "spread")],
        "rayleigh": [(G / "p256-a/dit-S-v_cbr48_qam16_blind0to20_rayleigh_s0", "spread")]}),
    ("DiT JSCC, trained -5..5 dB", "#eb6834", "-", "s", {
        "awgn": [(G / "p256-conv/dit-S-v_cbr48_qam16_blind-5to5_s0", s) for s in ("spread", "spread-curve")]}),
    ("CNN JSCC (perceptual), trained -5..5 dB", "#1baf7a", "-", "^", {
        "awgn": [(G / "p256-perc/cnn_cbr48_qam16_blind-5to5_lpips0.1_ft_s0", s) for s in ("spread", "spread-curve")]}),
    ("Separation: AVIF + LDPC + QAM", "#eda100", "--", "D", {
        "awgn": [(G / "separation/awgn/ldpc_all_bestpsnr", "spread")],
        "rayleigh": [(G / "separation/rayleigh/ldpc_all_bestpsnr", "spread")]}),
    ("Separation: AVIF at 16-QAM capacity", "#e87ba4", "--", "v", {
        "awgn": [(G / "separation/awgn/ideal_qam16_bestpsnr", "spread")],
        "rayleigh": [(G / "separation/rayleigh/ideal_qam16_bestpsnr", "spread")]}),
]
METRICS = [("psnr", "PSNR (dB)", 1), ("lpips", "LPIPS (lower is better)", 1),
           ("dino_sim", "DINOv2 similarity", 1), ("retrieval_top1", "Retrieval top-1 (%)", 100)]
CHANNELS = [("awgn", "AWGN"), ("rayleigh", "Block Rayleigh fading (no CSI at the transmitter)")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
VAE_CEILING = 24.68  # sanity.json vae_ceiling.psnr of the 256 px runs


def curve(runs: list[tuple[Path, str]]) -> dict[float, dict]:
    pts = {}
    for run, suffix in runs:
        f = run / f"eval-{suffix}.json"
        if not f.exists():
            continue
        for key, v in json.loads(f.read_text())["results"].items():
            snr, _, steps = key.partition("_")
            if steps in ("", "10"):  # DiT keys are SNR_STEPS; 10 sampling steps throughout
                pts[float(snr)] = v
    return dict(sorted(pts.items()))


plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(len(METRICS), len(CHANNELS), figsize=(9.6, 10.5), sharex=True,
                         constrained_layout=True)
table = {}
for c, (ch, ch_title) in enumerate(CHANNELS):
    for r, (m, ylabel, scale) in enumerate(METRICS):
        ax = axes[r][c]
        for label, colour, ls, marker, runs in SERIES:
            pts = curve(runs.get(ch, []))
            if not pts:
                continue
            xs = list(pts)
            ys = [pts[x][m] * scale for x in xs]
            ax.plot(xs, ys, ls=ls, color=colour, lw=2, marker=marker, ms=4.5, label=label, zorder=3)
            table.setdefault(ch, {}).setdefault(label, {})[m] = dict(zip(map(str, xs), ys))
        if m == "psnr":
            ax.axhline(VAE_CEILING, color=MUTED, lw=1, ls=":", zorder=2)
            ax.annotate("SD-VAE ceiling of the JSCC", (0.01, VAE_CEILING), xycoords=("axes fraction", "data"),
                        xytext=(0, 2), textcoords="offset points", fontsize=7, color=MUTED)
        ax.set_ylabel(ylabel)
        ax.grid(color=GRID, lw=0.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        if r == 0:
            ax.set_title(ch_title, fontsize=9, color=INK, loc="left")
        if r == len(METRICS) - 1:
            ax.set_xlabel("Average SNR (dB)")
for r in range(len(METRICS)):  # one y-scale per metric, so the channels compare by eye
    lo = min(axes[r][c].get_ylim()[0] for c in range(len(CHANNELS)))
    hi = max(axes[r][c].get_ylim()[1] for c in range(len(CHANNELS)))
    for c in range(len(CHANNELS)):
        axes[r][c].set_ylim(lo, hi)
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False, fontsize=8)
fig.suptitle("Diffusion JSCC against separate coding, 4096 16-QAM symbols per 256 px image "
             "(1/48 of a channel use per pixel), 1000 test images", fontsize=9, color=INK, x=0.01, ha="left")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=160)
out.with_suffix(".json").write_text(json.dumps(table, indent=1))
print("wrote", out, "and", out.with_suffix(".json"))
