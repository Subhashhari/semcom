"""Figures from semantic.py: what each system reconstructs, and whether it keeps the object class.

  montage_<snr>.png   one test image per class (columns), one system per row (original first);
                      each reconstruction is captioned with the class the k-NN classifier sees in
                      it, ticked when it is the true class
  class_vs_snr.png    class preservation against the channel SNR, one line per system
  per_class.png       class preservation per class (rows) and system (columns) at one SNR

usage: montage.py SEMANTIC_DIR OUT_DIR SNR[,SNR...] [PICK]
  PICK: which image of each class to show (0-based among that class's images; default 3)"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

src, out = Path(sys.argv[1]), Path(sys.argv[2])
snrs = [float(v) for v in sys.argv[3].split(",")]
pick = int(sys.argv[4]) if len(sys.argv) > 4 else 3
out.mkdir(parents=True, exist_ok=True)
summary = json.loads((src / "summary.json").read_text())
classes = summary["_original"]["classes"]
NAMES = {  # wnid -> readable label
    "n01440764": "tench", "n02102040": "springer", "n02979186": "cassette player", "n03000684": "chain saw",
    "n03028079": "church", "n03394916": "French horn", "n03417042": "garbage truck", "n03425413": "gas pump",
    "n03445777": "golf ball", "n03888257": "parachute"}
label = [NAMES.get(c, c) for c in classes]
SYSTEMS = [  # (key in summary, row label, colour for line plots)
    ("sep", "Separate (AVIF + code),\nre-designed per SNR", "#BDBAB2"),
    ("sep0", "Separate, designed\nfor 0 dB", "#7A776E"),
    ("cnn_cbr48_spec0_s0", "CNN JSCC (MSE),\ntrained at 0 dB", "#3D3B36"),
    ("cnn_cbr48_spec0_lpips0.1_ft_s0", "CNN JSCC (MSE + LPIPS),\ntrained at 0 dB", "#2a78d6"),
    ("dit-S-v_cbr48_blind-5to5_s0", "Generative JSCC,\nblind receiver", "#161613"),
    ("dit-S-v_cbr48_blind-5to5_s0@g1.5", "Generative JSCC,\n+ guidance 1.5", "#eb6834"),
    ("dit-S-v_cbr48_blind-5to5_txsnr_s0", "Generative JSCC,\nSNR-aware transmitter", "#C8501E"),
    ("dit-S-v_cbr48_qam16_blind-5to5_s0", "Generative JSCC,\n16-QAM", "#1baf7a"),
]
SYSTEMS = [s for s in SYSTEMS if s[0] in summary]
BG, INK, MUTED, GOOD, BAD = "#F9F8F5", "#161613", "#7A776E", "#1b7f4d", "#b3261e"
plt.rcParams.update({"font.family": "Calibri", "font.size": 10, "figure.facecolor": BG, "savefig.facecolor": BG,
                     "axes.facecolor": BG, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})


def load(key: str, snr: float):
    f = src / f"{key}__{snr:g}.npz"
    return np.load(f) if f.exists() else None


# ---- montages ----
ref = next(load(k, snrs[0]) for k, *_ in SYSTEMS if load(k, snrs[0]) is not None)
labels = ref["label"]
cols = [np.flatnonzero(labels == c)[pick] for c in range(len(classes))]
orig = np.load(src / "originals.npz")["images"] if (src / "originals.npz").exists() else None
for snr in snrs:
    rows = [(k, name) for k, name, _ in SYSTEMS if load(k, snr) is not None]
    fig, axes = plt.subplots(len(rows) + 1, len(cols), figsize=(1.35 * len(cols) + 1.9, 1.5 * (len(rows) + 1)),
                             gridspec_kw=dict(wspace=0.04, hspace=0.32))
    for j, i in enumerate(cols):
        ax = axes[0, j]
        if orig is not None:
            ax.imshow(orig[i].transpose(1, 2, 0))
        ax.set_title(label[labels[i]], fontsize=10, color=INK)
        for r, (k, name) in enumerate(rows, 1):
            d = load(k, snr)
            a = axes[r, j]
            a.imshow(d["recon"][i].transpose(1, 2, 0))
            ok = d["pred"][i] == labels[i]
            a.set_title(("✓ " if ok else "✗ ") + label[d["pred"][i]], fontsize=8.5, color=GOOD if ok else BAD, pad=2)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0, 0].set_ylabel("Original", rotation=0, ha="right", va="center", fontsize=10, color=INK, labelpad=8)
    for r, (k, name) in enumerate(rows, 1):
        acc = summary[k][f"{snr:g}"]["accuracy"]
        axes[r, 0].set_ylabel(f"{name}\nclass kept: {acc:.0%}", rotation=0, ha="right", va="center",
                              fontsize=9, color=INK, labelpad=8)
    fig.suptitle(f"Channel SNR {snr:+g} dB".replace("-", "−") + ", k = 1024 symbols (1/48 per pixel value). "
                 "Caption: the class a DINOv2 k-NN classifier sees in the reconstruction",
                 fontsize=11, color=INK, x=0.01, ha="left", y=0.995)
    fig.savefig(out / f"montage_{snr:g}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / f"montage_{snr:g}.png")

# ---- class preservation against SNR ----
fig, ax = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
for k, name, col in SYSTEMS:
    pts = sorted((float(s), v["accuracy"]) for s, v in summary[k].items())
    ls = "--" if k.startswith("sep") else "-"
    ax.plot([p[0] for p in pts], [100 * p[1] for p in pts], ls, color=col, lw=2, marker="o", ms=4,
            label=name.replace("\n", " "))
ax.axhline(100 * summary["_original"]["accuracy"], color=MUTED, lw=0.8, ls=":")
ax.axhline(10, color=MUTED, lw=0.8, ls=":")
ax.text(-10, 11, "chance (10 classes)", fontsize=8, color=MUTED)
ax.axvspan(-5, 5, color="#E6E4DE", alpha=0.5, lw=0, zorder=0)
ax.set_xlabel("Channel SNR (dB); shaded: training range of the generative models")
ax.set_ylabel("Reconstructions that keep their class (%)")
ax.set_ylim(0, 102)
ax.grid(color="#E6E4DE", lw=0.6)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="lower right", fontsize=8.5, frameon=False)
fig.savefig(out / "class_vs_snr.png", dpi=170)
plt.close(fig)
print("wrote", out / "class_vs_snr.png")

# ---- per class, at each SNR ----
for snr in snrs:
    keys = [(k, n) for k, n, _ in SYSTEMS if f"{snr:g}" in summary[k]]
    m = np.array([[summary[k][f"{snr:g}"]["per_class"][c] for k, _ in keys] for c in classes])
    fig, ax = plt.subplots(figsize=(1.15 * len(keys) + 2.2, 4.6), constrained_layout=True)
    ax.imshow(m, cmap="Greys", vmin=0, vmax=1.6, aspect="auto")
    for (r, c), v in np.ndenumerate(m):
        ax.text(c, r, f"{100 * v:.0f}", ha="center", va="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(keys)), [n for _, n in keys], fontsize=7.5, rotation=0)
    ax.set_yticks(range(len(classes)), label)
    ax.set_title(f"Class kept (%) per class at {snr:+g} dB".replace("-", "−"), loc="left", fontsize=11, color=INK)
    fig.savefig(out / f"per_class_{snr:g}.png", dpi=150)
    plt.close(fig)
print("wrote per-class tables")
