"""Analysis suite: what a set of systems does, in metrics and in pixels, against one reference.

Reads only finished outputs (CPU, no model runs):
  * each run's eval-spread.json / eval-spread-curve.json (means, FID) and eval_images-spread.npz
    (per-image PSNR, LPIPS, DINOv2 similarity, retrieval rank on the 1,000 class-spread images)
  * each run's history.json (validation curves)
  * a dump folder from scripts/gen_analysis/dump.py (reconstructions, class predictions, the
    refiner's CNN input, repeated samples), when given

and writes OUT/report.md with its figures:

  1  verdict        the pre-registered rule of 17 Nov, per candidate against the reference
  2  curves         PSNR, LPIPS, DINOv2 similarity, retrieval top-1, FID, class kept vs SNR
  3  paired         per-image paired differences with 95% bootstrap intervals and win rates
  4  plane          perception-distortion and perception-semantics planes
  5  per image      where a perceptual gain comes with a semantic loss on the same image
  6  per class      class kept per class
  7  images         one image per class per system; each candidate's best and worst images
  8  spectrum       radial power spectrum against the originals (blur vs invented texture)
  9  statistics     sharpness, contrast, colour shift, saturation against the originals
  10 diversity      how much repeated sampling changes the output (generative receivers)
  11 refiner        what the diffusion stage changed in its CNN input; class fixed / broken
  12 training       validation curves

usage: suite.py SPEC.json
SPEC: {"out": DIR, "dump": DIR (optional), "reference": KEY, "snrs": [-5, 0, 5],
       "systems": [{"key": KEY, "label": TEXT, "run": RUN_DIR}, ...]}
Systems are drawn in the order given (fixed colours: first = reference)."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = Path(__file__).resolve().parents[2]
spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
OUT = R / spec["out"]
OUT.mkdir(parents=True, exist_ok=True)
SYS = spec["systems"]
REF = spec["reference"]
KEYS = [s["key"] for s in SYS]
assert REF in KEYS, "the reference must be one of the systems"
LABEL = {s["key"]: s["label"] for s in SYS}
RUN = {s["key"]: R / s["run"] for s in SYS}
PAIR_SNRS = [float(v) for v in spec.get("snrs", [-5, 0, 5])]
DUMP = R / spec["dump"] if spec.get("dump") else None
CAND = [k for k in KEYS if k != REF]
rng = np.random.default_rng(0)
B = 2000

# Reference palette (dataviz skill, light mode), categorical slots in fixed order.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
COLOR = {k: SLOTS[i % len(SLOTS)] for i, k in enumerate(KEYS)}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.dpi": 130})
report: list[str] = []


def md(*lines):
    report.extend(lines)


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    md(f"![{name}]({name})", "")


# ------------------------------------------------------------------ loading

def means(key) -> dict[float, dict]:
    """SNR -> metric means at 10 sampling steps (or the CNN's single entry), spread set."""
    out = {}
    for f in ("eval-spread-curve.json", "eval-spread.json", "eval-spread-high.json"):
        p = RUN[key] / f
        if not p.exists():
            continue
        for k, v in json.loads(p.read_text())["results"].items():
            snr, _, steps = k.partition("_")
            if steps in ("", "10"):
                out[float(snr)] = v
    return dict(sorted(out.items()))


def per_image(key) -> dict:
    p = RUN[key] / "eval_images-spread.npz"
    if not p.exists():
        return {}
    d = np.load(p)
    out = {}
    for name in d.files:
        parts = name.split("_")
        steps = parts[-1] if len(parts) > 2 and parts[-2].lstrip("-").replace(".", "").isdigit() else ""
        if steps not in ("", "10"):
            continue
        snr = float(parts[-2] if steps else parts[-1])
        metric = "_".join(parts[:-2] if steps else parts[:-1])
        out.setdefault(snr, {})[metric] = d[name].astype(np.float64)
    for v in out.values():
        if "retrieval_rank" in v:
            v["top1"] = (v["retrieval_rank"] < 1).astype(float) * 100
            v["top5"] = (v["retrieval_rank"] < 5).astype(float) * 100
    return out


M = {k: means(k) for k in KEYS}
P = {k: per_image(k) for k in KEYS}
D, DMETA = {}, None
if DUMP and (DUMP / "dump.json").exists():
    DMETA = json.loads((DUMP / "dump.json").read_text())
    for k in KEYS:
        name = RUN[k].name
        for f in DUMP.glob(f"{name}__*.npz"):
            D.setdefault(k, {})[float(f.stem.split("__")[1])] = dict(np.load(f))
    ORIG8 = np.load(R / "data/gen_cache/imagenette_128_vitb14/test_images.npy", mmap_mode="r")
    KEPT_IDX = np.array(DMETA["index"])[DMETA["kept"]]
    ORIG = np.asarray(ORIG8[KEPT_IDX])
    LABELS = np.array(DMETA["labels"])
    CLASSES = DMETA["classes"]
NAMES = {"n01440764": "tench", "n02102040": "springer", "n02979186": "cassette player", "n03000684": "chain saw",
         "n03028079": "church", "n03394916": "French horn", "n03417042": "garbage truck", "n03425413": "gas pump",
         "n03445777": "golf ball", "n03888257": "parachute"}


def boot(d: np.ndarray) -> tuple[float, float, float]:
    i = rng.integers(0, len(d), (B, len(d)))
    m = d[i].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


#: Direction in which each metric improves.
BETTER = {"psnr": 1, "lpips": -1, "dino_sim": 1, "top1": 1, "top5": 1, "fid": -1, "class": 1}
UNIT = {"psnr": "dB", "lpips": "", "dino_sim": "", "top1": "pts", "top5": "pts"}


def paired(k, snr, metric):
    a, b = P[REF].get(snr, {}).get(metric), P[k].get(snr, {}).get(metric)
    if a is None or b is None or len(a) != len(b):
        return None
    d = b - a
    mean, lo, hi = boot(d)
    win = float(np.mean(BETTER[metric] * d > 0) * 100)
    sig = (lo > 0 or hi < 0)
    good = sig and BETTER[metric] * mean > 0
    return dict(mean=mean, lo=lo, hi=hi, win=win, sig=sig, good=good, bad=sig and not good)


def class_kept(k, snr):
    d = D.get(k, {}).get(snr)
    return None if d is None else float((d["pred"] == d["label"]).mean() * 100)


md(f"# Analysis: {spec.get('title', OUT.name)}", "",
   f"Reference: **{LABEL[REF]}** (`{RUN[REF].relative_to(R)}`). Test set: the 1,000 class-spread test images, "
   "10 sampling steps, paired channel and sampling noise; all runs scored on the same GPU type.", "")
md("| Key | System | Run |", "|---|---|---|", *[f"| {k} | {LABEL[k]} | `{RUN[k].relative_to(R)}` |" for k in KEYS], "")

# ------------------------------------------------------------------ 1 verdict (pre-registered 17 Nov)
md("## 1. Verdict against the pre-registered rule", "",
   "Rule fixed on 17 Nov before any result: a candidate **beats the reference** if, at 0 and +5 dB, it has a "
   "clear gain on LPIPS (95% interval below 0) or on FID (more than 5 points lower), and no clear loss "
   "(interval entirely on the wrong side) on DINOv2 similarity or retrieval top-1. Between candidates that pass, "
   "the higher DINOv2 similarity wins.", "",
   "| Candidate | " + " | ".join(f"{s:g} dB: perceptual gain / semantic loss" for s in (0.0, 5.0)) + " | Verdict |",
   "|---|---|---|---|")
passed = []
for k in CAND:
    cells, ok = [], True
    for s in (0.0, 5.0):
        lp, dn, t1 = paired(k, s, "lpips"), paired(k, s, "dino_sim"), paired(k, s, "top1")
        fr, fk = M[REF].get(s, {}).get("fid"), M[k].get(s, {}).get("fid")
        if lp is None or dn is None or t1 is None:
            cells.append("no data")
            ok = False
            continue
        gain = lp["good"] or (fr is not None and fk is not None and fk < fr - 5)
        loss = dn["bad"] or t1["bad"]
        ok &= gain and not loss
        cells.append(f"{'yes' if gain else 'no'} / {'yes' if loss else 'no'}")
    if ok:
        passed.append(k)
    md(f"| {LABEL[k]} | " + " | ".join(cells) + f" | {'**passes**' if ok else 'does not pass'} |")
if passed:
    best = max(passed, key=lambda k: M[k].get(0.0, {}).get("dino_sim", -1))
    md("", f"Chosen by the rule: **{LABEL[best]}**.")
md("")

# ------------------------------------------------------------------ 2 curves
md("## 2. Metrics against SNR", "", "Means over the 1,000 images. FID uses 1,000 images (relative only). "
   "Class kept: DINOv2 10-NN classifier on the reconstructions (from the dump).", "")
METRICS = [("psnr", "PSNR (dB)"), ("lpips", "LPIPS (lower is better)"), ("dino_sim", "DINOv2 similarity"),
           ("retrieval_top1", "Retrieval top-1"), ("fid", "FID (lower is better)"), ("class", "Class kept (%)")]
fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))
for ax, (m, title) in zip(axes.flat, METRICS):
    for k in KEYS:
        if m == "class":
            xs = sorted(D.get(k, {}))
            ys = [class_kept(k, s) for s in xs]
        else:
            xs = [s for s, v in M[k].items() if m in v]
            ys = [M[k][s][m] for s in xs]
        if xs:
            ax.plot(xs, ys, color=COLOR[k], lw=2 if k != REF else 2.4, marker="o", ms=4,
                    ls="--" if k == REF else "-", label=LABEL[k])
    ax.set_title(title, color=INK, loc="left", fontsize=9.5)
    ax.set_xlabel("SNR (dB)")
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=min(len(KEYS), 4), bbox_to_anchor=(0.5, -0.06))
fig.tight_layout()
save(fig, "curves.png")
rows = sorted({s for k in KEYS for s in M[k]})
md("| SNR | System | PSNR | LPIPS | DINOv2 | Top-1 | FID | Class kept |", "|---|---|---|---|---|---|---|---|")
for s in rows:
    for k in KEYS:
        v = M[k].get(s)
        if not v:
            continue
        ck = class_kept(k, s)
        md(f"| {s:g} | {LABEL[k]} | {v['psnr']:.2f} | {v.get('lpips', float('nan')):.3f} | {v['dino_sim']:.3f} | "
           f"{100 * v['retrieval_top1']:.0f}% | {v.get('fid', float('nan')):.0f} | "
           f"{'' if ck is None else f'{ck:.0f}%'} |")
md("")

# ------------------------------------------------------------------ 3 paired
md("## 3. Paired differences against the reference", "",
   "Candidate minus reference per image, mean with 95% bootstrap interval; **bold** = clear gain, "
   "*italic* = clear loss. Win rate: share of images where the candidate is better.", "",
   "| Candidate | SNR | ΔPSNR (dB) | ΔLPIPS | ΔDINOv2 | ΔTop-1 (pts) | ΔTop-5 (pts) | Win rate PSNR / LPIPS / DINOv2 |",
   "|---|---|---|---|---|---|---|---|")


def fmt(r, nd):
    if r is None:
        return "-"
    t = f"{r['mean']:+.{nd}f} [{r['lo']:+.{nd}f}, {r['hi']:+.{nd}f}]"
    return f"**{t}**" if r["good"] else f"*{t}*" if r["bad"] else t


for k in CAND:
    for s in PAIR_SNRS:
        r = {m: paired(k, s, m) for m in ("psnr", "lpips", "dino_sim", "top1", "top5")}
        if all(v is None for v in r.values()):
            continue
        wins = " / ".join(f"{r[m]['win']:.0f}%" if r[m] else "-" for m in ("psnr", "lpips", "dino_sim"))
        md(f"| {LABEL[k]} | {s:g} | {fmt(r['psnr'], 2)} | {fmt(r['lpips'], 3)} | {fmt(r['dino_sim'], 3)} | "
           f"{fmt(r['top1'], 1)} | {fmt(r['top5'], 1)} | {wins} |")
md("")

# ------------------------------------------------------------------ 4 planes
md("## 4. Perception-distortion and perception-semantics planes", "",
   "Each line is one system across SNR (marker size grows with SNR). Down-right is better in both panels: "
   "a system that only trades along a line moved along the trade-off, one that moves the whole line beat it.", "")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, (xm, xl) in zip(axes, [("psnr", "PSNR (dB)"), ("dino_sim", "DINOv2 similarity")]):
    for k in KEYS:
        pts = [(v[xm], v["lpips"], s) for s, v in M[k].items() if "lpips" in v]
        if not pts:
            continue
        x, y, s = map(np.array, zip(*pts))
        ax.plot(x, y, color=COLOR[k], lw=1.6, ls="--" if k == REF else "-", label=LABEL[k])
        ax.scatter(x, y, s=12 + 3 * (s - s.min()), color=COLOR[k], edgecolor="#fcfcfb", linewidth=1, zorder=3)
        ax.annotate(LABEL[k], (x[-1], y[-1]), xytext=(5, 0), textcoords="offset points",
                    fontsize=6.5, color=INK2, va="center")
    ax.set_xlabel(xl)
    ax.set_ylabel("LPIPS (lower is better)")
axes[0].legend(loc="upper right", fontsize=7.5)
fig.tight_layout()
save(fig, "plane.png")

# ------------------------------------------------------------------ 5 per image
md("## 5. Per image: does a perceptual gain cost semantics on the same image?", "",
   "Each dot is one test image: x = ΔLPIPS (left of 0 = sharper / closer perceptually), y = ΔDINOv2 "
   "(above 0 = semantically closer), candidate minus reference. The corner shares say how often a "
   "candidate wins on both, only perceptually, only semantically, or on neither.", "")
for s in [v for v in PAIR_SNRS if v >= 0] or PAIR_SNRS:
    cand = [k for k in CAND if P[k].get(s) and P[REF].get(s)]
    if not cand:
        continue
    fig, axes = plt.subplots(1, len(cand), figsize=(3.6 * len(cand), 3.4), squeeze=False)
    md(f"**{s:g} dB**", "", "| Candidate | Both better | Perceptual only | Semantic only | Neither |", "|---|---|---|---|---|")
    for ax, k in zip(axes[0], cand):
        dl = P[k][s]["lpips"] - P[REF][s]["lpips"]
        dd = P[k][s]["dino_sim"] - P[REF][s]["dino_sim"]
        ax.scatter(dl, dd, s=6, color=COLOR[k], alpha=0.45, linewidth=0)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(LABEL[k], fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("ΔLPIPS")
        ax.set_ylabel("ΔDINOv2")
        q = [np.mean((dl < 0) & (dd > 0)), np.mean((dl < 0) & (dd <= 0)), np.mean((dl >= 0) & (dd > 0)),
             np.mean((dl >= 0) & (dd <= 0))]
        md(f"| {LABEL[k]} | " + " | ".join(f"{100 * v:.0f}%" for v in q) + " |")
    md("")
    fig.tight_layout()
    save(fig, f"per_image_{s:g}.png")

# ------------------------------------------------------------------ dump-based sections
if not D:
    md("## 6-11. Image-level analysis", "", "No dump folder given or found: run scripts/gen_analysis/dump.py "
       "(Kaggle runner `dump`) for these sections.", "")
else:
    def snrs_of(k):
        return sorted(D.get(k, {}))

    common = sorted(set.intersection(*[set(snrs_of(k)) for k in KEYS if k in D]))
    # ---- 6 per class
    md("## 6. Class kept per class", "", "DINOv2 10-NN class of each reconstruction against the true class, "
       f"{len(LABELS)} images ({len(LABELS) // len(CLASSES)} per class on average).", "")
    for s in [v for v in common if v in PAIR_SNRS]:
        ks = [k for k in KEYS if k in D]
        mat = np.array([[np.mean(D[k][s]["pred"][LABELS == c] == c) * 100 for k in ks] for c in range(len(CLASSES))])
        fig, ax = plt.subplots(figsize=(1.3 * len(ks) + 2.2, 4.2))
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(ks)), [LABEL[k].split(",")[0] for k in ks], rotation=25, ha="right")
        ax.set_yticks(range(len(CLASSES)), [NAMES.get(c, c) for c in CLASSES])
        ax.grid(False)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=7,
                        color="#fcfcfb" if mat[i, j] > 60 else INK)
        ax.set_title(f"Class kept (%) at {s:g} dB", fontsize=9.5, loc="left")
        fig.colorbar(im, ax=ax, shrink=0.7)
        fig.tight_layout()
        save(fig, f"per_class_{s:g}.png")

    # ---- 7 images
    md("## 7. Images", "", "One test image per class. Under each reconstruction: the class the classifier "
       "sees (✓ = true class) and the image's PSNR.", "")
    pos_of_kept = {int(p): i for i, p in enumerate(DMETA["kept"])}
    lab_kept = LABELS[DMETA["kept"]]
    pick = [int(np.where(lab_kept == c)[0][min(1, int((lab_kept == c).sum()) - 1)]) for c in range(len(CLASSES))
            if (lab_kept == c).any()]

    def psnr8(a, b):
        return 10 * np.log10(255 ** 2 / max(np.mean((a.astype(float) - b.astype(float)) ** 2), 1e-9))

    def show(ax, img, title=None, sub=None):
        ax.imshow(img.transpose(1, 2, 0))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        if title:
            ax.set_title(title, fontsize=7, color=INK)
        if sub:
            ax.set_xlabel(sub, fontsize=6.5, color=INK2)

    for s in [v for v in common if v in PAIR_SNRS]:
        rows = [("Original", None)] + [(LABEL[k], k) for k in KEYS if k in D]
        fig, axes = plt.subplots(len(rows), len(pick), figsize=(1.25 * len(pick), 1.4 * len(rows)), squeeze=False)
        for j, i in enumerate(pick):
            for r, (name, k) in enumerate(rows):
                ax = axes[r, j]
                if k is None:
                    show(ax, ORIG[i], NAMES.get(CLASSES[lab_kept[i]], "") if r == 0 else None)
                else:
                    d = D[k][s]
                    full = DMETA["kept"][i]
                    pr = int(d["pred"][full])
                    ok = "✓" if pr == lab_kept[i] else "✗ " + NAMES.get(CLASSES[pr], "")[:10]
                    show(ax, d["recon"][i], None, f"{ok}  {psnr8(d['recon'][i], ORIG[i]):.1f} dB")
                if j == 0:
                    ax.set_ylabel(name.replace(", ", ",\n"), fontsize=7, color=INK, rotation=0, ha="right", va="center")
        fig.suptitle(f"{s:g} dB", fontsize=9, x=0.02, ha="left")
        fig.tight_layout()
        save(fig, f"images_{s:g}.png")

    # Best and worst images of each candidate against the reference (DINOv2 similarity), at 0 dB.
    s0 = 0.0 if 0.0 in common else common[len(common) // 2]
    for k in [c for c in CAND if c in D and P[c].get(s0) and P[REF].get(s0) and REF in D]:
        dd = (P[k][s0]["dino_sim"] - P[REF][s0]["dino_sim"])[DMETA["kept"]]
        order = np.argsort(dd)
        sel = list(order[:5]) + list(order[-5:])
        fig, axes = plt.subplots(3, len(sel), figsize=(1.25 * len(sel), 4.2), squeeze=False)
        for j, i in enumerate(sel):
            show(axes[0, j], ORIG[i], f"{'worst' if j < 5 else 'best'} {dd[i]:+.2f}")
            show(axes[1, j], D[REF][s0]["recon"][i])
            show(axes[2, j], D[k][s0]["recon"][i])
        for r, name in enumerate(["Original", LABEL[REF], LABEL[k]]):
            axes[r, 0].set_ylabel(name.replace(", ", ",\n"), fontsize=7, rotation=0, ha="right", va="center")
        fig.suptitle(f"{LABEL[k]} against the reference at {s0:g} dB: the 5 images it loses most and gains most "
                     "DINOv2 similarity on", fontsize=8.5, x=0.02, ha="left")
        fig.tight_layout()
        save(fig, f"extremes_{k}.png")

    # ---- 8 spectrum
    md("## 8. Power spectrum against the originals", "",
       "Radially averaged power spectrum of the reconstructions over that of the originals (kept images, "
       "grey level). Below 1 at high frequency = blur (detail removed); above 1 = texture or noise added that "
       "the original does not have.", "")

    def radial(imgs: np.ndarray) -> np.ndarray:
        g = imgs.astype(np.float64).mean(1)
        g = g - g.mean(axis=(1, 2), keepdims=True)
        p = np.abs(np.fft.fftshift(np.fft.fft2(g), axes=(1, 2))) ** 2
        h = g.shape[1]
        yy, xx = np.indices((h, h)) - h // 2
        r = np.sqrt(xx ** 2 + yy ** 2).astype(int)
        p = p.mean(0)
        return np.bincount(r.ravel(), p.ravel())[: h // 2] / np.bincount(r.ravel())[: h // 2]

    ref_spec = radial(ORIG)
    freqs = np.arange(len(ref_spec)) / (2 * len(ref_spec))
    show_snrs = [v for v in common if v in PAIR_SNRS]
    fig, axes = plt.subplots(1, len(show_snrs), figsize=(4 * len(show_snrs), 3.4), squeeze=False)
    spec_tab = {}
    for ax, s in zip(axes[0], show_snrs):
        for k in KEYS:
            if k in D and s in D[k]:
                ratio = radial(D[k][s]["recon"]) / ref_spec
                spec_tab[(k, s)] = float(ratio[len(ratio) // 2:].mean())
                ax.plot(freqs[1:], ratio[1:], color=COLOR[k], lw=1.8, ls="--" if k == REF else "-", label=LABEL[k])
        ax.axhline(1, color=INK2, lw=0.8)
        ax.set_yscale("log")
        ax.set_xlabel("spatial frequency (cycles / pixel)")
        ax.set_title(f"{s:g} dB: power / original power", fontsize=9, loc="left")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    save(fig, "spectrum.png")
    md("High-frequency power (upper half of the band) relative to the originals:", "",
       "| System | " + " | ".join(f"{s:g} dB" for s in show_snrs) + " |", "|---|" + "---|" * len(show_snrs),
       *[f"| {LABEL[k]} | " + " | ".join(f"{spec_tab.get((k, s), float('nan')):.2f}" for s in show_snrs) + " |"
         for k in KEYS if k in D], "")

    # ---- 9 statistics
    md("## 9. Image statistics against the originals", "",
       "Ratios to the original image (1 = same): sharpness = variance of the Laplacian, contrast = pixel standard "
       "deviation, saturation = mean HSV saturation; colour shift = mean absolute change of the per-channel means "
       "(0-255). Means over the kept images.", "")

    def stats(imgs: np.ndarray) -> dict:
        x = imgs.astype(np.float64)
        g = x.mean(1)
        lap = 4 * g[:, 1:-1, 1:-1] - g[:, :-2, 1:-1] - g[:, 2:, 1:-1] - g[:, 1:-1, :-2] - g[:, 1:-1, 2:]
        mx, mn = x.max(1), x.min(1)
        sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-9), 0)
        return {"sharp": lap.var(axis=(1, 2)), "contrast": x.std(axis=(1, 2, 3)), "sat": sat.mean(axis=(1, 2)),
                "means": x.mean(axis=(2, 3))}

    o = stats(ORIG)
    md("| System | SNR | Sharpness | Contrast | Saturation | Colour shift |", "|---|---|---|---|---|---|")
    for s in show_snrs:
        for k in KEYS:
            if k in D and s in D[k]:
                r = stats(D[k][s]["recon"])
                md(f"| {LABEL[k]} | {s:g} | {np.median(r['sharp'] / np.maximum(o['sharp'], 1e-9)):.2f} | "
                   f"{np.median(r['contrast'] / o['contrast']):.2f} | {np.median(r['sat'] / np.maximum(o['sat'], 1e-9)):.2f} | "
                   f"{np.abs(r['means'] - o['means']).mean():.1f} |")
    md("", "(Sharpness, contrast and saturation are medians of per-image ratios.)", "")

    # ---- 10 diversity
    gen = [k for k in KEYS if k in D and any("samples" in v for v in D[k].values())]
    if gen:
        md("## 10. Sample diversity", "",
           "The same received symbols decoded several times with different sampling noise. Spread = per-pixel "
           "standard deviation across draws (0-255), averaged; draw-to-draw PSNR = between two draws of the same "
           "image. A deterministic receiver has spread 0. High spread at high SNR means the receiver invents "
           "content the channel does carry.", "")
        md("| System | SNR | Spread | Draw-to-draw PSNR (dB) |", "|---|---|---|---|")
        for k in gen:
            for s in snrs_of(k):
                smp = D[k][s].get("samples")
                if smp is None:
                    continue
                x = smp.astype(np.float64)
                spread = x.std(1).mean()
                pp = [psnr8(x[i, a], x[i, b]) for i in range(len(x)) for a in range(x.shape[1])
                      for b in range(a + 1, x.shape[1])]
                md(f"| {LABEL[k]} | {s:g} | {spread:.1f} | {np.mean(pp):.1f} |")
        md("")
        k = gen[0]
        s = 0.0 if 0.0 in D[k] else snrs_of(k)[0]
        smp = D[k][s]["samples"][:6]
        fig, axes = plt.subplots(len(smp), smp.shape[1] + 2, figsize=(1.25 * (smp.shape[1] + 2), 1.3 * len(smp)),
                                 squeeze=False)
        for i in range(len(smp)):
            show(axes[i, 0], ORIG[i], "original" if i == 0 else None)
            for j in range(smp.shape[1]):
                show(axes[i, j + 1], smp[i, j], f"draw {j + 1}" if i == 0 else None)
            sd = smp[i].astype(np.float64).std(0).mean(0)
            axes[i, -1].imshow(sd, cmap="Greys", vmin=0, vmax=max(sd.max(), 1))
            axes[i, -1].set_xticks([])
            axes[i, -1].set_yticks([])
            axes[i, -1].grid(False)
            if i == 0:
                axes[i, -1].set_title("spread", fontsize=7)
        fig.suptitle(f"{LABEL[k]}, {s:g} dB: repeated draws", fontsize=8.5, x=0.02, ha="left")
        fig.tight_layout()
        save(fig, f"diversity_{k}.png")

    # ---- 11 refiner
    refs = [k for k in KEYS if k in D and any("cnn" in v for v in D[k].values())]
    for k in refs:
        md(f"## 11. What the refiner changed: {LABEL[k]}", "",
           "Class of the CNN output it started from against the class of its own output, on all images: "
           "fixed = CNN wrong, refiner right; broken = CNN right, refiner wrong.", "",
           "| SNR | CNN input kept | Refiner kept | Fixed | Broken | Both wrong, changed class | PSNR change (kept images, dB) |",
           "|---|---|---|---|---|---|---|")
        for s in snrs_of(k):
            d = D[k][s]
            if "cnn" not in d:
                continue
            cp, rp = d["cnn_pred"], d["pred"]
            fixed = np.mean((cp != LABELS) & (rp == LABELS)) * 100
            broken = np.mean((cp == LABELS) & (rp != LABELS)) * 100
            drift = np.mean((cp != LABELS) & (rp != LABELS) & (cp != rp)) * 100
            dpsnr = np.mean([psnr8(d["recon"][i], ORIG[i]) - psnr8(d["cnn"][i], ORIG[i]) for i in range(len(ORIG))])
            md(f"| {s:g} | {np.mean(cp == LABELS) * 100:.0f}% | {np.mean(rp == LABELS) * 100:.0f}% | {fixed:.1f}% | "
               f"{broken:.1f}% | {drift:.1f}% | {dpsnr:+.2f} |")
        md("")
        for s in [v for v in snrs_of(k) if v in PAIR_SNRS]:
            d = D[k][s]
            fig, axes = plt.subplots(4, len(pick), figsize=(1.25 * len(pick), 5.4), squeeze=False)
            for j, i in enumerate(pick):
                show(axes[0, j], ORIG[i], NAMES.get(CLASSES[lab_kept[i]], ""))
                show(axes[1, j], d["cnn"][i], None, f"{psnr8(d['cnn'][i], ORIG[i]):.1f} dB")
                show(axes[2, j], d["recon"][i], None, f"{psnr8(d['recon'][i], ORIG[i]):.1f} dB")
                diff = np.abs(d["recon"][i].astype(float) - d["cnn"][i].astype(float)).mean(0)
                axes[3, j].imshow(diff, cmap="Greys", vmin=0, vmax=64)
                axes[3, j].set_xticks([])
                axes[3, j].set_yticks([])
                axes[3, j].grid(False)
            for r, name in enumerate(["Original", "CNN input", "Refiner output", "|change|"]):
                axes[r, 0].set_ylabel(name, fontsize=7, rotation=0, ha="right", va="center")
            fig.suptitle(f"{LABEL[k]} at {s:g} dB", fontsize=8.5, x=0.02, ha="left")
            fig.tight_layout()
            save(fig, f"refiner_{k}_{s:g}.png")

# ------------------------------------------------------------------ 12 training
md("## 12. Training", "", "Validation curves from each run's history.json (EMA weights). The diffusion loss of "
   "the generative systems and the CNN's pixel MSE are different quantities: compare loss within a family only.", "")
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
for k in KEYS:
    hf = RUN[k] / "history.json"
    if not hf.exists():
        continue
    h = json.loads(hf.read_text())["history"]
    st = [e["step"] for e in h]
    q = [e.get("quality", {}) for e in h]
    if all("psnr@0" in e for e in q):
        axes[0].plot(st, [e["psnr@0"] for e in q], color=COLOR[k], lw=1.6, label=LABEL[k])
        axes[1].plot(st, [e["dino@0"] for e in q], color=COLOR[k], lw=1.6)
    if not RUN[k].name.startswith("cnn"):
        axes[2].plot(st, [e["val_loss"] for e in h], color=COLOR[k], lw=1.6)
for ax, t in zip(axes, ["val PSNR at 0 dB (32 images)", "val DINOv2 similarity at 0 dB", "val diffusion loss"]):
    ax.set_title(t, fontsize=9, loc="left")
    ax.set_xlabel("step")
axes[0].legend(fontsize=7)
fig.tight_layout()
save(fig, "training.png")

(OUT / "report.md").write_text("\n".join(report), encoding="utf-8")
print(f"wrote {OUT / 'report.md'} ({len(list(OUT.glob('*.png')))} figures)")
