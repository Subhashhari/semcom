"""Report for the exploratory DiT-vs-CNN metric analysis (protocol of 22 Nov in the month plan):
every metric, paired DiT minus CNN with 95% bootstrap intervals on all images and on the
exploration (even positions) and confirmation (odd) halves, oriented so that positive = DiT better,
plus distribution metrics, a content analysis and caption examples. CPU only.

usage: semantic_report.py METRICS_DIR DUMP_DIR OUT_DIR DIT_KEY:DIT_RUN CNN_KEY:CNN_RUN [--snrs -5,0,5]
  METRICS_DIR  output of semantic_metrics.py; DUMP_DIR the full dump (originals, DINOv2 embeddings)
  *_RUN        run folders whose eval_images-spread.npz adds PSNR, LPIPS, DINOv2, retrieval
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

argv = sys.argv[1:]
snrs = ["-5", "0", "5"]
if "--snrs" in argv:
    i = argv.index("--snrs")
    snrs = argv[i + 1].split(",")
    del argv[i:i + 2]
mdir, ddir, out = Path(argv[0]), Path(argv[1]), Path(argv[2])
(dk, drun), (ck, crun) = [a.split(":", 1) for a in argv[3:5]]
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
CLASSES = ["tench", "English springer", "cassette player", "chain saw", "church", "French horn",
           "garbage truck", "gas pump", "golf ball", "parachute"]


def load(k):
    return dict(np.load(mdir / f"{k}.npz"))


orig = load("original")
caps = json.loads((mdir / "captions.json").read_text())
dets = json.loads((mdir / "detections.json").read_text())["sets"]
odump = np.load(ddir / "_original.npz")
labels, n = odump["label"], len(odump["label"])
even, odd = np.arange(0, n, 2), np.arange(1, n, 2)


def det_f1(a, b, boxes):
    """F1 between the original's detections a and the reconstruction's b (label multisets, or
    greedy same-label box matches at IoU >= 0.5). Both empty counts as agreement (1)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if not boxes:
        from collections import Counter
        ca, cb = Counter(l for l, _ in a), Counter(l for l, _ in b)
        tp = sum((ca & cb).values())
    else:
        used, tp = set(), 0
        for la, ba in a:
            for j, (lb, bb) in enumerate(b):
                if j in used or lb != la:
                    continue
                ix = max(0, min(ba[2], bb[2]) - max(ba[0], bb[0])) * max(0, min(ba[3], bb[3]) - max(ba[1], bb[1]))
                u = (ba[2] - ba[0]) * (ba[3] - ba[1]) + (bb[2] - bb[0]) * (bb[3] - bb[1]) - ix
                if u > 0 and ix / u >= 0.5:
                    used.add(j)
                    tp += 1
                    break
    return 2 * tp / (len(a) + len(b))


def js(p, q):
    p, q = p.astype(np.float64) + 1e-12, q.astype(np.float64) + 1e-12
    m = (p + q) / 2
    return 0.5 * (p * np.log(p / m)).sum(1) + 0.5 * (q * np.log(q / m)).sum(1)


# (name, group, higher is better, description)
METRICS = [
    ("psnr", "pixel", True, "PSNR (dB)"),
    ("lpips", "perceptual", False, "LPIPS-Alex"),
    ("dists", "perceptual", False, "DISTS"),
    ("dreamsim", "perceptual", False, "DreamSim (human-aligned)"),
    ("dino_sim", "semantic", True, "DINOv2 cosine"),
    ("dino_top1", "semantic", True, "DINOv2 retrieval top-1"),
    ("clip_sim", "semantic", True, "CLIP L/14 cosine"),
    ("clip_top1", "semantic", True, "CLIP retrieval top-1"),
    ("in1k_agree", "semantic", True, "ImageNet-1k top-1 agrees with original"),
    ("in1k_top5", "semantic", True, "ImageNet-1k top-5 overlap"),
    ("in1k_js", "semantic", False, "ImageNet-1k JS divergence"),
    ("in1k_porig", "semantic", True, "prob. of the original's ImageNet class"),
    ("zs_agree", "semantic", True, "CLIP zero-shot class agrees"),
    ("caption_sim", "language", True, "caption similarity (CLIP text)"),
    ("capscore_rel", "language", True, "CLIP score vs original's caption (relative)"),
    ("det_set_f1", "objects", True, "detected object labels F1"),
    ("det_box_f1", "objects", True, "detected boxes F1 (IoU 0.5)"),
    ("det_count_err", "objects", False, "|#objects - original's|"),
    ("clipiqa", "no-reference", True, "CLIP-IQA quality"),
    ("brisque", "no-reference", False, "BRISQUE"),
]


def per_image(key, run, s):
    m = load(f"{key}__{s}")
    ev = np.load(Path(run) / "eval_images-spread.npz")
    evk = {k.rsplit("_", 1)[0] if k.endswith("_10") else k: k for k in ev.files}  # DiT keys end in _10
    g = lambda name: ev[evk[f"{name}_{s}"]].astype(np.float64)  # noqa: E731
    d = dets[f"{key}__{s}"]
    return {
        "psnr": g("psnr"), "lpips": g("lpips"), "dino_sim": g("dino_sim"),
        "dino_top1": (g("retrieval_rank") < 1).astype(float),
        "dists": m["dists"], "dreamsim": m["dreamsim"], "clip_sim": m["clip_sim"],
        "clip_top1": (m["clip_rank"] < 1).astype(float),
        "in1k_agree": (m["in1k_top1"] == orig["in1k_top1"]).astype(float),
        "in1k_top5": np.array([len(set(a) & set(b)) / 5 for a, b in zip(m["in1k_top5"], orig["in1k_top5"])]),
        "in1k_js": js(m["in1k_prob"], orig["in1k_prob"]),
        "in1k_porig": m["in1k_prob"].astype(np.float64)[np.arange(n), orig["in1k_top1"]],
        "zs_agree": (m["zs_top1"] == orig["zs_top1"]).astype(float),
        "caption_sim": m["caption_sim"], "capscore_rel": m["capscore"] / orig["capscore"],
        "det_set_f1": np.array([det_f1(a, b, False) for a, b in zip(dets["original"], d)]),
        "det_box_f1": np.array([det_f1(a, b, True) for a, b in zip(dets["original"], d)]),
        "det_count_err": np.array([abs(len(a) - len(b)) for a, b in zip(dets["original"], d)], float),
        "clipiqa": m["clipiqa"], "brisque": m["brisque"],
    }


def ci(x):
    idx = rng.integers(0, len(x), (2000, len(x)))
    b = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


table, per = {}, {}
for s in snrs:
    a, c = per_image(dk, drun, s), per_image(ck, crun, s)
    per[s] = (a, c)
    for name, group, hib, desc in METRICS:
        d = (a[name] - c[name]) * (1 if hib else -1)  # positive = DiT better
        row = {"dit": float(a[name].mean()), "cnn": float(c[name].mean()), "win_rate": float((d > 0).mean())}
        for part, ix in (("all", slice(None)), ("explore", even), ("confirm", odd)):
            m_, lo, hi = ci(d[ix])
            row[part] = {"diff": m_, "lo": lo, "hi": hi}
        table.setdefault(name, {"group": group, "higher_is_better": hib, "desc": desc})[s] = row

verdict = {}
for name, *_ in METRICS:
    ok = all(table[name][s][p]["lo"] > 0 for s in ("0", "5") for p in ("explore", "confirm"))
    bad = all(table[name][s][p]["hi"] < 0 for s in ("0", "5") for p in ("explore", "confirm"))
    verdict[name] = "DiT advantage (confirmed)" if ok else ("CNN advantage (confirmed)" if bad else "mixed / tie")


# ---- distribution metrics: precision, recall, density, coverage (Naeem et al. 2020), k = 5 ----
def prdc(real, fake, k=5):
    real, fake = real.astype(np.float32), fake.astype(np.float32)
    def d(a, b):
        return np.sqrt(np.maximum((a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2 * a @ b.T, 0))
    rr, ff, rf = d(real, real), d(fake, fake), d(real, fake)
    rk = np.sort(rr, 1)[:, k]
    fk = np.sort(ff, 1)[:, k]
    precision = float((rf < rk[:, None]).any(0).mean())
    recall = float((rf < fk[None, :]).any(1).mean())
    density = float((rf < rk[:, None]).sum(0).mean() / k)
    coverage = float((rf.min(1) < rk).mean())
    return {"precision": precision, "recall": recall, "density": density, "coverage": coverage}


dist = {}
for s in snrs:
    for key, run in ((dk, drun), (ck, crun)):
        dd = np.load(ddir / f"{Path(run).name}__{s}.npz")
        dist.setdefault(s, {})[key] = {"dinov2": prdc(odump["emb"], dd["emb"]),
                                       "clip": prdc(orig["clip_img"], load(f"{key}__{s}")["clip_img"])}

# ---- content analysis: what kind of image does the DiT win on? ----
img = odump["image"].astype(np.float64) / 255  # (N, 3, H, W): needs a --keep 1 dump
gray = img.mean(1)
gx, gy = np.diff(gray, axis=2)[:, :-1], np.diff(gray, axis=1)[:, :, :-1]
edge = np.sqrt(gx ** 2 + gy ** 2).mean((1, 2))
F_ = np.abs(np.fft.fftshift(np.fft.fft2(gray), axes=(1, 2))) ** 2
H = gray.shape[1]
yy, xx = np.mgrid[:H, :H] - H // 2
hf = F_[:, np.hypot(yy, xx) > H / 8].sum(1) / F_.sum((1, 2))
rg, yb = img[:, 0] - img[:, 1], 0.5 * (img[:, 0] + img[:, 1]) - img[:, 2]
colour = np.hypot(rg.std((1, 2)), yb.std((1, 2))) + 0.3 * np.hypot(rg.mean((1, 2)), yb.mean((1, 2)))
props = {"edge density": edge, "high-frequency energy": hf, "colourfulness": colour}
content = {}
for s in snrs:
    a, c = per[s]
    for name in ("dreamsim", "dino_sim", "clip_sim", "lpips", "caption_sim", "psnr"):
        hib = table[name]["higher_is_better"]
        d = (a[name] - c[name]) * (1 if hib else -1)
        row = {}
        for pn, pv in props.items():
            r, p = spearmanr(pv, d)
            q = np.quantile(pv, [0.25, 0.75])
            row[pn] = {"spearman": float(r), "p": float(p), "diff_low_quartile": float(d[pv <= q[0]].mean()),
                       "diff_high_quartile": float(d[pv >= q[1]].mean())}
        row["by class"] = {CLASSES[k]: float(d[labels == k].mean()) for k in range(10)}
        content.setdefault(s, {})[name] = row

# ---- caption examples: where the two disagree most about the original's meaning ----
examples = {}
for s in snrs:
    a, c = per[s]
    d = a["caption_sim"] - c["caption_sim"]
    pick = lambda ix: [{"i": int(i), "class": CLASSES[labels[i]], "original": caps["original"][i],  # noqa: E731
                        "dit": caps[f"{dk}__{s}"][i], "cnn": caps[f"{ck}__{s}"][i],
                        "dit_sim": float(a["caption_sim"][i]), "cnn_sim": float(c["caption_sim"][i])} for i in ix]
    order = np.argsort(d)
    examples[s] = {"dit_better": pick(order[::-1][:8]), "cnn_better": pick(order[:8]),
                   "random": pick(rng.choice(n, 6, replace=False))}

(out / "semantic_report.json").write_text(json.dumps(
    {"table": table, "verdict": verdict, "distribution": dist, "content": content, "examples": examples}, indent=1))

# ---- figure: standardised paired effect per metric and SNR (positive = DiT better) ----
names = [m[0] for m in METRICS]
fig, ax = plt.subplots(figsize=(7.5, 0.33 * len(names) + 1.2), constrained_layout=True)
colours = {"-5": "#2a78d6", "0": "#eb6834", "5": "#1baf7a"}
for j, s in enumerate(snrs):
    a, c = per[s]
    ys, xs, lo, hi = [], [], [], []
    for i, (name, _, hib, _) in enumerate(METRICS):
        d = (a[name] - c[name]) * (1 if hib else -1)
        sd = d.std() or 1.0
        m_, l_, h_ = table[name][s]["all"]["diff"], table[name][s]["all"]["lo"], table[name][s]["all"]["hi"]
        ys.append(i + (j - 1) * 0.25)
        xs.append(m_ / sd)
        lo.append((m_ - l_) / sd)
        hi.append((h_ - m_) / sd)
    ax.errorbar(xs, ys, xerr=[lo, hi], fmt="o", ms=4, color=colours.get(s, "#555"), label=f"{s} dB", capsize=2)
ax.axvline(0, color="#52514e", lw=0.8)
ax.set_yticks(range(len(names)), [m[3] for m in METRICS], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Paired effect size (mean / s.d. of the per-image difference); right = DiT better")
ax.grid(axis="x", color="#e4e3df", lw=0.6)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_title("DiT-S against the converged CNN, 1000 spread images (95% bootstrap intervals)", fontsize=9, loc="left")
fig.savefig(out / "semantic_effects.png", dpi=160)

# ---- report ----
L = ["# What the DiT does better than the converged CNN (exploratory)", "",
     "Protocol fixed before computing (month plan, 22 Nov). Positive differences favour the DiT. A metric",
     "counts as a DiT advantage only if its paired 95% interval favours the DiT on both the exploration",
     "(even) and confirmation (odd) halves at 0 and +5 dB.", "", "![effects](semantic_effects.png)", "",
     "| metric | group | " + " | ".join(f"{s} dB: DiT / CNN, diff [95%], win rate" for s in snrs) + " | verdict |",
     "|---|---|" + "---|" * len(snrs) + "---|"]
for name, group, hib, desc in METRICS:
    cells = []
    for s in snrs:
        r = table[name][s]
        cells.append(f"{r['dit']:.3f} / {r['cnn']:.3f}, {r['all']['diff']:+.3f} [{r['all']['lo']:+.3f}, "
                     f"{r['all']['hi']:+.3f}], {100 * r['win_rate']:.0f}%")
    L.append(f"| {desc} | {group} | " + " | ".join(cells) + f" | {verdict[name]} |")
L += ["", "## Distribution (precision = realism, recall / coverage = diversity of the outputs)", "",
      "| SNR | system | DINOv2 P / R / D / C | CLIP P / R / D / C |", "|---|---|---|---|"]
for s in snrs:
    for key in (dk, ck):
        v = dist[s][key]
        f = lambda x: " / ".join(f"{x[k]:.3f}" for k in ("precision", "recall", "density", "coverage"))  # noqa: E731
        L.append(f"| {s} | {key} | {f(v['dinov2'])} | {f(v['clip'])} |")
L += ["", "## Content: on which originals does the DiT gain? (Spearman of the per-image gain with the property;",
      "mean gain in the lowest and highest quartile of the property)", ""]
for s in snrs:
    L.append(f"**{s} dB**")
    L.append("")
    L.append("| metric | " + " | ".join(props) + " | best classes for the DiT | worst |")
    L.append("|---|" + "---|" * len(props) + "---|---|")
    for name, row in content[s].items():
        cells = [f"r = {row[p]['spearman']:+.2f}, {row[p]['diff_low_quartile']:+.3f} -> {row[p]['diff_high_quartile']:+.3f}"
                 for p in props]
        bc = sorted(row["by class"].items(), key=lambda kv: -kv[1])
        L.append(f"| {name} | " + " | ".join(cells) + f" | {', '.join(k for k, _ in bc[:3])} | {', '.join(k for k, _ in bc[-3:])} |")
    L.append("")
L += ["## Captions where the two systems disagree most (0 dB)", ""]
for kind in ("dit_better", "cnn_better", "random"):
    L.append(f"**{kind}**")
    L.append("")
    for e in examples.get("0", {}).get(kind, []):
        L.append(f"- [{e['class']}] original: *{e['original']}* | DiT: *{e['dit']}* ({e['dit_sim']:.2f}) | "
                 f"CNN: *{e['cnn']}* ({e['cnn_sim']:.2f})")
    L.append("")
(out / "report.md").write_text("\n".join(L), encoding="utf-8")
print("wrote", out / "report.md")
for name, v in verdict.items():
    print(f"  {name:14s} {v}")
