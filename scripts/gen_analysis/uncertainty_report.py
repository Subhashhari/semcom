"""Report for uncertainty.py: does the disagreement between several draws from the same received
symbols predict the evaluation draw's semantic errors? CPU only.

usage: uncertainty_report.py UNCERTAINTY.npz OUT_DIR"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

d = np.load(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
labels, K = d["label"], int(d["draws"])
od, oc = d["orig_dino"].astype(np.float32), d["orig_clip"].astype(np.float32)
rng = np.random.default_rng(0)


def auroc(score, positive):
    """P(score of a random error > score of a random correct one), ties counted half."""
    pos, neg = score[positive], score[~positive]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auroc_ci(score, positive):
    vals = []
    for _ in range(1000):
        i = rng.integers(0, len(score), len(score))
        vals.append(auroc(score[i], positive[i]))
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def disagreement(E):  # (K+1, N, D) unit vectors -> 1 - mean pairwise cosine per image
    G = np.einsum("knd,lnd->nkl", E, E)
    k = E.shape[0]
    return 1 - (G.sum((1, 2)) - k) / (k * (k - 1))


table, curves = {}, {}
for s in [f"{v:g}" for v in d["snrs"]]:
    D, C, P = d[f"dino_{s}"].astype(np.float32), d[f"clip_{s}"].astype(np.float32), d[f"pred_{s}"]
    u_clip, u_dino = disagreement(C), disagreement(D)
    e0 = D[0]  # the evaluation draw
    sim = (e0 * od).sum(1)
    S = e0 @ od.T
    rank = (S > S.diagonal()[:, None]).sum(1)
    err = {"class flip (DINOv2 k-NN vs label)": P[0] != labels,
           "retrieval miss (DINOv2)": rank > 0}
    row = {"error_rates": {k: float(v.mean()) for k, v in err.items()}}
    for uname, u in (("CLIP draws", u_clip), ("DINOv2 draws", u_dino)):
        for ename, e in err.items():
            a = auroc(u, e)
            row[f"AUROC {uname} -> {ename}"] = {"auroc": a, "ci": auroc_ci(u, e)}
        r, p = spearmanr(u, sim)
        row[f"Spearman {uname} vs DINOv2 similarity"] = {"r": float(r), "p": float(p)}
    # selective reception: flag the most uncertain q (CLIP draws), error rate on the rest
    order = np.argsort(u_clip)
    sel = {}
    for q in (0.0, 0.1, 0.2, 0.3):
        keep = order[:int(round(len(order) * (1 - q)))]
        sel[f"{int(q * 100)}%"] = {k: float(v[keep].mean()) for k, v in err.items()}
    row["error after flagging the most uncertain (CLIP draws)"] = sel
    curves[s] = (u_clip, err["class flip (DINOv2 k-NN vs label)"])
    # medoid of the draws (closest to the others, CLIP space): a receiver-side choice, no original needed
    G = np.einsum("knd,lnd->nkl", C, C)
    med = G.sum(2).argmax(1)
    em = D[med, np.arange(D.shape[1])]
    Sm = em @ od.T
    row["medoid draw vs evaluation draw"] = {
        "DINOv2 similarity": [float(sim.mean()), float((em * od).sum(1).mean())],
        "DINOv2 retrieval top-1": [float((rank == 0).mean()), float(((Sm > Sm.diagonal()[:, None]).sum(1) == 0).mean())],
        "class kept": [float((P[0] == labels).mean()), float((P[med, np.arange(len(labels))] == labels).mean())]}
    # class agreement across draws (majority vote): another uncertainty, interpretable as a vote
    agree = (P == P[0][None]).mean(0)
    row["AUROC (1 - draw class agreement) -> class flip"] = auroc(1 - agree, err["class flip (DINOv2 k-NN vs label)"])
    table[s] = row

lp = {k: float(d[k].mean()) for k in d.files if k.startswith("lpips_vgg_")}
(out / "uncertainty_report.json").write_text(json.dumps({"table": table, "lpips_vgg": lp}, indent=1))

fig, axes = plt.subplots(1, len(curves), figsize=(4 * len(curves), 3.2), constrained_layout=True, squeeze=False)
for ax, (s, (u, e)) in zip(axes[0], curves.items()):
    bins = np.quantile(u, np.linspace(0, 1, 11))
    which = np.clip(np.digitize(u, bins[1:-1]), 0, 9)
    rate = [e[which == b].mean() * 100 for b in range(10)]
    ax.bar(range(10), rate, color="#2a78d6", width=0.8, zorder=2)
    ax.axhline(e.mean() * 100, color="#52514e", lw=1, ls="--", zorder=3)
    ax.set_xticks(range(10), [f"{10 * b + 10}" for b in range(10)], fontsize=7)
    ax.set_xlabel("draw disagreement (CLIP), decile (10 = most)")
    ax.set_ylabel("class flips in the decile (%)")
    ax.set_title(f"{s} dB (dashed: average)", fontsize=9, loc="left")
    ax.grid(axis="y", color="#e4e3df", lw=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("DiT receiver: disagreement between 5 draws from the same symbols against its class errors",
             fontsize=9, x=0.01, ha="left")
fig.savefig(out / "uncertainty_deciles.png", dpi=160)

L = ["# Can the diffusion receiver flag its own errors? (exploratory, 23 Nov)", "",
     f"{K + 1} draws per image from the same received symbols (the evaluation's draw first). Uncertainty =",
     "1 - mean pairwise cosine of the draws' embeddings; it needs no original. AUROC 0.5 = no signal.", "",
     "![deciles](uncertainty_deciles.png)", ""]
for s, row in table.items():
    L += [f"## {s} dB", "", f"Error rates of the evaluation draw: " +
          ", ".join(f"{k} {100 * v:.1f}%" for k, v in row["error_rates"].items()), ""]
    for k, v in row.items():
        if k.startswith("AUROC ") and isinstance(v, dict):
            L.append(f"- {k}: **{v['auroc']:.3f}** [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]")
        elif k.startswith("Spearman"):
            L.append(f"- {k}: r = {v['r']:+.3f}")
    L.append(f"- AUROC (1 - draw class agreement) -> class flip: {row['AUROC (1 - draw class agreement) -> class flip']:.3f}")
    L.append("- Error after flagging the most uncertain (CLIP draws): " + "; ".join(
        f"{q}: " + ", ".join(f"{100 * v:.1f}%" for v in r.values()) for q, r in
        row["error after flagging the most uncertain (CLIP draws)"].items()))
    L.append("- Medoid draw vs evaluation draw: " + "; ".join(
        f"{k} {a:.3f} -> {b:.3f}" for k, (a, b) in row["medoid draw vs evaluation draw"].items()))
    L.append("")
if lp:
    L += ["## LPIPS-VGG (the CNN's own fine-tuning loss)", ""] + [f"- {k}: {v:.4f}" for k, v in sorted(lp.items())]
(out / "uncertainty.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
