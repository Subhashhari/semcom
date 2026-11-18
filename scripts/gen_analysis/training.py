"""Training analysis: how each run trained, from its logs (STEP / VAL lines) and history.json.

Used by scripts/gen_analysis/suite.py (section 12) and on its own, also mid-run (history.json is
only written at the end; the VAL lines carry the same validation fields):

  training.py OUT_DIR KEY=RUN_DIR[:LOG[,LOG...]] [KEY=...]
  e.g. training.py results/gen/analysis/p256-live dit256=results/gen/p256-conv/dit-S-v_cbr48_qam16_blind-5to5_s0:logs/gen_256_try2.log,logs/gen_256.log

Logs (plain text, or Kaggle kernel logs in their JSON form) are read in the order given; a step logged twice (a run resumed from a checkpoint redoes
the steps after it) keeps the later log's value. Only lines of that run's name are used, so one
queue log can serve several runs.

Writes OUT_DIR/training.md and its figures:
  training_loss.png         training loss (every 100 steps; mean over 1k steps) and validation loss
  training_jitter.png       loss jitter (coefficient of variation over 2k steps), gradient norm,
                            AMP-skipped steps
  training_val.png          validation PSNR / DINOv2 similarity at each SNR (32 images, EMA weights)
                            and the EMA-vs-raw-weights gap
  training_convergence.png  validation-loss gain per validation against the stopping rule
  training_diffusion.png    generative runs: conditioning gain (what the channel signal is worth to
                            the denoiser) and sampled-latent spread (1 = the VAE's own statistics)
  training_samples_<key>.png the run's validation samples over training (samples/step*.png)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
STYLE = {"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
         "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False, "figure.facecolor": BG,
         "axes.facecolor": BG, "savefig.dpi": 130}
SMOOTH, JITTER, TAIL = 10, 20, 10_000  # log points (100 steps each) for the mean and the CV; tail steps
NUM = re.compile(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$")


def parse_line(line: str) -> tuple[str, dict]:
    kind, _, rest = line.partition(" ")
    out = {}
    for tok in rest.split():
        k, eq, v = tok.partition("=")
        if eq and NUM.match(v):
            out[k] = float(v)
        elif eq:
            out[k] = v
    return kind, out


def read_logs(name: str, logs: list[Path]) -> tuple[dict[int, dict], dict[int, dict]]:
    """STEP and VAL entries of run `name`, by step; later logs win."""
    steps, vals = {}, {}
    for f in logs:
        text = f.read_text(encoding="utf-8", errors="replace")
        if text.lstrip().startswith("[{"):  # a Kaggle kernel log: JSON records of output chunks
            try:
                text = "".join(r.get("data", "") for r in json.loads(text))
            except json.JSONDecodeError:
                pass
        for line in text.replace("\r", "\n").splitlines():
            if not (line.startswith("STEP run=") or line.startswith("VAL run=")):
                continue
            kind, d = parse_line(line)
            if d.get("run") != name or "step" not in d:
                continue
            (steps if kind == "STEP" else vals)[int(d["step"])] = d
    return steps, vals


def load(run: Path, logs: list[Path]) -> dict:
    steps, vals = read_logs(run.name, logs)
    hist_f = run / "history.json"
    finished = hist_f.exists()
    if finished:  # authoritative, and complete even if a log is missing
        h = json.loads(hist_f.read_text())
        vals = {int(e["step"]): {"val_loss": e["val_loss"], **e.get("quality", {}),
                                 **({"cond_gain": e["cond_gain"]} if "cond_gain" in e else {})}
                for e in h["history"]}
        summary = h.get("summary", {})
    else:
        summary = {}
    s = np.array(sorted(steps))
    v = np.array(sorted(vals))
    # The trainer's skipped-step counter restarts at 0 in each process (resume): accumulate it.
    skipped, off, prev = [], 0.0, 0.0
    for i in s:
        x = steps[i].get("skipped", 0)
        if x < prev:
            off += prev
        prev = x
        skipped.append(off + x)
    return {"run": run, "finished": finished, "summary": summary,
            "step": s, "loss": np.array([steps[i]["loss"] for i in s]) if len(s) else np.array([]),
            "gnorm": np.array([steps[i].get("gnorm", np.nan) for i in s]) if len(s) else np.array([]),
            "skipped": np.array(skipped),
            "it_s": np.array([steps[i].get("it_s", np.nan) for i in s]) if len(s) else np.array([]),
            "vstep": v, "val": [vals[i] for i in v],
            "generative": not run.name.startswith("cnn")}


def rolling(x: np.ndarray, w: int, fn) -> np.ndarray:
    return np.array([fn(x[max(0, i - w + 1):i + 1]) for i in range(len(x))]) if len(x) else x


def rule_state(vloss: list[float], patience: int, min_delta: float) -> tuple[list[float], int, bool]:
    """Per validation, the relative gain over the best before it (%); validations since the last
    qualifying best; converged (as semcom.gen.train.converged)."""
    best, best_i, gains = float("inf"), -1, []
    for i, x in enumerate(vloss):
        gains.append(100 * (best - x) / best if np.isfinite(best) else np.nan)
        if x < best * (1 - min_delta):
            best, best_i = x, i
    since = len(vloss) - 1 - best_i
    return gains, since, bool(patience) and since >= patience


def analyse(systems: list[dict], out: Path, colors: dict | None = None) -> list[str]:
    """systems: [{"key", "label", "run": Path, "logs": [Path, ...]}]. Returns markdown lines."""
    plt.rcParams.update(STYLE)
    out.mkdir(parents=True, exist_ok=True)
    colors = colors or {s["key"]: SLOTS[i % len(SLOTS)] for i, s in enumerate(systems)}
    T = {s["key"]: load(Path(s["run"]), [Path(p) for p in s.get("logs", [])]) for s in systems}
    T = {k: t for k, t in T.items() if len(t["step"]) or len(t["vstep"])}
    label = {s["key"]: s["label"] for s in systems}
    md: list[str] = []
    if not T:
        return ["No training logs or histories found.", ""]

    def save(fig, name):
        fig.savefig(out / name, bbox_inches="tight")
        plt.close(fig)
        md.extend([f"![{name}]({name})", ""])

    # ---- summary table
    md += ["Training loss is logged every 100 steps (mean over those steps); validation uses the EMA weights: "
           "loss on the validation split, quality on 32 fixed validation images (noisy: compare trends, "
           "not single points). Loss jitter = standard deviation / mean of the logged training loss over "
           "2k-step windows.", "",
           "| Run | Steps | State | Best val loss (step) | Last val loss | Validations since best | "
           f"Loss jitter, last {TAIL // 1000}k | Grad norm, last {TAIL // 1000}k | AMP-skipped | "
           "Val DINOv2 @0 dB (best / last) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for k, t in T.items():
        cfgf = t["run"] / "config.yaml"
        pat, md_ = 0, 0.002
        if cfgf.exists():
            txt = cfgf.read_text()
            m1, m2 = re.search(r"^patience: (\d+)", txt, re.M), re.search(r"^min_delta: ([\d.e-]+)", txt, re.M)
            pat, md_ = int(m1.group(1)) if m1 else 0, float(m2.group(1)) if m2 else 0.002
        vl = [e["val_loss"] for e in t["val"]]
        _, since, conv = rule_state(vl, pat, md_) if vl else ([], 0, False)
        bi = int(np.argmin(vl)) if vl else None
        tail = t["step"] >= (t["step"].max() - TAIL) if len(t["step"]) else np.array([], bool)
        cv = np.std(t["loss"][tail]) / np.mean(t["loss"][tail]) if tail.any() else np.nan
        last_step = int(max(t["step"].max() if len(t["step"]) else 0, t["vstep"].max() if len(t["vstep"]) else 0))
        state = ("finished" if t["finished"] else "running") + (", converged" if conv else "")
        d0 = [e.get("dino@0") for e in t["val"]]
        md.append(f"| {label[k]} | {last_step:,} | {state} | "
                  + (f"{vl[bi]:.4f} ({int(t['vstep'][bi]):,}) | {vl[-1]:.4f} | {since}"
                     + (f" of {pat}" if pat else "") if vl else "- | - | -")
                  + f" | {100 * cv:.1f}% | {np.nanmedian(t['gnorm'][tail]) if tail.any() else np.nan:.3f} | "
                  f"{int(t['skipped'].max()) if len(t['skipped']) else 0} | "
                  + (f"{d0[bi]:.3f} / {d0[-1]:.3f}" if vl and d0[-1] is not None and d0[bi] is not None else "-")
                  + " |")
    md.append("")

    # ---- loss
    n = len(T)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 3.3), squeeze=False)
    for ax, (k, t) in zip(axes[0], T.items()):
        if len(t["step"]):
            ax.plot(t["step"], t["loss"], color=colors[k], lw=0.6, alpha=0.3)
            ax.plot(t["step"], rolling(t["loss"], SMOOTH, np.mean), color=colors[k], lw=1.6,
                    label="train (1k-step mean)")
            lo, hi = np.percentile(t["loss"][len(t["loss"]) // 20:], [1, 99]) if len(t["loss"]) > 40 else (None, None)
        else:
            lo = hi = None
        if len(t["vstep"]):
            vl = [e["val_loss"] for e in t["val"]]
            ax.plot(t["vstep"], vl, color=INK, lw=1.2, marker="o", ms=2.5, label="validation (EMA)")
            if lo is not None:
                lo, hi = min(lo, min(vl[len(vl) // 20:])), max(hi, max(vl[len(vl) // 20:]))
        if lo is not None:
            pad = 0.08 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(label[k], fontsize=9, loc="left")
        ax.set_xlabel("step")
        ax.legend(fontsize=7)
    axes[0, 0].set_ylabel("loss (each run's own objective)")
    fig.tight_layout()
    save(fig, "training_loss.png")

    # ---- jitter
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.3))
    for k, t in T.items():
        if not len(t["step"]):
            continue
        cv = rolling(t["loss"], JITTER, lambda w: np.std(w) / np.mean(w) * 100 if len(w) > 3 else np.nan)
        axes[0].plot(t["step"], cv, color=colors[k], lw=1.4, label=label[k])
        g = t["gnorm"]
        axes[1].plot(t["step"], rolling(g, SMOOTH, np.nanmedian), color=colors[k], lw=1.4)
        axes[1].fill_between(t["step"], rolling(g, SMOOTH, lambda w: np.nanpercentile(w, 10)),
                             rolling(g, SMOOTH, lambda w: np.nanpercentile(w, 90)), color=colors[k], alpha=0.15, lw=0)
        axes[2].plot(t["step"], t["skipped"], color=colors[k], lw=1.4)
    for ax, ttl in zip(axes, ["training-loss jitter (CV over 2k steps, %)", "gradient norm (median, 10-90% band)",
                              "AMP-skipped steps (cumulative)"]):
        ax.set_title(ttl, fontsize=9, loc="left")
        ax.set_xlabel("step")
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(fontsize=7)
    fig.tight_layout()
    save(fig, "training_jitter.png")

    # ---- validation quality
    snrs = sorted({float(m.group(1)) for t in T.values() for e in t["val"] for key in e
                   for m in [re.match(r"^dino@(-?[\d.]+)$", key)] if m})
    if snrs:
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4))
        styles = ["-", "--", ":"]
        for k, t in T.items():
            for j, s in enumerate(snrs[:3]):
                key = f"@{s:g}"
                st = [x for x, e in zip(t["vstep"], t["val"]) if f"dino{key}" in e]
                if not st:
                    continue
                axes[0].plot(st, [e[f"dino{key}"] for e in t["val"] if f"dino{key}" in e], color=colors[k],
                             ls=styles[j], lw=1.4, label=f"{label[k]}, {s:g} dB")
                axes[1].plot(st, [e[f"psnr{key}"] for e in t["val"] if f"psnr{key}" in e], color=colors[k],
                             ls=styles[j], lw=1.4)
            gap = [(x, e["dino@0"] - e["raw_dino@0"]) for x, e in zip(t["vstep"], t["val"])
                   if "dino@0" in e and "raw_dino@0" in e]
            if gap:
                axes[2].plot(*zip(*gap), color=colors[k], lw=1.4)
        axes[2].axhline(0, color=INK2, lw=0.8)
        for ax, ttl in zip(axes, ["val DINOv2 similarity (32 images)", "val PSNR (dB)",
                                  "EMA minus raw weights, DINOv2 @0 dB"]):
            ax.set_title(ttl, fontsize=9, loc="left")
            ax.set_xlabel("step")
        axes[0].legend(fontsize=6.5, ncol=1)
        fig.tight_layout()
        save(fig, "training_val.png")
        md += ["Line style is the SNR (solid / dashed / dotted = " + " / ".join(f"{s:g}" for s in snrs[:3])
               + " dB). Above zero in the right panel, the EMA weights the run is evaluated with beat its raw "
               "weights.", ""]

    # ---- convergence
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    thr = None
    for k, t in T.items():
        vl = [e["val_loss"] for e in t["val"]]
        if len(vl) < 2:
            continue
        gains, _, _ = rule_state(vl, 0, 0.002)
        ax.plot(t["vstep"][1:], gains[1:], color=colors[k], lw=1.4, marker="o", ms=2.5, label=label[k])
        thr = 0.2
    if thr is not None:
        ax.axhline(thr, color=INK2, lw=1, ls="--")
        ax.text(ax.get_xlim()[1], thr, " stop rule:\n 0.2%", fontsize=7, color=INK2, va="center")
        ax.set_ylim(-0.5, max(2.0, ax.get_ylim()[1] if ax.get_ylim()[1] < 3 else 3))
        ax.set_title("val-loss gain over the best so far, per validation (%)", fontsize=9, loc="left")
        ax.set_xlabel("step")
        ax.legend(fontsize=7)
        fig.tight_layout()
        save(fig, "training_convergence.png")
        md += ["A run stops once `patience` validations in a row stay under the dashed line (no gain above "
               "0.2% over its best). Points near the line mean it is still improving slowly.", ""]
    else:
        plt.close(fig)

    # ---- diffusion diagnostics
    gen = {k: t for k, t in T.items() if t["generative"] and any("cond_gain" in e for e in t["val"])}
    if gen:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
        for k, t in gen.items():
            axes[0].plot([x for x, e in zip(t["vstep"], t["val"]) if "cond_gain" in e],
                         [e["cond_gain"] for e in t["val"] if "cond_gain" in e], color=colors[k], lw=1.4,
                         label=label[k])
            for j, s in enumerate(snrs[:3]):
                z = [(x, e[f"zstd@{s:g}"]) for x, e in zip(t["vstep"], t["val"]) if f"zstd@{s:g}" in e]
                if z:
                    axes[1].plot(*zip(*z), color=colors[k], ls=["-", "--", ":"][j], lw=1.4)
        axes[1].axhline(1, color=INK2, lw=0.8)
        for ax, ttl in zip(axes, ["conditioning gain (unconditional minus conditional val loss)",
                                  "sampled-latent spread / true (1 = matches the VAE)"]):
            ax.set_title(ttl, fontsize=9, loc="left")
            ax.set_xlabel("step")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        save(fig, "training_diffusion.png")

    # ---- samples over training
    for k, t in T.items():
        files = sorted((t["run"] / "samples").glob("step*.png"))
        if not files:
            continue
        pick = [files[int(round(i))] for i in np.linspace(0, len(files) - 1, min(4, len(files)))]
        pick = list(dict.fromkeys(pick))
        imgs = [plt.imread(f) for f in pick]
        # Each snapshot is the trainer's grid: originals on top, then one row per validation SNR.
        fig, axes = plt.subplots(len(imgs), 1, squeeze=False,
                                 figsize=(10, sum(10 * i.shape[0] / i.shape[1] + 0.35 for i in imgs)))
        for ax, im, f in zip(axes[:, 0], imgs, pick):
            ax.imshow(im)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_title(f"step {int(f.stem[4:]):,}", fontsize=8, loc="left")
        fig.tight_layout()
        md += [f"**{label[k]}**: validation samples over training. In each block the top row is the original, "
               "then one row per validation SNR, low to high. These are the trainer's 8 fixed validation images "
               "(the first ones of the validation split, so mostly one class); the class-spread images are in "
               "section 7.", ""]
        save(fig, f"training_samples_{k}.png")
    return md


if __name__ == "__main__":
    out = Path(sys.argv[1])
    systems = []
    for i, a in enumerate(sys.argv[2:]):
        key, _, rest = a.partition("=")
        # RUN_DIR:LOGS; ";" also accepted (Git Bash on Windows rewrites "a:b" path lists to "a;b").
        m = re.match(r"^((?:[A-Za-z]:[\\/])?[^:;]*)(?:[:;](.*))?$", rest)
        run, logs = m.group(1), m.group(2) or ""
        systems.append({"key": key, "label": key, "run": run, "logs": [p for p in re.split(r"[,;]", logs) if p]})
    lines = ["# Training analysis", ""] + analyse(systems, out)
    (out / "training.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out / 'training.md'}")
