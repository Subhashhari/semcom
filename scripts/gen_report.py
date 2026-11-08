"""Tables and figures for the generative-JSCC runs: specialists vs the SNR-adaptive arms.

    python scripts/gen_report.py [--root results/gen/p128] [--steps 20] [--out results/gen/report]

Reads every <root>/<run>/eval.json (and history.json for validation curves) and writes:
    report.md          one table per metric: runs x test SNR, plus the "specialist envelope"
                       (each fixed-SNR specialist scored at its own training SNR) and the gap of the
                       told / blind arms to it, which is the week-2 question
    metrics_vs_snr.png PSNR, LPIPS, DINO, FID against test SNR, one panel each (no shared axis)
    val_curves.png     EMA validation PSNR/DINO against training step, per run
DiT results are read at --steps DDIM steps; CNN runs have one result per SNR.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRICS = [("psnr", "PSNR (dB)", True), ("lpips", "LPIPS", False), ("dino_sim", "DINOv2 similarity", True),
           ("fid", "FID", False)]
# Reference categorical palette (dataviz skill), fixed slot order: envelope, told, blind, CNN, extra.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4", "#008300", "#e34948", "#eda100"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(root: Path, steps: int) -> dict[str, dict]:
    runs = {}
    for ev in sorted(root.glob("*/eval.json")):
        e = json.loads(ev.read_text())
        by_snr = {}
        for key, r in e["results"].items():
            parts = key.split("_")
            snr = float(parts[0])
            if len(parts) > 1 and parts[1] not in ("None", "") and int(parts[1]) != steps:
                continue
            by_snr[snr] = r
        hist = ev.parent / "history.json"
        runs[e["run_name"]] = {"model": e["model"], "arm": e["arm"], "by_snr": by_snr,
                               "history": json.loads(hist.read_text())["history"] if hist.exists() else []}
    return runs


def spec_snr(arm: str) -> float | None:
    m = re.fullmatch(r"spec(-?\d+(?:\.\d+)?)", arm)
    return float(m.group(1)) if m else None


def envelope(runs: dict, model: str) -> dict[float, dict]:
    """Each specialist of `model` scored at its own training SNR."""
    env = {}
    for r in runs.values():
        s = spec_snr(r["arm"])
        if r["model"] == model and s is not None and s in r["by_snr"]:
            env[s] = r["by_snr"][s]
    return env


def series(runs: dict) -> list[tuple[str, dict[float, dict], str]]:
    """(label, {snr: metrics}, linestyle) in fixed colour-slot order."""
    out = []
    dit = {n: r for n, r in runs.items() if r["model"] == "dit"}
    env = envelope(runs, "dit")
    if len(env) > 1:
        out.append(("DiT specialists (each at its SNR)", env, "-"))
    for kind in ("told", "blind"):
        for n, r in dit.items():
            if r["arm"].startswith(kind):
                out.append((f"DiT {kind} ({r['arm'][len(kind):]} dB)", r["by_snr"], "-"))
    for n, r in runs.items():
        if r["model"] == "cnn" and r["arm"] == "spec0":
            out.append(("CNN spec 0 dB", r["by_snr"], "-"))
    for n, r in dit.items():
        if r["arm"] == "spec0":
            out.append(("DiT spec 0 dB (all SNRs)", r["by_snr"], "--"))
    return out


def table(runs: dict, steps: int) -> str:
    snrs = sorted({s for r in runs.values() for s in r["by_snr"]})
    env = envelope(runs, "dit")
    lines = [f"# Generative JSCC results (DiT at {steps} DDIM steps)\n"]
    for key, label, higher in METRICS:
        lines.append(f"## {label} ({'higher' if higher else 'lower'} is better)\n")
        lines.append("| run | " + " | ".join(f"{s:+g} dB" for s in snrs) + " |")
        lines.append("|---" * (len(snrs) + 1) + "|")
        for name, r in sorted(runs.items()):
            cells = [f"{r['by_snr'][s][key]:.3f}" if s in r["by_snr"] else "" for s in snrs]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        if len(env) > 1:
            lines.append("| **specialist envelope** | " + " | ".join(
                f"**{env[s][key]:.3f}**" if s in env else "" for s in snrs) + " |")
            for name, r in sorted(runs.items()):
                if r["model"] == "dit" and spec_snr(r["arm"]) is None:
                    gap = [(r["by_snr"][s][key] - env[s][key]) * (1 if higher else -1)
                           if s in env and s in r["by_snr"] else None for s in snrs]
                    lines.append(f"| {name} vs envelope | " + " | ".join(
                        "" if g is None else f"{g:+.3f}" for g in gap) + " |")
        lines.append("")
    lines.append("Gap rows are signed so that positive means the adaptive arm is better than the specialist.\n")
    return "\n".join(lines)


def style(ax, title: str):
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_metrics(runs: dict, path: Path, steps: int):
    import matplotlib.pyplot as plt

    ser = series(runs)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), facecolor="#fcfcfb")
    for ax, (key, label, higher) in zip(axes, METRICS):
        style(ax, f"{label} ({'↑' if higher else '↓'})")
        ax.set_facecolor("#fcfcfb")
        for i, (name, by, ls) in enumerate(ser):
            xs = sorted(by)
            ax.plot(xs, [by[s][key] for s in xs], ls, color=COLORS[i], lw=2, marker="o", ms=6,
                    markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=name)
        ax.set_xlabel("test SNR (dB)", color=MUTED, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=8)
    fig.suptitle(f"Reconstruction quality vs channel SNR (DiT: {steps} DDIM steps, CBR 1/48, 128 px)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_curves(runs: dict, path: Path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), facecolor="#fcfcfb")
    names = [n for n, r in sorted(runs.items()) if r["history"]]
    for ax, metric in zip(axes, ("psnr", "dino")):
        style(ax, f"validation {metric.upper()} (EMA weights), at the run's first validation SNR")
        ax.set_facecolor("#fcfcfb")
        for i, n in enumerate(names):
            h = [x for x in runs[n]["history"] if x.get("quality")]
            keys = [k for k in h[-1]["quality"] if k.startswith(metric + "@")] if h else []
            if not keys:
                continue
            ax.plot([x["step"] for x in h], [x["quality"].get(keys[0]) for x in h],
                    color=COLORS[i % len(COLORS)], lw=2, label=f"{n} [{keys[0]}]")
        ax.set_xlabel("training step", color=MUTED, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.06 * max(1, (len(labels) + 1) // 2), 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", type=Path,
                    help="results folder(s) of run directories (default results/gen/p128)")
    ap.add_argument("--steps", type=int, default=20, help="DDIM steps to report for DiT runs")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "gen" / "report")
    args = ap.parse_args()
    runs = {}
    for root in args.root or [ROOT / "results" / "gen" / "p128"]:
        runs.update(load(root, args.steps))
    if not runs:
        raise SystemExit("no eval.json found")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text(table(runs, args.steps), encoding="utf-8")
    plot_metrics(runs, args.out / "metrics_vs_snr.png", args.steps)
    plot_curves(runs, args.out / "val_curves.png")
    print(f"{len(runs)} runs -> {args.out}")


if __name__ == "__main__":
    main()
