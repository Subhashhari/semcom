"""Did each arm actually converge, or did the epoch budget decide the comparison?

The 2x2 gives every arm the same number of epochs, which keeps the comparison fair only
if every arm is near its own asymptote at the end. It is not automatically so: an adaptive
model fits a harder function (all SNRs) than a specialist (one SNR), so at equal epochs it
sits further from convergence. That asymmetry pushes the matched-point test toward the
specialists for reasons that have nothing to do with the mechanism under study.

This reads the run records only - no GPU, no model loading - and reports, per run, how
much validation PSNR was still being gained at the end. Interpretation:

  adaptive still gaining, specialists flat  -> budget-limited; re-run the adaptive arms
                                               longer before believing a null result
  everything flat                           -> the null result is real

Usage:  python scripts/convergence_report.py [results/adjscc-q/r12]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# How many validation checkpoints back to measure the tail gain over. At eval_every=10
# this is the last 30 epochs, long enough that per-eval noise does not dominate.
TAIL = 4

# Below this, a run is treated as having stopped improving. Chosen well under the ~0.1 dB
# margins the matched-point test turns on, so "flat" never hides a decisive gain.
FLAT_DB = 0.05


def load(run_dir: Path) -> dict | None:
    f = run_dir / "history.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text(encoding="utf-8"))
    evals = [e for e in d["history"] if "val_psnr" in e]
    if len(evals) < 2:
        return None
    tail = evals[-min(TAIL, len(evals)) :]
    s = d["summary"]
    return {
        "arm": s["arm"],
        "snr": s.get("snr_train_fixed"),
        "best_epoch": s["best_epoch"],
        "epochs_trained": s["epochs_trained"],
        "final_val": evals[-1]["val_psnr"],
        "tail_gain": tail[-1]["val_psnr"] - tail[0]["val_psnr"],
        "tail_epochs": tail[-1]["epoch"] - tail[0]["epoch"],
        "adaptive": s.get("snr_train_fixed") is None,
    }


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/adjscc-q/r12")
    runs = [r for r in (load(d) for d in sorted(root.iterdir()) if d.is_dir()) if r]
    if not runs:
        print(f"no run records under {root}")
        return

    span = runs[0]["tail_epochs"]
    print(f"convergence over the final {span} epochs ({root})\n")
    print(f"{'arm':<12} {'train SNR':>9} {'best ep':>8} {'of':>5} {'final val':>10} {'tail gain':>10}")
    for r in runs:
        snr = "adaptive" if r["adaptive"] else f"{r['snr']:g} dB"
        mark = "" if abs(r["tail_gain"]) < FLAT_DB else "   <- still climbing"
        print(
            f"{r['arm']:<12} {snr:>9} {r['best_epoch']:>8} {r['epochs_trained']:>5} "
            f"{r['final_val']:>10.3f} {r['tail_gain']:>+10.3f}{mark}"
        )

    adaptive = [r for r in runs if r["adaptive"]]
    special = [r for r in runs if not r["adaptive"]]
    if not (adaptive and special):
        return

    a = sum(r["tail_gain"] for r in adaptive) / len(adaptive)
    s = sum(r["tail_gain"] for r in special) / len(special)
    print(f"\nmean tail gain:  adaptive {a:+.3f} dB   specialists {s:+.3f} dB")

    if a > FLAT_DB and s < FLAT_DB:
        print(
            "\nBUDGET-LIMITED. The adaptive arms were still improving while the specialists\n"
            "had plateaued, so the equal-epoch budget favours the specialists and the\n"
            "matched-point loss is not attributable to the mechanism. Re-run the two\n"
            "adaptive arms longer before reporting a null result."
        )
    elif a < FLAT_DB and s < FLAT_DB:
        print(
            "\nCONVERGED. Every arm had stopped improving, so the epoch budget is not the\n"
            "explanation and the null result stands on its own."
        )
    else:
        print(
            "\nINCONCLUSIVE. The specialists had not plateaued either, so neither side of\n"
            "the comparison is at its asymptote. Both need a longer budget."
        )


if __name__ == "__main__":
    main()
