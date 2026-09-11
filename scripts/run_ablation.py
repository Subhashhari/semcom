"""Run the full 2x2 ablation and emit the comparison figure and tables.

    python scripts/run_ablation.py --config configs/cifar_r12.yaml

Trains, in order:
  * two adaptive arms   - ADJSCC (analog) and ADJSCC-Q (digital), one model each
  * two specialist sets - BDJSCC and DeepJSCC-Q, one model per fixed SNR

then evaluates every run over the full SNR sweep and answers the three questions the
ablation exists to answer:

  1. Does ADJSCC-Q beat every DeepJSCC-Q specialist across the range, including at each
     specialist's own matched point?
  2. Is the ADJSCC-Q -> DeepJSCC-Q gap larger than the ADJSCC -> BDJSCC gap? (The
     hypothesis: quantisation starves the encoder further, so allocation should matter
     more, not less.)
  3. What does one adaptive model cost in storage versus the specialist ensemble?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semcom.config import Config  # noqa: E402
from semcom.evaluate import evaluate_run  # noqa: E402
from semcom.separation import (  # noqa: E402
    choose_fixed_mcs,
    cifar_test_images,
    evaluate_separation,
    measure_codec_floor,
)
from semcom.train import train  # noqa: E402

SPECIALIST_SNRS = [1.0, 4.0, 7.0, 13.0, 19.0]


def arm_configs(base: Config, specialist_snrs=SPECIALIST_SNRS) -> list[Config]:
    """The full set of runs: 2 adaptive models + 2 x len(specialist_snrs) specialists."""
    runs = [
        base.replace(snr_adaptive=True, digital=False),
        base.replace(snr_adaptive=True, digital=True),
    ]
    for snr in specialist_snrs:
        runs.append(base.replace(snr_adaptive=False, digital=False, snr_train_fixed=snr))
        runs.append(base.replace(snr_adaptive=False, digital=True, snr_train_fixed=snr))
    return runs


def envelope(results: list[dict], snrs: list[float]) -> dict[float, float]:
    """Best PSNR achieved by any run in the set, at each SNR.

    This is the fair way to show a specialist ensemble: it assumes an oracle that always
    picks the right specialist for the current channel, which is strictly more generous
    than any real system could be.
    """
    return {
        s: max(r["psnr_by_snr"][str(s)] if str(s) in r["psnr_by_snr"] else r["psnr_by_snr"][s]
               for r in results)
        for s in snrs
    }


def curve_of(result: dict, snrs: list[float]) -> dict[float, float]:
    by_snr = result["psnr_by_snr"]
    return {s: by_snr[str(s)] if str(s) in by_snr else by_snr[s] for s in snrs}


def analyse(results: list[dict], snrs: list[float]) -> dict:
    by_arm: dict[str, list[dict]] = {}
    for r in results:
        by_arm.setdefault(r["arm"], []).append(r)

    missing = {"ADJSCC", "ADJSCC-Q", "BDJSCC", "DeepJSCC-Q"} - set(by_arm)
    if missing:
        raise RuntimeError(f"ablation incomplete; missing arms: {sorted(missing)}")

    adjscc_q = curve_of(by_arm["ADJSCC-Q"][0], snrs)
    adjscc = curve_of(by_arm["ADJSCC"][0], snrs)
    deepjscc_q_env = envelope(by_arm["DeepJSCC-Q"], snrs)
    bdjscc_env = envelope(by_arm["BDJSCC"], snrs)

    # Q1: the matched-point test. For each digital specialist, compare at the SNR it was
    # trained for - the specialist's home turf, and the sharpest version of the claim.
    matched = []
    for r in by_arm["DeepJSCC-Q"]:
        snr = r["snr_train_fixed"]
        if snr is None:
            continue
        if snr not in adjscc_q:
            # Loudly, not silently. Skipping here would leave `matched` short or empty,
            # and `all([])` is True - so a silent skip reports "adaptive wins everywhere"
            # on zero comparisons, which is the most misleading answer available.
            raise RuntimeError(
                f"specialist trained at {snr} dB was not evaluated at that SNR "
                f"(eval grid: {sorted(adjscc_q)}). The matched-point test cannot run."
            )
        specialist = curve_of(r, [snr])[snr]
        matched.append(
            {
                "snr_db": snr,
                "specialist_psnr": specialist,
                "adjscc_q_psnr": adjscc_q[snr],
                "adaptive_wins": adjscc_q[snr] > specialist,
                "margin_db": adjscc_q[snr] - specialist,
            }
        )

    # Q2: does conditioning pay more once the channel input is quantised?
    digital_gain = sum(adjscc_q[s] - deepjscc_q_env[s] for s in snrs) / len(snrs)
    analog_gain = sum(adjscc[s] - bdjscc_env[s] for s in snrs) / len(snrs)

    # Q3: storage. One adaptive model against the specialist ensemble it replaces.
    def megabytes(params: int) -> float:
        return params * 4 / (1024**2)

    adaptive_params = by_arm["ADJSCC-Q"][0]["parameters"]
    ensemble_params = sum(r["parameters"] for r in by_arm["DeepJSCC-Q"])

    return {
        "curves": {
            "ADJSCC-Q": adjscc_q,
            "ADJSCC": adjscc,
            "DeepJSCC-Q (specialist envelope)": deepjscc_q_env,
            "BDJSCC (specialist envelope)": bdjscc_env,
        },
        "q1_matched_point": {
            "per_snr": matched,
            # Explicitly requires at least one comparison: a vacuous truth here would be
            # read as the paper's headline claim reproducing.
            "adaptive_wins_everywhere": bool(matched) and all(m["adaptive_wins"] for m in matched),
            "wins": sum(m["adaptive_wins"] for m in matched),
            "of": len(matched),
        },
        "q2_conditioning_gain_db": {
            "digital (ADJSCC-Q - DeepJSCC-Q)": digital_gain,
            "analog (ADJSCC - BDJSCC)": analog_gain,
            "quantisation_amplifies_conditioning": digital_gain > analog_gain,
        },
        "q3_storage": {
            "adaptive_models": 1,
            "adaptive_mb": megabytes(adaptive_params),
            "adaptive_mean_psnr": sum(adjscc_q.values()) / len(snrs),
            "ensemble_models": len(by_arm["DeepJSCC-Q"]),
            "ensemble_mb": megabytes(ensemble_params),
            "ensemble_mean_psnr": sum(deepjscc_q_env.values()) / len(snrs),
            "storage_fraction": adaptive_params / max(ensemble_params, 1),
        },
    }


def separation_reference(k: int, snrs: list[float], n_images: int = 64) -> dict:
    """The external reference point: classical source + channel coding, same symbol budget.

    Without this the 2x2 is entirely self-referential — it can say whether conditioning
    survives quantisation, but not whether any of it beats what a conventional radio
    already does, and it cannot show the cliff at all.
    """
    images = cifar_test_images(n_images)
    floor = measure_codec_floor(images)

    ideal = evaluate_separation(images, k=k, snrs=snrs, mode="ideal")

    mcs = choose_fixed_mcs(k, snrs, floor)
    fixed = None
    if mcs is not None:
        fixed = evaluate_separation(
            images, k=k, snrs=snrs, mode="fixed_mcs",
            bits_per_symbol=mcs["bits_per_symbol"], code_rate=mcs["code_rate"],
        )

    return {"codec_floor_bytes": floor, "mcs": mcs, "ideal": ideal, "fixed_mcs": fixed}


def separation_curve(sweep: dict | None, require_all: bool = True) -> dict[float, float]:
    """Extract a plottable {snr: psnr} curve for the separation reference.

    Drops SNRs where nothing was delivered, and by default also drops SNRs where only
    *some* images could be encoded. That second filter matters more than it looks: when
    the byte budget sits near the codec floor, the images that still fit are precisely
    the most compressible ones, so a mean over that subset is biased upward. Plotting it
    made the rate-distortion curve non-monotonic — quality appeared to *fall* as the
    channel improved, purely because the harder images rejoined the average.
    """
    if not sweep:
        return {}
    return {
        s: v["psnr"]
        for s, v in sweep["by_snr"].items()
        if v["psnr"] is not None and (not require_all or v.get("feasible_fraction", 1.0) == 1.0)
    }


def plot(analysis: dict, out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed; skipping figure")
        return

    styles = {
        "ADJSCC-Q": dict(color="#c2410c", lw=2.4, marker="o", ms=4, zorder=5),
        "ADJSCC": dict(color="#0369a1", lw=1.8, marker="s", ms=3),
        "DeepJSCC-Q (specialist envelope)": dict(color="#c2410c", lw=1.6, ls="--"),
        "BDJSCC (specialist envelope)": dict(color="#0369a1", lw=1.6, ls="--"),
    }

    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)
    for label, curve in analysis["curves"].items():
        snrs = sorted(curve)
        ax.plot(snrs, [curve[s] for s in snrs], label=label, **styles.get(label, {}))

    # The external reference, drawn with gaps rather than interpolation: where the
    # separation scheme delivers nothing, the line should be absent, not imputed.
    sep = analysis.get("separation")
    if sep:
        ideal = separation_curve(sep.get("ideal"))
        if ideal:
            xs = sorted(ideal)
            ax.plot(xs, [ideal[s] for s in xs], color="#3f3f46", lw=1.6, ls=":",
                    marker="^", ms=3, label="Separation (capacity bound, oracle AMC)")
        fixed = separation_curve(sep.get("fixed_mcs"))
        if fixed:
            xs = sorted(fixed)
            mcs = sep.get("mcs") or {}
            ax.plot(xs, [fixed[s] for s in xs], color="#3f3f46", lw=2.0,
                    label=f"Separation (fixed {mcs.get('label', 'MCS')}) — the cliff")
            if mcs.get("threshold_db") is not None:
                ax.axvline(mcs["threshold_db"], color="#3f3f46", lw=0.8, alpha=0.5)

    ax.set_xlabel("Test SNR (dB)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("ADJSCC-Q ablation: does SNR conditioning survive quantisation?")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"figure -> {out_path}")


def report(analysis: dict) -> None:
    q1 = analysis["q1_matched_point"]
    print("\n=== Q1: matched-point test (ADJSCC-Q vs each DeepJSCC-Q specialist) ===")
    print(f"{'SNR':>6} {'specialist':>12} {'ADJSCC-Q':>10} {'margin':>9}")
    for m in q1["per_snr"]:
        flag = "" if m["adaptive_wins"] else "   <- specialist wins"
        print(
            f"{m['snr_db']:6.1f} {m['specialist_psnr']:12.3f} "
            f"{m['adjscc_q_psnr']:10.3f} {m['margin_db']:+9.3f}{flag}"
        )
    print(f"adaptive wins at {q1['wins']}/{q1['of']} matched points")

    q2 = analysis["q2_conditioning_gain_db"]
    print("\n=== Q2: does quantisation amplify the value of conditioning? ===")
    print(f"  digital  (ADJSCC-Q - DeepJSCC-Q): {q2['digital (ADJSCC-Q - DeepJSCC-Q)']:+.3f} dB")
    print(f"  analog   (ADJSCC   - BDJSCC)    : {q2['analog (ADJSCC - BDJSCC)']:+.3f} dB")
    print(f"  hypothesis holds: {q2['quantisation_amplifies_conditioning']}")

    sep = analysis.get("separation")
    if sep:
        print("\n=== External reference: separation-based coding ===")
        print(f"  CIFAR-10 codec floor: {sep['codec_floor_bytes']:.0f} bytes/image")
        ideal = separation_curve(sep.get("ideal"))
        if ideal:
            lo, hi = min(ideal), max(ideal)
            print(f"  capacity bound (oracle AMC): {ideal[lo]:.2f} dB @ {lo:g} dB"
                  f"  ->  {ideal[hi]:.2f} dB @ {hi:g} dB")
        blocked = [s for s, v in sep["ideal"]["by_snr"].items() if v["psnr"] is None]
        if blocked:
            print(f"  infeasible at or below {max(blocked):g} dB — the 32x32 codec"
                  f" floor, not the channel")
        mcs, fixed_sweep = sep.get("mcs"), sep.get("fixed_mcs")
        if mcs and fixed_sweep:
            fixed = separation_curve(fixed_sweep)
            flat = next(iter(fixed.values())) if fixed else float("nan")
            print(f"  fixed {mcs['label']}: cliff at {mcs['threshold_db']:.2f} dB,"
                  f" flat {flat:.2f} dB above it (surplus SNR wasted)")
        else:
            print("  no MCS puts a cliff inside the swept range at this rate")

    q3 = analysis["q3_storage"]
    print("\n=== Q3: storage ===")
    print(f"{'strategy':>28} {'models':>7} {'MB':>9} {'mean PSNR':>11}")
    print(f"{'ADJSCC-Q (adaptive)':>28} {q3['adaptive_models']:7d} "
          f"{q3['adaptive_mb']:9.2f} {q3['adaptive_mean_psnr']:11.3f}")
    print(f"{'DeepJSCC-Q (ensemble)':>28} {q3['ensemble_models']:7d} "
          f"{q3['ensemble_mb']:9.2f} {q3['ensemble_mean_psnr']:11.3f}")
    print(f"  adaptive uses {q3['storage_fraction']:.2%} of the ensemble's storage")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--skip-training", action="store_true", help="evaluate existing runs only")
    p.add_argument("--no-wandb", dest="wandb", action="store_false")
    p.add_argument("-M", "--modulation-order", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--no-separation", action="store_true",
                   help="skip the classical separation reference curves")
    args = p.parse_args()

    overrides = {"modulation_order": args.modulation_order, "epochs": args.epochs}
    if not args.wandb:
        overrides["wandb"] = False
    base = Config.from_yaml(args.config, **overrides)
    runs = arm_configs(base)

    print(f"ablation: {len(runs)} runs at R={base.bandwidth_ratio:.4f}\n")
    for i, cfg in enumerate(runs, 1):
        print(f"--- [{i}/{len(runs)}] {cfg.run_name}")
        if not args.skip_training:
            train(cfg)

    results = [evaluate_run(cfg.run_dir, use_wandb=args.wandb) for cfg in runs]

    analysis = analyse(results, base.eval_snrs)
    if not args.no_separation:
        print("\ncomputing separation baseline ...")
        analysis["separation"] = separation_reference(base.k, base.eval_snrs)

    out_dir = Path(base.out_dir)
    (out_dir / "ablation.json").write_text(
        json.dumps({"results": results, "analysis": analysis}, indent=2), encoding="utf-8"
    )
    plot(analysis, out_dir / "ablation.png")
    report(analysis)
    print(f"\nfull results -> {out_dir / 'ablation.json'}")


if __name__ == "__main__":
    main()
