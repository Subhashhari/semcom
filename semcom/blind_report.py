"""Apply the pre-registered blind-JSCC decision rules across seeds.

    python -m semcom.blind_report results/blind/r12 [--n-boot 10000]

Reads every run directory under the root that has a `blind_eval.json` (written by
`semcom.blind_eval`), pairs runs by seed, and reports:

  * the week-1 gate (proposal section 5.1): G = B - C-att at 18 dB and the plug-in energy
    penalty P, 16-QAM primary;
  * every arm's gap to B at each SNR, as paired t-intervals with equivalence verdicts
    (secondary; analog is secondary throughout);
  * the RQ2 cue-fit decision (section 4.3) when power-mismatch data exist.

Writes `blind_report.json` into the root.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .blind_stats import classify, cue_fit, gate

PRIMARY_SNR = 18.0
RQ2_ALPHAS = (1.1, 1.2, 1.3)


def _mod(rec: dict) -> str:
    return f"qam{rec['modulation_order']}" if rec["digital"] else "analog"


def load_runs(root: Path) -> dict:
    """{(arm, modulation): {seed: (run_dir, record)}} for every evaluated run."""
    runs: dict = defaultdict(dict)
    for path in sorted(Path(root).glob("*/blind_eval.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        runs[(rec["arm"], _mod(rec))][rec["seed"]] = (path.parent, rec)
    return runs


def paired(runs: dict, a: tuple, b: tuple) -> list:
    """Seeds present for both arms, in order."""
    return sorted(set(runs.get(a, {})) & set(runs.get(b, {})))


def gaps_to_b(runs: dict, mod: str) -> dict:
    """Every arm's per-seed gap to B (B minus arm) at each SNR, classified."""
    out = {}
    b_key = ("B", mod)
    for (arm, m), _ in runs.items():
        if m != mod or arm == "B":
            continue
        seeds = paired(runs, b_key, (arm, m))
        if len(seeds) < 2:
            continue
        snrs = runs[b_key][seeds[0]][1]["curve"].keys()
        out[arm] = {
            "seeds": seeds,
            "by_snr": {
                s: classify(
                    [runs[b_key][sd][1]["curve"][s] - runs[(arm, m)][sd][1]["curve"][s] for sd in seeds]
                )
                for s in snrs
            },
        }
    return out


def week1_gate(runs: dict, mod: str = "qam16") -> dict | None:
    b, c = ("B", mod), ("C-att", mod)
    seeds = paired(runs, b, c)
    if len(seeds) < 2:
        return None
    key = str(PRIMARY_SNR)
    g = [runs[b][s][1]["curve"][key] - runs[c][s][1]["curve"][key] for s in seeds]
    energy = {s: runs[b][s][1]["plugin"]["energy"] for s in seeds}
    snrs = list(energy[seeds[0]].keys())
    p_all = {snr: [energy[s][snr]["penalty"] for s in seeds] for snr in snrs}
    result = gate(g, p_all[key], p_all)
    result["seeds"] = seeds
    result["modulation"] = mod
    return result


def rq2(runs: dict, mod: str = "qam16", snr: float = PRIMARY_SNR, alphas=RQ2_ALPHAS,
        n_boot: int = 10_000) -> dict | None:
    """The cue-fit decision on the double-difference readout."""
    b, c = ("B", mod), ("C-att", mod)
    seeds = paired(runs, b, c)
    if len(seeds) < 2:
        return None
    r, preds = [], defaultdict(list)
    cues = None
    for s in seeds:
        bi = np.load(runs[b][s][0] / "blind_eval_images.npz")
        ci = np.load(runs[c][s][0] / "blind_eval_images.npz")
        if cues is None:
            prefix = f"pm_{snr:g}_1_"
            cues = [k[len(prefix):] for k in bi.files if k.startswith(prefix) and not k.endswith("own")]

        def own(arr_file, a):
            return arr_file[f"pm_{snr:g}_{a:g}_own"]

        r.append(np.stack(
            [(own(ci, a) - own(ci, 1)) - (own(bi, a) - own(bi, 1)) for a in alphas], axis=-1
        ))
        for cue in cues:
            preds[cue].append(np.stack(
                [
                    (bi[f"pm_{snr:g}_{a:g}_{cue}"] - own(bi, a))
                    - (bi[f"pm_{snr:g}_1_{cue}"] - own(bi, 1))
                    for a in alphas
                ],
                axis=-1,
            ))
    result = cue_fit(np.stack(r), {k: np.stack(v) for k, v in preds.items()}, n_boot=n_boot)
    result.update(seeds=seeds, snr=snr, alphas=list(alphas), modulation=mod)
    return result


def report(root: Path, n_boot: int = 10_000) -> dict:
    runs = load_runs(root)
    out = {
        "runs": {f"{a}|{m}": sorted(v) for (a, m), v in runs.items()},
        "gate_qam16": week1_gate(runs, "qam16"),
        "gaps_to_B": {m: gaps_to_b(runs, m) for m in ("qam16", "analog")},
        "rq2_qam16": rq2(runs, "qam16", n_boot=n_boot),
        "notes": [
            "Primary: 16-QAM at 18 dB. Analog is secondary: with 3 seeds it can support "
            "'clearly nonzero' claims, not 'no penalty' claims.",
            "P is an upper bound on the energy penalty (B was trained on the true SNR).",
        ],
    }
    (Path(root) / "blind_report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("runs:", {k: v for k, v in out["runs"].items()})
    g = out["gate_qam16"]
    if g:
        ci = g["G"]["ci95"]
        print(
            f"\nGATE (16-QAM, {PRIMARY_SNR:g} dB, seeds {g['seeds']}): {g['outcome']}\n"
            f"  G = B - C-att = {ci['mean']:+.3f} dB  95% CI [{ci['lo']:+.3f}, {ci['hi']:+.3f}]  "
            f"({g['G']['verdict']})\n"
            f"  P = {g['P']['ci95']['mean']:+.3f} dB  ({g['P']['verdict']})\n"
            f"  -> {g['action']}"
        )
    else:
        print("\nGATE: needs B and C-att (16-QAM) evaluated for at least 2 paired seeds")
    q = out["rq2_qam16"]
    if q:
        print(
            f"\nRQ2 cue fit: {q['outcome']} (best {q['best_cue']}, win rate "
            f"{q['best_win_rate']:.2f}, discriminable={q['discriminable']})"
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", type=Path)
    p.add_argument("--n-boot", type=int, default=10_000)
    args = p.parse_args()
    report(args.root, n_boot=args.n_boot)


if __name__ == "__main__":
    main()
