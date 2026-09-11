"""Record what the AF modules actually learned.

    python -m semcom.analyze_gates results/r12/adjsccq_r0.0833_m16_snr0-20

ADJSCC reports two patterns in the scaling factors S, and this reproduces both for a
trained model. They are the most interesting part of that paper and the part most often
skipped in citations:

  Pattern 1 - the gates get more selective as SNR rises. At low SNR every feature is
  compromised roughly equally, so there is little to gain by discriminating. At high SNR
  some features matter far more than others, so it pays to boost those and suppress the
  rest. Measured here as the across-channel std of S, which should climb with SNR.

  Pattern 2 - SNR-dependence concentrates in the early layers. The spread between the
  curves at different SNRs shrinks with depth: channel noise damages low-level features
  (texture, edges) far more than high-level ones. Measured as the range of per-module
  mean gate values across SNR, which should shrink from module 1 to module 4.

Pattern 2 is the interesting one for semantic communication: it is the network
independently rediscovering that meaning is more robust to channel noise than pixels are,
from a model that was only ever asked to minimise MSE.

The open question this script answers is whether either pattern survives quantisation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import cifar10_loaders
from .evaluate import load_run


@torch.no_grad()
def collect_gate_statistics(model, loader, device, snrs, max_batches: int = 8) -> dict:
    """Mean and across-channel std of S, per encoder AF module, per SNR."""
    if not model.snr_adaptive:
        raise ValueError(f"{model.name} has no AF modules; nothing to analyse")

    stats: dict[float, list[dict]] = {}
    for snr in snrs:
        # Accumulate the gate vectors themselves, then reduce, so the across-channel std
        # is computed on the mean gate profile rather than averaged over batches.
        totals: list[torch.Tensor] = []
        count = 0
        for i, (x, _) in enumerate(loader):
            if i >= max_batches:
                break
            x = x.to(device)
            gates = model(x, torch.full((x.shape[0],), float(snr), device=device),
                          collect_gates=True)["gates"]
            if not totals:
                totals = [torch.zeros(g.shape[1], device=device) for g in gates]
            for acc, g in zip(totals, gates):
                acc += g.sum(dim=0)
            count += x.shape[0]

        stats[float(snr)] = [
            {
                "module": idx + 1,
                "mean": (acc / count).mean().item(),
                "std_across_channels": (acc / count).std().item(),
            }
            for idx, acc in enumerate(totals)
        ]
    return stats


def summarise(stats: dict) -> dict:
    """Reduce the raw statistics to the two patterns, with an explicit verdict on each."""
    snrs = sorted(stats)
    n_modules = len(stats[snrs[0]])

    selectivity = {
        f"module_{m + 1}": [stats[s][m]["std_across_channels"] for s in snrs]
        for m in range(n_modules)
    }
    # Pattern 1 holds for a module if selectivity is higher at high SNR than at low SNR.
    pattern_1 = {k: v[-1] > v[0] for k, v in selectivity.items()}

    # Pattern 2: how much a module's mean gate moves across the SNR range.
    snr_spread = [
        max(stats[s][m]["mean"] for s in snrs) - min(stats[s][m]["mean"] for s in snrs)
        for m in range(n_modules)
    ]
    pattern_2 = snr_spread[0] > snr_spread[-1]

    return {
        "snrs": snrs,
        "selectivity_by_module": selectivity,
        "snr_spread_by_module": snr_spread,
        "pattern_1_gates_more_selective_at_high_snr": pattern_1,
        "pattern_2_snr_dependence_concentrates_early": pattern_2,
    }


def analyse(run_dir: Path, snrs=(1.0, 4.0, 7.0, 13.0, 19.0), max_batches: int = 8) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_run(run_dir, device)
    _, _, test_loader = cifar10_loaders(
        cfg.data_root, cfg.batch_size, cfg.num_workers, seed=cfg.seed
    )

    stats = collect_gate_statistics(model, test_loader, device, snrs, max_batches)
    result = {"run_name": cfg.run_name, "arm": cfg.arm, "raw": stats, **summarise(stats)}
    (run_dir / "gate_analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\n{cfg.arm} - AF gate statistics")
    print(f"{'SNR':>6} | " + " | ".join(f"mod{m + 1} mean/std" for m in range(len(stats[snrs[0]]))))
    for snr in sorted(stats):
        row = "  ".join(f"{d['mean']:.3f}/{d['std_across_channels']:.3f}" for d in stats[snr])
        print(f"{snr:6.1f} | {row}")

    p1 = result["pattern_1_gates_more_selective_at_high_snr"]
    print(f"\nPattern 1 (selectivity rises with SNR): {sum(p1.values())}/{len(p1)} modules")
    print(f"Pattern 2 (SNR-dependence concentrates early): {result['pattern_2_snr_dependence_concentrates_early']}")
    print(f"  per-module SNR spread: {[f'{v:.4f}' for v in result['snr_spread_by_module']]}")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dirs", type=Path, nargs="+")
    p.add_argument("--max-batches", type=int, default=8)
    args = p.parse_args()
    for run_dir in args.run_dirs:
        analyse(run_dir, max_batches=args.max_batches)


if __name__ == "__main__":
    main()
