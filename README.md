# ADJSCC-Q

SNR-adaptive, constellation-constrained deep joint source-channel coding — composing the
**ADJSCC** attention-feature module with the **DeepJSCC-Q** soft-to-hard quantiser, plus
the controlled ablation that tests whether the two mechanisms actually compose.

It targets a gap the DeepJSCC-Q authors name explicitly in their own future-work section:

> **No SNR conditioning.** Models are trained per SNR_train. Composing the ADJSCC AF
> module with the soft-to-hard quantiser is the obvious next step and is not done here.

ADJSCC removes the fixed-SNR idealisation but keeps analog symbols. DeepJSCC-Q removes the
analog-symbol debt but keeps per-SNR specialist models. ADJSCC-Q removes both: one set of
weights, SNR-conditioned at inference, emitting legal M-QAM symbols.

## Prior art — read this before claiming novelty

The **idea** is not novel. Framed honestly, this is a reproduction-and-composition study;
the contribution is the evidence and the ablation, not the architecture.

- No paper titled "ADJSCC-Q" exists, and the canonical works remain separate:
  [ADJSCC (arXiv:2012.00533)](https://arxiv.org/abs/2012.00533),
  [DeepJSCC-Q (arXiv:2206.08100)](https://arxiv.org/abs/2206.08100).
- But the combination has been done under other names.
  [VQ-DeepISC (arXiv:2508.03740)](https://arxiv.org/html/2508.03740) uses an "SNR ModNet"
  that is structurally an AF module in front of a vector-quantised digital pipeline;
  [JSCM (arXiv:2511.15699)](https://arxiv.org/html/2511.15699) combines Gumbel-softmax
  with soft quantisation; [VQ-DSC-R (arXiv:2602.15045)](https://arxiv.org/html/2602.15045v1)
  adds attention-based channel adaptation to differentiable VQ.
- What none of them do is the minimal composition of **ADJSCC's exact AF module** with
  **DeepJSCC-Q's exact soft-to-hard quantiser against a fixed, standards-legal M-QAM
  lattice**. They all use *learned* VQ codebooks, which forfeits the deployability point
  that DeepJSCC-Q exists to make. And none run the 2x2 that isolates whether SNR
  conditioning still pays once the channel input is quantised.

## The 2x2

One model class, two booleans ([semcom/models.py](semcom/models.py)):

| | `digital=False` (analog) | `digital=True` (M-QAM) |
|---|---|---|
| `snr_adaptive=False` | BDJSCC | DeepJSCC-Q |
| `snr_adaptive=True` | ADJSCC | **ADJSCC-Q** |

Fixed-SNR arms are trained as specialists at {1, 4, 7, 13, 19} dB. Adaptive arms get one
model trained on SNR ~ U[0,20] dB, resampled *per example*.

The digital arms add **zero trainable parameters** (the constellation is fixed) and the AF
modules add well under 1%, so the comparison is controlled on capacity.

## The hypothesis

ADJSCC reports its largest margin at low bandwidth ratio — where the encoder is most
starved of dimensions and allocation decisions matter most. Quantisation starves the
encoder *further*. So conditioning should pay **more** under a finite constellation, not
less. That is what `run_ablation.py`'s Q2 measures. A null result is a legitimate outcome
and is reported as one.

## Install and run

```bash
pip install -r requirements.txt
pytest tests/ -q                      # 142 tests, ~1 min on CPU

# one arm
python -m semcom.train --config configs/cifar_r12.yaml --snr-adaptive --digital -M 16

# the whole 2x2 (12 runs), then curves, tables and figure
python scripts/run_ablation.py --config configs/cifar_r12.yaml

# afterwards
python -m semcom.evaluate results/r12/<run>       # full SNR sweep -> evaluation.json
python -m semcom.analyze_gates results/r12/<run>  # AF gate statistics
```

Add `--no-wandb` to any of these to run without logging. Training is resumable: rerun the
same command and it picks up from `checkpoint.pt`.

### Logging

Weights & Biases is on by default (`wandb: true` in the configs). Runs are named by arm,
rate, modulation order and training SNR, and tagged by arm so the 2x2 groups cleanly.
Logged: loss, MSE, PSNR, and for digital arms the KL term and the quantiser's `sigma_q`
annealing — worth watching, since that schedule climbs fast once it starts. `wandb init`
failures are caught and never kill a training run.

## Design decisions worth knowing

**Ordering: AF gating → power normalisation → quantisation.** This is the one decision the
composition forces and neither source paper had to make. AF gates are sigmoid-bounded, so
they attenuate at low SNR. Normalising *before* gating would let the gates control
transmit power rather than resource allocation, and quantisation against a fixed-power
constellation would then silently change meaning with SNR. Guarded by
`test_af_gating_does_not_control_transmit_power`.

**The straight-through splice is written `hard + (soft - soft.detach())`**, not the more
common `soft + (hard - soft).detach()`. Algebraically identical, same gradients — but only
this form is bit-exact in float32, because `soft - soft.detach()` is exactly zero. The
other form leaves transmitted values a few ULPs off the constellation, silently violating
the one invariant that makes the scheme standards-legal. The test suite caught this.

**`torch.cdist` is avoided in the quantiser.** Its matmul backend reports errors of ~5e-4
on values that are provably exact constellation points — enough to blur the
nearest-neighbour decision. Distances use the quadratic expansion instead, which is also
far cheaper in memory at training batch sizes.

**M is a resolution knob, not a rate knob.** `k` is fixed by `c_out`, so raising M does not
raise the rate — it refines the codebook, and must be monotonically better at every SNR.
Any non-monotonicity in M is a bug, not a finding.

## Testing

142 tests. The suite is structured around the invariants that, if broken, would produce
*plausible but wrong* results rather than crashes:

- `tests/test_constellation.py` — the alphabet invariant (transmitted values are bit-exact
  constellation members), gradient flow through the splice, `sigma_q` annealing, the KL
  anti-collapse term, and monotonic quantisation error in M.
- `tests/test_channel.py` — power normalisation hits its target exactly and per-example;
  empirical channel SNR matches the requested SNR; Rayleigh equalisation inverts the fade.
- `tests/test_models.py` — the 2x2 wiring, rate arithmetic, the gating/power ordering
  guard, gradient reach in every arm, capacity parity across arms, and an overfit smoke
  test on all four arms.
- `tests/test_ablation.py` — the analysis logic on synthetic curves where the answer is
  known by construction, including that a failed hypothesis is reported as failed.
- `tests/test_integration.py` — train → checkpoint → reload → evaluate → analyse, on a
  small in-memory dataset, for every arm.

Three real bugs were caught this way before any training run:

1. The straight-through splice emitted off-constellation values (float rounding).
2. `Config.save()` wrote derived keys that `Config.from_yaml()` rejected — **no run
   directory could be reloaded**, so every evaluation would have failed *after* training.
3. `analyse()` silently skipped specialists missing from the eval grid, and `all([])` is
   `True` — so it would report "adaptive wins everywhere" having made zero comparisons.

`evaluate.py` also re-asserts the constellation invariant on the *trained* model before
reporting any number, and refuses to emit results if it fails. Training moves the
encoder's output distribution a long way; the unit tests only check an untrained quantiser.

## Caveats

- **Epoch budget.** The ADJSCC results being reproduced were trained for 1280 epochs.
  Early stopping usually fires far sooner; `history.json` records `epochs_trained`
  alongside `epochs_configured`, and any writeup should quote the actual number rather
  than implying protocol parity.
- **Scope.** CIFAR-10, AWGN and flat Rayleigh only. No OFDM, no frequency selectivity, no
  MIMO, no bit interface — so no CRC, HARQ, or ciphering. This makes the *modulator*
  standards-compatible; it does not make the *stack* compatible. That distinction is the
  subject of standardisation-roadmap work, not this repo.
- The learned-constellation (L-M) variant from DeepJSCC-Q is not implemented.

## Layout

```
semcom/
  constellation.py   M-QAM lattice + soft-to-hard quantiser (DeepJSCC-Q)
  modules.py         FL modules + AF module (ADJSCC)
  models.py          the 2x2, as two booleans
  channel.py         power normalisation, AWGN, Rayleigh
  gdn.py             generalised divisive normalisation
  config.py          dataclass config + YAML
  data.py            CIFAR-10 loaders, PSNR, SNR sampling
  train.py           training loop + wandb
  evaluate.py        SNR sweep + trained-model invariant check
  analyze_gates.py   AF gate statistics (ADJSCC patterns 1 and 2)
scripts/
  run_ablation.py    all 12 runs -> curves, tables, figure
configs/             cifar_r12.yaml (primary), cifar_r6.yaml
tests/
```
