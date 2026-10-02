# R=1/12 Ablation: In-Depth Results Analysis

Forward plan lives in [future-work-plan.md](future-work-plan.md). This document is analysis
only — what the numbers say, how confident to be, and what would overturn each claim.

---

## 0. Provenance and status

| | |
|---|---|
| Runs | 8 arms: 2 adaptive (ADJSCC, ADJSCC-Q) + 3 specialist SNRs × 2 families |
| Config | CIFAR-10 32×32, R=1/12 (k=256, c_out=8), AWGN, M=16, hidden=256 |
| Training | 150 epochs/arm, Adam 1e-4, batch 128, AMP, ~28.8 s/epoch, ~10 h total on RTX 3050 |
| Specialist SNRs | 1, 7, 19 dB (reduced from the planned 1/4/7/13/19 to fit the compute budget) |
| Evaluation | 10k test images × 10 channel realisations × 21 SNR points (0–20 dB, 1 dB steps) |
| Parameters | 10,567,527 adaptive / 10,499,687 fixed-SNR (AF modules = +0.65%) |

**Data integrity caveat.** The run directory was deleted. Records were reconstructed from the
local wandb logs with `scripts/recover_from_wandb.py`; the per-SNR curves below come from the
`ablation.json` captured before loss. **Model weights are unrecoverable**, so nothing here can
be re-evaluated at new SNRs, probed for gate behaviour, or attacked. Every number is a
read-only artifact.

**The single most important caveat, stated up front:** no arm converged. Mean validation gain
over the final 30 epochs was +0.215 dB (adaptive) and +0.102 dB (specialists), and 7 of 8 arms
set a new best at epoch 149. Claims below are tagged by how much this threatens them. Curve
*shape* findings survive it much better than *absolute level* or *small-margin* findings,
because undertraining shifts levels roughly uniformly within an arm while the shape is set by
the rate/noise structure.

---

## 1. Implementation correctness

Everything that could be checked, was, and nothing failed.

| Invariant | Result | Why it matters |
|---|---|---|
| Transmitted symbols lie exactly on the 16-QAM alphabet | `on_constellation: true`, both digital arms, post-training | The entire deployability claim is void if a single off-alphabet value is transmitted |
| Specialists peak near their training SNR | Yes, all three pairs | A specialist that didn't would mean SNR conditioning was leaking |
| Low-SNR specialist saturates | BDJSCC@1 flat at ~24.93 dB above 8 dB | Learned robustness, structurally unable to exploit a good channel |
| High-SNR specialist collapses at low SNR | BDJSCC@19 drops 13.33 dB from 20→0 dB | The cliff that motivates adaptivity, reproduced |
| AF parameter cost | +0.65% (67,840 params) | Matches ADJSCC's reported ~+0.6%; a fatter MLP would invalidate the comparison |
| Separation reference internally consistent | Codec floor 92.9 B; capacity bound monotone over feasible range; fixed-MCS cliff at 8.45 dB | An independent external anchor; all four arms could be uniformly wrong without it |

The only structural gap: **no gate-interpretability analysis was run**, and the weights are
gone, so whether the AF modules learned SNR-dependent behaviour at all is **unknown**. This is
a real hole — it is the mechanism check for everything adaptivity-related, and §5's null result
cannot be fully diagnosed without it.

---

## 2. Finding 1 — The modulation ceiling (confidence: **high**)

All three digital arms converge to the same high-SNR value regardless of training SNR, with
slopes collapsed 4–10× relative to analog:

| arm | PSNR @ 20 dB | slope 19→20 dB |
|---|---|---|
| ADJSCC-Q | 26.596 | **+0.030** |
| DeepJSCC-Q@7 | 26.283 | **+0.019** |
| DeepJSCC-Q@19 | 26.673 | **+0.052** |
| ADJSCC (analog) | 30.313 | +0.102 |
| BDJSCC@19 (analog) | 31.156 | +0.190 |

Spread across digital arms: **0.39 dB**. The analytic bound matches: `log₂(16) × 256 = 1024`
bits over 3072 pixels = **0.333 bpp**, and ~26.6 dB is the expected PSNR for a learned codec at
that rate on CIFAR-10.

**Mechanism.** Above ~13 dB the digital arms are *rate-limited, not noise-limited*. Improving
the channel cannot help because the alphabet, not the noise, is binding. Analog arms have no
such cap and keep climbing.

**Why undertraining doesn't threaten this:** three independently trained models landing within
0.39 dB with collapsed slopes, at a value matching the analytic bit budget, is not something
undertraining produces. If anything, longer training raises all three toward the same ceiling
and tightens the spread.

**What would overturn it:** the M-sweep. If PSNR @ 20 dB does *not* rise with M (4→16→64→256→
1024), the ceiling is not the alphabet and this explanation is wrong.

---

## 3. Finding 2 — Quantisation buys robustness under train/test mismatch (confidence: **moderate**)

This is the most unexpected result in the run and it was not in any hypothesis.

**At 0 dB, the mismatched digital specialists beat their analog twins:**

| pair | analog @ 0 dB | digital @ 0 dB | digital advantage |
|---|---|---|---|
| trained @1 (matched) | 22.357 | 21.595 | −0.762 (analog wins) |
| trained @7 | 18.618 | 18.915 | **+0.297** |
| trained @19 | 17.823 | 18.009 | **+0.185** |

The crossover is sharp: at **1 dB the @7 and @19 pairs are exactly tied** (−0.000 and −0.004
dB), and analog pulls ahead monotonically above that.

**Cliff depth confirms it, and scales with the degree of mismatch:**

| trained at | analog loss (train SNR → 0 dB) | digital loss | digital shallower by |
|---|---|---|---|
| 1 dB | 0.702 dB (3.0% of peak) | 0.568 dB (2.6%) | +0.134 |
| 7 dB | 7.889 dB (29.8%) | 5.916 dB (23.8%) | **+1.974** |
| 19 dB | 13.142 dB (42.4%) | 8.612 dB (32.4%) | **+4.530** |

Monotone in mismatch severity, in both absolute dB and as a fraction of peak. The worse the
train/test mismatch, the more the quantiser protects.

**Mechanism (proposed, not proven).** A hard quantiser is a projection onto a finite set.
Perturbations smaller than half the lattice spacing — whether from channel noise or from
operating at the wrong SNR — snap back to the same constellation point and vanish. This is
*feature squeezing*, a known adversarial defence, arising here as a side effect of the
deployability constraint. At low SNR, channel noise dominates quantisation error, so the
constraint costs almost nothing (+0.436 dB for the adaptive pair at 0 dB) while the projection
still suppresses mismatch-induced error.

**Threats to this finding:**
- The effect (0.185–0.297 dB) is only 2–3× the ~0.1 dB sensitivity floor. Not marginal, but not
  comfortable either.
- It is partly entangled with the ceiling: a lower-dynamic-range arm is mechanically less able
  to fall far. The *relative* numbers (32.4% vs 42.4%) control for this and the effect
  survives, but imperfectly.
- Undertraining could plausibly affect the two families asymmetrically. Unquantified.

**This is the single most promising lead for the security programme**, because adversarial
robustness is exactly robustness-to-perturbation and this is a measured instance of it.

---

## 4. Finding 3 — Digital loses to classical separation at high SNR (confidence: **moderate-high**)

| SNR | capacity bound (oracle AMC) | ADJSCC margin | ADJSCC-Q margin |
|---|---|---|---|
| 11 dB | 23.64 | +4.64 | +2.09 |
| 13 dB | 25.09 | +3.90 | +0.99 |
| 15 dB | 26.22 | +3.32 | +0.09 |
| **16 dB** | 26.67 | +3.08 | **−0.27** |
| 18 dB | 27.66 | +2.43 | **−1.13** |
| 20 dB | 28.45 | +1.86 | **−1.85** |

**ADJSCC-Q crosses below the separation baseline at ~15–16 dB and is 1.85 dB behind by 20 dB.**
Analog ADJSCC stays ahead everywhere (+1.86 dB at worst).

This is the quantitative statement of the *deployability regression*: constellation-constraining
to M=16 at R=1/12 surrenders the JSCC advantage precisely in the regime where classical systems
already work well. The practical reading is that a standards-legal digital JSCC at this rate and
alphabet is only worth deploying below ~15 dB — which is, to be fair, where the interesting
operating points are, but it is a much narrower claim than "JSCC beats separation."

Note the baseline is a *capacity bound with oracle AMC* — an upper bound on all separation
schemes, so this is the hardest possible version of the comparison. Against the realistic
fixed-MCS curve (flat 23.06 dB above its 8.45 dB threshold), ADJSCC-Q wins everywhere.

**What would overturn it:** larger M. The crossover should move up in SNR and eventually
disappear. The M-sweep settles this and is the main reason to run it.

---

## 5. Finding 4 — Adaptivity is *easier* under quantisation (confidence: **moderate**)

Adaptive arm minus oracle specialist envelope, at each SNR:

| SNR | analog (ADJSCC − BDJSCC env) | digital (ADJSCC-Q − DeepJSCC-Q env) |
|---|---|---|
| 0 | −0.382 | −0.056 |
| 4 | +0.324 | +0.308 |
| 8 | −0.095 | −0.198 |
| 12 | +0.316 | +0.030 |
| 16 | −0.403 | +0.029 |
| 18 | −0.654 | −0.031 |
| 20 | **−0.843** | **−0.077** |
| wins | 7/21 | 9/21 |

The shapes differ qualitatively. The **analog** deficit grows monotonically and without bound
at high SNR (−0.079 at 14 dB → −0.843 at 20 dB). The **digital** margin stays inside ±0.08 dB
across the entire top half of the range.

**Mechanism.** The analog adaptive model must cover a dynamic range of 8.34 dB with one set of
weights, against specialists that each cover a narrow band — and BDJSCC@19 alone spans 13.33
dB. The digital adaptive model only has to cover 5.06 dB, because the ceiling compresses the
range that needs covering. **Quantisation makes the adaptation problem smaller.**

This is the honest version of Q2. The reported `digital_gain = −0.010 dB` vs
`analog_gain = −0.174 dB` is directionally consistent with the hypothesis that conditioning
matters more when quantised — but both are negative, so nothing is amplified. The correct
statement is: *conditioning loses ~4× less to the oracle envelope under quantisation, because
the ceiling shrinks the range it must cover.* The `quantisation_amplifies_conditioning` boolean
in the original output was `digital > analog`, which is true of two negative numbers and reads
as the opposite of what happened; it has been replaced.

A secondary observation supporting the same mechanism: the digital envelope switches from the
@7 to the @19 specialist at **14 dB**, versus **12 dB** for analog, because the digital @7
specialist holds up better for longer.

---

## 6. Finding 5 — Storage parity (confidence: **moderate-high**)

| | mean PSNR, 0–20 dB | storage |
|---|---|---|
| ADJSCC-Q, one model | 24.957 dB | 40.31 MB |
| DeepJSCC-Q oracle 3-specialist envelope | 24.967 dB | 120.16 MB |

**0.0098 dB behind, at 33.5% of storage**, against an envelope assuming a selector that always
picks correctly — strictly more generous than any deployable system. The storage ratio is exact
arithmetic. Parity is a weaker and more robust claim than a win, so it survives the convergence
caveat better than §7 does.

Caveat: with 5 specialists rather than 3 the ensemble would be 200 MB and the envelope slightly
higher. The storage ratio improves; the PSNR comparison gets marginally harder.

---

## 7. Non-finding — The matched-point claim (confidence: **not established**)

| SNR | analog margin (ADJSCC − BDJSCC@that SNR) | digital margin |
|---|---|---|
| 1 dB | −0.425 | −0.077 |
| 7 dB | −0.164 | −0.194 |
| 19 dB | −0.754 | −0.055 |
| mean | −0.448 | −0.108 |

0/3 wins in both regimes. **The analog pair is ADJSCC's published headline result**, so a
failure there indicates a cause common to both arms rather than anything about quantisation.
Adding this analog control was what made the result diagnosable; testing only the digital pair
would have invited the wrong conclusion.

The convergence data identifies the cause:

| | mean val gain, final 30 epochs |
|---|---|
| adaptive arms | **+0.215 dB** |
| specialists | **+0.102 dB** |

Adaptive arms were improving **twice as fast** at the cutoff, and the remaining gains
(+0.14 to +0.29 dB) **exceed the margins being measured** (0.055–0.194 dB). An adaptive model
fits a harder function and converges later, so an equal-epoch budget systematically favours
specialists. The comparison was decided by where training stopped.

**Verdict: inconclusive, not null.** Reporting this as a failed reproduction would be wrong.

---

## 8. Measured harness properties

Useful independent of any finding, and necessary to design what comes next.

- **Sensitivity floor ≈ 0.1 dB.** Validation shows non-monotone dips of 0.02–0.07 dB between
  checkpoints at 10 repeats × 10k images. Effects below ~0.1 dB are not distinguishable from
  evaluation noise. Any future claim must clear this.
- **Throughput:** 28.8 s/epoch steady state (351 steps), ~2,950 eval forward passes/s, so a
  full 21×10 sweep is ~11 min/model. Epoch 0 costs ~47 s (cuDNN autotune, worker spawn).
- **`persistent_workers` matters on Windows:** 28.9 s/epoch with 4 persistent workers vs 36.4 s
  with none; without persistence, respawning per epoch made 4 workers *slower* than 0.
- **Convergence horizon:** >150 epochs; extrapolation from the first 40 epochs underestimated
  the asymptote by ~0.9 dB because the tail is heavier than geometric. Do not extrapolate these
  curves from early epochs.
- **σ_q anneal saturates at epoch ~29** (increment accumulates: `+5·⌊t/10000⌋` per step), so the
  quantiser is fully hardened for ~80% of a 150-epoch run.

---

## 9. Confidence summary

| Claim | Confidence | Principal threat |
|---|---|---|
| Implementation correct; constellation invariant holds | **High** | None identified |
| 16-QAM ceiling at ~26.6 dB = 0.333 bpp | **High** | Would need M-sweep to fail to rise with M |
| Digital loses to separation capacity bound above ~15 dB | **Moderate-high** | Undertraining lowers digital more than the bound; direction unlikely to flip |
| Storage parity at 33.5% cost | **Moderate-high** | Convergence; 3 vs 5 specialists |
| Quantisation aids robustness under train/test mismatch | **Moderate** | Effect 2–3× the noise floor; entangled with dynamic range |
| Adaptivity easier under quantisation (margin stability) | **Moderate** | Convergence; mechanism inferred not measured |
| Conditioning gain ~4× smaller when quantised | **Moderate** | Both gains negative; undertrained |
| Matched-point claim fails | **Not established** | Models not converged; gains exceed margins |
| AF gates learn SNR-dependent behaviour | **Unknown** | Never measured; weights lost |

---

## 10. What a reviewer would attack first

1. **"Your models aren't converged."** Correct, and fatal to §7. Must be fixed before any
   adaptivity claim is made.
2. **"You claim matched capacity but M=16 caps the digital arm."** Correct. The analog−digital
   gap is 0.44 dB at 0 dB and 3.72 dB at 20 dB — the arms are matched in *symbol budget*, not
   information rate, and the mismatch is SNR-dependent. Any comparison at high SNR conflates
   quantisation effects with the digital arm simply having less to lose.
3. **"Three specialists is not the published protocol."** Correct; 1/4/7/13/19 was planned,
   1/7/19 was run. The matched-point claim is tested at 3 points rather than 5.
4. **"150 epochs vs the paper's 1280."** Correct, and 12% of the reference protocol.
5. **"No gate analysis, so the mechanism is unverified."** Correct, and now unfixable for these
   weights.
6. **"PSNR is not semantic."** Fair in principle; PSNR is what ADJSCC and DeepJSCC-Q report, so
   it is the right choice for comparability, but the limitation should be stated rather than
   defended.
