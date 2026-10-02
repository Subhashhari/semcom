# ADJSCC-Q, explained

*What is being built in this repository, why, what is genuinely new about it, and what
it deliberately does not attempt. Written to be read alongside
[the complete roadmap](semantic-communication-roadmap-complete.md) and
[the SOTA survey](semcom-sota-and-future-work.md).*

---

## 1. The one-paragraph version

Two papers each removed one idealisation from deep joint source-channel coding and left the
other in place. **ADJSCC** made a single model work across a range of channel SNRs, by
conditioning convolutional feature gates on the SNR — but it still emits arbitrary complex
numbers, which no commercial radio can transmit. **DeepJSCC-Q** constrained the channel input
to a real M-QAM constellation, making the scheme standards-legal — but it trains a separate
model for every SNR. This repository builds **ADJSCC-Q**: both mechanisms in one model, one
set of weights, SNR-conditioned at inference, emitting legal QAM symbols. It then runs the
2×2 ablation that asks whether the two mechanisms actually compose, or whether conditioning
stops paying once the encoder's output is snapped to a coarse lattice.

---

## 2. Why this particular gap

The roadmap's central observation is that the field climbed its practicality staircase one
step per paper, and the steps do not compose for free:

> The honest reading of this table: the field has, collectively, addressed every one of the
> major practicality gaps — but usually in *separate* papers, each removing one idealization
> while often re-assuming others. **Integration, not any single missing capability, is the
> current frontier.**

ADJSCC and DeepJSCC-Q are the cleanest instance of that problem, because they are *adjacent*
steps that were never joined. Both papers say so themselves:

| Paper | What it fixed | What it left |
|---|---|---|
| ADJSCC | Fixed SNR → one conditioned model | "It still emits arbitrary complex symbols. The analog-symbol debt is untouched." |
| DeepJSCC-Q | Analog symbols → finite M-QAM | "No SNR conditioning. Models are trained per SNR_train. Composing the ADJSCC AF module with the soft-to-hard quantiser is the obvious next step and is not done here." |

Each names the other's contribution as its own missing piece. That is an unusually
well-specified gap, and it is what this repo fills.

### Why it is not merely additive

The tempting assumption is that bolting two orthogonal modules together yields the sum of
their gains. There is a concrete reason to doubt it, and a concrete reason to expect the
opposite — which is what makes the experiment worth running rather than merely worth coding.

**The case for interference.** The AF module works by *re-allocating* representational
capacity across feature channels: suppress fragile high-frequency detail when the channel is
bad, restore it when the channel is good. That re-allocation is expressed as continuous
rescaling of a continuous latent. Quantisation then snaps the result to a coarse lattice —
at 16-QAM, four levels per dimension. A gate that scales a feature by 0.83 instead of 0.88
may make no difference at all after the latent is power-normalised and rounded to the
nearest of sixteen points. If the quantiser discards the very resolution the gates operate
in, conditioning should *stop paying*.

**The case for amplification.** ADJSCC reports its largest margin at *low bandwidth ratio* —
"precisely where the encoder is most starved of dimensions and allocation decisions matter
most." Quantisation is another form of starvation: it reduces the effective information per
channel use. On that reading, a quantised encoder has *less* to spend and therefore *more*
to gain from spending it wisely, so conditioning should pay **more**, not less.

These predict opposite outcomes. The repo's primary experiment measures which is right, and
reports a null or negative result as readily as a positive one.

---

## 3. What is actually built

### 3.1 The 2×2

One model class, two booleans, so that the comparison is a flag change rather than four
separately-written architectures that might differ in a dozen incidental ways:

| | `digital=False` (analog) | `digital=True` (M-QAM) |
|---|---|---|
| `snr_adaptive=False` | **BDJSCC** — the plain baseline | **DeepJSCC-Q** |
| `snr_adaptive=True` | **ADJSCC** | **ADJSCC-Q** ← the target |

Fixed-SNR arms are trained as *specialists* at {1, 4, 7, 13, 19} dB, one model each, and are
plotted as an oracle envelope — assuming a system that always picks the right specialist for
the current channel. That is strictly more generous than any real deployment, which is the
point: the adaptive model should have to beat the best possible version of its rival.

Two properties make this a controlled comparison rather than a confounded one:

- **The digital arms add zero trainable parameters.** The constellation is fixed, so
  ADJSCC and ADJSCC-Q have *identical* capacity. Any difference is the quantiser, not width.
- **The AF modules add +0.65%** (10,499,687 → 10,567,527 parameters at R=1/12), against the
  +0.6% ADJSCC reports. The adaptive arm cannot be winning on capacity.

### 3.2 The transmit chain

```
image ──► encoder (FL modules, AF gates conditioned on SNR)
            │
            ├─► power normalise to unit average symbol power
            │
            ├─► soft-to-hard quantise to M-QAM        [digital arms only]
            │
            ├─► channel: y = h·z + n                  [AWGN or Rayleigh]
            │
            └─► decoder network, directly on noisy symbols
                  (no demapping to LLRs, no channel decoding, no CRC)
```

The receiver side is where the architectures diverge most sharply from a 5G pipeline. The
equalised complex symbols go *straight into the decoder network*. It never asks "which
constellation point was intended?" — the question that dominates the entire receive chain in
a conventional radio. It asks only "what image best explains these observations?"

### 3.3 The AF module (from ADJSCC)

Three steps, and the whole thing is about fifteen lines:

1. Global-average-pool each of `c` feature maps → `c` scalars.
2. Concatenate the SNR → a context vector in ℝ^(c+1).
3. Two-layer MLP → sigmoid → per-channel gates in (0,1); multiply.

The sigmoid is load-bearing: it makes this a *gate* that can only attenuate, never amplify.
Structurally it is a Squeeze-and-Excitation block with one extra scalar in the squeeze
vector. The contribution is not the architecture — it is the *interpretation*: a learned,
continuous, per-feature analogue of the classical source/channel bit split.

A naming trap worth repeating, since it trips up nearly every reader: **"channel-wise" here
means the feature channels of the tensor, not the wireless channel.** The module attends over
convolutional feature maps and uses the wireless channel's SNR only as a conditioning scalar.

### 3.4 The soft-to-hard quantiser (from DeepJSCC-Q)

The mechanism that makes a finite alphabet trainable at all. Mapping a network output to the
nearest constellation point is a step function — gradient zero almost everywhere, undefined
on the boundaries. Insert it naively and training stops.

- **Forward (hard):** `z̄ᵢ = argmin_j ‖zᵢ − c_j‖²` — an actual, legal QAM symbol.
- **Backward (soft):** the softmax-weighted average of *all* constellation points, weighted
  by negative squared distance and sharpened by inverse temperature `σ_q`.
- **Annealed:** `σ_q` climbs from 5 toward 100 during training. Early on the problem is soft
  and the encoder can explore; by the end the gradient matches the discrete reality.

The property that matters: the forward pass is *always* the true transmitted signal, and only
the gradient is approximated. The encoder never develops a mismatch between what it thinks it
sent and what actually went over the air.

A **KL regulariser** pulls the empirical symbol distribution toward uniform, countering
codebook collapse (the encoder finding a handful of convenient points and wasting the rest).

### 3.5 The one design decision neither paper had to make

Composing the two forces a choice about **ordering**, and it is not cosmetic.

AF gates are sigmoid-bounded, so at low SNR they attenuate — shrinking the latent's overall
magnitude. If power normalisation ran *before* gating, that shrinkage would pass straight
through to the transmitted signal, and the gates would be controlling **transmit power**
rather than **resource allocation**. Quantisation against a fixed-power constellation would
then silently change meaning with SNR: the same lattice would represent a differently-scaled
latent at every operating point, and the SNR axis on every plot would stop meaning what it
claims.

**The chain is therefore: AF gating → power normalisation → quantisation.** Normalisation
sits immediately before the quantiser so the latent is always dimensionally matched to the
constellation's fixed scale, and the gates do only the job they are meant to do. There is no
post-quantisation normalisation, which would move symbols off the constellation and defeat
the entire exercise.

This is enforced by a test, not just a comment
(`test_af_gating_does_not_control_transmit_power`): transmit power must be identical across
SNRs while the gates themselves demonstrably respond to SNR.

### 3.6 The claim the whole thing rests on

Everything about deployability reduces to one invariant: **what goes over the air must be an
exact member of the constellation alphabet.** If that fails, the scheme is not
standards-legal, EVM conformance is undefined, and every comparison against a 5G baseline is
void. So it is asserted in three places:

1. On an untrained quantiser, across five constellation orders (unit tests).
2. On the real encoder with real power normalisation, across SNRs (model tests).
3. On the **trained** model, on real test data, before any number is reported — and
   `evaluate.py` *refuses to emit results* if it fails.

Point 3 matters because training moves the encoder's output distribution a long way from
where the unit tests looked.

---

## 4. What is genuinely new here — triple-checked

The honest answer has three layers, and the middle one is uncomfortable.

**The idea is not novel.** Searching the 2025–2026 literature turns up several works that
combine channel-adaptive conditioning with some form of discretisation:

- [VQ-DeepISC (arXiv:2508.03740)](https://arxiv.org/html/2508.03740) uses an "SNR ModNet"
  that is structurally an AF module — global average pooling, factor prediction conditioned
  on SNR — in front of a vector-quantised digital pipeline.
- [VQ-DSC-R (arXiv:2602.15045)](https://arxiv.org/html/2602.15045v1) adds attention-based
  channel adaptation on top of differentiable VQ with adaptive noise variance.
- [JSCM (arXiv:2511.15699)](https://arxiv.org/html/2511.15699) combines Gumbel-softmax with
  soft quantisation for differentiable modulation in token communications.
- [Task-Oriented DeepJSCC with Semantic-Aware Adaptive Quantization (Sensors, Aug 2026)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13517896/)
  uses a "ChannelAdapter" — a two-layer MLP mapping SNR to channel-wise gates — with
  adaptive bit-width quantisation.

Anyone claiming "ADJSCC-Q is a new architecture" would be wrong, and should be corrected.

**But the specific composition is still not available.** Checking each of the above against
what this repo does:

| Work | SNR-conditioned gating | Discretised | **Fixed standards-legal QAM alphabet** |
|---|---|---|---|
| ADJSCC (2022) | ✔ | ✘ (analog) | ✘ |
| DeepJSCC-Q (2022) | ✘ (per-SNR models) | ✔ | ✔ |
| VQ-DeepISC (2025) | ✔ | ✔ | ✘ — *learned* VQ codebook |
| VQ-DSC-R (2026) | ✔ | ✔ | ✘ — *learned* VQ codebook |
| Sensors (Aug 2026) | ✔ | ✔ bit-width | ✘ — transmits unconstrained baseband symbols |
| SA-RA-JSCC (Jun 2026) | ✔ | ✘ | ✘ — continuous fading model, no quantisation |
| **This repo** | ✔ | ✔ | **✔ fixed M-QAM** |

The distinction in the last column is not pedantry — it is the entire point of DeepJSCC-Q's
existence. A *learned* VQ codebook is not a constellation a commercial modulator can emit; it
reintroduces exactly the problem that made analog DeepJSCC undeployable, one level up. And
quantising feature *bit-width* while still transmitting unconstrained baseband symbols is a
different operation altogether from constraining the *channel input alphabet*. Several recent
papers describe themselves as "digital semantic communication" while doing the latter.

**What is actually contributed is evidence, not architecture.** No work found runs the
controlled 2×2 that isolates whether SNR conditioning still pays *once the channel input is
quantised to a fixed lattice* — with capacity held constant, against an oracle specialist
envelope, including the matched-point test. That question has an argued case on both sides
(§2), and it is currently unanswered.

So this should be framed, and is framed in the README, as a **reproduction-and-composition
study**. The contribution is the measurement.

### A caveat on the search itself

This survey was run in September 2026 against a fast-moving literature, and absence of
evidence is weak evidence here. Searches covered: DeepJSCC-Q follow-ups and citing work,
SNR-adaptive and channel-adaptive JSCC, digital/constellation-constrained semantic
communication, and learned-constellation work. What would change the verdict is any paper
combining AF-style conditioning with a *fixed* QAM alphabet and reporting the ablation.
Anyone building on this should re-run that check rather than trusting this table's date.

---

## 5. The experiment

Three questions, each with a pre-registered interpretation so that the analysis cannot be
retrofitted to whatever the data says.

**Q1 — Does ADJSCC-Q beat every DeepJSCC-Q specialist, including at each specialist's own
training SNR?** The matched-point comparison is the sharp version. ADJSCC's most striking
claim is that one generalist beats each specialist *on the specialist's home turf* — the
natural reading being that exposure to a range of noise levels acts as a regulariser and
forces a better-organised feature hierarchy. Whether that survives quantisation is unknown.

**Q2 — Is the ADJSCC-Q → DeepJSCC-Q gap larger than the ADJSCC → BDJSCC gap?** This is the
amplification hypothesis from §2.3, stated as a measurable quantity: the mean PSNR gain from
conditioning, computed separately in the digital and analog halves of the 2×2. If the digital
gain is larger, quantisation amplifies the value of conditioning. If smaller, the
interference argument wins. Both are publishable; only one is expected.

**Q3 — Storage.** One adaptive model against the specialist ensemble it replaces. ADJSCC's
version of this argument is the one that most directly kills the "just train several models"
workaround: ten specialists bought 29.826 dB, one ADJSCC bought 29.831 dB, at ~10% of the
storage and training cost.

Plus a fourth, non-comparative question:

**Q4 — What did the gates learn, and does quantisation change it?** ADJSCC reports two
patterns: gates become *more selective* as SNR rises (at low SNR every feature is compromised
roughly equally, so there is little to gain by discriminating), and SNR-dependence
*concentrates in the early layers* (channel noise damages low-level features far more than
high-level ones).

The second is the interesting one for semantic communication as a field, and worth stating
bluntly: **the network independently rediscovered that meaning is more robust to channel
noise than pixels are** — from a model that was only ever asked to minimise MSE. The whole
premise of SemCom appears there not as an assumption but as an emergent property.
`analyze_gates.py` tests whether it survives a finite alphabet.

---

## 6. What this deliberately does not attempt

Scope discipline matters more here than coverage, because the value of the result depends on
the comparison being clean.

- **No OFDM, no frequency selectivity, no MIMO.** AWGN and flat Rayleigh only. Adding these
  would be climbing further steps of the staircase, and would confound the one question being
  asked.
- **No bit interface, therefore no CRC, HARQ, ciphering, or rate matching.** This makes the
  *modulator* standards-compatible; it does not make the *stack* compatible. That distinction
  is the honest limit of this entire line of work, and it is where the remaining
  standardisation effort actually lies.
- **No learned constellations.** The L-M variant is not implemented — deliberately, since
  fixed standards-legal QAM is the property that distinguishes this from the recent VQ-based
  literature. (Learned constellations are, separately, where the 2026 field is moving; see
  the SOTA document.)
- **CIFAR-10 only.** Small images, so the whole 12-run ablation is an overnight job on one
  consumer GPU rather than a cluster booking. The architecture is resolution-agnostic, but
  no Kodak or ImageNet numbers are claimed.
- **Not a state-of-the-art claim.** MambaJSCC, SwinJSCC and NTSCC++ are all stronger
  *architectures*. This repo deliberately uses the original ADJSCC backbone, because
  swapping in a better backbone would improve the numbers while destroying the ability to
  attribute any difference to the mechanism under test.

That last point is worth dwelling on. It would be easy to make the numbers look better and
the result mean less.

---

## 7. How the correctness argument is structured

The failure mode this codebase is most exposed to is not a crash — it is **a plausible but
wrong number**. A quantiser that emits near-constellation values, a power normalisation that
drifts, an analysis that compares the wrong pairs: all of these produce clean-looking curves
that are quietly meaningless. The test suite is organised around that risk, and it earned its
keep before a single training run:

1. **The straight-through splice emitted off-constellation values.** Written the conventional
   way, `soft + (hard − soft).detach()`, float32 rounding leaves transmitted values a few ULPs
   off the lattice. Rewritten as `hard + (soft − soft.detach())` — algebraically identical,
   same gradients, but `soft − soft.detach()` is *exactly* zero, so the forward value is
   exactly `hard`. The invariant test caught this immediately.

2. **`torch.cdist` was lying.** Its matmul backend reported ~5×10⁻⁴ error on values that were
   provably exact constellation points. The secondary check that used it failed while the
   primary check passed, which is how the tool under suspicion was identified. Distances now
   use the quadratic expansion, which is also far cheaper in memory at batch scale.

3. **No run directory could be reloaded.** `Config.save()` wrote derived keys that
   `Config.from_yaml()` then rejected as unknown. Every evaluation would have failed *after*
   training completed — the most expensive possible time to discover it.

4. **The ablation could report a vacuous win.** `analyse()` silently skipped specialists
   missing from the eval grid, and `all([])` is `True` — so with a mismatched grid it would
   have reported "adaptive wins everywhere" having made **zero comparisons**, reproducing the
   paper's headline claim from no data at all. It now raises.

Bugs 1 and 4 are the ones worth noting: both would have produced *publishable-looking*
results. Neither would have crashed anything.

---

## 8. Reading order

For someone picking this up cold:

1. This document — what and why.
2. [`semantic-communication-roadmap-complete.md`](semantic-communication-roadmap-complete.md)
   §2.3 and §3.1 and §3.6 — the two source mechanisms in full, and the debt they address.
3. §1 of `semantic-communication-roadmap-complete.md` — if the wireless vocabulary is
   unfamiliar; it introduces the pipeline in ML terms.
4. `semcom/constellation.py` and `semcom/modules.py` — the two mechanisms, ~150 lines each.
5. `semcom/models.py` — how they compose, and where the ordering decision lives.
6. [`semcom-sota-and-future-work.md`](semcom-sota-and-future-work.md) — where this sits in
   the wider field, and what else is open.
