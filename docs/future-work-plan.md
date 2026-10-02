# Future Work Plan

Evidence base: [r12-results-analysis.md](r12-results-analysis.md). Security landscape and
threat taxonomy: [semcom-security-research-programme.md](semcom-security-research-programme.md)
— that document's §§1–5 (landscape, taxonomy, existing defences, adversarial audit of SOTA,
verified gaps) stand unchanged. This plan revises its §6 (P1) design against real data and
sequences everything.

---

## 0. The strategic situation

The R=1/12 sweep produced two things of unequal value.

**The architecture contribution is thin.** ADJSCC-Q as "the missing composition" was already
weak on novelty — VQ-DeepISC, VQ-DSC-R and JSCM do similar combinations with learned codebooks
— and the matched-point result that would have carried it is unresolved. Storage parity at 1/3
cost is real but modest.

**The measurement contribution is stronger than expected.** Three findings emerged that were
not in any hypothesis and that reframe the work:

1. The **modulation ceiling** is measurable and matches its analytic bound (0.333 bpp, ~26.6 dB,
   three arms within 0.39 dB).
2. **Digital loses to the separation capacity bound above ~15 dB** — the deployability
   regression, quantified for the first time in this repo.
3. **Quantisation buys robustness under train/test mismatch** — digital specialists degrade
   4.53 dB less than analog twins when run 19 dB from their training point, monotone in
   mismatch severity.

(3) is a robustness result arrived at by accident, and robustness-to-perturbation is exactly
what the security programme is about. That is the thread to pull.

**Recommended reframing.** Drop "ADJSCC-Q is the missing composition." Adopt: **"What does
constellation-constraining actually cost, and what does it buy?"** — a controlled audit with a
quality axis (ceiling, separation crossover) and a robustness axis (mismatch, then adversarial).
This is honest about novelty, supported by the data already in hand, and leads directly into P1.

---

## 1. Prerequisites — do these before any new experiment

### 1.1 Make runs survivable (≤1 h)

The last sweep lost ~650 MB of weights and 10 h of compute to a deleted directory. Non-negotiable
before spending 20+ h again:

- `wandb.save(str(run_dir / "best.pt"))` after each best-checkpoint write, so weights survive
  local loss. Note wandb's default file size limits; use an artifact if `best.pt` exceeds them.
- Commit run records between arms. `.gitignore` now tracks `history.json` / `config.yaml` /
  `ablation.json` / plots and excludes `*.pt`, so this is one command per arm.
- Optional `--commit-each-run` flag on `scripts/run_ablation.py`.

### 1.2 Retrain to convergence (~20–25 h)

```
python scripts/run_ablation.py --config configs/cifar_r12.yaml --epochs 500 --specialist-snrs 1 7 19
```

With `patience: 50` and `eval_every: 10`, early stopping fires per-arm, so each arm trains to
*its own* convergence rather than an arbitrary shared cutoff. Worst case 8 × 500 × 28.8 s ≈ 32 h;
expect ~20–25 h since specialists will stop well short of 500.

"Equal budget" becomes "until converged," which is the stronger protocol and removes the
asymmetry that invalidated the matched-point test. **Gate on `scripts/convergence_report.py`
returning CONVERGED before trusting any comparison.**

### 1.3 Run the gate analysis (~10 min)

`python semcom/analyze_gates.py` on both adaptive arms. This was never done and the previous
weights are gone. It is the mechanism check for every adaptivity claim: if gates are
near-constant in SNR, the AF modules aren't conditioning and no amount of training fixes it.
Do this **immediately after** the retrain, before building anything on top.

---

## 2. WP1 — The modulation sweep (5 runs, ~8–10 h)

The cheapest genuinely novel result available, and a prerequisite for WP2.

```
for M in 4 16 64 256 1024: train ADJSCC-Q at R=1/12, 500 epochs
```

**Rate implications:** M=4 → 0.167 bpp, 16 → 0.333, 64 → 0.5, 256 → 0.667, 1024 → 0.833.

**Three questions, all answerable from one sweep:**

1. **Does the ceiling rise with M as predicted?** This is the falsification test for
   §2 of the analysis. If PSNR @ 20 dB does not rise monotonically with M, the ceiling
   explanation is wrong and much of the analysis needs revisiting.
2. **Where does the separation crossover go?** ADJSCC-Q currently falls below the capacity
   bound at ~15–16 dB. The crossover should move up and eventually vanish. The M at which a
   standards-legal digital JSCC stops being beaten by classical separation at 20 dB is a
   concrete, quotable deployability number.
3. **Does digital converge to analog as M→∞?** This is hypothesis H-B from the security
   programme, and it doubles as an end-to-end correctness check — failure means the quantiser or
   power normalisation is wrong, not that the field is surprising.

**Deliverable:** PSNR-vs-SNR family over M, with the analog ADJSCC curve as the asymptote and
the separation bound overlaid. One figure that states the cost of standards-legality as a
function of alphabet size.

---

## 3. WP2 — P1: adversarial robustness, analog vs digital (~3–4 weeks)

Hypotheses H-A through H-D from
[semcom-security-research-programme.md §6.4](semcom-security-research-programme.md) are
**unchanged** — they were pre-registered before any data and should stay that way. What follows
revises design only.

### 3.1 The confound that must be handled

The programme document asserts the 2×2 compares arms "at matched capacity." **At M=16 this is
false in the information sense, and the error is SNR-dependent:**

| | analog − digital gap |
|---|---|
| 0 dB | 0.44 dB |
| 10 dB | 2.36 dB |
| 20 dB | **3.72 dB** |

Symbol budget is matched (k=256 both); information rate is not. A robustness comparison at high
SNR would conflate *quantisation changing decoder sensitivity* (the question) with *the digital
arm having 3.7 dB less quality to lose* (an artifact). This is the kind of thing that sinks a
paper at review.

**Three mitigations, to be used together:**

| | Approach | Role |
|---|---|---|
| (a) | Make M the primary axis, not a sanity check | Spans ceiling-binding to not-binding; the transition is itself the interesting variable |
| (b) | Pair arms by matched *unattacked* PSNR, not matched SNR | Report attack power to cause a Δ dB drop *from each arm's own operating point* |
| (c) | Restrict the primary claim to 0–8 dB | Gap is 0.44–1.9 dB there; arms are near-matched. Report 9–20 dB separately and label it rate-confounded |

### 3.2 The attack scale is now known, not guessed

16-QAM normalised to unit average power puts points at `(±1, ±3)·a`, `a = 1/√10 ≈ 0.3162`:

- minimum distance `2a ≈ 0.632`
- **half-lattice spacing `a ≈ 0.316`** — the H2/H3 boundary

Channel noise per real dimension is `σ = √(10^(−SNR/10)/2)`: **0.224 at 10 dB (0.71× the
half-spacing)**, 0.112 at 16 dB (0.35×).

**The H2/H3 crossover sits inside the normal operating range.** This turns H-D (a non-monotonic
robustness curve — quantisation protecting below half-spacing, betraying above) from speculation
into a sweep with known bounds: **perturbation std 0.05 → 0.6, i.e. 0.15× to 2× half-spacing.**

Generalisation: half-spacing scales as `1/√(2(M−1)/3)` for square M-QAM at unit power, so the
protective band narrows as M grows — which predicts the robustness advantage of digital should
*shrink with M*, testable on the same WP1 models.

### 3.3 Revised design table

| Element | Programme doc | Revised | Reason |
|---|---|---|---|
| Models | existing 2×2 | retrained to convergence (§1.2) | Undertrained models conflate robustness with distance-from-convergence |
| M | sanity check for H-B | **primary axis**, {4,16,64,256,1024} | Fixes the capacity confound; tests the narrowing-band prediction |
| Pairing | matched SNR | matched SNR **and** matched unattacked PSNR | Separates sensitivity from headroom |
| Primary SNR range | 0–20 dB | **0–8 dB** primary, 9–20 dB labelled confounded | Where arms are information-matched |
| Attack magnitudes | unspecified | **0.15×–2× half-spacing** (std 0.05–0.6) | Derived from constellation geometry |
| Repeats | 10 | **50–100** near the crossover | Protective regime may sit near the 0.1 dB noise floor |
| Covariate | decoder Lipschitz estimate | unchanged | Still the mechanism test for H-C |

### 3.4 Attacks, in build order

1. **Symbol-domain perturbation** — first, because it is cheapest, is the realistic over-the-air
   threat, and is the only one where the quantiser sits *between* attacker and decoder. Sweep
   across §3.2's range.
2. **Boundary-aware attack (digital only)** — minimise perturbation power to flip each symbol's
   nearest-neighbour decision, with `a ≈ 0.316` as the known threshold. This is the concrete
   instantiation of "quantisation boundary crossing" and appears not to have been implemented
   anywhere.
3. **PGD on the input image** — white-box, matches arXiv:2603.24082's methodology so numbers are
   comparable to theirs; reproducing their semantic-vs-classical headline validates the harness.
4. **[Magmaw](https://github.com/juc023/Magmaw)** — last. Universal, modality-agnostic, NDSS,
   with code. Pre-empts the "your hand-rolled attacks are too weak" objection.

### 3.5 Controls

- Identical channel realisations across arms (the pattern already used in `erasure_importance`).
- Attack budget matched in **power**, never L∞ pixel norm, or the comparison is meaningless.
- Assert symbols remain on-constellation **under attack** — an attack that works by pushing
  symbols off-alphabet is not a valid attack on a deployable system.
- Always report the unattacked operating point alongside, so robustness is never confused with a
  model that was simply worse to start with.
- **Free head start:** §3 of the analysis is already a robustness result on the mismatch axis
  (digital 4.53 dB shallower cliff at 19 dB mismatch, monotone in severity). Reproducing it on
  converged models is the cheapest possible first data point, needs no attack code, and if it
  survives it is direct evidence for H3 before a single adversarial example is generated.

---

## 4. WP3 — P2: importance-aware protection leaks importance (~3 weeks)

Unchanged from
[semcom-security-research-programme.md §7](semcom-security-research-programme.md). Downstream of
WP2 because it needs converged models and the same attack harness.

The machinery exists: `semcom/importance.py` has erasure-based importance measurement
(`erasure_importance`, `symbol_groups`, `spearman`, `concentration`) with the validity
precondition already documented — importance is meaningless on an untrained model, where erasing
anything moves PSNR by ~5e-4 dB.

One thing the R=1/12 run adds: because the digital arms are rate-limited above ~13 dB, the
importance *profile* may be flatter there (every symbol is carrying its full share of a saturated
budget). Worth measuring importance concentration as a function of SNR **and** M — if
concentration collapses when the ceiling binds, that is a precondition result for UEP schemes
generally, not just a side-channel finding.

---

## 5. WP4 — P3: leakage as a third axis (~4+ weeks)

Unchanged from
[semcom-security-research-programme.md §8](semcom-security-research-programme.md). Most
speculative, furthest out, and should not start until WP2 has produced a result.

---

## 6. Sequencing and compute budget

RTX 3050, ~28.8 s/epoch at R=1/12, ~11 min per full 21×10 evaluation sweep.

| # | Work | Compute | Gate to proceed |
|---|---|---|---|
| 0 | Run safety (§1.1) | <1 h, no GPU | — |
| 1 | Retrain to convergence (§1.2) | 20–25 h | `convergence_report.py` says CONVERGED |
| 2 | Gate analysis (§1.3) | ~10 min | Gates vary with SNR; if not, stop and diagnose |
| 3 | WP1 modulation sweep | 8–10 h | Ceiling rises monotonically with M |
| 4 | WP2 symbol-domain attack + mismatch replication | ~1 week incl. code | Any effect >0.1 dB floor |
| 5 | WP2 boundary-aware + PGD + Magmaw | ~2–3 weeks | — |
| 6 | WP3 | ~3 weeks | WP2 produced a result |
| 7 | WP4 | ~4 weeks | WP2/WP3 landed |

Steps 0–3 are ~30–36 h of GPU time and produce a complete, publishable quality-side story on
their own, independent of whether the security work succeeds. That matters: it means the
programme has a floor.

---

## 7. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Matched-point claim still fails after convergence | Medium | Then it is a genuine failed reproduction — publishable with the analog control as evidence it is not about quantisation. Do not bury it |
| Gates turn out not to vary with SNR | **High** | Would invalidate the adaptivity half of the work. Check at step 2, before investing further |
| Robustness effect is below the 0.1 dB floor | Medium | Raise repeats to 100; widen the perturbation sweep; report bounds rather than a point estimate |
| Ceiling does not rise with M | **High** | The analysis's central mechanism is wrong. Stop and re-derive before proceeding |
| arXiv:2603.24082's result does not reproduce in our harness | Medium | Diagnose before comparing against it; their claim is the anchor for the whole P1 framing |
| Someone publishes the digital-robustness comparison first | Medium | It is a visible gap. Steps 0–4 are ~5 weeks; the mismatch replication (step 4) is the fastest path to a stakeable result |
| Compute exhaustion on one 3050 | Medium | R=1/6 secondary sweep and the 5-specialist protocol are both droppable; state what was omitted |

---

## 8. One-paragraph version

Retrain to convergence with checkpoint safety, verify the AF gates actually condition on SNR,
then run the modulation sweep — those three steps cost ~35 h, fix the capacity confound, and
produce a self-contained result about what constellation-constraining costs. Then build the
security work on top, starting with the cheapest and most realistic attack and with the mismatch
robustness result as a free first data point. The reframing from "the missing composition" to "a
controlled audit of what going digital costs and buys" is what makes the already-collected data
worth something, and the accidental finding that quantisation helps under train/test mismatch is
the bridge from the quality story to the security story.
