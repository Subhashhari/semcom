# Shaping versus Estimability in Digital JSCC

**Research proposal, v2: can a learned digital JSCC receiver do without being told the SNR, and
what does learned constellation shaping cost in estimability?**

Status: proposal, not started. v2 revises v1 after an independent review; every concern and the
change it produced is listed in §9. Builds on this repo's fixed M-QAM quantiser, 2×2 ablation and
evaluation harness, and reuses the converged models planned in
[future-work-plan.md](future-work-plan.md).

---

## 0. Summary

**The question.** Digital fixed-constellation JSCC systems assume perfect SNR or CSI knowledge. A
digital receiver knows its constellation, so it could estimate SNR itself — but the encoder of a
learned JSCC system *chooses* how often each constellation point is used, and that choice changes
the statistics blind estimators rely on. This project measures that interaction.

**The core result sought (RQ1).** A **shaping–estimability trade-off**: learned non-uniform symbol
usage improves reconstruction (JCM reports +0.66 dB over DeepJSCC-Q at 18 dB) but pushes the
signal's kurtosis toward the Gaussian value 2, which erodes moment-based blind SNR estimation. In
this repo the amount of shaping is a single knob — the KL usage regulariser weight λ — so the
trade-off can be swept directly.

**The supporting result (RQ2).** The first **seed-replicated, digital** test of the SIJSCC/CBJSCC
ablation (SNR at encoder+decoder / decoder only / none), which has so far been run once, without
variance, on analog JSCC.

**Scope.** AWGN only for all primary claims; broadcast / no-feedback setting with fixed M. Fading
and closed-loop rate adaptation are secondary and explicitly limited (§4.1).

**Kind of novelty.** Measurement and consequence. The estimators are classical, learned shaping in
digital semantic communication exists (JCM), and the kurtosis sensitivity of moment estimators is
known. What is new is connecting them: quantifying, in learned digital JSCC, how shaping trades
reconstruction quality against the receiver's ability to estimate SNR, and what that implies for
SNR-free versus SNR-adaptive designs.

**Verdict (§3).** Still worth doing, now hinged on one cheap measurement (§6, Step 0). Letter-length
paper plus a thesis chapter. About 10 weeks and 130–200 GPU-hours — roughly double v1, because the
review correctly demanded seeds, equal-window comparisons and a larger-image evaluation.

---

## 1. Background

### 1.1 Where SNR enters a JSCC system

| Use of SNR | Needs feedback? | In the literature |
|---|---|---|
| AF gating in the **encoder** (ADJSCC) | **Yes** (CSIT) | Assumed perfect |
| AF gating in the **decoder** | No — receiver can estimate | Assumed known |
| Modulation-order selection (uJSCC, Park et al.) | **Yes** — the transmitter must be told M | Assumed known |
| LLR scaling (Park et al.) | No | Assumed known |
| Diffusion timestep (Zhang et al.) | No | Learned blind estimator, analog only |

### 1.2 Classical SNR estimation

| Family | Mechanism | Weakness |
|---|---|---|
| Data-aided (pilots) | Known reference symbols. In 5G NR, SINR measured on reference signals (DMRS/CSI-RS) and reported as CQI | Overhead; amortised over a slot or coherence interval, so cheap in practice |
| Decision-directed (DD / EVM) | Residual `|y − Q(y)|²` to nearest constellation point | Biased upward at low SNR (wrong decisions snap to closer points); bias grows with M. Pooling does not remove bias |
| Moment-based (M2M4) | `S = √((2M2² − M4)/(2 − k_a))`, `N = M2 − S` | Needs signal kurtosis `k_a`; ill-conditioned as `k_a → 2` and at high SNR with few samples |
| Learned | Network maps received samples/latents to SNR | Training data; analog only so far |

Canonical reference: N. Pauluzzi and N. C. Beaulieu, "A comparison of SNR estimation techniques for
the AWGN channel," *IEEE Trans. Commun.*, vol. 48, no. 10, pp. 1681–1691, 2000 (verified). It
treats **M-PSK**; QAM requires the non-constant-modulus extensions.

### 1.3 Why learned digital JSCC is different

- **The receiver has a reference lattice** — but the decoder already gets soft `y` (this repo feeds
  the continuous channel output straight into the decoder, *"no demapping to LLRs"*), so an
  SNR-blind decoder can in principle learn the same lattice-residual cue internally. This makes the
  SNR-blind arm an *implicit estimator*, which is central to the design (§4.3).
- **Symbol usage is learned.** JCM (fixed standard QAM, learned symbol probabilities) reports that
  the learned distribution *"has the appearance of a two-dimensional Gaussian distribution at low SNR
  and gradually changes towards a uniform distribution."* In this repo, usage is pulled toward
  uniform by a KL regulariser with weight λ = 0.05.
- **Shaping moves kurtosis toward the Gaussian value.** For Maxwell–Boltzmann-shaped square QAM
  (`p_j ∝ exp(−ν|c_j|²)`):

| M | ν = 0 (uniform) | ν = 1 | ν = 2 | ν = 4 |
|---|---|---|---|---|
| 16 | k_a 1.320, H 4.00 b | 1.565, 3.79 b | **1.814**, 3.31 b | 1.698, 2.48 b |
| 64 | 1.381, 6.00 b | 1.603, 5.78 b | 1.801, 5.32 b | **1.975**, 4.48 b |
| 256 | 1.395, 8.00 b | 1.612, 7.77 b | 1.800, 7.32 b | **1.970**, 6.49 b |

  The M2M4 identifiability margin `2 − k_a` shrinks from 0.68 to ~0.19 at M = 16 (non-monotone: very
  strong shaping concentrates on the inner ring and pulls k_a back down) and to ~0.03 at M = 64/256.
  A Gaussian-like signal in Gaussian noise cannot be separated by moments at all.

---

## 2. Novelty audit

### 2.1 Method

About 25 targeted searches and 18 papers read in full or in part. Every quote below was checked
against the source text. Two search-engine summaries misattributed quotes during this process; both
were traced or dropped.

### 2.2 Closest work

| Paper | Digital? | SNR / CSI source | Decoder SNR-conditioned? | Learned usage? | SNR-error / estimation study? |
|---|---|---|---|---|---|
| [SIJSCC, arXiv:2306.15183](https://arxiv.org/abs/2306.15183) | No (`z∈ℂᵏ`) | Not used | Fig. 7: enc+dec / dec / none, **same backbone**, single run | — | No |
| [CBJSCC, *Sensors* 2024](https://www.mdpi.com/1424-8220/24/12/4005) (same first author) | No | Not used | Same three conditions | — | Gap attributed to randomness; no error bars |
| [Zhang et al., arXiv:2501.01138](https://arxiv.org/html/2501.01138) | No | Learned blind CNN → diffusion timestep | Via timestep | — | No |
| [SNR-EQ-JSCC, arXiv:2501.04732](https://arxiv.org/abs/2501.04732) | Not stated | Instantaneous or **average** SNR | Yes | — | Imperfect-feedback variant |
| [JCM, Bo et al., *IEEE TCOM* 2024, arXiv:2310.06690](https://arxiv.org/abs/2310.06690) | **Yes**, fixed QAM | Perfect channel knowledge; trained per SNR | — | **Yes** (Gaussian-like at low SNR) | **No** |
| [Park et al., arXiv:2311.08146](https://arxiv.org/abs/2311.08146) | **Yes**, fixed 4/16/64-QAM | Perfect CSI via pilots; SNR in LLRs | No (robust training) | — | No |
| [uJSCC, arXiv:2405.10749](https://arxiv.org/abs/2405.10749) | **Yes**, fixed BPSK–256-QAM | "known in advance to both the transmitter and receiver" | Yes (SNR and M) | — | No |
| [D²-JSCC, arXiv:2403.07338](https://arxiv.org/abs/2403.07338) | **Yes**, BPSK/QPSK | "perfectly known at both the transmitter and receiver" | No | — | No |
| [Importance-aware constellation, arXiv:2605.14940](https://arxiv.org/abs/2605.14940) | Learned **geometry** | — | — | Geometric | No |
| [Receiver-side enhancer, *Sensors* 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13517682/) | Not stated | Pilots / RSSI | Yes | — | ±6 dB, robust |

Context: [SecDiff, arXiv:2511.01466](https://arxiv.org/abs/2511.01466) (pilot spoofing);
[robustness survey, arXiv:2604.04413](https://arxiv.org/html/2604.04413); classical adaptive
modulation: A. J. Goldsmith and S.-G. Chua, "Variable-rate variable-power MQAM for fading channels,"
*IEEE Trans. Commun.*, vol. 45, no. 10, pp. 1218–1230, 1997 (verified), and the imperfect-CSI
adaptive-modulation literature that followed it.

### 2.3 Already known — not claimed

- DD bias at low SNR; kurtosis dependence of moment estimators; ill-conditioning of M2M4 at high SNR.
- Blind estimation under probabilistic shaping (EM, [arXiv:1806.10062](https://arxiv.org/pdf/1806.10062));
  shaping penalties for blind DSP in optical PS-QAM.
- Learned non-uniform symbol usage in digital semantic communication (JCM), and its PSNR benefit.
- SNR removal in analog JSCC with a matched-backbone ablation (SIJSCC Fig. 7, CBJSCC).
- Learned blind SNR estimation from analog latents (Zhang et al.).
- Effects of imperfect CSI on adaptive M-QAM (classical AMC literature).

### 2.4 Open, as far as found

1. **The shaping–estimability trade-off in learned digital JSCC** — JCM shows shaping helps
   reconstruction and assumes perfect channel knowledge; nobody measures what shaping does to the
   receiver's ability to estimate SNR blindly.
2. **The SIJSCC ablation in digital JSCC, with seeds.** It has been run once, without variance, on
   analog systems.
3. **How the SNR-blind decoder's implicit estimation compares with explicit estimators** when the
   decoder sees soft symbols on a known lattice.

Phrasing for the paper: *"digital fixed-constellation JSCC systems assume perfect SNR or CSI
knowledge"* — not "genie SNR", which Park et al. could rebut (their SNR comes from pilot-based CSI).

### 2.5 Residual risk

- ~25 searches and 18 papers; IEEE-only and non-English venues under-sampled.
- Whether *this repo's* trained usage is non-uniform enough to matter is unmeasured (§6, Step 0).
- Before submission: check citations of JCM and uJSCC for any later estimation study.

---

## 3. Is it worth doing?

### 3.1 What the review removed

Two of v1's four supporting arguments did not survive:

- **"Pilots cost 6.2% at JSCC block lengths."** An artifact of comparing a per-image pilot estimate
  with a pooled blind one, on 32×32 images. SNR is a link property estimated per slot or coherence
  interval; pooled over 64 images, 16 pilots per image give ~0.14 dB RMSE, and on a Kodak-size image
  (~98,000 symbols at R = 1/12) 16 pilots cost 0.016%. Dropped.
- **"DD-driven modulation selection errs high."** True, but decision-directed CQI loops are
  classical, real systems use reference signals precisely to avoid this, and modulation selection
  needs feedback, contradicting the feedback-free framing. Demoted to a secondary scenario (§4.1b).

### 3.2 What remains

1. **A new, clean trade-off with a single control knob (λ)** and a mechanism with a closed form
   (`2 − k_a`), motivated by a published observation (JCM's Gaussian-like low-SNR distributions).
2. **A real methodological gap** in the SNR-free claim: single-run, no-variance, analog-only.
3. **Low implementation cost.** Estimators are inference-only; usage statistics already exist in
   the quantiser (`_last_usage`); λ is a config value.
4. **Shared infrastructure** with the security work (converged M-sweep models).

### 3.3 Verdict

**Worth doing, conditional on Step 0.** If trained usage is near-uniform at λ = 0.05 *and* lowering
λ does not produce meaningful shaping, RQ1 collapses and the project reduces to RQ2 — a modest
replication result, probably not worth a standalone paper. If shaping appears (JCM suggests it will
once the regulariser is relaxed), the trade-off is a clean letter-length contribution.

Strong version of the paper: lead with the trade-off curve (PSNR gain vs estimability loss as λ
varies), explain it with the kurtosis mechanism, and use the seed-replicated ablation to say whether
the SNR-blind decoder makes explicit estimation unnecessary.

---

## 4. The idea in detail

### 4.1 Scenarios

**(a) Broadcast / no feedback — primary.** Fixed M. Only decoder-side adaptation is possible, so
the question is whether the receiver needs an SNR estimate at all, and if so which estimator works
under learned usage. All primary claims live here.

**(b) Point-to-point with a CQI link — secondary.** Modulation order selected by SNR threshold. The
question is *blind vs pilot-based CQI at equal overhead and equal observation window*, with **PSNR**
(not SER) as the cost. Related work: Goldsmith & Chua and the imperfect-CSI AMC literature.

**Fading — limitation.** If `h` is known, the effective SNR is essentially known (σ² is a slowly
varying, well-characterised thermal floor; the uncertain part is |h|²), so blind SNR estimation
after genie-`h` equalisation solves a non-problem. Primary claims are AWGN only. A fading study is
worth doing only as joint blind estimation of `h` and σ², or pilot-based `h` with SNR derived from
it — listed as an extension, not in scope.

### 4.2 Research questions and hypotheses

| | Question | Hypothesis |
|---|---|---|
| **RQ1** (core) | As usage regularisation λ is relaxed, how does reconstruction quality trade against blind SNR estimability? | **H1a:** lower λ → lower usage entropy, higher k_a, higher PSNR. **H1b:** M2M4 error grows sharply as `2 − k_a` shrinks; usage-aware M2M4 removes the *bias* but not the *variance* blow-up. **H1c:** DD is less sensitive to shaping than M2M4 |
| **RQ2** | At matched architecture and with seeds, how do SNR at enc+dec (A), dec only (B) and none (C) compare in digital JSCC? | **H2:** A ≈ B ≈ C within the measured seed variance at mid/high SNR; any gap concentrated at low SNR |
| **RQ3** | Does the SNR-blind decoder (C) match a decoder given an explicit estimate (B-est)? | **H3:** C ≈ B-est wherever the estimator is accurate; C beats B-est where the estimator is biased (DD at low SNR) |
| **RQ4** (scenario b) | Blind vs pilot CQI at equal overhead and window: PSNR cost of mis-selected M? | **H4:** at equal window, pilot-based CQI dominates; blind CQI needs a backoff margin |

**Central logic (from the review).** If C ≈ B-genie, every explicit-estimator arm has no practical
stake for AF gating — one would simply deploy C. So the decisive comparison for decoder adaptation
is **B-est vs C**, not B-est vs B-genie. Explicit estimators matter for RQ1 (the estimability
measurement in its own right) and RQ4 (where a number must be produced and fed back), not for
deciding whether to condition the decoder.

### 4.3 Arms

Trained, each at M = 16 primary; M ∈ {4, 64} secondary; M = 256 conditional (§6, Risk 5). AWGN,
R = 1/12. **The decoder receives soft `y` in every arm.**

| Arm | Encoder SNR | Decoder SNR | Notes |
|---|---|---|---|
| **A** | genie | genie | ADJSCC-Q as now |
| **B** | — | genie | Decoder-only adaptation |
| **B-loop** | — | estimate | **Trained with the estimator in the loop** (main arm, not optional). Removes the train/test mismatch that would otherwise inflate "estimation regret" |
| **B-avg** | — | long-run average SNR | SNR-EQ-JSCC-style cheap baseline |
| **C** | — | — | SNR-blind; an implicit estimator, since it sees soft `y` |
| Analog A/B/C | as above | as above | For the analog–digital comparison with SIJSCC |

λ sweep for RQ1: λ ∈ {0, 0.01, 0.05, 0.2} on arms B and C at M = 16 (and M = 64 if time allows).

Inference-only estimator arms on B (and B-loop), each evaluated at observation windows of
**1, 4, 16 and 64 images**:

| Estimator | Notes |
|---|---|
| DD/EVM | Empirical signal power, not 1 |
| M2M4, textbook `k_a` | Uniform-usage kurtosis |
| M2M4, usage-aware `k_a` | From measured training usage; report **per-image** fit too, since usage is image-dependent |
| DD bias-corrected | Invert the DD bias curve; needs usage information; **report variance after inversion** (slope ~0.25 dB/dB at the top of the M = 256 curve amplifies variance ~4×) |
| Learned CNN | Zhang-style, trained on received symbols |
| Data-aided, Np pilots | Np ∈ {8, 16}; same window; overhead reported |

### 4.4 Estimator definitions

- **DD/EVM:** `ĉ_i = Q(y_i)`, `SNR̂ = mean|ĉ_i|² / mean|y_i − ĉ_i|²`.
- **M2M4:** `M2 = mean|y|²`, `M4 = mean|y|⁴`, `Ŝ = √((2M2² − M4)/(2 − k_a))`, `N̂ = M2 − Ŝ`.
- **Usage-aware k_a:** `k_a = Σ_j p_j|c_j|⁴ / (Σ_j p_j|c_j|²)²` with `p` from `quantiser._last_usage`.
- **Range:** the AF module was trained on `μ/20`, μ ∈ [0, 20]. Report results **with and without**
  clipping, or train AF on an extended range, so clipping does not hide DD's over-estimation.

### 4.5 Metrics and statistics

- PSNR vs true SNR, 0–20 dB, 10k test images × 10 channel realisations; **plus one Kodak-scale
  evaluation** (§4.7).
- Estimator bias and RMSE (dB) vs SNR, M, λ and window.
- Usage entropy and k_a, training-set average and per image.
- RQ4: wrong-M fraction and direction; PSNR loss; backoff margin needed.
- **Power analysis before the main runs.** Measure seed variance on one arm first (3 seeds), then set
  the seed count `n ≥ 2(z₀.₉₇₅ + z₀.₈)² σ²/δ²` to detect δ = 0.15 dB:

| seed σ (dB) | seeds per arm |
|---|---|
| 0.05 | 2 |
| 0.10 | 7 |
| 0.15 | 16 |

  Three seeds is adequate only if σ ≲ 0.06 dB. Use identical data order and initialisation seeds
  across arms where possible to pair comparisons.
- `scripts/convergence_report.py` must return CONVERGED for every arm.

### 4.6 Decision rules (defined before running)

| Measured A/B/C headroom (95% CI) | Action |
|---|---|
| ≥ 0.2 dB somewhere | Report as a real effect; explicit-estimator arms for AF gating become relevant |
| 0.1–0.2 dB | Add seeds up to the power-analysis count; if still in band, report as "bounded below 0.2 dB" |
| < 0.1 dB everywhere | Report as a digital, seed-replicated confirmation of SIJSCC; drop AF-gating estimator comparisons |

| Step 0 usage result (§6) | Action |
|---|---|
| Entropy clearly below log₂M at λ = 0.05, or falls clearly as λ → 0 | Proceed with RQ1 as the core |
| Near-uniform at every λ | RQ1 collapses; project reduces to RQ2 (+ RQ4); reconsider whether to continue |

### 4.7 Larger-image evaluation

CIFAR-only results are weak evidence for a 2026 letter. Add one Kodak-scale evaluation at M = 16:
arms B and C trained on 128×128 crops from a larger dataset, evaluated on Kodak. Check first that the
encoder/decoder path handles sizes other than 32×32 (the symbol count `k` is currently derived from
`image_size` in the config).

### 4.8 Preliminary simulations (estimator layer; uniform usage; AWGN)

**Equal observation windows**, K = 256 data symbols per image, 16 pilots per image:

| M | SNR | Window (images) | DD bias / RMSE | M2M4 bias / RMSE | 16-pilot RMSE |
|---|---|---|---|---|---|
| 16 | 4 dB | 1 | +4.22 / 4.24 | +0.09 / 1.48 | 1.12 |
| 16 | 4 dB | 64 | +4.21 / 4.21 | −0.01 / 0.16 | 0.15 |
| 16 | 10 dB | 1 | +1.56 / 1.59 | +0.42 / 3.67 | 1.10 |
| 16 | 10 dB | 64 | +1.56 / 1.56 | −0.03 / 0.19 | 0.14 |
| 16 | 16 dB | 1 | +0.06 / 0.31 | +16.87 / 41.66 | 1.08 |
| 16 | 16 dB | 64 | +0.05 / 0.06 | −0.04 / 0.55 | 0.13 |
| 64 | 10 dB | 1 | +5.76 / 5.77 | +0.97 / 7.24 | 1.10 |
| 64 | 10 dB | 64 | +5.74 / 5.74 | +0.06 / 0.30 | 0.13 |
| 256 | 10 dB | 64 | +9.21 / 9.21 | −0.02 / 0.32 | 0.12 |

Readings: DD is bias-limited (pooling does not help) but excellent above its SER threshold even per
image; M2M4 is unbiased with uniform usage but very noisy per image at high SNR; pilots behave
exactly as sample size predicts (~4.34/√N dB). At equal windows, no estimator dominates everywhere,
and pilots are competitive once amortised.

**SNR for target symbol error rate** (square M-QAM, AWGN) — the SER = 10⁻² column is where DD becomes
unbiased (an estimator property, H1c); the SER = 0.1 column is uJSCC's selection rule (an operating
rule, RQ4). They are different quantities by design.

| M | SER = 0.1 | SER = 0.01 |
|---|---|---|
| 4 | 4.3 dB | 8.2 dB |
| 16 | 12.2 dB | 15.7 dB |
| 64 | 18.8 dB | 22.0 dB |
| 256 | 25.0 dB | 28.2 dB |

---

## 5. Implementation plan

### 5.1 Code changes

| Change | Where | Size |
|---|---|---|
| Split `snr_adaptive` into `snr_at_encoder` / `snr_at_decoder` | `models.py`, `config.py`, run naming | Small |
| Decoder-SNR override in `forward`, leaving transmitted symbols untouched | `models.py` | Small |
| Estimator-in-the-loop training (B-loop) and average-SNR arm (B-avg) | `train.py` | Small–medium |
| Estimators with a window parameter: DD, M2M4, usage-aware M2M4, bias-corrected DD, data-aided | new `semcom/snr_estimation.py` | Medium |
| Learned CNN estimator | new module | Medium |
| Usage logging: full histogram, entropy, k_a, per image | `train.py`, `evaluate.py` | Small |
| Arbitrary image sizes for the Kodak-scale arm | `models.py`, `data.py` | Small–medium |
| CQI selection policy for scenario (b) | `scripts/` | Small |

### 5.2 Tests to write first

- DD and M2M4 reproduce the §4.8 table within tolerance at each window.
- Usage-aware M2M4 is unbiased for a known non-uniform usage; textbook M2M4 is not.
- Estimator variance after DD bias inversion is reported and matches simulation.
- Decoder-SNR override changes `x_hat` but leaves transmitted symbols bit-identical and on-constellation.
- `snr_at_encoder=False` arms have no encoder AF parameters.
- Usage entropy and k_a computed from the histogram match closed-form values for uniform QAM.

### 5.3 Compute

| Item | Run-equivalents |
|---|---|
| Seed-variance pilot (3 seeds, arm B, M = 16) | 3 |
| Main arms at M = 16 (A, B, B-loop, C), n seeds (n = 5 assumed) | 20 |
| λ sweep (B, C × 3 new λ × 2 seeds) | 12 |
| M = 4, 64 (A, B, B-loop, C, 1 seed) | 8 |
| Analog A/B/C | 3 |
| Kodak-scale (B, C at 128×128; ~16× cost per epoch) | ~8 |
| M = 256 (conditional) | 4 |
| **Total** | **~55–60 runs ≈ 130–200 GPU-hours** |

At ~3 h per CIFAR run on the RTX 3050 this is 6–8 days of continuous training; with parallel
Kaggle sessions, ~3–4 days of wall clock. If the power analysis returns n = 7, add ~8 runs. Cuts
available if needed: drop analog arms (keep the existing R=1/12 analog curves as reference), run
the λ sweep at one seed first, drop M = 4.

---

## 6. Step 0 — do this before committing

**Measure how non-uniform trained usage actually is.** The R=1/12 weights were lost, so this cannot
be done by inference right now. Two routes:

1. **Today, no compute:** read `train/kl` at the end of training for the two digital runs
   (`adjsccq_r0.0833_m16_snr0-20`, `deepjsccq_*`) in the wandb `adjscc-q` project. The repo computes
   `KL(P̂ ‖ U) = Σ p log(p·M)` in nats on the batch-mean soft usage, so usage entropy is
   **`H = ln 16 − KL`** nats. At σ_q = 100 soft usage ≈ hard usage. This gives entropy but not k_a.
2. **First new run:** log the full usage histogram, entropy and k_a (average and per image), and do
   one short λ = 0 run to see how much shaping appears without the regulariser.

---

## 7. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| **1.** Trained usage near-uniform at every λ | **High** | Step 0; decision rule §4.6; JCM's results suggest shaping appears when unregularised |
| **2.** Seed variance swamps RQ2 | Medium | Power analysis before main runs; paired seeds; defined 0.1–0.2 dB band |
| **3.** Weak practical motivation | **High** | Restrict to broadcast/no-feedback (scenario a); state AWGN-only scope; do not claim pilot-overhead savings |
| **4.** Reviewers call estimators textbook | Medium | Say so (§2.3); lead with the trade-off, not the estimators |
| **5.** M = 256 usage collapse at R = 1/12 with straight-through quantisation | Medium | Monitor entropy; fall back to dropping M = 256 or using R = 1/6 |
| **6.** Not converged | High | Convergence gate on every arm |
| **7.** Losing runs | High | `wandb.save()` on `best.pt`; commit run records between runs |
| **8.** Kodak-scale arm too expensive on a 3050 | Medium | Kaggle; reduce epochs with convergence check; single M |

---

## 8. Timeline (~10 weeks, part-time, RTX 3050 + optional Kaggle)

| Week | Work | Deliverable | Gate |
|---|---|---|---|
| **0** | Pre-reading (§10). Run-safety fixes. **Step 0 route 1** (wandb `train/kl`) | Usage entropy from existing runs; safe pipeline | Usage not obviously uniform, or decide to test λ = 0 |
| **1** | Encoder/decoder flags; decoder override; estimators with windows; usage logging; tests | Estimator module matching §4.8 | Tests pass |
| **2** | Seed-variance pilot (3× arm B); one λ = 0 run (Step 0 route 2) | σ_seed; required n; shaping at λ = 0 | **§4.6 decision rules** |
| **3–4** | Main arms at M = 16 with n seeds; B-loop; B-avg; λ sweep | RQ2 with confidence intervals; RQ1 trade-off curve | Convergence report |
| **5** | M = 4, 64 arms (M = 256 if stable); analog A/B/C | Secondary M results | — |
| **6** | Inference: all estimators × windows × λ × M; learned estimator; scenario (b) CQI | RQ1, RQ3, RQ4 results | — |
| **7** | Kodak-scale B and C | Larger-image check | — |
| **8** | Analysis, figures, related work (§2) | Draft | — |
| **9–10** | Buffer; final novelty re-check; verify every citation | Submission-ready draft | — |

---

## 9. Revision history — v1 → v2 (independent review)

| # | Concern | Verdict | Change |
|---|---|---|---|
| 1 | SIJSCC Fig. 7 *is* matched-architecture; the confound is in Fig. 4 vs ADJSCC | **Accepted** — v1 mischaracterised it | Novelty restated: first seed-replicated digital test of a single-run analog ablation. CBJSCC quote added (gap attributed to "inherent unpredictability", no error bars) |
| 2 | Pilot vs blind compared at unequal windows | **Accepted**, confirmed by simulation | All estimators at equal windows of 1/4/16/64 images (§4.8) |
| 3 | Pilot-overhead motivation is a CIFAR artifact; SNR is a link property | **Accepted** | Pilot-overhead argument dropped; Kodak-scale evaluation added |
| 4 | Genie `h` makes blind SNR estimation under fading a non-problem | **Accepted** | Primary claims AWGN only; fading as a stated limitation; motivation risk raised to High |
| 5 | Modulation selection needs feedback; DD-driven CQI is classical | **Accepted** | Two scenarios separated; RQ4 recast as blind vs pilot CQI at equal overhead, PSNR cost; AMC literature added (Goldsmith & Chua verified) |
| 6 | Emergent shaping may be weak; elevate the λ sweep | **Accepted** | λ sweep is now RQ1 (core). JCM found and cited as motivation; kurtosis mechanism added. Step 0 moved before commitment |
| 7 | Arm C is an implicit estimator; specify soft vs hard `y` | **Accepted** — repo verified to feed soft `y` | B-est vs C made the central comparison |
| 8 | Statistical power inadequate; 0.1–0.2 dB band undefined | **Accepted** | Seed-variance pilot and power analysis; decision rules for all bands |
| 9 | Estimation regret overstated by train/test mismatch | **Accepted** | B-loop (estimator in the loop) is a main arm |
| m1 | DD bias correction amplifies variance; needs usage info | Accepted | Report post-inversion variance; usage-aware correction |
| m2 | Clipping hides DD over-estimation | Accepted | Report with and without clipping, or extend AF range |
| m3 | "Dynamic range 5.06 vs 8.34 dB" undefined | Accepted | It was PSNR(20 dB) − PSNR(0 dB) of the adaptive curves; removed from the hypotheses as too weak to support them |
| m4 | SER thresholds misaligned | Accepted | Explained as different quantities (estimator property vs operating rule) |
| m5 | M = 256 collapse risk | Accepted | Conditional arm with fallback |
| m6 | Missing related work | Accepted | JCM, importance-aware constellation design, classical AMC added |
| m7 | SNR-EQ-JSCC average-SNR baseline | Accepted | B-avg arm |
| m8 | Unverified citations | Accepted | Pauluzzi & Beaulieu and Goldsmith & Chua verified; remaining from-memory items marked |
| m9 | "Genie SNR" overstated for Park et al. | Accepted | "Perfect SNR or CSI knowledge" |

---

## 10. Prerequisites and pre-reading

### 10.1 Core — in this order

| # | Paper | What to take from it |
|---|---|---|
| 1 | Bourtsoulatze, Kurka, Gündüz, "Deep Joint Source-Channel Coding for Wireless Image Transmission," *IEEE TCCN* 2019, arXiv:1809.01733 *(from memory; verify)* | DeepJSCC baseline; graceful degradation |
| 2 | ADJSCC, [arXiv:2012.00533](https://arxiv.org/abs/2012.00533) | AF modules; SNR at encoder and decoder |
| 3 | DeepJSCC-Q, [arXiv:2206.08100](https://arxiv.org/abs/2206.08100) | Fixed-QAM soft-to-hard quantiser; the KL usage regulariser (the λ knob) |
| 4 | JCM, [arXiv:2310.06690](https://arxiv.org/abs/2310.06690) | Learned symbol probabilities on fixed QAM; Figs. 7–8 (shape vs SNR); assumes perfect channel knowledge |
| 5 | SIJSCC, [arXiv:2306.15183](https://arxiv.org/abs/2306.15183) — Fig. 4 and **Fig. 7** | The matched-backbone ablation being replicated, and where the real confound is (Fig. 4) |
| 6 | CBJSCC, [*Sensors* 2024](https://www.mdpi.com/1424-8220/24/12/4005) §4.4 | The randomness attribution and absence of variance |
| 7 | Zhang et al., [arXiv:2501.01138](https://arxiv.org/html/2501.01138) §V-C | Learned blind SNR estimation |

### 10.2 Classical

- Pauluzzi & Beaulieu, *IEEE TCOM* 48(10):1681–1691, 2000 — estimator families, M2M4, Cramér–Rao
  bounds (M-PSK).
- A non-constant-modulus / QAM extension of moment-based estimation (e.g.
  [this paper](https://www.researchgate.net/publication/4080481_SNR_estimation_for_non-constant_modulus_constellations)).
- Goldsmith & Chua, *IEEE TCOM* 45(10):1218–1230, 1997 — adaptive M-QAM; then an imperfect-CSI AMC
  follow-up.
- [Blind estimation under probabilistic shaping, arXiv:1806.10062](https://arxiv.org/pdf/1806.10062).
- 3GPP TS 38.214 CQI/MCS reporting *(from memory; optional)*.

### 10.3 Secondary

uJSCC ([arXiv:2405.10749](https://arxiv.org/abs/2405.10749)), Park et al.
([arXiv:2311.08146](https://arxiv.org/abs/2311.08146)), SNR-EQ-JSCC
([arXiv:2501.04732](https://arxiv.org/abs/2501.04732)), SecDiff
([arXiv:2511.01466](https://arxiv.org/abs/2511.01466)).

### 10.4 Concepts

- Square M-QAM geometry, unit-power normalisation, symbol error rate vs SNR.
- Complex Gaussian noise; second and fourth moments; kurtosis; why a Gaussian signal in Gaussian
  noise is not identifiable by moments.
- Probabilistic constellation shaping (Maxwell–Boltzmann), entropy of the usage distribution.
- Estimator bias, variance, RMSE, Cramér–Rao bound; why pooling reduces variance but not bias.
- Power analysis for comparing means across random seeds.

### 10.5 Repo

- [r12-results-analysis.md](r12-results-analysis.md) §§5 and 8 (sensitivity floor, harness).
- [future-work-plan.md](future-work-plan.md) §1.1 (run safety).
- `semcom/constellation.py` (`_last_usage`, `kl_to_uniform`), `semcom/models.py` (AF placement,
  soft `y` into the decoder), `semcom/channel.py`.
