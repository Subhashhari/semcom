# How Does a Blind JSCC Receiver Know the Noise Level?

**Research proposal, v2: why deep JSCC decoders work without being told the SNR, which cue they use
in its place, what it costs in model capacity, and an SNR-independent digital model built on the
answer.**

Status: proposal, not started. v2 revises v1 after an independent review; every concern and the
change it produced is in §11. Builds on this repo's 2×2 harness (analog/digital × fixed/adaptive)
and keeps the digital shaping study as an optional second paper. That study's proposal is archived:
`git show 976db88:docs/snr-self-estimation-proposal.md`. The first week is a kill test with outcomes fixed in advance (§6).

---

## 0. Summary

**The problem.** Most SNR-adaptive deep JSCC systems give the decoder the true SNR, and often the
encoder too. Several papers have since shown that removing the SNR costs almost nothing. Nobody has
explained *why*, so nobody can say *when* it stops being true, or whether the answer changes for
digital (fixed-constellation) systems or for small, cheap models.

**The observation behind this proposal.** Deep JSCC encoders normalise transmit power for every
image, so the received energy reveals the noise level: `σ̂² = (‖y‖² − k)/k`. A blind decoder may
simply be reading that. The estimate is poor when the decoder sees few symbols at high SNR. The
number of symbols that matters is the decoder's **receptive field** (about 1,000 symbols for this
repo's CNN), not the image size, so the poor regime exists even for large images. In digital JSCC,
quantisation makes the transmitted energy vary from image to image, so the energy cue degrades,
while the constellation offers a better cue (the residual to the nearest point).

**What we do.**

1. **Kill test (week 1):** a matched blind vs SNR-conditioned comparison, with seeds, against a
   quantitative prediction of what an energy-reading decoder would lose.
2. **Mechanism:** *cue-conflict* tests that make the candidate cues disagree, to see which one the
   blind decoder follows. Probes trained on natural data are read out under conflict.
3. **Capacity:** separate encoder and decoder width sweeps across SNR. They measure whether
   blindness costs decoder capacity and which side of the link needs capacity at which SNR.
4. **Deliverable: SI-JSCC-Q,** an SNR-independent digital JSCC model with fixed 16-QAM, one set of
   weights and no SNR or pilots anywhere. It comes in two versions: plain blind, and
   self-conditioned on statistics the receiver computes itself.

**Contribution type.** Mechanism, design guidance and a concrete model. The guidance takes the form
*"blindness is safe above effective k ≈ X at SNR Y with per-image power normalisation; below that,
or for digital QAM, use estimator Z; and if the decoder is small, it needs W extra width to stay
blind."*

**Cost.** Week 1: about 40 GPU-hours. Full study: about 9 weeks and 150–220 GPU-hours on the
RTX 3050, less with parallel Kaggle sessions.

---

## 1. Motivation

### 1.1 Why the SNR matters

Deep JSCC maps an image straight to complex channel symbols with a learned encoder, and back with a
learned decoder. It degrades gracefully: a worse channel gives a blurrier image, not a lost packet.
But the best encoding and decoding depend on channel quality. A decoder trained at 1 dB denoises
aggressively, while one trained at 19 dB trusts its input. In this repo's R = 1/12 results, digital
specialists lose several dB when tested away from their training SNR
([r12-results-analysis.md](../prior-work/adjscc-q/r12-results-analysis.md)). The field therefore moved to one model
conditioned on the SNR, and that assumes somebody supplies the SNR.

### 1.2 Where the SNR comes from

| Use of SNR | Who must know it | In practice |
|---|---|---|
| Encoder conditioning (ADJSCC) | Transmitter | Feedback from the receiver: latency, overhead, staleness; impossible in broadcast |
| Decoder conditioning | Receiver | Pilots (overhead) or blind estimation (error) |
| Neither (blind) | Nobody | The network copes on its own |

In broadcast, multicast or any feedback-free link, encoder conditioning is unavailable. Decoder
conditioning needs an estimator, and every estimator has error. The blind option is simplest to
deploy, but only if it is safe.

### 1.3 What a practitioner cannot currently answer

The published evidence says blindness is nearly free (§2.2), but it comes from analog systems, from
single training runs, from large images, from full-size models and without an explanation. So these
questions have no answer beyond "try it":

- *Will dropping the SNR input hurt my system at my SNR?*
- *Does it hurt more with QAM symbols?*
- *Does it hurt more if I shrink the decoder to fit a handset?*

A mechanism that predicts the loss turns them into design rules.

---

## 2. Previous approaches

### 2.1 SNR-conditioned JSCC

| Work | What it does | SNR assumption |
|---|---|---|
| DeepJSCC, Bourtsoulatze et al., *IEEE TCCN* 2019, [arXiv:1809.01733](https://arxiv.org/abs/1809.01733) | First CNN image JSCC | Trained per SNR |
| ADJSCC, [arXiv:2012.00533](https://arxiv.org/abs/2012.00533) | Attention-feature (AF) modules: squeeze-and-excitation with the SNR appended | True SNR at both ends |
| DeepJSCC-Q, [arXiv:2206.08100](https://arxiv.org/abs/2206.08100) | Soft-to-hard quantiser onto fixed M-QAM; soft symbols into the decoder | Trained per SNR |
| uJSCC / euJSCC, [arXiv:2405.10749](https://arxiv.org/abs/2405.10749), [arXiv:2602.14018](https://arxiv.org/abs/2602.14018) | Single digital model across modulation orders; learned VQ codebooks | SNR/CQI at both ends (*"the receiver uses the CQI generated δ blocks earlier"*) |
| JCM, Bo et al., *IEEE TCOM* 2024, [arXiv:2310.06690](https://arxiv.org/abs/2310.06690) | Fixed QAM, learned symbol probabilities | Perfect channel knowledge |
| SNR-EQ-JSCC, [arXiv:2501.04732](https://arxiv.org/abs/2501.04732) | Instantaneous or long-run average SNR | Includes imperfect feedback |

### 2.2 SNR-free JSCC: it works, unexplained

| Work | Finding | Limitation |
|---|---|---|
| DeepJSCC, slow Rayleigh fading | *"we do not assume channel state information either at the receiver or the transmitter, or consider the transmission of pilot signals"*; *"the network learns to estimate the channel state"* | Inferred from performance; mechanism not examined |
| SIJSCC, [arXiv:2306.15183](https://arxiv.org/abs/2306.15183) | No SNR anywhere; matches or beats ADJSCC. Fig. 7: SNR at encoder+decoder / decoder only / none, same backbone | Single run, no variance, analog. Fig. 4 compares different architectures |
| CBJSCC, [*Sensors* 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11209452/) (same first author) | §4.4: ADJSCC's AF module in the same three settings, *"no method was significantly superior"*, attributed to *"the inherent unpredictability of deep learning models"* | No error bars; analog; 128×128 crops |
| STARJSCC, [*Sci. Rep.* 2025](https://www.nature.com/articles/s41598-025-16753-4) | Table 2: SE / CBAM / CSA attention with and without SNR; SNR adds 0.26 dB to CSA | One operating point (CBR 1/6, 13 dB, Kodak) |

### 2.3 Digital JSCC without the SNR

| Work | What it does | Why it is not a matched blind soft-symbol digital study |
|---|---|---|
| Low-SNR-robust JSCC, [arXiv:2604.20278](https://arxiv.org/abs/2604.20278) | Uniform quantisation + M-QAM; *"minimum distance demodulation is first performed on y"* | Hard decisions discard soft information, so there is nothing for an SNR to condition; uses channel equalisation |
| Park et al., [arXiv:2311.08146](https://arxiv.org/abs/2311.08146) | Single model, robust training over erasure channels; fixed 4/16/64-QAM | SNR enters via the LLRs and modulation selection; pilot CSI |
| BlindSC, [arXiv:2501.02273](https://arxiv.org/abs/2501.02273) | "Blind" means training without specifying the channel | Inference *"adaptively selects power and modulation levels based on... channel conditions"* |
| D²-JSCC, [arXiv:2403.07338](https://arxiv.org/abs/2403.07338) | Digital; decoder not SNR-conditioned | Channel *"perfectly known at both the transmitter and receiver"* |
| DeepJSCC-Q per-SNR models | No SNR input | One model per SNR, so blind only in a trivial sense |

### 2.4 Model capacity, depth and asymmetry

| Work | What it does | What it does not do |
|---|---|---|
| Implicit-JSCC / effective depth, [arXiv:2606.29737](https://arxiv.org/abs/2606.29737) | A depth–SNR model: *"the SNR-dependent refinement depth required to reach a prescribed perturbation tolerance"*; receiver-side refinement grows with noise | Uses SNR as input; equilibrium (implicit) networks; does not vary encoder and decoder capacity independently or study blindness |
| DD-JSCC, [arXiv:2507.20467](https://arxiv.org/abs/2507.20467) (ICC 2025) | Dynamic layer activation by device capability and channel | Forces equal depths: *"we align their layer configurations... 𝐋d = 𝐋e"*; SNR input at both ends |
| G-UNet-JSCC, [arXiv:2602.22691](https://arxiv.org/abs/2602.22691) | *"a relatively simple encoder paired with a powerful U-Net-based decoder"*, for constrained transmitters | Design choice, no ablation over the split; trained per SNR |

### 2.5 SNR estimation

| Family | Mechanism | Weakness |
|---|---|---|
| Data-aided (pilots) | Known reference symbols | Overhead; needs a frame structure |
| Energy | Known transmit power, measured receive power | Assumes known transmit power; cross term dominates at high SNR |
| Decision-directed (DD) | Residual to the nearest constellation point | Over-estimates SNR below the symbol-error threshold; needs a lattice and correct scale |
| Moment-based (M2M4) | Second and fourth moments with known kurtosis | Breaks for Gaussian-like signals; noisy at high SNR. **Does not assume known signal power** |
| Learned | Network regresses SNR; Zhang et al., [arXiv:2501.01138](https://arxiv.org/html/2501.01138) | Analog; explicit module |

Reference: N. Pauluzzi and N. C. Beaulieu, *IEEE Trans. Commun.* 48(10):1681–1691, 2000.

### 2.6 Known vs open

**Known; not claimed here:**

- Removing the SNR costs little in analog JSCC on large images.
- Attention with vs without SNR has been ablated (CBJSCC, STARJSCC).
- Classical estimators and their error behaviour are textbook.
- Decoder refinement depth grows with noise, for SNR-conditioned implicit networks (Implicit-JSCC).
- Asymmetric "light encoder, heavy decoder" designs exist (G-UNet-JSCC), without analysis.

**Open, as far as about 40 searched papers show:**

1. **Which cue a blind decoder uses**, tested by interventions rather than correlation.
2. **A predictive account of the blind penalty** in terms of effective k, SNR and decoder
   sensitivity.
3. **A matched, seed-replicated blind vs conditioned comparison in digital JSCC** with soft
   symbols. Blind digital models exist, but not this comparison.
4. **Independent encoder and decoder capacity sweeps across SNR**, and whether blindness has a
   capacity cost.
5. **A digital decoder conditioned on receiver-computed lattice statistics instead of the SNR.**

---

## 3. The idea

### 3.1 The energy hypothesis

Every encoder in this repo normalises power per image ([channel.py:40](../../semcom/channel.py#L40)),
so `‖z‖² = k`. Over AWGN, `‖y‖² = k + 2·Re⟨z, n⟩ + ‖n‖²`, giving `σ̂² = (‖y‖² − k)/k` with
relative standard deviation `√((2·SNR + 1)/k)`. The `2·SNR` cross term makes energy estimation worse
as the channel gets better.

**Effective k.** A convolutional decoder pools only over its receptive field. This repo's decoder
(three stride-1 5×5 layers, then two upsampling layers) sees about 16×16 latent positions, or about
1,000 complex symbols at 4 symbols per position. At CIFAR scale (8×8 latent, k = 256) the field
covers the whole latent. On larger images effective k stays near 1,000 however big the image is.
That is why the high-SNR, small-k regime matters beyond CIFAR. The *effective* receptive field is
smaller than this geometric bound.

### 3.2 Simulated estimator error

SNR estimation error in dB (estimate − truth). Each cell gives the **failure rate** (noise estimate
≤ 0, SNR undefined), then the **median and interquartile range** among valid estimates. AWGN,
4,000 trials; Gaussian latent for analog; uniform symbol usage for digital.

**Small effective k (256):**

| Estimator | 0 dB | 10 dB | 20 dB |
|---|---|---|---|
| Energy, analog | 0% · +0.02 [−0.30, +0.34] | 0% · +0.05 [−0.74, +1.00] | **13%** · −0.59 [−2.27, +1.96] |
| Energy, 16-QAM | 0% · +0.02 [−0.31, +0.37] | 1% · −0.08 [−1.21, +1.45] | **41%** · −4.54 [−6.75, −1.44] |
| Energy, 16-PSK (constant energy) | 0% · +0.02 [−0.28, +0.35] | 0% · +0.02 [−0.73, +0.92] | 12% · −0.51 [−2.21, +1.89] |
| DD, 16-QAM | 0% · **+4.63** [+4.32, +4.97] | 0% · +1.56 [+1.40, +1.73] | 0% · 0.00 [−0.18, +0.18] |
| Hybrid, 16-QAM: mean ‖y‖² − mean ‖Q(y)‖² | 0% · +0.38 [+0.09, +0.72] | 0% · 0.00 [−0.71, +0.85] | 12% · −0.57 [−2.24, +1.85] |

**At 20 dB, larger effective k:**

| Estimator | k = 1,024 | k = 4,096 |
|---|---|---|
| Energy, analog | 1% · −0.08 [−1.17, +1.36] | 0% · 0.00 [−0.60, +0.69] |
| Energy, 16-QAM | **28%** · −2.14 [−4.26, +0.51] | **13%** · −0.64 [−2.39, +1.91] |
| Energy, 16-PSK | 1% · −0.04 [−1.14, +1.45] | 0% · +0.02 [−0.58, +0.74] |
| DD, 16-QAM | 0% · 0.00 [−0.09, +0.10] | 0% · 0.00 [−0.05, +0.04] |

Readings:

- **Energy fails at high SNR and small k.** In analog, failures fall from 13% to about 1% between
  k = 256 and k = 1,024.
- **For 16-QAM energy is much worse at high SNR,** because the transmitted energy is no longer
  fixed: 41% failures at k = 256 and still 13% at k = 4,096. At low and mid SNR it is almost as good
  as analog, which corrects v1's claim of a 6 dB error at 10 dB (an RMS figure dominated by
  outliers).
- **16-PSK behaves like analog,** since every symbol has the same energy. That makes it the
  constant-energy control (§4.3).
- **DD is exact at high SNR and badly biased at low SNR.** Energy is the reverse. The two fail in
  regions that barely overlap.
- **The hybrid repairs the 16-QAM energy cue** at a small low-SNR bias.

### 3.3 Why digital is different

In analog JSCC the energy cue is clean and the only structure cue is the learned latent manifold.
Well-trained encoders make their latents fill the space, which limits that cue. In 16-QAM JSCC the
energy cue degrades at high SNR, while the lattice provides an exact structure cue there. **The
prediction:** analog blind decoders lose at high SNR and small effective k. Digital blind decoders
either lose more there (if they read energy) or less (if they learn a DD-like cue). Which one
happens is the first digital result.

### 3.4 Capacity: does blindness cost decoder capacity?

A conditioned decoder is handed the noise level. A blind one must work it out from y with the same
layers it uses to denoise, so blindness may compete for capacity. Two consequences can be tested:

- **The blind penalty should grow as the decoder shrinks.** A large decoder can spare capacity for
  estimation; a small one cannot. If so, "blindness is free" holds only for large models, which
  matters exactly where blind receivers are attractive: cheap broadcast receivers.
- **Encoder and decoder capacity should matter at different SNRs.** At low SNR the decoder's job is
  heavy inference under noise, which suggests decoder capacity dominates. Implicit-JSCC's depth–SNR
  result points the same way for conditioned models. At high SNR the channel is nearly clean and
  quality is limited by how well the encoder compresses, which suggests encoder capacity dominates.
  If true, this gives a capacity-allocation rule for asymmetric links. Uplink IoT has a weak
  encoder; a broadcast handset has a weak decoder.

These questions also supply a control the deliverable needs. If self-conditioned SI-JSCC-Q (§3.6)
beats plain blind, is that because of the information in the statistics, or just the extra
parameters? A plain blind decoder made slightly wider separates the two.

### 3.5 Research questions

| | Question | Hypothesis |
|---|---|---|
| **RQ1** | Is the matched blind penalty C − B distinguishable from zero, and how does it compare with the penalty predicted for an energy-reading decoder? | **H1:** analog C − B is near zero at low SNR and positive at high SNR for k = 256 |
| **RQ2** | Which cue does the blind decoder follow when the cues disagree? | **H2a:** analog C follows energy. **H2b:** 16-QAM C follows a DD-like cue at high SNR |
| **RQ3** | Does the blind penalty depend on decoder capacity? | **H3:** C − B grows as decoder width shrinks |
| **RQ4** | How does the marginal value of encoder vs decoder capacity change with SNR? | **H4:** decoder capacity matters more at low SNR, encoder capacity more at high SNR |
| **RQ5** | Can a digital model with no SNR anywhere match the genie-decoder model? | **H5:** self-conditioned SI-JSCC-Q is within seed noise of B at every SNR, and its advantage over plain blind exceeds that of an equally sized wider plain blind model |
| RQ6 *(below the cut line)* | How do non-stationary noise, unknown-gain fading and non-Gaussian noise break blindness? | Each breaks a specific cue (§7.3) |
| RQ7 *(second paper)* | Shaping–estimability trade-off | See the archived proposal (commit 976db88) |

**Crossover (H2b) defined in advance:** for each system (analog, 16-QAM), plot the blind penalty
C − B in dB against SNR. A crossover means the two penalty curves' 95% confidence bands cross,
with analog worse at high SNR and 16-QAM worse at low SNR.

### 3.6 The deliverable: SI-JSCC-Q

One model, fixed 16-QAM symbols (DeepJSCC-Q's quantiser), trained over SNR ~ U[0, 20] dB, with
**no SNR at either end and no pilots**.

**Plain.** DeepJSCC-Q trained over the SNR range with no SNR input. It runs on the current code:

```bash
python -m semcom.train --config configs/cifar_r12.yaml --digital -M 16
```

Leaving out `--snr-adaptive` and `--snr-train-fixed` makes it sample U[0, 20] dB with no AF
modules. A second variant keeps the attention modules but with no SNR input (C-att). It separates
"no SNR" from "no attention".

**Self-conditioned.** ADJSCC-Q's decoder attention modules, with the SNR slot filled by statistics
the receiver computes from the soft symbols it receives:

| Statistic | Strong where (§3.2) |
|---|---|
| Received energy `(‖y‖² − k)/k` | Low and mid SNR |
| DD residual: mean of ‖y − Q(y)‖² | High SNR |
| Hybrid: mean ‖y‖² − mean ‖Q(y)‖² | Mid and high SNR |
| Moments `M2`, `M4` | Scale-invariant; mid SNR |
| Local (windowed) versions | Non-stationary noise |

The statistics enter the attention MLP as a small vector, and the network learns to combine them.
The encoder stays blind, so the scheme needs no feedback. Energy and DD fail in regions that barely
overlap (§3.2), so together they should cover 0–20 dB, which the plain blind model may not learn on
its own.

**Comparisons:**

- DeepJSCC-Q specialists (envelope)
- ADJSCC-Q (genie SNR at both ends; needs feedback)
- arm B (genie decoder; the realistic ceiling)
- plain, plain-wide and self-conditioned SI-JSCC-Q
- the separation baseline in `semcom/separation.py`

Results: PSNR over 0–20 dB with 3+ seeds, plus storage (one model vs specialists).

**Novelty, honestly.** The plain version is SIJSCC's recipe moved onto DeepJSCC-Q. It is open as a
matched, seed-replicated study, but a modest step. The self-conditioned version is the more
original part: no paper found feeds receiver-computed lattice statistics to a JSCC decoder in place
of the SNR.

---

## 4. Experimental design

### 4.1 Arms

Shared backbone; R = 1/12, CIFAR-10, AWGN. **B and C differ only in whether the decoder receives
the SNR.**

| Arm | Encoder SNR | Decoder input besides y | Role |
|---|---|---|---|
| **A** | true | true SNR | ADJSCC / ADJSCC-Q reference |
| **B** | — | true SNR | Genie decoder; source of the sensitivity curve |
| **C** | — | nothing | Blind; mechanism subject. Digital C = plain SI-JSCC-Q |
| **C-att** | — | nothing (attention kept, no SNR input) | "No SNR" vs "no attention" |
| **C-wide** | — | nothing; decoder widened to match D's extra parameters | Information-vs-capacity control |
| **B-loop(energy)** | — | energy estimate, also in training | Does an explicit energy statistic help? |
| **D** | — | statistics vector (§3.6), end to end | Self-conditioned SI-JSCC-Q (digital) |

**What B-loop and D can and cannot show.** A decoder that sees y *and* a statistic can do anything
C can, so B-loop and D bound C from above, up to optimisation noise. They answer *"does handing the
decoder this statistic help?"*, which is the deliverable's question. They cannot show which cue C
uses: at CIFAR scale C can compute the same statistic itself. The mechanism question is answered
only by the cue-conflict tests of §4.3.

**Modulations.** Analog and 16-QAM throughout; **16-PSK** for B and C as the constant-energy
control; QPSK optional (see §4.3).

### 4.2 Measurements

1. **Sensitivity curve** (inference on B): PSNR when the decoder is given SNR_true + Δ, for
   Δ ∈ [−10, +10] dB, at each true SNR. Reported **with and without** clipping to the training
   range, and B is trained with its SNR input sampled over a wider range (−5 to 25 dB) so that
   clipping is rarely needed.
2. **Predicted energy penalty:** average the sensitivity curve over the energy estimator's error
   distribution at the matching effective k and SNR, and also feed each image its own estimate.
   Failures (noise estimate ≤ 0) are floored at the top of B's input range, and the failure rate is
   reported.
3. **Matched gaps:** C − B and every other arm against B, as PSNR vs SNR over 0–20 dB, 10k test
   images × 10 channel realisations, with **paired** noise across arms.
4. **Cue-conflict tests** (§4.3).
5. **Probes** trained to read log σ² from C's decoder layers **on natural data**, then **applied to
   the cue-conflict inputs**, where the cues disagree. Probes trained where all cues are collinear
   cannot discriminate between them; reading them out under conflict can.
6. **Usage statistics** (digital): histogram, entropy and kurtosis, overall and per image.

### 4.3 Cue-conflict tests (inference only)

Each test builds inputs on which the candidate cues imply different SNRs, then checks which one the
blind decoder C follows. The behavioural reference: C's PSNR under the conflict compared with B
given each cue's implied SNR, on the same inputs.

| Test | Construction | Cues it separates | Readout |
|---|---|---|---|
| **Power mismatch** | Transmit `αz` (α ∈ {0.7, 0.85, 1.2, 1.4}) with noise set so the true SNR `α²/σ²` is unchanged | Cues that **assume unit transmit power** (energy, DD, re-encoding residual), which are all misled, vs **scale-invariant** cues (M2M4-like signal-and-noise estimation), which are not | PSNR vs B given the energy-implied SNR and vs B given the true SNR; probe readout |
| **Symbol shuffling** | Permute the received symbols' positions (energy and lattice membership kept) | **Manifold** cues (re-encoding residual) break; energy and DD survive | Probe readout only: the image is destroyed, so PSNR says nothing about C's internal SNR |
| **16-PSK vs 16-QAM** | Same bits per symbol; PSK has constant energy, so per-image energy stays exact after quantisation | Energy cue clean (PSK) vs degraded (QAM) | Blind penalty C − B for each |
| QPSK *(optional)* | Constant energy, 2 bits/symbol | As above | Weak: QPSK decisions are almost error-free above ~8 dB, so its decoder barely depends on the SNR in the high-SNR regime that matters |

**The power-mismatch test needs care.** The decoder's denoising depends on input scale whatever its
SNR estimate is. Comparing C with B on the *same* scaled inputs cancels that only if B and C are
equally robust to scale. Using several values of α and checking the trend, rather than one point,
separates "C reads energy" from "C is fragile to scale".

**Interpretation:**

- **C follows the energy-implied SNR under power mismatch, and 16-PSK's blind penalty differs from
  16-QAM's at high SNR as §3.2 predicts:** the energy story holds.
- **C follows the true SNR under power mismatch:** C estimates signal and noise separately. This is
  the more interesting paper, and the probes then have a specific target.
- **The probe readout moves under shuffling but not under power mismatch, or the reverse:** that
  identifies a manifold cue or an energy cue respectively.

### 4.4 Capacity sweeps (RQ3–4)

Split `hidden` into `hidden_enc` and `hidden_dec` (the encoder and decoder classes already take
width separately; [models.py](../../semcom/models.py)). Each run trains one model over U[0, 20] dB and
yields a whole PSNR-vs-SNR curve, so a grid of widths costs one run per cell, not per SNR.

| Sweep | Grid | Arms | Purpose |
|---|---|---|---|
| **Decoder width** | `hidden_dec` ∈ {64, 128, 256}, `hidden_enc` = 256 | B and C, analog and 16-QAM | RQ3: blind penalty vs decoder size |
| **Asymmetric grid** | `hidden_enc` × `hidden_dec` ∈ {64, 128, 256}², 9 cells | B (conditioned), 16-QAM | RQ4: marginal value of each side across SNR |
| **Capacity control** | C-wide at D's parameter count | 16-QAM | RQ5: information vs capacity |

**Outputs:**

- **Capacity-allocation map:** PSNR(`hidden_enc`, `hidden_dec`, SNR), with the marginal gain of
  doubling each side, `∂PSNR/∂log₂(width)`, plotted against SNR.
- **Iso-parameter comparison:** the diagonal cells (64, 256), (128, 128) and (256, 64) have
  similar *total* parameter counts, since conv parameters scale with width squared. Comparing them
  at each SNR answers "given a fixed budget, which side should get it?" Parameters and FLOPs are
  reported separately, because the decoder works at higher spatial resolution in its last layers.
- **Blind penalty vs decoder width,** with confidence intervals.

**Caveats.**

- Smaller models converge at different speeds, so every cell uses the same learning-rate decay and
  must pass the convergence check (§4.5).
- AF module size scales with width, so ADJSCC's "under 1% overhead" is re-checked per cell.
- Width is varied, not depth. Depth changes the receptive field and so confounds RQ3 with effective
  k. Depth is the second receptive-field arm (§4.6) and kept separate.

### 4.5 Training and statistics

- **Learning-rate decay.** The R = 1/12 runs used a constant learning rate (no scheduler in
  `semcom/train.py`), and none converged in 150 epochs. All v2 runs use cosine decay. One
  calibration run (arm B) checks whether 200–300 epochs reach CONVERGED in
  `scripts/convergence_report.py` before committing compute.
- **Arms are compared at convergence, not at matched training loss.** The training loss is the MSE
  being compared, so matching it would erase the gap by construction. Gaps are also reported at
  intermediate checkpoints, so a gap that is still closing is visible.
- **Seeds.** Three paired seeds per arm (same data order and channel noise). Seed variance from the
  kill test feeds the power analysis (δ = 0.15 dB needs 2, 7 or 16 seeds for σ = 0.05, 0.10 or
  0.15 dB). Capacity grid cells start at one seed, with three seeds on the diagonal.
- **Estimator error** is reported as failure rate plus median and interquartile range, never RMS
  alone.

### 4.6 Generality of the design rule

The safe-blindness region depends on the receptive field, so the rule is plotted against
**effective k**, not image size, and stated as specific to the architecture family. Extra
effective-k points come cheaply from:

- **a second decoder with a smaller receptive field** (3×3 kernels in the stride-1 layers, about
  9×9 latent positions instead of 16×16);
- **R = 1/6 at CIFAR scale** (k = 512), as a secondary axis since it also changes the rate;
- **128×128 crops** (k = 4,096 total, about 1,000 effective) in Stage 2.

---

## 5. Decision rules (fixed before running)

### 5.1 The gate (end of week 1)

Let P be the energy penalty predicted from B's sensitivity curve (§4.2), and G the measured C − B
gap, each with a 95% confidence interval.

| Outcome | Meaning | Action |
|---|---|---|
| G within noise everywhere **and** P clearly nonzero somewhere | C does better than naive energy reading | Continue: the cue-conflict tests decide how; this is the more interesting paper |
| G clearly nonzero **and** consistent with P | C reads energy | Continue: the energy story, the digital contrast and capacity |
| G clearly nonzero **and** larger than P | Blind training costs more than estimation error explains (optimisation or capacity) | Continue with the capacity sweeps first (RQ3) |
| P ≈ 0 everywhere (decoder insensitive) | Blindness is free for a trivial reason at this scale | Drop the mechanism study; ship plain SI-JSCC-Q and the capacity map as a short paper |

"Clearly" means the 95% interval excludes zero.

### 5.2 What SI-JSCC-Q becomes

| Result | Deliverable |
|---|---|
| Plain blind ≈ B at every SNR | Plain SI-JSCC-Q is the result; the self-conditioned version is reported as unnecessary |
| Plain blind < B somewhere, D closes the gap, C-wide does not | Self-conditioned SI-JSCC-Q is the method; the gap it closes is the headline number |
| C-wide closes the gap as well as D | The gap was capacity, not information: report that and recommend widening the decoder |

### 5.3 Combined with Step 0 (usage entropy from wandb)

| | Gate passes | Decoder insensitive |
|---|---|---|
| **Usage clearly non-uniform** | This paper; the shaping study (RQ7) as a second paper | The shaping study alone, plus plain SI-JSCC-Q |
| **Usage near-uniform** | This paper; RQ7 dropped | Plain SI-JSCC-Q + capacity map as a short paper, or pivot to impairment-blind decoding |

---

## 6. Week 1: the kill test

| Step | Work | Compute |
|---|---|---|
| 0 | Run safety: `wandb.save` on `best.pt`. The R = 1/12 weights were lost, so nothing can reuse them | — |
| 0 | Step 0: final `train/kl` from wandb, usage entropy `H = ln 16 − KL` nats (soft usage, ≈ hard at σ_q = 100) | — |
| 1 | Code: encoder/decoder SNR flags; decoder SNR override; wider SNR input range for B; cosine decay; 16-PSK constellation; energy/DD/hybrid/M2M4 estimators; sensitivity and prediction evaluation; power-mismatch evaluation | — |
| 2 | Calibration: one arm B run with cosine decay; confirm convergence within 200–300 epochs | ~2 h |
| 3 | B and C: analog, 16-QAM, 16-PSK × 3 seeds (18 runs) | ~30–40 h |
| 4 | Sensitivity curves, prediction P, gaps G, power-mismatch test | Inference |
| 5 | Gate (§5.1) | — |

Runs take about 72 min per 150 epochs on the RTX 3050, so 200–300 epochs means about 1.6–2.4 h
each. Step 3 is two overnight runs, or about one day across three Kaggle sessions.

---

## 7. Stage 2 onward (if the gate passes)

### 7.1 Core (weeks 2–5)

| Weeks | Work | Compute |
|---|---|---|
| 2 | Symbol-shuffling test; probes trained on natural data and read out under conflict; second (3×3) decoder for B and C, 2 seeds | ~10 h |
| 2–3 | **SI-JSCC-Q:** C-att, D, C-wide, B-loop(energy), 16-QAM, 3 seeds | ~20–30 h |
| 3–4 | **Capacity:** decoder-width sweep (B, C × analog, 16-QAM × 3 widths, 2 seeds); asymmetric 3×3 grid (B, 16-QAM, 1 seed + diagonal seeds) | ~35–50 h (small models run faster) |
| 5 | 128×128 crops, B and C, 16-QAM; R = 1/6 at CIFAR | ~40–60 h |

### 7.2 Write-up (weeks 6–9)

| Weeks | Work |
|---|---|
| 6–7 | Analysis: safe-blindness region vs effective k, capacity-allocation map, SI-JSCC-Q table |
| 8 | Draft. Lead with effective k and the digital contrast, not CIFAR's k = 256 |
| 9 | Buffer; final novelty re-check; verify every citation |

### 7.3 Below the cut line (only if time remains)

RQ6 perturbations: piecewise SNR (SNR changes every w symbols, with the symbol-to-space mapping
stated), Rayleigh fading with unknown gain, impulsive and narrowband noise. RQ7 shaping study:
second paper.

**Total:** about 150–220 GPU-hours for the core, before the cut line.

### 7.4 Code changes

| Change | Where |
|---|---|
| `snr_at_encoder` / `snr_at_decoder` flags; decoder SNR override; wider SNR input range | `semcom/models.py`, `semcom/config.py`, `semcom/data.py` |
| `hidden_enc` / `hidden_dec`; 3×3-kernel decoder option | `semcom/config.py`, `semcom/models.py` |
| Cosine learning-rate decay | `semcom/train.py` |
| 16-PSK constellation (fixed point list; the quantiser is constellation-agnostic) | `semcom/constellation.py` |
| Estimators: energy, DD, hybrid, M2M4, re-encoding residual | new `semcom/snr_estimation.py` |
| B-loop training; AF modules that take a statistics vector (D); attention without SNR (C-att) | `semcom/train.py`, `semcom/modules.py` |
| Sensitivity, prediction, matched-gap and cue-conflict evaluation | new `scripts/blind_eval.py` |
| Probes trained on natural data, applied to conflict inputs | new `scripts/probe.py` |
| Capacity grid driver and allocation map | new `scripts/capacity_sweep.py` |

**Tests to write first:**

- the estimators reproduce §3.2 within tolerance;
- 16-PSK symbols are bit-exact and constant-energy;
- the decoder SNR override leaves transmitted symbols identical;
- power mismatch preserves the true SNR;
- shuffling preserves energy and lattice membership;
- encoder-blind arms have no encoder AF parameters;
- `hidden_enc` ≠ `hidden_dec` builds and trains.

---

## 8. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Decoder insensitive to SNR error at CIFAR scale | **High** | Gate row 4; the capacity map and plain SI-JSCC-Q still give a short paper |
| Runs still not converged with cosine decay | High | Calibration run before committing; gaps reported across checkpoints |
| Mechanism-only papers are hard to place in communications venues | High | Lead with the design rule, the capacity map and SI-JSCC-Q |
| Power-mismatch result confounded by scale fragility | Medium | Several α values; compare trends, not single points |
| Prior art appears in a later search (novelty has shrunk twice) | Medium | Re-search before Stage 2 and before writing; claims limited to §2.6 |
| Scope creep: mechanism, capacity and deliverable in one letter | **High** | Capacity grid limited to one seed off-diagonal; RQ6–7 below the cut line; if needed, split the capacity map into a separate short paper |
| Capacity-grid cells differ in convergence speed | Medium | Same schedule; convergence check per cell |
| Seed variance swamps small gaps | Medium | Paired seeds; power analysis after week 1 |

---

## 9. Pre-reading

| # | Reading | What to take from it |
|---|---|---|
| 1 | DeepJSCC, [arXiv:1809.01733](https://arxiv.org/abs/1809.01733), §IV-A | The "learns to estimate the channel state" claim |
| 2 | ADJSCC, [arXiv:2012.00533](https://arxiv.org/abs/2012.00533) | AF modules; where the SNR enters |
| 3 | SIJSCC, [arXiv:2306.15183](https://arxiv.org/abs/2306.15183), Figs. 4 and 7 | The matched ablation and its single-run limitation |
| 4 | CBJSCC, [*Sensors* 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11209452/), §4.4–4.5 | AF ablation; training range |
| 5 | STARJSCC, [*Sci. Rep.* 2025](https://www.nature.com/articles/s41598-025-16753-4), Table 2 | Attention with and without SNR |
| 6 | DeepJSCC-Q, [arXiv:2206.08100](https://arxiv.org/abs/2206.08100) | The quantiser and usage regulariser |
| 7 | Implicit-JSCC, [arXiv:2606.29737](https://arxiv.org/abs/2606.29737) | Depth–SNR model; closest work for RQ4 |
| 8 | DD-JSCC, [arXiv:2507.20467](https://arxiv.org/abs/2507.20467); G-UNet-JSCC, [arXiv:2602.22691](https://arxiv.org/abs/2602.22691) | Dynamic and asymmetric architectures |
| 9 | Pauluzzi & Beaulieu, *IEEE TCOM* 2000 | Estimator families, M2M4 |
| 10 | Alain & Bengio, "Understanding intermediate layers using linear classifier probes", 2016 *(from memory; verify)* | Probe methodology and its limits |
| 11 | Archived shaping proposal, `git show 976db88:docs/snr-self-estimation-proposal.md`, §§1.2, 4.4, 4.8 | Estimator definitions and simulations for the digital half |

Concepts: complex Gaussian noise and moments; why the cross term dominates energy detection at high
SNR; constant-modulus vs QAM constellations; receptive fields; probing vs intervention; paired
comparisons and power analysis.

---

## 10. Provenance of claims

- Quotes from DeepJSCC, CBJSCC, STARJSCC, the low-SNR-robust JSCC paper, BlindSC, euJSCC,
  Implicit-JSCC, DD-JSCC and G-UNet-JSCC were checked against fetched text, partly via page
  summaries for the last five; the key sentences are quoted.
- SIJSCC's figure contents come from an earlier reading in this project. The claim that its
  attention is local and windowed was not re-checked.
- D²-JSCC and Park et al. details come from the earlier novelty audit (archived shaping proposal §2.2, commit 976db88).
- The §3.2 simulations are idealised (Gaussian latent or uniform usage, AWGN, 4,000 trials).
- The receptive field (~16×16 latent positions) is a geometric bound from the layer configuration.
- Capacity scaling statements (parameters ∝ width²) are standard for convolutions; FLOPs will be
  measured, not assumed.

---

## 11. Revision history: v1 → v2 (independent review)

| # | Concern | Verdict | Change |
|---|---|---|---|
| 1 | B-loop(energy) sees y plus a statistic, so it bounds C from above; "C better than B-loop" is unreachable and "C ≈ B-loop" is the default | **Accepted** | B-loop and D moved to the deliverable question ("does a statistic help?"); mechanism answered by cue conflict (§4.1, §4.3) |
| 2 | Cues are collinear on natural data; probes show presence, not use | **Accepted** | Cue-conflict tests; probes trained on natural data and read out under conflict |
| 2a | Power mismatch leaves residual cues unaffected | **Partly rejected:** scaling also misleads DD and re-encoding, which assume unit power | Test reframed as power-assuming vs scale-invariant cues; several α values for the scale-fragility confound |
| 2b | Structure destruction breaks DD | **Rejected for DD:** shuffled or sign-flipped QAM symbols stay on the lattice; another image's latent is still a valid latent | Symbol shuffling kept as a manifold-cue test, with a probe readout because PSNR is meaningless there |
| 2c | QPSK as the constant-modulus control | **Partly accepted:** QPSK's decoder is nearly SNR-insensitive above ~8 dB | 16-PSK (same bits as 16-QAM, constant energy) is the main control; QPSK optional |
| 3 | The plug-in gate passes regardless | **Accepted** | New gate: measured gap vs predicted energy penalty, both with confidence intervals (§5.1) |
| 4 | Non-convergence biases the gap against blind arms | **Accepted**; "matched training loss" rejected because it erases the gap by construction | Cosine decay, calibration run, comparison at convergence, multi-checkpoint reporting |
| 5 | Rule rests on one architecture and two sizes | **Accepted** | Plotted against effective k; 3×3-kernel decoder; R = 1/6; stated as architecture-specific |
| m1 | "Blind digital JSCC untested" too strong | Accepted | §2.3 table; claim restated as matched, seed-replicated comparison |
| m2 | RMS in dB dominated by the floor convention | Accepted | §3.2 re-simulated with failure rate, median and interquartile range; v1's "6 dB at 10 dB" corrected |
| m3 | Clipping shapes the plug-in penalty | Accepted | Reported with and without; B trained on a wider SNR input range |
| m4 | Crossover undefined | Accepted | Defined in §3.5 |
| m5 | Add the hybrid estimator | Accepted | In §3.2 and as a statistic for D |
| m6 | Lead with effective k, not CIFAR's k = 256 | Accepted | Summary and §3.1 |
| m7 | Keep RQ5 / shaping behind the cut line | Accepted | Now RQ7, second paper |
| new | Capacity question raised by the author | Added | §3.4, §4.4, RQ3–4, C-wide control |
