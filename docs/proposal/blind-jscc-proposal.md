# How Does a Blind JSCC Receiver Know the Noise Level?

**Research proposal: why deep JSCC decoders work without being told the SNR, which cue they use
in its place, what it costs in model capacity, and an SNR-independent digital model built on the
answer.**

Status: proposal, not started; revised through three rounds of independent review. Builds on this
repo's 2×2 harness (analog/digital × fixed/adaptive)
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
number of symbols that matters is what the decoder pools over: for a plain CNN decoder that is its
**receptive field** (about 1,000 symbols here), not the image size, so the poor regime exists even
for large images. Attention modules with global pooling see the whole block instead. In digital JSCC,
quantisation makes the transmitted energy vary from image to image, so the energy cue degrades,
while the constellation offers a better cue (the residual to the nearest point).

**What we do.**

1. **Kill test (week 1):** a matched blind vs SNR-conditioned comparison (identical architecture,
   only the SNR input removed), with seeds and an equivalence margin, against a quantitative
   prediction of what an energy-reading decoder would lose.
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

**Cost.** Week 1: about 45–60 GPU-hours. Full study: about 9–11 weeks and 180–260 GPU-hours, with
the 128×128 runs warm-started from CIFAR checkpoints; parallel Kaggle sessions are assumed for those. Seed counts are set by
the primary contrasts' power (§4.5), not by a fixed three per arm. The capacity-allocation map (RQ4) is a separable
module: it reuses the same infrastructure and can be split into its own short paper if needed.

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
| SIJSCC, [arXiv:2306.15183](https://arxiv.org/abs/2306.15183) | No SNR anywhere; matches or beats ADJSCC. Fig. 7: SNR at encoder+decoder / decoder only / none, same backbone. Uses an ACmix self-attention module at the encoder output and decoder input, which the authors describe as *"global self-attention"* for *"long-range dependency relationships"*. The reference ACmix code, however, uses 7×7 local-window attention (`kernel_att=7`, `nn.Unfold`), so unless SIJSCC changed it, its decoder has no truly global path | Single run, no variance, analog. Fig. 4 compares different architectures |
| CBJSCC, [*Sensors* 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11209452/) (same first author) | §4.4: ADJSCC's AF module in the same three settings, *"no method was significantly superior"*, attributed to *"the inherent unpredictability of deep learning models"* | No error bars; analog; 128×128 crops |
| STARJSCC, [*Sci. Rep.* 2025](https://www.nature.com/articles/s41598-025-16753-4) | Table 2: SE / CBAM / CSA attention with and without SNR; SNR adds 0.26 dB to CSA | One operating point (CBR 1/6, 13 dB, Kodak) |

### 2.3 Digital JSCC without the SNR

| Work | What it does | Why it is not a matched blind soft-symbol digital study |
|---|---|---|
| Low-SNR-robust JSCC, [arXiv:2604.20278](https://arxiv.org/abs/2604.20278) | Uniform quantisation + M-QAM; *"minimum distance demodulation is first performed on y"* | Hard decisions discard soft information, so there is nothing for an SNR to condition; uses channel equalisation |
| Park et al., [arXiv:2311.08146](https://arxiv.org/abs/2311.08146) | Single model, robust training over erasure channels; fixed 4/16/64-QAM | SNR enters via the LLRs and modulation selection; pilot CSI |
| BlindSC, [arXiv:2501.02273](https://arxiv.org/abs/2501.02273) | "Blind" means training without specifying the channel | At inference, bit errors are matched to the trained probabilities *"by adaptively selecting power and modulation levels based on practical requirements and channel conditions"* |
| D²-JSCC, [arXiv:2403.07338](https://arxiv.org/abs/2403.07338) | Digital; decoder not SNR-conditioned | Channel *"perfectly known at both the transmitter and receiver"* |
| DeepJSCC-Q per-SNR models | No SNR input | One model per SNR, so blind only in a trivial sense |

### 2.4 Model capacity, depth and asymmetry

| Work | What it does | What it does not do |
|---|---|---|
| Implicit-JSCC / effective depth, [arXiv:2606.29737](https://arxiv.org/abs/2606.29737) | A depth–SNR model: *"the SNR-dependent refinement depth required to reach a prescribed perturbation tolerance"*; receiver-side refinement grows with noise | Uses SNR as input; equilibrium (implicit) networks; does not vary encoder and decoder capacity independently or study blindness |
| DD-JSCC, [arXiv:2507.20467](https://arxiv.org/abs/2507.20467) (ICC 2025) | Dynamic layer activation by device capability and channel | Forces equal depths: *"we align their layer configurations during training and inference: Ld = Le"*; SNR input at both ends |
| G-UNet-JSCC, [arXiv:2602.22691](https://arxiv.org/abs/2602.22691) | *"an asymmetric design: a relatively simple encoder paired with a powerful U-Net decoder"*, for constrained transmitters | Design choice, no ablation over the split; trained per SNR |

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
smaller than this geometric bound. This applies to decoders **without global pooling** (arm C). The
attention modules of ADJSCC-style decoders contain global average pooling, so for those arms
effective k is the whole block (§4.1).

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
  as analog. (RMS error would suggest otherwise, about 6 dB at 10 dB, but it is dominated by
  outliers.)
- **16-PSK behaves like analog,** since every symbol has the same energy. It is not used as an arm,
  because it also changes the lattice geometry; the genie-energy arm C+E is the clean energy control
  (§4.1).
- **DD is exact at high SNR and badly biased at low SNR.** Energy is the reverse. The two fail in
  regions that barely overlap.
- **The hybrid repairs the 16-QAM energy cue only up to analog quality.** With correct decisions it
  still contains the cross term 2·Re⟨z, n⟩, so at 20 dB it fails as often as analog energy (12%).
  DD dominates at high SNR.

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

These questions also bear on the deliverable. If self-conditioned SI-JSCC-Q (§3.6) beats plain
blind, is that because of the information in the statistics, or because of the attention modules
that carry them? §4.1 separates the two with architecture-matched controls.

### 3.5 Research questions and primary contrasts

Each research question has **one primary contrast**, fixed in advance. Everything else is
secondary and reported as such, so the many arms × SNR points × α values do not produce a
"significant" difference by chance. Where one primary contrast consists of two tests (RQ4), Holm
correction is applied.

**Analog is secondary throughout.** The analog arms B and C-att run with 3 seeds. With three seeds
the 90% equivalence interval fits inside ±0.15 dB only if the seed σ of the paired difference is
about 0.09 dB or less (§4.5), so analog results can support **"clearly nonzero"** claims but not
**"no penalty"** claims. They are worded accordingly: an analog gap whose interval includes zero is
reported as inconclusive, never as equivalence.

**Primary SNR point: 18 dB, not 20 dB.** Every arm trains on 0–20 dB, so 20 dB is the edge of the
training range. A blind decoder has learned that the SNR never exceeds 20 dB, so near the edge its
implicit estimate is truncated and can only err downward (over-denoising). That edge effect is real
but is not the estimation-noise mechanism under test. The primary contrasts are therefore at
18 dB; 20 dB is reported as secondary.

| | Question | Hypothesis | Primary contrast |
|---|---|---|---|
| **RQ1** | Is the matched blind penalty distinguishable from zero, and how does it compare with the penalty predicted for an energy-reading decoder? | **H1:** in 16-QAM, the penalty is near zero at low SNR and positive at high SNR for k = 256 | G = B − C-att at **18 dB**, **16-QAM**: two-sided test and equivalence (§4.5). Analog is secondary (see below) |
| **RQ2** | Which cue does the blind decoder follow when the cues disagree? | **H2b:** 16-QAM C-att follows a DD-like cue at high SNR. *H2a (analog C-att follows energy) is exploratory: the analog implied-SNR table can only be filled in after training* | The cue-fit decision rule of §4.3 across α ∈ {1.1, 1.2, 1.3} at a true **18 dB** (16-QAM) |
| **RQ3** | Does the blind penalty depend on decoder capacity? | **H3:** the penalty grows as decoder width shrinks | (B − C-att) at width 64 minus (B − C-att) at width 256, 16-QAM, 18 dB; **one-sided superiority** (penalty larger at width 64) |
| **RQ4** | How does the marginal value of encoder vs decoder capacity change with SNR? | **H4:** decoder capacity matters more at low SNR, encoder capacity more at high SNR | PSNR(64, 169) − PSNR(169, 64) at 0 dB (predicted > 0) and at 18 dB (predicted < 0); two one-sided tests |
| **RQ5** | Can a digital model with no SNR anywhere match the genie-decoder model? | **H5:** self-conditioned SI-JSCC-Q beats its shuffled-statistics control where global information matters, and is equivalent to B at CIFAR scale | **D − D-shuf > 0 at 128×128, 18 dB (one-sided superiority).** Secondary: D vs B equivalence at CIFAR, 18 dB |
| RQ6 *(below the cut line)* | How do non-stationary noise, unknown-gain fading and non-Gaussian noise break blindness? | Each breaks a specific cue (§7.3) | — |
| RQ7 *(second paper)* | Shaping–estimability trade-off | See the archived proposal (commit 976db88) | — |

**Crossover (H2b) defined in advance:** for each system (analog, 16-QAM), plot the blind penalty
B − C-att in dB against SNR. A crossover means the two penalty curves' 95% confidence bands cross,
with analog worse at high SNR and 16-QAM worse at low SNR. Secondary.

### 3.6 The deliverable: SI-JSCC-Q

One model, fixed 16-QAM symbols (DeepJSCC-Q's quantiser), trained over SNR ~ U[0, 20] dB like every
other arm (§4.1), with **no SNR at either end and no pilots**.

**Plain.** Two variants, both with no SNR input:

- **C**, DeepJSCC-Q with no attention modules. It runs on the current code:
  ```bash
  python -m semcom.train --config configs/cifar_r12.yaml --digital -M 16
  ```
  Leaving out `--snr-adaptive` and `--snr-train-fixed` makes it sample U[0, 20] dB with no AF
  modules.
- **C-att**, the same decoder attention modules as B, with no input in the SNR slot. This is the
  architecture-matched blind model and the primary blind arm (§4.1).

**Self-conditioned (D).** C-att's attention modules, with the empty SNR slot filled by statistics
the receiver computes from the soft symbols it receives:

| Statistic | Strong where (§3.2) |
|---|---|
| Received energy `(‖y‖² − k)/k` | Low and mid SNR |
| DD residual: mean of ‖y − Q(y)‖² | High SNR |
| Hybrid: mean ‖y‖² − mean ‖Q(y)‖² | Mid SNR (restores analog-quality energy; not better than DD at high SNR) |
| Moments `M2`, `M4` | Scale-invariant; mid SNR |
| Local (windowed) versions | Non-stationary noise |

The statistics enter the attention MLP as a small vector, and the network learns to combine them.
The encoder stays blind, so the scheme needs no feedback. Energy and DD fail in regions that barely
overlap (§3.2), so together they should cover 0–20 dB.

**Controls for D.** D and C-att have identical architecture; the only difference is what enters the
SNR slot. **D-shuf** has the same architecture and input dimensionality, but each image receives
the statistics of a different image at a different SNR, in training and evaluation. It carries no
information, so **D − D-shuf is the information effect**, and D-shuf ≈ C-att is the expected sanity
check. (A wider-C control would not work here: D adds only a few inputs to an MLP whose overhead is
under 1%, so widening C to match would change it negligibly and control nothing.)

**Where D can win, and where it cannot.** At CIFAR scale the decoder's receptive field covers the
whole 8×8 latent, and C-att's global pooling sees all of it anyway, so C-att can compute every
statistic in D's vector itself. There, any D advantage can only come from inductive bias or easier
optimisation, and **D ≈ C-att is the expected CIFAR outcome, pre-registered as such, not a negative
result**. D's structural advantage is explicit, precise statistics: the attention module's global
average pooling sees the whole block, but only through learned features. Whether hand-computed
statistics over the whole block beat that is the question, and it can only bite on large images.
The decisive D vs C-att vs D-shuf comparison is therefore at **128×128 crops** (k = 4,096), inside
the SI-JSCC-Q block (§7.1). Its primary test is **superiority, D − D-shuf > 0**, not D-vs-B
equivalence. The 128×128 runs are warm-started from CIFAR checkpoints (§7.1), where D ≈ C-att by
design, and a limited fine-tuning budget biases the arms toward looking alike. That bias works
*against* a superiority claim, so a positive result stays trustworthy; it would work *for* an
equivalence claim, which is why D vs B equivalence is not tested at 128×128.

**Comparisons:**

- DeepJSCC-Q specialists at the two endpoints, 1 and 19 dB (envelope and storage comparison;
  budgeted in §7.1)
- ADJSCC-Q, arm A (genie SNR at both ends; needs feedback; budgeted)
- arm B (genie decoder; the realistic ceiling)
- C, C-att, D, D-shuf
- the separation baseline in `semcom/separation.py` (CPU only)

Results: PSNR over 0–20 dB, plus storage (one model vs specialists).

**Novelty, honestly.** The plain version is SIJSCC's recipe moved onto DeepJSCC-Q. It is open as a
matched, seed-replicated study, but a modest step. The self-conditioned version is a modest step
beyond plugging in a single SNR estimate. It is phrased as *receiver-computed statistics instead of
the SNR, with an architecture-matched information control*, not as a new class of method. No paper
found does this for a JSCC decoder.

---

## 4. Experimental design

### 4.1 Arms

Shared backbone; R = 1/12, CIFAR-10, AWGN. **Every arm is trained on channel SNR ~ U[0, 20] dB and
evaluated over 0–20 dB.**

| Arm | Encoder SNR | Decoder attention modules | Input to the attention SNR slot | Role |
|---|---|---|---|---|
| **A** | true | yes | true SNR | ADJSCC / ADJSCC-Q reference |
| **B** | — | yes | true SNR | Genie decoder; source of the sensitivity curve |
| **C-att** | — | yes | nothing (constant) | **Primary blind arm**; matched to B. Digital C-att = plain SI-JSCC-Q |
| **C** | — | no | — | Blind, no global pooling (local receptive field only); the no-attention variant of plain SI-JSCC-Q |
| **C+E** | — | yes | genie: true per-image transmitted energy ‖z̄‖² | 16-QAM energy control |
| **C+E-shuf** | — | yes | another image's ‖z̄‖² | **Run only if C+E shows a gain:** confirms the gain is the energy value, not the input path |
| **D** | — | yes | statistics vector (§3.6) | Self-conditioned SI-JSCC-Q (16-QAM) |
| **D-shuf** | — | yes | another image's statistics | Information control for D |

**B and C-att differ only in what enters the SNR slot.** B vs C would not be matched: C has no
attention modules, so that gap would mix "told the SNR" with "has attention and global pooling".
CBJSCC and STARJSCC show attention alone changes PSNR. B − C-att is the matched blind penalty; C is
kept as the local-pooling-only variant.

**Global pooling changes effective k.** The attention modules' global average pooling gives B, A,
C-att, C+E and D statistics over the whole latent. Only C (and its 3×3-kernel variant) is limited
to the receptive field's ~1,000 symbols. At CIFAR scale the two coincide. At 128×128:

- **C vs C-att** isolates local vs global pooling with no SNR information involved;
- **B vs C-att** isolates the SNR information at matched pooling;
- **D vs C-att** and **D vs D-shuf** isolate the statistics' information at matched architecture.

**What C+E and D can and cannot show.** A decoder that sees y *and* a statistic can do
anything C-att can, so these arms bound C-att from above, up to optimisation noise. They answer
*"does handing the decoder this statistic help?"*. They cannot show which cue C-att uses, since it
can compute the same statistic itself. The mechanism question is answered only by the cue-conflict
tests of §4.3. C+E's interpretation is one-directional: if C+E barely beats C-att at high SNR, the
blind decoder was not limited by the degraded energy cue. A large gain shows energy matters to the
decoder, not that it uses nothing else.

**No analog C+E.** In analog, per-image transmitted energy is exactly k by construction, so a
genie-energy arm would carry no information; C+E exists only for 16-QAM.

**Modulations.** Analog and 16-QAM only. A constant-energy constellation such as 16-PSK is not
used: its minimum distance at unit power is 2·sin(π/16) ≈ 0.39, against 2/√10 ≈ 0.63 for 16-QAM
(about 4.2 dB worse), so it would change geometry as well as the energy cue. C+E is the clean
control. (An explicit energy-estimate arm is also omitted: D, whose statistics include energy, and
C+E answer the same question.)

**Why U[0, 20] dB and not a wider range.** Widening every arm's training range (say to −5–25 dB)
would let B's input range cover estimation errors, but it has a cost: under an MSE loss the lowest-SNR samples carry the largest
errors and dominate the gradients, shifting capacity away from high SNR, where the blind penalty is
predicted to live. Every arm uses 0–20 dB, matching the code default, the earlier R = 1/12 runs
and the published baselines. The price is clipping in the sensitivity curve, which is reported.

### 4.2 Measurements

1. **Sensitivity curve** (inference on B): PSNR when the decoder is given SNR_true + Δ, for
   Δ ∈ [−10, +10] dB, at each true SNR. Inputs outside 0–20 dB are clipped. Curves are reported
   with and without clipping (unclipped values extrapolate the attention MLP), together with the
   clipping rate.
2. **Predicted energy penalty P:** average the sensitivity curve over the energy estimator's error
   distribution at the matching effective k and SNR, and also feed each image its own estimate.
   Failures (noise estimate ≤ 0) are floored at 20 dB, and the failure rate is reported. **P is an
   upper bound.** B was trained on the true SNR and overreacts to estimation error, so P overstates
   what an energy-reading decoder trained under its own uncertainty would lose. There is no clean
   energy-*limited* reference, since any decoder that sees y can learn implicit estimation. Clipping
   biases P further near the top of the range: at a true 18–20 dB most of the positive-Δ half of the
   sensitivity curve is clipped to 20 dB, so P there mainly captures the over-denoising side.
3. **Matched gaps:** B − C-att (primary), B − C and every other arm against B, as PSNR vs SNR over
   0–20 dB, 10k test images × 10 channel realisations, with **paired** noise across arms.
4. **Cue-conflict tests** (§4.3).
5. **Probes** trained to read log σ² from C-att's and C's decoder layers **on natural data**, then
   **applied to the cue-conflict inputs**. Probes trained where all cues are collinear cannot
   discriminate between them; reading them out under conflict can. The conflict inputs are out of
   the probes' training distribution, so the probes are **calibrated first**: readout error on
   held-out natural data at matched SNR, and readout under mild perturbations whose effect on each
   cue is known (small α, partial shuffles). A shift is attributed to a change in the internal
   estimate only if it exceeds the calibration error. **Layers, fixed in advance:** the output of
   each of the four decoder FL modules (after the attention module where present). The **primary**
   readout is the first decoder module, the earliest point at which the whole received block has
   been pooled; the other three are secondary.
6. **Usage statistics** (digital): histogram, entropy and kurtosis, overall and per image.

### 4.3 Cue-conflict tests (inference only)

Each test builds inputs on which the candidate cues imply different SNRs, then checks which one the
blind decoder follows. Subject: C-att (primary), C (secondary).

| Test | Construction | Cues it separates | Readout |
|---|---|---|---|
| **Power mismatch** | Transmit `αz`, α ∈ {0.85, 1.1, 1.2, 1.3, 1.4}, with noise set so the true SNR `α²/σ²` is unchanged | Every power-assuming cue is misled, **but by different amounts** (table below); scale-invariant cues are not | **Quantitative double difference** (decision rule below); probe readout |
| **Symbol shuffling** | Permute the received symbols' positions (energy and lattice membership kept) | **Manifold** cues (re-encoding residual) break; energy and DD survive | Probe readout only: the image is destroyed, so PSNR says nothing about the internal SNR |
| **C+E vs C-att** (16-QAM) | Genie true transmitted energy in the attention slot | Repairs only the energy cue; geometry and architecture unchanged | Gain of C+E over C-att vs SNR; C+E-shuf as its control if there is a gain |

**Implied SNR under power mismatch** (simulated; 16-QAM, k = 4,096, median, receiver assumes unit
transmit power; "fail" = negative noise estimate, floored at 20 dB). Each cell is energy / DD /
M2M4, with DD's symbol-error rate in brackets:

| α | True 10 dB | **True 18 dB (primary)** | True 20 dB |
|---|---|---|---|
| 0.85 | fail → 20 / 12.2 / 10.0 (27%) | fail → 20 / 14.7 / 17.8 (1.1%) | fail → 20 / 15.3 / 20.0 (0.2%) |
| 1.00 | 10.0 / 11.6 / 10.1 (22%) | 17.9 / 18.0 / 18.1 (0.1%) | 20.2 / 20.0 / 19.7 (0%) |
| 1.10 | 4.8 / 10.2 / 10.0 (23%) | **6.4 / 15.4 / 18.2 (0.2%)** | 6.5 / 16.6 / 19.6 (0%) |
| 1.20 | 2.3 / 8.7 / 10.1 (26%) | **3.4 / 12.0 / 18.0 (0.9%)** | 3.4 / 12.6 / 19.7 (0.1%) |
| 1.30 | 0.7 / 7.1 / 10.0 (30%) | **1.4 / 9.4 / 18.0 (2.8%)** | 1.5 / 9.7 / 19.9 (0.8%) |
| 1.40 | −0.6 / 5.6 / 10.0 (33%) | 0.0 / 7.3 / 17.8 (6.3%) | 0.1 / 7.5 / 19.7 (2.7%) |

α = 0.7 is not used: DD's decisions break down (24% symbol errors even at 20 dB).

Readings:

- **α = 1.1–1.3 at a true 18 dB is the primary window.** Energy, DD and M2M4 imply 6.4, 15.4 and
  18.2 dB at α = 1.1, with 0.2% DD symbol errors. α = 1.4 is secondary (6.3% DD errors at 18 dB).
- **α < 1 adjudicates at mid SNR, not at high SNR.** At a true 18–20 dB, energy's failure is
  floored at 20 dB, close to the truth, so energy and M2M4 predict nearly the same thing. At a true 10 dB the floor
  is 10 dB away from the truth, so α = 0.85 separates energy from M2M4 there (DD is already biased by
  22% symbol errors at α = 1).
- **The hybrid tracks energy** to within 1 dB at every α, so it is not a separate column.
- **The M2M4 column assumes uniform symbol usage** (kurtosis 1.32). The evaluation
  recomputes it with each run's measured usage kurtosis.

**Readout: a double difference.** For the blind arm at power scale α,

```
r(α) = [PSNR_C-att(α) − PSNR_C-att(1)] − [PSNR_B(α; true SNR) − PSNR_B(1; true SNR)]
```

The first bracket is C-att's change under scaling. The second removes pure scale fragility, measured
by B given the true SNR on the same scaled inputs. Differencing against α = 1 removes C-att's
unscaled gap G, which would otherwise shift the whole curve by a constant and bias the fit toward
whichever cue predicts a similar offset. Each cue j predicts

```
p_j(α) = [PSNR_B(α; s_j(α)) − PSNR_B(α; true SNR)] − [PSNR_B(1; s_j(1)) − PSNR_B(1; true SNR)]
```

where s_j(α) is the cue's implied SNR from the table, computed per image on the actual inputs.

**Discriminability check (pre-registered, inference only).** Once B is trained, compute every p_j
across α at each true SNR. Let m be the 95% confidence half-width of r(α) across seeds. An SNR
point is used to adjudicate H2b only if every pair of cue curves differs by more than 2m at one α or
more. Points that fail the check are reported but not used.

**Cue-fit decision rule (RQ2 primary, pre-registered).**

1. **Metric:** for each cue j, `S_j = Σ_α (r(α) − p_j(α))²` over α ∈ {1.1, 1.2, 1.3} at a true
   18 dB.
2. **Uncertainty:** a hierarchical bootstrap, 10,000 resamples: resample seeds, then test images
   within each seed, recomputing r, p_j and S_j each time.
3. **Winner:** the cue with the lowest S_j wins only if it beats the runner-up in at least 95% of
   bootstrap resamples. Otherwise the outcome is **not resolved**.
4. **None:** if even the winner's root-mean-square residual `√(S_j/3)` exceeds m, the outcome is
   **tracks none**, whatever the ranking.

**Analog.** The analog latent is roughly complex Gaussian, with kurtosis near 2, which makes M2M4
unidentifiable (its margin 2 − k_a goes to zero). There is therefore **no scale-invariant cue in
analog**: the analog test separates energy from the re-encoding residual. The re-encoding residual's
implied SNR depends on the trained model, so the **analog implied-SNR table is filled in post hoc**
from the trained B and the measured latent kurtosis. An analog result where C-att tracks the true
SNR would point to a cue not on this list, and is reported as such rather than as one of the
expected outcomes. **All analog mechanism results (H2a) are exploratory,** since the analog table
cannot be fixed before training.

**Interpretation (16-QAM), applying the decision rule:**

- **Energy wins, and C+E clearly beats C-att at high SNR (and C+E-shuf does not):** the 16-QAM
  blind decoder reads energy, against H2b.
- **DD wins, and C+E adds little:** a lattice cue (H2b supported).
- **M2M4 (the true-SNR curve) wins:** the blind decoder estimates signal and noise separately. This
  is the more interesting paper, and the probes then have a specific target.
- **Not resolved, tracks none, or the discriminability check fails:** the mismatch test is
  inconclusive for mechanism; rely on the shuffling test and probes.
- **The probe readout moves under shuffling but not under power mismatch, or the reverse:** that
  identifies a manifold cue or an energy cue respectively.

### 4.4 Capacity sweeps (RQ3–4)

Split `hidden` into `hidden_enc` and `hidden_dec` (the encoder and decoder classes already take
width separately; [models.py](../../semcom/models.py)). Each run trains one model over U[0, 20] dB and
yields a whole PSNR-vs-SNR curve, so a grid of widths costs one run per cell, not per SNR.

| Sweep | Grid | Arms | Purpose |
|---|---|---|---|
| **Decoder width** | `hidden_dec` ∈ {64, 128, 256}, `hidden_enc` = 256 | B and C-att, analog and 16-QAM. Width 256 reuses the week-1 runs; width 64, 16-QAM gets 5 seeds (primary); the rest 2 seeds | RQ3: blind penalty vs decoder size |
| **Asymmetric grid** | `hidden_enc` × `hidden_dec` ∈ {64, 128, 256}², 9 cells, plus (64, 169) and (169, 64) | B (conditioned), 16-QAM. (64, 169) and (169, 64): 5 seeds (primary); other cells 1 seed | RQ4: marginal value of each side across SNR |

**Outputs:**

- **Capacity-allocation map:** PSNR(`hidden_enc`, `hidden_dec`, SNR), with the marginal gain of
  doubling each side, `∂PSNR/∂log₂(width)`, plotted against SNR.
- **Iso-parameter comparison**, chosen from measured counts (C_out = 8, R = 1/12):

  | Width | Encoder params | Decoder params | FLOPs per side |
  |---|---|---|---|
  | 64 | 0.34M | 0.34M | 47M |
  | 128 | 1.33M | 1.33M | 180M |
  | 169 | 2.30M | 2.30M | 311M |
  | 256 | 5.25M | 5.25M | 704M |

  The anti-diagonal is **not** iso-parameter: (64, 256) and (256, 64) have 5.59M against 2.66M for
  (128, 128). The iso-parameter set is **(64, 169), (128, 128), (169, 64)** at 2.65–2.66M. In this
  mirrored CNN, encoder and decoder FLOPs are equal at equal width (measured), so iso-parameter is
  also iso-FLOPs.
- **Blind penalty vs decoder width,** with confidence intervals.

**Caveats.**

- Smaller models converge at different speeds; see the shared epoch budget in §4.5.
- AF module size scales with width, so ADJSCC's "under 1% overhead" is re-checked per cell.
- Width is varied, not depth. Depth changes the receptive field and so confounds RQ3 with effective
  k. Depth is the second receptive-field arm (§4.6) and kept separate.

### 4.5 Training and statistics

- **Learning-rate decay.** The R = 1/12 runs used a constant learning rate (no scheduler in
  `semcom/train.py`), and none converged in 150 epochs. All new runs use cosine decay.
- **Calibration and a shared epoch budget.** Three calibration runs (arm B, arm C-att, and the
  width-64 decoder) find the epochs each needs to reach CONVERGED in
  `scripts/convergence_report.py`. **Every arm in a comparison then trains for the maximum of those
  budgets,** with no per-arm early stopping. Otherwise a slower-converging blind arm would be
  compared before it finished.
- **Arms are compared at convergence, not at matched training loss.** The training loss is the MSE
  being compared, so matching it would erase the gap by construction. Gaps are also reported at
  intermediate checkpoints, so a gap that is still closing is visible.
- **Equivalence, not non-significance.** Claims of "no penalty" (gate row 1, H5's CIFAR secondary)
  use **two one-sided tests at 5%**: the **90%** confidence interval of the paired difference,
  computed with **Student's t on n − 1 degrees of freedom**, must lie inside **±0.15 dB**. A
  difference that is neither significant nor equivalent is reported as **inconclusive**, not as
  equality.
- **Seeds, sized with t quantiles.** With few seeds the interval uses Student's t, which changes the
  numbers drastically compared with normal quantiles. σ below is the seed-to-seed standard deviation
  of the **paired difference** (same data order and channel noise across arms). Simulated
  probability that a truly zero difference passes the ±0.15 dB equivalence test:

  | σ (dB) | n = 3 | n = 4 | n = 5 | n = 6 | n = 7 |
  |---|---|---|---|---|---|
  | 0.05 | 0.88 | 0.99 | 1.00 | 1.00 | 1.00 |
  | 0.08 | 0.50 | 0.77 | 0.92 | 0.97 | 0.99 |
  | 0.10 | 0.33 | 0.54 | 0.74 | 0.86 | 0.93 |
  | 0.12 | 0.22 | 0.36 | 0.53 | 0.68 | 0.79 |

  The largest σ that can pass at all is 0.034, 0.089, 0.127 and 0.157 dB for n = 2, 3, 4 and 5, so
  **equivalence with two seeds is impossible in practice**. Superiority tests (one-sided, 5%) are
  far cheaper. Power to detect a true 0.15 dB difference:

  | σ (dB) | n = 3 | n = 4 | n = 5 |
  |---|---|---|---|
  | 0.08 | 0.67 | 0.88 | 0.96 |
  | 0.10 | 0.53 | 0.74 | 0.86 |
  | 0.15 | 0.32 | 0.46 | 0.58 |

  **Seed plan, set by the primary contrasts:**

  | Contrast | Test | Seeds |
  |---|---|---|
  | RQ1: B − C-att, 16-QAM | Two-sided + equivalence | **5** for B and C-att |
  | RQ1 secondary: B − C-att, analog | Two-sided only ("clearly nonzero"; no equivalence claims) | **3** for B and C-att |
  | RQ2: cue fit | Bootstrap rule (§4.3) | Uses the RQ1 runs |
  | RQ3: penalty at width 64 − at width 256 | One-sided superiority | **5** at width 64 (width 256 reuses RQ1) |
  | RQ4: (64, 169) vs (169, 64) | Two one-sided tests | **5** per cell |
  | RQ5: D − D-shuf at 128×128 | One-sided superiority | **3** for B, C-att, D, D-shuf, warm-started (power 0.67 for a 0.15 dB effect at σ = 0.08 dB; 0.94 for 0.25 dB) |
  | Effective-k claim: C vs C-att at 128×128 | Secondary | **2** for C, warm-started |
  | RQ5 secondary: D vs B at CIFAR | Equivalence | **5** for D, D-shuf |
  | Everything else | Secondary | 1–3 |

  RQ3 is a difference of differences, so its σ is larger than a single gap's. The week-1 measurement
  of σ re-checks every row; if a primary contrast is underpowered, seeds are added before its
  stage starts.
- **Primary contrasts** as listed in §3.5; Holm correction for RQ4's two tests.
- **Estimator error** is reported as failure rate plus median and interquartile range, never RMS
  alone.

### 4.6 Generality of the design rule

The safe-blindness region depends on how much of the block the decoder pools over. For C that is
the receptive field; for the attention arms it is the whole block. The rule is therefore plotted
against **effective k** and stated as specific to the architecture family. Extra effective-k points
come cheaply from:

- **a second decoder with a smaller receptive field** (3×3 kernels in the stride-1 layers, about
  9×9 latent positions instead of 16×16), for C;
- **R = 1/6 at CIFAR scale** (k = 512), as a secondary axis since it also changes the rate;
- **128×128 crops** (k = 4,096 total; about 1,000 effective for C, all 4,096 for C-att).

---

## 5. Decision rules (fixed before running)

### 5.1 The gate (end of week 1)

Let P be the energy penalty predicted from B's sensitivity curve (§4.2), and G = B − C-att the
measured matched gap at 18 dB, both from paired seeds with Student's t intervals.

| Outcome | Meaning | Action |
|---|---|---|
| G **equivalent to zero** (90% t-interval inside ±0.15 dB) at 18 dB **and** P clearly nonzero somewhere | C-att **beats plug-in energy**. This is the likely outcome by construction, since P is an upper bound (§4.2), so it says nothing about *which* cue is used | Continue: mechanism claims come only from the cue-conflict tests |
| G clearly nonzero **and** consistent with P | The blind decoder reads energy | Continue: the energy story, the digital contrast and capacity |
| G clearly nonzero **and** larger than P | Unlikely, because P overstates the penalty; if seen, blind training costs more than estimation error explains (optimisation or capacity) | Continue with the decoder-width sweep first (RQ3) |
| G inconclusive (neither equivalent nor clearly nonzero) | Underpowered | Add contingency seeds to the primary contrast, then re-apply the gate |
| P ≈ 0 everywhere (decoder insensitive) | Blindness is free for a trivial reason at this scale | Drop the mechanism study; ship plain SI-JSCC-Q and the capacity map as a short paper |

"Clearly nonzero" means the 95% t-interval excludes zero; "equivalent" means the 90% t-interval
lies inside ±0.15 dB (§4.5). The gate's main job is to catch an
insensitive decoder (last row). It does not decide the mechanism.

### 5.2 What SI-JSCC-Q becomes

| Result | Deliverable |
|---|---|
| D ≈ C-att at CIFAR scale | Expected (§3.6); no decision taken. The decision is made at 128×128 |
| C-att equivalent to B at CIFAR, and D − D-shuf not significant at 128×128 | Plain SI-JSCC-Q (C-att) is the result; D is reported as unnecessary |
| D − D-shuf > 0 at 128×128 (primary), and D closes some of the B − C-att gap there | Self-conditioned SI-JSCC-Q is the method; the gap it closes is the headline number |
| D − D-shuf not significant at 128×128 | The statistics' information does not help; any D gain over C was the attention modules. Recommend C-att |

### 5.3 Combined with the measured symbol usage

The week-1 16-QAM runs log hard symbol usage (entropy and kurtosis) at every validation pass.
That measurement, not the earlier runs' wandb logs, decides whether the shaping study is worth
pursuing. The old runs used a constant learning rate, never converged, and logged only
batch-averaged soft usage.

| | Gate passes | Decoder insensitive |
|---|---|---|
| **Usage clearly non-uniform** | This paper; the shaping study (RQ7) as a second paper | The shaping study alone, plus plain SI-JSCC-Q |
| **Usage near-uniform** | This paper; RQ7 dropped | Plain SI-JSCC-Q + capacity map as a short paper, or pivot to impairment-blind decoding |

---

## 6. Week 1: the kill test

| Step | Work | Compute |
|---|---|---|
| 0 | Run safety: `best.pt`, `config.yaml` and `history.json` uploaded to wandb at the end of every run. The R = 1/12 weights were lost, so nothing can reuse them | — |
| 1 | Code: encoder/decoder SNR flags; decoder SNR override; attention with an empty or arbitrary input slot (C-att, C+E, D); cosine decay; energy/DD/hybrid/M2M4 estimators; sensitivity and prediction evaluation; quantitative power-mismatch evaluation with the scale-fragility reference | — |
| 2 | Calibration: arm B and arm C-att, 16-QAM, cosine decay, in a separate results folder; set the shared epoch budget | ~5 h |
| 3 | B and C-att, 16-QAM × 5 seeds; B and C-att, analog × 3 seeds; C, analog and 16-QAM × 3 seeds; C+E 16-QAM × 3 seeds (25 runs) | ~40–55 h |
| 4 | Sensitivity curves, prediction P, gaps G; discriminability check; power-mismatch test | Inference |
| 5 | Gate (§5.1) | — |

Runs take about 72 min per 150 epochs on the RTX 3050, so 200–300 epochs means about 1.6–2.4 h
each. Step 3 is about two days on the 3050, or about one day across three Kaggle sessions.

---

## 7. Stage 2 onward (if the gate passes)

### 7.1 Core (weeks 2–5)

| Weeks | Work | Compute |
|---|---|---|
| 2 | Symbol-shuffling test; probe calibration, then probes read out under conflict; C+E-shuf × 3 (only if C+E showed a gain); 3×3-kernel decoder for C, 2 seeds; contingency seeds for the primary contrasts if week 1 was inconclusive | ~5–15 h |
| 2–4 | **SI-JSCC-Q at CIFAR:** D, D-shuf × 5 seeds. **Baselines:** arm A × 1 seed; DeepJSCC-Q specialists at 1 and 19 dB × 1 seed | ~25–30 h |
| 3–6 | **SI-JSCC-Q at 128×128, warm-started:** each arm fine-tuned from **its own** CIFAR checkpoint at the same seed (B from B, D from D, and so on), all for the same number of epochs. B, C-att, D, D-shuf, 16-QAM × 3 seeds (the primary RQ5 contrast); C × 2 seeds (the only test of the effective-k claim, secondary). **Warm-start check first:** C-att seed 1 trained both cold and warm with matched epochs; warm starts are adopted only if the two agree within ±0.15 dB at 18 dB, otherwise the block runs cold. Training data: random 128×128 crops from the **DIV2K** training set (800 images), about 20k fresh crops per epoch; evaluation on full **Kodak** images and DIV2K validation crops. Fine-tune cost estimated at 3–5 h, a cold run at 8–12 h | ~55–85 h |
| 4–5 | **RQ3:** decoder-width sweep; width 64, 16-QAM, B and C-att × 5 seeds (primary); width 128 and analog × 2 seeds; calibration run at width 64 | ~25–30 h (small models run faster) |
| 5–6 | **RQ4 (separable module):** asymmetric grid, 9 cells × 1 seed; (64, 169) and (169, 64) × 5 seeds; R = 1/6 at CIFAR | ~30–40 h |

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

**Total:** about 180–260 GPU-hours before the cut line: week 1 45–60, 128×128 runs 55–85, RQ4
30–40, the rest CIFAR-scale. If the warm-start check fails and the 128×128 block runs cold, add
about 60–90 h. The 128×128 block uses 3 seeds: RQ5 is powered for effects of about
0.25 dB (power 0.94 at σ = 0.08 dB) but only weakly for 0.15 dB (0.67). A 0.15 dB effect that is
missed is reported as inconclusive, not as absent.

**128×128 content check.** DIV2K crops range from smooth to highly textured, and earlier results in
this repo showed smooth content behaves very differently. Test PSNR is therefore also reported by
content quartile (mean gradient energy of the crop), per arm, to check that no arm's advantage comes
from one content type.

### 7.4 Code changes

| Change | Where |
|---|---|
| `snr_at_encoder` / `snr_at_decoder` flags; decoder SNR override | `semcom/models.py`, `semcom/config.py` |
| `hidden_enc` / `hidden_dec`; 3×3-kernel decoder option | `semcom/config.py`, `semcom/models.py` |
| Cosine learning-rate decay | `semcom/train.py` |
| Estimators: energy, DD, hybrid, M2M4, re-encoding residual | new `semcom/snr_estimation.py` |
| AF modules with a configurable slot: nothing (C-att), true SNR (B), genie energy (C+E), statistics vector (D), or another image's value (C+E-shuf, D-shuf) | `semcom/train.py`, `semcom/modules.py` |
| Per-run evaluation: curve, sensitivity, plug-in prediction, power mismatch, usage (paired noise) | new `semcom/blind_eval.py` |
| Decision rules across seeds: t-intervals, equivalence, gate, cue-fit bootstrap | new `semcom/blind_stats.py`, `semcom/blind_report.py` |
| Run plans by stage, sharded by seed | new `scripts/run_blind.py`, `configs/blind/cifar_r12.yaml` |
| Probes trained on natural data, applied to conflict inputs | new `scripts/probe.py` |
| Capacity grid driver and allocation map | new `scripts/capacity_sweep.py` |

**Tests to write first:**

- the estimators reproduce §3.2 within tolerance;
- the decoder SNR override leaves transmitted symbols identical;
- power mismatch preserves the true SNR;
- shuffling preserves energy and lattice membership;
- encoder-blind arms have no encoder AF parameters;
- B and C-att have identical parameter counts apart from the SNR slot;
- the shuffled-input arms never receive an image's own value;
- `hidden_enc` ≠ `hidden_dec` builds and trains.

---

## 8. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Decoder insensitive to SNR error at CIFAR scale | **High** | Gate row 4; the capacity map and plain SI-JSCC-Q still give a short paper |
| Runs still not converged with cosine decay | High | Calibration runs; shared epoch budget per comparison; gaps reported across checkpoints |
| Equivalence not reachable even with five seeds (σ > 0.12 dB) | Medium | Seeds re-planned from week 1's measured σ; RQ5 primary is superiority, not equivalence; "inconclusive" reported as such |
| 128×128 runs cost more than estimated | High | Warm starts from CIFAR checkpoints; calibrate with one run first; Kaggle; reduce the crop set with a convergence check. Two seeds is not a fallback: superiority power at n = 2 is below 0.5 |
| Warm starts bias the 128×128 arms toward looking alike | Medium | Cold-vs-warm check on C-att before adopting; primary test is superiority, which the bias works against; no equivalence claims at 128×128 |
| RQ5 underpowered for small effects with 3 seeds | Medium | Powered for ~0.25 dB; a smaller effect is reported as inconclusive |
| Mechanism-only papers are hard to place in communications venues | High | Lead with the design rule, the capacity map and SI-JSCC-Q |
| Power-mismatch result confounded by scale fragility | Medium | Several α values; compare trends, not single points |
| Prior art appears in a later search (novelty has shrunk twice) | Medium | Re-search before Stage 2 and before writing; claims limited to §2.6 |
| Scope: mechanism, capacity and deliverable in one study | Medium | RQ4 is a separable module on shared infrastructure; RQ6–7 below the cut line |
| Power-mismatch readout ambiguous | Medium | Discriminability check before adjudicating; scale-fragility reference; shuffling test and calibrated probes as independent evidence |
| D tested where it cannot win | Medium | D ≈ C-att at CIFAR pre-registered as expected; decision at 128×128 against C-att and D-shuf |
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
  Implicit-JSCC, DD-JSCC and G-UNet-JSCC were checked against the full paper text (PDF).
- SIJSCC's figure contents come from an earlier reading in this project. Its attention design was
  checked against the full text: the authors describe the ACmix module as *"global
  self-attention"* for long-range dependencies. The reference ACmix implementation
  (LeapLabTHU/ACmix, `ResNet/test_bottleneck.py`) uses 7×7 local-window attention
  (`kernel_att=7`, `nn.Unfold`). SIJSCC's own code was not checked, so whether it changed this is
  unknown.
- D²-JSCC and Park et al. details come from the earlier novelty audit (archived shaping proposal §2.2, commit 976db88).
- The §3.2 simulations are idealised (Gaussian latent or uniform usage, AWGN, 4,000 trials).
- The receptive field (~16×16 latent positions) is a geometric bound from the layer configuration.
- Parameter and FLOP counts in §4.4 were measured from `semcom/models.py` (`torch` FLOP counter,
  one 32×32 image, C_out = 8).
- The power-mismatch table (§4.3) is simulated with uniform 16-QAM usage, AWGN, k = 4,096.
- The equivalence and superiority power tables (§4.5) are Monte Carlo simulations (200,000 trials
  per cell) of t-based tests on normally distributed paired differences.
- The 128×128 per-run cost (§7.1) is an estimate, not a measurement.
