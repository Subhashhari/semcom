# One Generative JSCC Receiver for Every SNR: a Small DiT-JSCC Prototype

**Status (9 Nov 2026):** week-1 gate passed (DiT-S beats the CNN at 0 dB on LPIPS, DINOv2 and
FID). Every run planned for 18 Nov is trained and evaluated (`results/gen/report/report.md`):
the analog SNR study (specialists, told, blind; told and the ±5 dB specialists replicated with a
second training seed), the CNN baselines, and the digital arm (section 7: 4/16/64-QAM at 0 dB,
16-QAM specialists, told and blind, CNN with 16-QAM, and the energy-convention ablation).

## 1. Question

DiT-JSCC (Tan et al., arXiv:2601.03112) trains **a separate model for every SNR**: "we train separate
models for each SNR under both AWGN and Rayleigh fading channels." Its receiver is a 50-step diffusion
transformer (5.3 s per image).

> Can **one** DiT-JSCC-style receiver, either told the SNR or blind, match a set of per-SNR
> specialists across −5 to 5 dB, and what does that save in storage and decoding steps?

## 2. What is already known (and must not be claimed)

| Work | What it already does |
|---|---|
| SGD-JSCC (arXiv:2501.01138) | DiT denoiser of the *received features*; maps an SNR estimated from the channel output to a diffusion timestep; "nearly identical, whether the SNR is available or not" |
| DiffJSCC (arXiv:2404.17736) | Stable Diffusion refiner with an SNR-conditioned control module, trained over U[0, 14] dB |
| LRISC (arXiv:2504.21577) | SNR-adaptive JSCC codec feeding a latent-conditioned diffusion decoder |
| Pei et al., TWC 2025 (arXiv:2406.06644) | Consistency distillation for one- or few-step latent-diffusion semantic communication |
| DriftDecode (arXiv:2605.02325) | SNR-conditioned one-step wireless image decoder |

**The gap.** In DiT-JSCC the diffusion process starts from pure Gaussian noise and the received
symbols are only *conditions*, so the channel noise is not the diffusion noise and SGD-JSCC's
SNR-to-timestep mapping does not apply. None of the works above compares one SNR-adaptive
generative receiver against per-SNR specialists at each SNR, or tests SNR mismatch. That matched
comparison is what this repo's methodology does (ADJSCC-Q ablation, blind-JSCC study).

## 3. The prototype

```
 transmitter                                          channel        receiver
 DINOv2 ViT-B features (frozen, cached) -> semantic enc. -> 256 symbols -+
                                                                         +-> conditional DiT-S -> SD-VAE latent -> image
 image (128x128) -> detail enc. (4 x stride 2)         -> 768 symbols -+     (DDIM, 50 / 20 / 10 steps)
```

| | DiT-JSCC (paper) | Prototype |
|---|---|---|
| Data | ImageNet, 256 / 512 px | Imagenette (10 ImageNet classes, 9,469 train), 128 px |
| Bandwidth | 1/384 to 1/48 | 1/48 (k = 1,024: 256 semantic + 768 detail) |
| Semantic branch | frozen DINOv2 ViT-B + JSCC encoder | same; features pooled to the 8x8 symbol grid |
| Detail branch | SwinJSCC on pixels | 4-stage strided CNN on pixels |
| Receiver | 21-block DiT, hidden 1024, long skips, from a pretrained DiT | DiT-S (12 blocks, hidden 384, 35M params), no skips, from scratch |
| Fusion | channel fusion, coarse-to-fine | same: symbols concatenated and projected into blocks 1–3 (semantic) and 4–6 (detail), zero-initialised |
| Bandwidth split | per image, from a BLIP-2 caption | fixed |
| CFG | yes | condition dropout 0.1 in training; guidance scale configurable |
| SNR | one model per SNR | **three arms (below)** |

Each branch is power-normalised separately, so the semantic/detail split moves bandwidth, never
power. The frozen models run once per image to build a cache (`semcom/gen/data.py`), so training
runs only the encoders and the DiT.

**Arms** (same split as the blind-JSCC study):

| Arm | Training SNR | SNR slot | Role |
|---|---|---|---|
| `spec−5`, `spec0`, `spec5` | one fixed SNR | none | specialists, as in DiT-JSCC |
| `told` | U[−5, 5] dB | the SNR (into the adaLN conditioning, like the timestep) | one model, informed |
| `blind` | U[−5, 5] dB | a constant; identical architecture to `told` | one model, blind (matched to `told`) |
| `cnn_*` | as above | as above (repo's CNN JSCC, ADJSCC AF modules) | MSE deep-JSCC baseline at the same bandwidth |

**Metrics** (`semcom/gen/evaluate.py`), on the first 1,000 test images at −5, 0 and 5 dB: PSNR,
LPIPS, DINOv2 similarity (CLS cosine), FID, and decoding seconds per image at 50, 20 and 10 DDIM
steps. Channel and sampling noise are seeded per (SNR, batch), so every arm sees the same noise on
the same images and differences between arms are paired.

**Honest limits of the scale-down.** Absolute numbers will not be comparable to the paper's (FID
60 at 256 px on ImageNet); comparisons *between arms* are what the study needs. Imagenette is small,
so overfitting is watched through the validation loss.

## 4. Compute

RTX 3050, RTX 5060 and 2–3 Kaggle T4 accounts. Every run of one comparison uses one GPU type.
DiT-S at 128 px has only 64 tokens, so a run should fit comfortably in 4–8 GB. Expected cost per
run is measured in week 1 (first job) before week 2 is sized. The RTX 5060 (Blackwell) needs a
PyTorch build for CUDA 12.8 or newer.

```bash
python scripts/run_gen.py --stage smoke                 # offline CPU check, a few minutes
python scripts/run_gen.py --stage week1 --evaluate      # on a local GPU
python scripts/kaggle_jobs.py push g-w1 --runner gen --plan "--stage week1 --evaluate --max-hours 10"
```

## 5. Gate (end of week 1, fixed in advance)

Continue to the SNR study only if:

1. training is stable (validation loss decreasing, no divergence);
2. at 1/48 and 0 dB, the DiT specialist beats the CNN baseline on LPIPS **and** DINOv2 similarity;
3. sample grids keep the right objects and layout.

If it fails: (a) a larger receiver (DiT-B) or longer training, then (b) a refinement design
(SGD-JSCC / DiffJSCC style) where a pretrained Stable Diffusion carries the generative prior.

## 6. Schedule and deliverables

| Date | Meeting | Deliverable |
|---|---|---|
| Tue 10 Nov | short update | Design (this document): question, novelty position, architecture and deviations; pipeline implemented and tested; data cache built; first GPU run launched, with throughput and early loss curves |
| Wed 11 Nov | short update | First results: DiT-S specialist vs CNN baseline at 0 dB (sample grid; LPIPS, DINOv2, PSNR, FID); decoding time at 50 / 20 / 10 steps; gate verdict |
| Wed 18 Nov | comprehensive | The SNR study: three specialists vs `told` vs `blind` at −5, 0, 5 dB, with paired comparisons; CNN baselines; storage (1 model vs 3) and decoding-cost table; SNR-mismatch result (specialists away from their SNR); **digital transmission (section 7): the cost of sending 16-QAM instead of analog symbols, the modulation-order sweep, and the told/blind question with 16-QAM symbols**; limitations and next steps (few-step distillation) |

## 7. Digital transmission (added 8 Nov)

**Why.** DiT-JSCC, like every image-generating diffusion JSCC we found (DiffJSCC, CDM-JSCC, LRISC,
DiffCom, SGD-JSCC, OSD-JSCC, CDDM), sends continuous-valued complex symbols, which no standard
modulator can emit. Digital diffusion work either cleans up QAM symbols in a classical link (SCDM,
SSCDM, CD3M) or uses a separation design (tokens or bits plus a generative decoder). We found no
work that quantises a DiT-JSCC-style receiver's learned symbols, where diffusion starts from pure
noise and the received symbols are only conditions, to M-QAM (`reports/Digital diffusion JSCC
novelty.md`, gap 1).

**What must not be claimed.** Quantising learned JSCC symbols to QAM is DeepJSCC-Q (Tung et al.,
arXiv:2111.13042, 2206.08100), including one SNR-adaptive quantised model over 0–10 dB with a
mismatch study, for a CNN decoder. MDJCM and SemQAM already do QAM-native JSCC with GAN-type or
SNR-conditioned decoders. The new part is the combination: a pure-noise conditional diffusion
receiver with QAM symbols, and (with section 3's arms) told vs blind vs specialists in that setting.

**Design** (`semcom/gen/model.py`, `digital: true`):

- Both branches are power-normalised as before, then every symbol is snapped to the nearest
  point of one square M-QAM constellation with unit average energy under uniform use. Only exact
  constellation points are transmitted; validation and evaluation verify this and refuse to report
  otherwise.
- Gradients use DeepJSCC-Q's softmax relaxation. Its inverse temperature is held fixed at
  2/d_min² (5 for 16-QAM, DeepJSCC-Q's starting value) instead of DeepJSCC-Q's rise to 100. On
  unit-power Gaussian inputs, the rise leaves 14% of 16-QAM symbols with a surrogate gradient
  above 0.1; the fixed value leaves about 91% for 4-, 16- and 64-QAM alike.
- A KL-to-uniform penalty (weight 0.05, training only) counters codebook collapse. Symbol-usage
  entropy, kurtosis and mean transmit energy are logged at every validation.
- The receiver conditions on the noisy received symbols, as an equaliser would hand them to a
  demapper. There is no channel code.
- **SNR is Es/N0 with Es the energy the symbols actually carry** (`power_ref: actual`): the batch
  mean in training, one constant measured on the test images at evaluation (a fixed constellation
  scaled by one amplifier gain). Analog runs send exactly unit energy per image, so the two compare
  at equal transmitted energy. The first digital job (g-w4) used DeepJSCC-Q's nominal Es = 1:
  clipping onto the outer points left it sending 0.83, an unstated 0.8 dB handicap. It is kept as
  the ablation of that convention.

**First measurement (g-w4, nominal Es).** Mapping validation loss onto the analog specialists'
curves (−5 / 0 / 5 dB, same step), the 16-QAM model trained at 0 dB behaves like analog at
−1.9 dB, at 4k, 6k and 10k steps alike: about 0.8 dB from the energy convention and about 1.1 dB
from quantisation (an untrained Gaussian latent would predict 0.5 dB: signal-to-quantisation-noise
ratio 9.1 dB for 16-QAM, 3.9 dB for QPSK, 12.5 dB for 64-QAM). Constellation-constrained mutual
information at 0 dB is 0.97 / 0.99 / 0.99 bits for 4 / 16 / 64-QAM against 1.00 for a Gaussian
input, so the loss is in the uncoded mapping, not in what the constellation can carry.

**Arms and jobs** (40k steps, v-target DiT-S as section 3; actual-Es results in
`results/gen/p128-qam-es`, the nominal-Es ablation in `results/gen/p128-qam`):

| Job | Account | Runs | Question |
|---|---|---|---|
| g-w4 | own | 16-QAM `spec0`, `told`, nominal Es | ablation: the energy convention |
| g-d1 | teammate | 16-QAM `spec0`, `told` | cost of going digital at 0 dB; does one told model still work with 16-QAM symbols |
| g-p2 | teammate | 0.3 h probes: 4-QAM, 64-QAM `spec0`; 16-QAM with DeepJSCC-Q's temperature rise (control) | are the other orders healthy; does the fixed temperature beat the rise; probes resume as full runs |
| g-d2, g-d3; g-w5b | teammate; own | 4-QAM, 64-QAM `spec0` (resumed), 16-QAM `blind`, `spec−5`, `spec5`; CNN 16-QAM `spec0` | modulation sweep; full told/blind/specialist study with 16-QAM; DiT vs CNN under digital transmission |
| g-d4 | teammate | analog `spec−5`, `spec5`, `told`, seed 1 | do the told-vs-specialist gaps survive retraining |

**Results (9 Nov, paired 95% bootstrap intervals over 1,000 test images, 20 DDIM steps).**

- *Cost of 16-QAM at 0 dB* (spec0 minus analog spec0): DiT −0.57 dB PSNR, +0.019 LPIPS,
  −0.015 DINOv2; CNN −0.42 dB, +0.015, −0.031. QPSK and 64-QAM cost DiT −0.97 and −0.32 dB.
  DiT keeps its LPIPS lead over the CNN (−0.156 with 16-QAM, −0.160 analog), and its DINOv2 lead
  becomes significant (+0.024; analog +0.009, not significant). Both 0 dB digital models are
  better than their analog versions when tested at −5 dB (+0.54 / +0.33 dB); not yet explained.
- *Told vs specialists, analog, two seeds:* at −5 dB −0.73 (seed 0) and −0.66 dB (seed 1);
  at +5 dB +0.15 and +0.25 dB; at 0 dB −0.01 (not significant, seed 0 only). Retraining the
  same arm moves PSNR by at most 0.06 dB, so the gaps are properties of the arms, not of one seed.
- *With 16-QAM:* told minus specialist −0.62 / +0.03 (not significant) / +0.17 dB; blind within
  0.09 dB of told (analog: within 0.06 dB).

**What 18 Nov shows.** (1) Analog vs 4/16/64-QAM DiT at 0 dB, evaluated at −5, 0, 5 dB, on PSNR,
LPIPS, DINOv2 and FID. (2) 16-QAM told vs the 16-QAM specialist(s), and blind if g-w6 finishes in
time. (3) Symbol usage and transmit energy per run. Expected by 14–16 Nov at the current per-run
cost of about 1.5 h of training plus about 0.3 h of evaluation on a T4.

## 8. Status on 13 Nov: results since 9 Nov and what runs next

All numbers below come from paired comparisons over the same test images, with the same channel
noise and the same sampling seeds; brackets are 95% bootstrap intervals. Unless stated otherwise,
they use the **10-class set**: 1,000 test images spaced evenly over the whole test split
(`--eval-subset spread`, files `eval-spread*.json`). The first 1,000 test images, used up to 9 Nov,
cover only 3 classes. Sampling uses 10 DDIM steps: on this set, 10 steps gives better LPIPS and FID
than 20 or 50 (`eval-fewstep.json`).

To recompute a comparison, run
`python scripts/gen_analysis/paired.py spread REF_RUN RUN [RUN:EVAL_NAME ...]`.
It reads the per-image metrics in each run's `eval_images-*.npz`.

**Where the receiver needs the SNR** (`results/gen/analysis/paired_spread_vs_blind_s0.json`).
PSNR differences against the blind receiver at −5 / 0 / +5 dB:

| Model | −5 dB | 0 dB | +5 dB |
|---|---|---|---|
| told (receiver given the SNR) | −0.04 | −0.03 | +0.02 |
| best specialist at that SNR | +0.64 | +0.005 (ns) | −0.07 |
| SNR-aware transmitter, blind receiver | **+0.48** [0.40, 0.56] | −0.03 (ns) | −0.16 |

So telling the receiver the SNR adds nothing. One blind model matches the specialists everywhere
except at −5 dB. Feeding the SNR to the transmitter instead (FiLM, 117k parameters) recovers about
75% of that −5 dB gap.

Seed 1 replicates every row on the first 1,000 images
(`results/gen/analysis/paired_first_txsnr_vs_blind_s{0,1}.json`), within 0.06 dB:

| SNR-aware minus blind | −5 dB | 0 dB | +5 dB |
|---|---|---|---|
| seed 0 | +0.51 | −0.06 | −0.14 |
| seed 1 | +0.51 | −0.07 | −0.15 |

The fine-tunes from told and from blind (`results/gen/p128-ft`) explain why. The gap belongs to the
transmitter: after fine-tuning, the transmitter accounts for 78% of the gain and the receiver 11%.

**Seed 1 and −10 to +10 dB, analog** (g-e2).

- **Seed 1 replicates on the 10-class set** (`paired_spread_vs_blind_s1.json`):
  - told minus blind is −0.03 / −0.01 / +0.03 dB;
  - the −5 dB specialist leads by +0.68 dB;
  - the SNR-aware transmitter gains **+0.50** [0.42, 0.57] dB, which is 73% of that gap (seed 0: 75%).
- **Outside the training range the SNR-aware gain grows** (`paired_spread_curve_vs_blind_s0.json`): +1.58 dB [1.47, 1.71] at −10 dB and +1.07 dB at −7.5 dB. It costs 0.12–0.17 dB above +2.5 dB.
- **Telling the receiver hurts outside the training range.** Told minus blind is −1.50 dB at −10 dB, −1.01 dB at −7.5 dB and −0.18 dB at +10 dB. An SNR input the receiver never saw in training misleads it; the blind receiver degrades gracefully. This is a second reason to keep the SNR out of the receiver.
- **On the −10 to +10 dB curve the perceptual CNN loses on LPIPS at every SNR**, by 0.05–0.15. It leads on PSNR only from −2.5 dB up.
- Figure: `results/gen/analysis/cliff.png` (analog) and `cliff_qam16.png` (16-QAM).

**A wrong SNR at the transmitter** (`results/gen/analysis/feedback_cpu200.json`, CPU, 200 images):

- **Too low is safe.** It trades PSNR for LPIPS.
- **2 dB too high still beats blind.**
- **5 dB or more too high is worse than blind**, so feedback should be conservative.
- **Below the training range the gain grows.** At −10 dB the SNR-aware transmitter beats blind by
  +1.67 dB.

**Guidance at the receiver** (`eval-spread-g15.json`;
`results/gen/analysis/paired_spread_guidance15_s0.json`). Classifier-free guidance 1.5, with no
retraining:

| | −5 dB | 0 dB | +5 dB |
|---|---|---|---|
| LPIPS | −0.046 | −0.034 | −0.025 |
| FID | 187 → 173 | 142 → 129 | 112 → 100 |
| retrieval top-1 | +1.0 pt (ns) | +2.7 pts | +8.0 pts |
| PSNR | −0.13 dB | −0.16 dB | −0.18 dB |

Retrieval top-1 counts the images whose DINOv2 embedding is nearest to their own original among
the 1,000. The SNR-aware transmitter gets the same gains.

**Sampling steps are a receiver-side setting, not a cost** (`results/gen/analysis/steps.png`,
`steps.json`). These numbers are for the blind analog model on the first 1,000 images, on a T4,
with batched decoding:

| Steps | PSNR at 0 dB | LPIPS at 0 dB | FID at 0 dB | Time per image |
|---|---|---|---|---|
| 1 | **19.22** | 0.290 | 176 | 41 ms |
| 10 | 18.06 | **0.267** | **146** | 51 ms |
| 50 | 17.08 | 0.275 | 165 | 103 ms |

- One step gives the best PSNR: it behaves like a posterior-mean (MMSE-like) decoder.
- Ten steps give the best LPIPS and FID.
- More steps make every metric worse and double the decoding time.
- The same trained model therefore covers distortion-first and perception-first decoding, chosen
  at the receiver.
- Guidance (above) is a second setting of this kind.

**Baselines: what limits the claim.**

- *Separation* (`results/gen/analysis/separation_curve_fbl.json`, `cliff.png`). The separate design
  is AVIF plus an ideal channel code at the finite-blocklength rate for n = 1024 and
  block error 10⁻³.
  - *Re-designed for every SNR* (it needs the SNR at the sender), it wins PSNR at every SNR and
    retrieval from 0 dB up.
  - *The generative receiver wins LPIPS* up to about +10 dB: 0.27 vs 0.59 at 0 dB.
  - *A fixed separate design falls to the mean-colour floor below its design SNR.* The generative
    receiver degrades gracefully and needs no SNR at the sender.
- *Perceptual CNN* (`results/gen/p128-perc`). The 0 dB CNN, fine-tuned with MSE + 0.1·LPIPS-VGG,
  beats the DiT at 0 dB: +2.3 dB PSNR, FID 104 vs 142, retrieval top-1 36% vs 14%. LPIPS is about a
  tie. The DiT wins away from the CNN's training SNR: +1.1 dB PSNR and −0.15 LPIPS at −5 dB, −0.09
  LPIPS at +5 dB (`paired_spread_dit_vs_cnnperc.json`).

So the claim is about what a diffusion JSCC receiver needs: digital symbols, no SNR at the
receiver, one model for all SNRs, and few steps. It is not a claim that the DiT beats CNN or
separate designs at a fixed SNR.

**Digital 16-QAM: the same picture** (g-q1, g-q2; `results/gen/analysis/paired_spread_qam16_vs_blind_s0.json`,
10-class set, 10 steps; figure `results/gen/analysis/snr_placement.png`). Every transmitted symbol
is a 16-QAM point. PSNR differences against the blind 16-QAM receiver (seed 0):

| 16-QAM model | −5 dB | 0 dB | +5 dB |
|---|---|---|---|
| told | −0.04 | 0.00 (ns) | +0.07 |
| specialist at −5 / 0 / +5 dB | +0.57 | −0.03 (ns) | −0.02 (ns) |
| SNR-aware transmitter, blind receiver, seed 0 | **+0.37** [0.31, 0.45] | 0.00 (ns) | −0.06 |
| SNR-aware transmitter, blind receiver, seed 1 | **+0.36** [0.30, 0.42] | −0.01 (ns) | −0.04 |

- **The analog findings carry over to digital.** Telling the receiver adds nothing; one blind
  model matches the specialists at 0 and +5 dB.
- **The SNR-aware transmitter recovers about 65% of the −5 dB gap** in both seeds. It also lowers
  LPIPS at every SNR (−0.013 at −5 dB).
- **Energy is held fair.**
  - Each training batch has one SNR.
  - Evaluation measures Es separately at every SNR: 0.92 / 0.90 / 0.89 (seed 0) and
    0.93 / 0.91 / 0.90 (seed 1) at −5 / 0 / +5 dB.
  - The transmitter therefore cannot buy SNR by choosing higher-energy points.
  - The first 1,000 images give the same result: +0.38 and +0.34 dB at −5 dB
    (`paired_first_qam16_vs_blind_s0.json`).
- **The price of digital symbols** (blind receiver, analog minus 16-QAM): 0.31 / 0.53 / 0.76 dB
  PSNR and 0.016 / 0.024 / 0.029 LPIPS at −5 / 0 / +5 dB.
- **The 16-QAM CNN at 0 dB** (MSE) has +3.1 dB PSNR at 0 dB, but much worse LPIPS (0.45 vs 0.30)
  and retrieval top-1 (5.4% vs 10.0%). Below its training SNR it loses on every metric
  (−0.5 dB at −5 dB).
- **Outside the training range the SNR-aware gain grows** (g-q3,
  `paired_spread_curve_qam16_vs_blind_s0.json`). The gain over the blind 16-QAM receiver is:
  - +0.94 dB [0.85, 1.03] at −10 dB;
  - +0.62 dB at −7.5 dB;
  - +0.13 dB at −2.5 dB;
  - −0.10 dB at +10 dB.

  The energy sent stays at 0.88–0.92 at every SNR.
- **Guidance 1.5 works on 16-QAM as well** (`paired_spread_qam16_guidance15_s0.json`), for blind
  and SNR-aware alike. It gains 0.03–0.05 LPIPS, 14 FID points, and +3.6 / +5.9 points of
  retrieval top-1 at 0 / +5 dB, for 0.15–0.2 dB PSNR.
- **Against separation, both digital** (`results/gen/analysis/cliff_qam16.png`):
  - The generative 16-QAM link degrades gradually from +10 to −10 dB with no cliff.
  - It has the lowest LPIPS of every system at every SNR, even against a separate design
    re-designed for every SNR: 0.22 vs 0.28 at +10 dB and 0.29 vs 0.59 at 0 dB.
  - The 16-QAM CNN, at any SNR, also loses to it on LPIPS.
  - Separation keeps the PSNR lead everywhere it decodes (20.8 vs 17.3 dB at 0 dB).
  - Separation leads retrieval top-1 from 0 dB up (18% vs 10% at 0 dB; 83% vs 26% at +10 dB).
- Tables: `results/gen/report-spread-qam16/report.md`; analog tables:
  `results/gen/report-spread/report.md`.

**Running next** (job plans and pinned commits: `results/gen/kaggle_jobs.json`):

- All planned jobs have finished (g-q1, g-q2, g-q3, g-e2).
