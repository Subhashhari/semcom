# One Generative JSCC Receiver for Every SNR: a Small DiT-JSCC Prototype

**Status (8 Nov 2026):** week-1 gate passed (DiT-S beats the CNN at 0 dB on LPIPS, DINOv2 and
FID); the week-2 SNR study (g-w3) is running; digital M-QAM transmission added (section 7), first
digital job (g-w4) running.

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
| next | teammate | 4-QAM, 64-QAM `spec0` (resumed), 16-QAM `blind`, `spec−5`, `spec5`; CNN 16-QAM `spec0` | modulation sweep; full told/blind/specialist study with 16-QAM; DiT vs CNN under digital transmission |

**What 18 Nov shows.** (1) Analog vs 4/16/64-QAM DiT at 0 dB, evaluated at −5, 0, 5 dB, on PSNR,
LPIPS, DINOv2 and FID. (2) 16-QAM told vs the 16-QAM specialist(s), and blind if g-w6 finishes in
time. (3) Symbol usage and transmit energy per run. Expected by 14–16 Nov at the current per-run
cost of about 1.5 h of training plus about 0.3 h of evaluation on a T4.
