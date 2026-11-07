# One Generative JSCC Receiver for Every SNR: a Small DiT-JSCC Prototype

**Status (7 Nov 2026):** pipeline implemented and tested (`semcom/gen/`, `scripts/run_gen.py`,
`tests/test_gen.py`); no GPU runs yet.

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
| Wed 18 Nov | comprehensive | The SNR study: three specialists vs `told` vs `blind` at −5, 0, 5 dB, with paired comparisons; CNN baselines; storage (1 model vs 3) and decoding-cost table; SNR-mismatch result (specialists away from their SNR); limitations and next steps (few-step distillation) |
