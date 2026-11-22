# Analysis: 256 px re-check, 16-QAM, CBR 1/48: blind DiT-S against the blind two-stage CNN, both trained to convergence by the same rule (same images, same GPU)

Reference: **CNN, MSE+LPIPS, blind, converged, 256 px** (`results\gen\p256-conv\cnn_cbr48_qam16_blind-5to5_lpips0.1_ft_s0`). Test set: the 1,000 class-spread test images, 10 sampling steps, paired channel and sampling noise; all runs scored on the same GPU type.

| Key | System | Run |
|---|---|---|
| cnn | CNN, MSE+LPIPS, blind, converged, 256 px | `results\gen\p256-conv\cnn_cbr48_qam16_blind-5to5_lpips0.1_ft_s0` |
| dit | DiT-S, blind, 256 px | `results\gen\p256-conv\dit-S-v_cbr48_qam16_blind-5to5_s0` |
| cnnpx | CNN, MSE only (stage 1), converged, 256 px | `results\gen\p256-conv\cnn_cbr48_qam16_blind-5to5_s0` |

## 1. Verdict against the pre-registered rule

Rule fixed on 17 Nov before any result: a candidate **beats the reference** if, at 0 and +5 dB, it has a clear gain on LPIPS (95% interval below 0) or on FID (more than 5 points lower), and no clear loss (interval entirely on the wrong side) on DINOv2 similarity or retrieval top-1. Between candidates that pass, the higher DINOv2 similarity wins.

| Candidate | 0 dB: perceptual gain / semantic loss | 5 dB: perceptual gain / semantic loss | Verdict |
|---|---|---|---|
| DiT-S, blind, 256 px | yes / yes | yes / yes | does not pass |
| CNN, MSE only (stage 1), converged, 256 px | no data | no data | not scored yet |

## 2. Metrics against SNR

Means over the 1,000 images. FID uses 1,000 images (relative only). Class kept: DINOv2 10-NN classifier on the reconstructions (from the dump).

![curves.png](curves.png)

Dotted lines are context only (CNN, MSE+LPIPS, 40k + 10k steps, 256 px): scored on their own images, so they are not paired with or judged against the systems above.

| SNR | System | PSNR | LPIPS | DINOv2 | Top-1 | FID | Class kept |
|---|---|---|---|---|---|---|---|
| -10 | DiT-S, blind, 256 px | 12.64 | 0.580 | 0.055 | 1% | 244 |  |
| -10 | CNN, MSE+LPIPS, 40k + 10k steps, 256 px (context) | 14.29 | 0.603 | 0.026 | 0% | 289 |  |
| -7.5 | DiT-S, blind, 256 px | 15.06 | 0.457 | 0.165 | 9% | 172 |  |
| -7.5 | CNN, MSE+LPIPS, 40k + 10k steps, 256 px (context) | 16.46 | 0.524 | 0.084 | 4% | 238 |  |
| -5 | CNN, MSE+LPIPS, blind, converged, 256 px | 18.46 | 0.402 | 0.297 | 25% | 129 | 78% |
| -5 | DiT-S, blind, 256 px | 17.05 | 0.371 | 0.304 | 27% | 120 | 78% |
| -5 | CNN, MSE+LPIPS, 40k + 10k steps, 256 px (context) | 18.25 | 0.447 | 0.218 | 16% | 174 |  |
| -2.5 | DiT-S, blind, 256 px | 18.55 | 0.300 | 0.443 | 57% | 90 |  |
| -2.5 | CNN, MSE+LPIPS, 40k + 10k steps, 256 px (context) | 19.75 | 0.380 | 0.395 | 46% | 116 |  |
| 0 | CNN, MSE+LPIPS, blind, converged, 256 px | 21.29 | 0.288 | 0.604 | 83% | 59 | 95% |
| 0 | DiT-S, blind, 256 px | 19.72 | 0.249 | 0.561 | 81% | 70 | 93% |
| 0 | CNN, MSE+LPIPS, 40k + 10k steps, 256 px (context) | 20.95 | 0.332 | 0.533 | 71% | 82 |  |
| 2.5 | DiT-S, blind, 256 px | 20.50 | 0.216 | 0.647 | 90% | 59 |  |
| 2.5 | CNN, MSE+LPIPS, 40k + 10k steps, 256 px (context) | 21.88 | 0.302 | 0.619 | 87% | 64 |  |
| 5 | CNN, MSE+LPIPS, blind, converged, 256 px | 22.79 | 0.246 | 0.720 | 94% | 43 | 98% |
| 5 | DiT-S, blind, 256 px | 21.02 | 0.196 | 0.703 | 95% | 51 | 97% |
| 5 | CNN, MSE+LPIPS, 40k + 10k steps, 256 px (context) | 22.51 | 0.284 | 0.672 | 90% | 56 |  |
| 7.5 | DiT-S, blind, 256 px | 21.33 | 0.185 | 0.724 | 96% | 45 |  |
| 7.5 | CNN, MSE+LPIPS, 40k + 10k steps, 256 px (context) | 22.91 | 0.274 | 0.699 | 92% | 52 |  |
| 10 | DiT-S, blind, 256 px | 21.50 | 0.178 | 0.748 | 97% | 44 |  |
| 10 | CNN, MSE+LPIPS, 40k + 10k steps, 256 px (context) | 23.15 | 0.269 | 0.713 | 94% | 49 |  |

## 3. Paired differences against the reference

Candidate minus reference per image, mean with 95% bootstrap interval; **bold** = clear gain, *italic* = clear loss. Win rate: share of images where the candidate is better.

| Candidate | SNR | ΔPSNR (dB) | ΔLPIPS | ΔDINOv2 | ΔTop-1 (pts) | ΔTop-5 (pts) | Win rate PSNR / LPIPS / DINOv2 |
|---|---|---|---|---|---|---|---|
| DiT-S, blind, 256 px | -5 | *-1.42 [-1.48, -1.35]* | **-0.031 [-0.035, -0.028]** | +0.008 [-0.003, +0.019] | +1.8 [-1.6, +5.2] | +3.4 [-0.1, +6.9] | 7% / 73% / 52% |
| DiT-S, blind, 256 px | 0 | *-1.57 [-1.61, -1.52]* | **-0.039 [-0.042, -0.036]** | *-0.043 [-0.053, -0.034]* | -1.7 [-4.6, +1.3] | *-2.1 [-3.9, -0.2]* | 2% / 85% / 41% |
| DiT-S, blind, 256 px | 5 | *-1.77 [-1.81, -1.72]* | **-0.050 [-0.052, -0.047]** | *-0.017 [-0.024, -0.009]* | +1.3 [-0.4, +3.0] | +0.3 [-0.4, +1.0] | 1% / 89% / 47% |

## 4. Perception-distortion and perception-semantics planes

Each line is one system across SNR (marker size grows with SNR). Down-right is better in both panels: a system that only trades along a line moved along the trade-off, one that moves the whole line beat it.

![plane.png](plane.png)

## 5. Per image: does a perceptual gain cost semantics on the same image?

Each dot is one test image: x = ΔLPIPS (left of 0 = sharper / closer perceptually), y = ΔDINOv2 (above 0 = semantically closer), candidate minus reference. The corner shares say how often a candidate wins on both, only perceptually, only semantically, or on neither.

**0 dB**

| Candidate | Both better | Perceptual only | Semantic only | Neither |
|---|---|---|---|---|
| DiT-S, blind, 256 px | 37% | 49% | 4% | 10% |

![per_image_0.png](per_image_0.png)

**5 dB**

| Candidate | Both better | Perceptual only | Semantic only | Neither |
|---|---|---|---|---|
| DiT-S, blind, 256 px | 43% | 46% | 4% | 7% |

![per_image_5.png](per_image_5.png)

## 6. Class kept per class

DINOv2 10-NN class of each reconstruction against the true class, 1000 images (100 per class on average).

![per_class_-5.png](per_class_-5.png)

![per_class_0.png](per_class_0.png)

![per_class_5.png](per_class_5.png)

## 7. Images

One test image per class. Under each reconstruction: the class the classifier sees (✓ = true class) and the image's PSNR.

![images_-5.png](images_-5.png)

![images_0.png](images_0.png)

![images_5.png](images_5.png)

![extremes_dit.png](extremes_dit.png)

### Close-ups at full resolution

4 images at their native 256 px (one image pixel per figure pixel), each followed by its centre third enlarged: blur, invented texture and artefacts show here. Under each reconstruction: its PSNR | DINOv2 similarity to the original.

![closeups_0.png](closeups_0.png)

![closeups_5.png](closeups_5.png)

## 8. Power spectrum against the originals

Radially averaged power spectrum of the reconstructions over that of the originals (kept images, grey level). Below 1 at high frequency = blur (detail removed); above 1 = texture or noise added that the original does not have.

![spectrum.png](spectrum.png)

High-frequency power (upper half of the band) relative to the originals:

| System | -5 dB | 0 dB | 5 dB |
|---|---|---|---|
| CNN, MSE+LPIPS, blind, converged, 256 px | 0.17 | 0.19 | 0.19 |
| DiT-S, blind, 256 px | 0.87 | 0.98 | 1.04 |
| CNN, MSE only (stage 1), converged, 256 px | 0.10 | 0.11 | 0.12 |

## 9. Image statistics against the originals

Ratios to the original image (1 = same): sharpness = variance of the Laplacian, contrast = pixel standard deviation, saturation = mean HSV saturation; colour shift = mean absolute change of the per-channel means (0-255). Means over the kept images.

| System | SNR | Sharpness | Contrast | Saturation | Colour shift |
|---|---|---|---|---|---|
| CNN, MSE+LPIPS, blind, converged, 256 px | -5 | 0.15 | 0.87 | 0.79 | 3.4 |
| DiT-S, blind, 256 px | -5 | 0.90 | 0.94 | 0.92 | 5.1 |
| CNN, MSE only (stage 1), converged, 256 px | -5 | 0.04 | 0.89 | 0.85 | 2.4 |
| CNN, MSE+LPIPS, blind, converged, 256 px | 0 | 0.18 | 0.94 | 0.86 | 2.0 |
| DiT-S, blind, 256 px | 0 | 1.01 | 0.96 | 0.96 | 3.3 |
| CNN, MSE only (stage 1), converged, 256 px | 0 | 0.07 | 0.94 | 0.86 | 1.4 |
| CNN, MSE+LPIPS, blind, converged, 256 px | 5 | 0.20 | 0.95 | 0.88 | 1.5 |
| DiT-S, blind, 256 px | 5 | 1.07 | 0.98 | 0.99 | 2.7 |
| CNN, MSE only (stage 1), converged, 256 px | 5 | 0.09 | 0.95 | 0.88 | 1.1 |

(Sharpness, contrast and saturation are medians of per-image ratios.)

## 10. Sample diversity

The same received symbols decoded several times with different sampling noise. Spread = per-pixel standard deviation across draws (0-255), averaged; draw-to-draw PSNR = between two draws of the same image. A deterministic receiver has spread 0. High spread at high SNR means the receiver invents content the channel does carry.

| System | SNR | Spread | Draw-to-draw PSNR (dB) |
|---|---|---|---|
| DiT-S, blind, 256 px | -5 | 14.3 | 19.0 |
| DiT-S, blind, 256 px | 0 | 9.5 | 22.6 |
| DiT-S, blind, 256 px | 5 | 8.3 | 23.7 |

![diversity_dit.png](diversity_dit.png)

## 12. Training

From each run's logs (spec `logs`, in order; resumed runs merged) and history.json. The diffusion loss of the generative systems and the CNN's pixel loss are different quantities: compare loss within a family only.

Training loss is logged every 100 steps (mean over those steps); validation uses the EMA weights: loss on the validation split, quality on 32 fixed validation images (noisy: compare trends, not single points). Loss jitter = standard deviation / mean of the logged training loss over 2k-step windows.

| Run | Steps | State | Lowest val loss (step) | Last val loss | Validations since a 0.2% gain | Loss jitter, last 10k | Grad norm, last 10k | AMP-skipped | Val DINOv2 @0 dB (best / last) |
|---|---|---|---|---|---|---|---|---|---|
| CNN, MSE+LPIPS, blind, converged, 256 px | 36,000 | finished | 0.0461 (36,000) | 0.0461 | 1 of 10 (since 34,000) | nan% | nan | 0 | 0.671 / 0.671 |
| DiT-S, blind, 256 px | 140,000 | finished, converged | 0.2840 (140,000) | 0.2840 | 10 of 10 (since 120,000) | 0.8% | 0.158 | 51 | 0.664 / 0.664 |
| CNN, MSE only (stage 1), converged, 256 px | 40,000 | running | 0.0091 (40,000) | 0.0091 | 0 (since 40,000) | 0.6% | 0.038 | 13 | 0.410 / 0.410 |

![training_loss.png](training_loss.png)

![training_jitter.png](training_jitter.png)

![training_val.png](training_val.png)

Line style is the SNR (solid / dashed / dotted = -5 / 0 / 5 dB). Above zero in the right panel, the EMA weights the run is evaluated with beat its raw weights.

![training_convergence.png](training_convergence.png)

A run stops once `patience` validations in a row stay under the dashed line (no gain above 0.2% over its best). Points near the line mean it is still improving slowly.

![training_diffusion.png](training_diffusion.png)

**DiT-S, blind, 256 px**: validation samples over training. In each block the top row is the original, then one row per validation SNR, low to high. These are the trainer's 8 fixed validation images (the first ones of the validation split, so mostly one class); the class-spread images are in section 7.

![training_samples_dit.png](training_samples_dit.png)
