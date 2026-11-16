# Generative JSCC: review of 16 Nov 2026 and the plan to 13 Dec 2026

This document records a critical review of the DiT-JSCC prototype (`semcom/gen`, results in
`results/gen`), the position the work can defend, and the four-week experiment plan that follows
from it. It is written so that a later task can pick up any experiment without the conversation
that produced it. The earlier design and status notes are in
[gen-jscc-prototype.md](gen-jscc-prototype.md); where the two disagree, this document is newer.

Every number below was recomputed on 16 Nov 2026 from the committed per-image result files
(`eval_images-*.npz`) unless marked otherwise. Literature statements come from paper text fetched
that day; section 3 says which ones could not be confirmed verbatim.

## 1. Verdict of the review

The evaluation method is sound and the measurements reproduce. The model they are measured on is
too weak to carry a claim, and the headline finding ("one diffusion receiver for a range of SNRs")
is already published. The work needs a better model and one missing control before anything is
written up.

### What is solid

- Paired evaluation: every arm sees the same test images, channel noise and sampling noise.
  Seed 0 and seed 1 of the blind model are different models (per-image PSNR differs by 0.8 to
  1.4 dB) whose means agree within 0.01 dB.
- Telling the receiver the SNR adds nothing: at most 0.04 dB PSNR against the blind receiver, on
  analog symbols, 16-QAM and Rayleigh fading.
- Giving the SNR to the transmitter gains +0.48 dB at -5 dB [0.41, 0.56], in two seeds.
- The Kaggle pipeline, the test-leak exclusion for extra data and the separation baseline are
  reusable.

### What undermines it

All rows use the 1,000-image 10-class set (`--eval-subset spread`), 10 sampling steps, 0 dB.

| | DiT-S blind | CNN, MSE + 0.1 LPIPS, trained at 0 dB only |
|---|---|---|
| PSNR | 17.8 dB | 20.1 dB |
| DINOv2 similarity | 0.19 | 0.38 |
| Retrieval top-1 | 14% | 36% |
| Class kept (300 images) | 59% | 82% |
| LPIPS (lower is better) | 0.271 | 0.287 |
| FID (1,000 images) | 142 | 104 |

1. **The model barely transmits the image.** An 8x8 thumbnail, upscaled, scores 17.6 dB PSNR on the
   same images. The VAE alone reaches 22.8 dB and DINOv2 similarity 0.81.
2. **A cheap CNN beats it.** The perceptual CNN above is a single-SNR model, yet it leads on PSNR,
   DINOv2 similarity and retrieval from -2.5 to +10 dB.
3. **The DiT's remaining win is confounded.** It wins at -5 dB and below against a CNN that never
   saw those SNRs. No CNN was trained over the SNR range with a perceptual loss.
4. **The SNR findings are expected.** A receiver can read the noise level off the received power.
   SNR at the encoder is the ADJSCC idea, and its gain here is PSNR only: DINOv2 similarity is
   unchanged and LPIPS moves by 0.008.
5. **Every quality lever tried so far came back flat:** DiT-B (4x parameters), 120k steps, the
   semantic/detail bandwidth split, and 6x more data from other classes (validation loss fell from
   0.302 to 0.285, with mixed test results). DiT-B and guidance 1.5 also leave class preservation
   unchanged (61% / 60% / 59% at 0 dB for DiT-S / guidance / DiT-B, GPU run g-c1).
6. **The scale-down removed the generative prior.** DiT-JSCC starts from a pretrained
   class-conditional generator; this is a 35M-parameter model trained from scratch on 9,469 images.
7. **Smaller issues.** FID uses 1,000 images with 2,048 features, so absolute values are
   unreliable. Class preservation used 300 images. The week-1 gate was passed on a 3-class subset.

### Where quality is lost (what the plan targets)

- Noise is not the main limit: at +10 dB the blind model reaches only 19.9 dB and DINOv2
  similarity 0.34.
- Training is small: 40k steps at batch 64, constant learning rate 1e-4, and the transmitter learns
  only through the denoising loss.
- 128 px hurts twice: the SD-VAE runs below its native resolution, and a bandwidth ratio of 1/48
  gives 1,024 symbols (4,096 at 256 px).

## 2. Claims and their status

| # | Claim | Evidence | Novel? | Useful? |
|---|---|---|---|---|
| 1 | One DiT receiver works over -5 to 5 dB | Yes | No (DiffJSCC, FlowSem, OSD-JSCC) | Low |
| 2 | Its gap to per-SNR specialists: 0 at 0 and +5 dB, 0.64 dB at -5 dB | Yes, 2 seeds | Yes, narrowly: no paper read compares against separately trained specialists | Moderate |
| 3 | Telling the receiver the SNR adds nothing | Yes, 3 settings | Weak: SIJSCC shows it for a non-generative codec; OSD-JSCC has a partial ablation | Small |
| 4 | SNR at the transmitter only recovers 75% of the -5 dB gap | Yes, 2 seeds | Not found anywhere, but a search cannot prove absence | PSNR only |
| 5 | Wrong-SNR sensitivity at the transmitter | Weak (200 images, CPU) | Partly (DeepJSCC-Q has one for CNNs) | Minor |
| 6 | 16-QAM symbols into a diffusion receiver that generates from noise, 0.3 to 0.8 dB cost | Yes | Yes, still open | Best claim; uncoded, toy scale |
| 7 | Sampling steps trade distortion for perception | Yes | No | Low |
| 8 | Guidance improves perception | Yes | No | Low |
| 9 | Graceful degradation; class kept at low SNR | Confounded | No | Low |
| 10 | Nominal-energy SNR hides a 0.8 dB handicap in quantised JSCC | Yes | Supported by DeepJSCC-Q's own definition | Footnote |

## 3. Literature position (checked 16 Nov 2026)

**Target position:** a constellation-constrained (16-QAM) diffusion JSCC that needs no SNR at
either end. No paper found combines QAM-constrained symbols, a diffusion receiver that generates
the image, and no SNR input. Each pair of those is taken.

| Work | What it already has | What it lacks |
|---|---|---|
| [DiT-JSCC](https://arxiv.org/html/2601.03112) (v2, 18 Jun 2026) | The architecture this prototype scales down; starts from a pretrained class-conditional generator | "we train separate models for each SNR"; no SNR input; analog |
| [DiffJSCC](https://arxiv.org/html/2404.17736) | Starts from N(0, I); one model over [0, 14] or [-5, 5] dB; SNR to encoder, decoder and control module | Analog; no specialist comparison, no mismatch test |
| [FlowSem](https://arxiv.org/html/2608.21651) (Aug 2026) | Generates from Gaussian noise; one model over a wide SNR range; SNR at both ends | Real-valued analog symbols |
| [OSD-JSCC](https://arxiv.org/html/2611.06437) (4 Nov 2026) | One model over [0, 20] dB, SNR at both ends; "w/o SNR modulation" costs 0.66 dB | Refines an estimate; analog; no per-end ablation |
| [CDM-JSCC](https://arxiv.org/html/2409.02597v3) | Conditional diffusion from Gaussian noise | One model per SNR; analog (QAM only in baselines, all three versions read) |
| [SGD-JSCC](https://arxiv.org/html/2501.01138) | Estimates the SNR from the channel output; transmitter has no CSI | Denoises the received features; analog |
| [SC-CDM](https://arxiv.org/html/2410.02121) | One model over 0 to 15 dB, no SNR input found | Restores a decoded image; analog |
| [SIJSCC](https://arxiv.org/abs/2306.15183) | "SNR-Independent" JSCC; SNR at both ends / decoder only / neither | Not generative; no encoder-only case |
| [SemQAM](https://arxiv.org/html/2610.29643) (Oct 2026) | QAM-native generative JSCC (1024-QAM, no channel code) | GAN-trained VQ decoder; per-operating-point codebooks |
| [SSCDM](https://arxiv.org/html/2603.22983), [SCDM](https://arxiv.org/html/2501.17876), [DSRDM](https://arxiv.org/abs/2605.27730) | QAM symbols plus diffusion | Diffusion cleans symbols; needs the SNR to pick the start step |
| [TokCom](https://arxiv.org/html/2502.12096), [Text-guided TokCom](https://arxiv.org/html/2507.05781) | Standard coded digital link, generative receiver with no SNR input, -5 to 5 dB, ImageNet 256 px | Not JSCC, not diffusion: coded bits plus masked-token filling |
| [DeepJSCC-Q](https://arxiv.org/html/2111.13042) | Soft-to-hard QAM quantiser; one model over 0 to 10 dB with an SNR input; mismatch study | CNN decoder; SNR defined against nominal constellation power |

**Wording to use and to avoid.**

- Do not claim "first single-model, SNR-adaptive diffusion JSCC" or "SNR-independent JSCC".
- Do not say "digitally deployable": there are no bits and no channel code, and the receiver needs
  the raw noisy samples. Use DeepJSCC-Q's term, "channel input constrained", or
  "constellation-constrained".
- "Needs no SNR" means no noise-level knowledge. The simulation assumes the receiver knows the
  signal gain exactly (experiment B4 tests this).
- The SNR-aware transmitter (claim 4) needs SNR feedback, so in this position it appears only as a
  contrast: what feedback would buy.
- TokCom is answered in related work plus the existing separation baseline
  (`scripts/gen_analysis/separation.py`), not with a new experiment.

**Not verified.**

- CD3M ("Channel Denoising in Digital Semantic Communication Systems via Discrete Diffusion",
  IEEE document 11305152) could not be opened. A search snippet says it is a discrete-diffusion
  symbol denoiser with one codec for all channel states.
- Citation lists (Google Scholar) for DiT-JSCC, SemQAM and DeepJSCC-Q were not checked.
- SGD-JSCC's "nearly identical, whether the SNR is available or not" and SIJSCC's scenario
  wording came from the fetch tool's summary, not verbatim text.
- The proposal says DeepJSCC-Q raises its quantiser temperature to 100. The conference paper uses
  a constant ("σq=100"); the journal version (arXiv:2206.08100) was not read.

## 4. Compute

| Machine | Budget | Use for |
|---|---|---|
| Kaggle T4, 4 accounts (own, teammates) | About 120 h a week in total (30 h each, reset on Saturdays UTC), 12 h per session, jobs in parallel | Short and medium runs: recipe tests, controls, seeds, the final arms |
| RTX 5060 (separate machine) | Unmetered, one run at a time; 8 GB memory assumed | Long single runs: pretraining, long training, 256 px |

- The development PC has CPU-only PyTorch and about 10 GB of free disk. Do not run the full test
  suite there: run targeted tests with `--basetemp` in a scratch folder and delete it.
- The RTX 5060 needs a PyTorch build for CUDA 12.8 or newer and about 40 GB of disk for the
  ImageNet-100 cache. Its speed is unmeasured; measure it before planning around it.
- Measured T4 costs: DiT-S 40k steps 1.5 h, DiT-B 40k steps 3.8 h, DiT-S 100k steps on 59k images
  3.7 h, evaluation about 0.3 h.
- **Ruled out:** fine-tuning a public pretrained generator. The released checkpoints are
  [DiT-XL/2](https://github.com/facebookresearch/DiT) (256 and 512 px, non-commercial licence),
  [SiT-XL/2](https://github.com/willisma/SiT) (256 px, MIT) and
  [U-ViT-L/2, H/2](https://github.com/baofff/U-ViT) (ImageNet 256 px). XL/2 costs 119 GFLOPs per
  image, roughly 80 times the current model by estimate: too large for 8 GB, and an estimated 15
  to 30 h per model on a T4.
- **Full ImageNet is not needed.** Imagenette already holds nearly all ImageNet training images of
  its 10 classes, and caching ImageNet-1k with DINOv2 features would take about 320 GB by
  estimate. The full ImageNet-100 set (about 126k images, Kaggle dataset `ambityga/imagenet100`)
  fits.
- **Recipe cap:** the final recipe must cost at most about 4 h per run on a T4, so the 12 final
  runs fit in one Kaggle week.

## 5. Levers for a better model, ranked by expected gain per GPU hour

| # | Lever | Cost | Why | Code |
|---|---|---|---|---|
| 1 | Two ceiling diagnostics | 4 h | Shows whether the transmitter or the generator is the limit | Small |
| 2 | REPA alignment loss | 2 h per run | Aligns the DiT's hidden states with DINOv2 patch features. The [paper](https://arxiv.org/html/2410.06940) reports FID 18.8 to 9.7 for SiT-L/2 at 400k steps with DINOv2-B, depth 8, weight 0.5. The cache already holds DINOv2-B features on the 8x8 token grid. Untested on small data or conditional decoding | New loss |
| 3 | Training recipe | 2 h per variant | Timestep weighting, learning-rate decay, and a direct regression loss so the transmitter gets a stronger signal | New options |
| 4 | Pretrain broad, then fine-tune on Imagenette | 1.5 h first test | The 59k-image model was still improving at 100k steps; `--init-from` exists | None |
| 5 | 256 px | About 6 h per run (estimate) | 4x the symbols; VAE at its native resolution; matches the literature | Check config |
| 6 | Longer training, DiT-B | 10 to 15 h | Only after 2 to 4: both overfitted on 9.5k images | None |

## 6. Timeline

Dates are 2026. "Needs code" marks work to do before the run can start; the notes on existing
flags are expectations and should be checked with `--dry-run` first.

### Week 1 (16 to 22 Nov): find the bottleneck, try the cheap levers

Kaggle, about 25 h:

| ID | Experiment | Runs | Needs code |
|---|---|---|---|
| W1-D1 | Rate ceiling: blind-architecture DiT-S trained and tested at 30 dB (effectively noise-free) | 1 | No (`--arms spec` at a fixed SNR, or `--snr-range 30 30`) |
| W1-D2 | Generator ceiling: evaluate with the true DINOv2 features and clean detail features fed to the receiver | 1 | Yes, small |
| W1-R1 | REPA loss on the blind analog DiT-S, 40k steps | 1 | Yes |
| W1-R2 | Recipe: timestep weighting (for example min-SNR or logit-normal sampling) | 1 | Yes |
| W1-R3 | Recipe: direct regression loss from the received symbols to the clean latent | 1 | Yes |
| W1-F1 | Fine-tune the 59k-image model (`results/gen/p128-long/dit-S-v_cbr48_blind-5to5_ximagenet100-500_s0`) on Imagenette only | 1 | No (`--init-from`) |
| W1-C1 | Control: 16-QAM CNN, MSE + 0.1 LPIPS, blind over -5 to 5 dB, seeds 0 and 1 | 2 | Check that `--qam`, `--cnn-lpips-weight` and the blind arm combine |
| W1-C2 | Same CNN told the SNR; and the analog blind version | 2 | As above |
| W1-E1 | Evaluation upgrade: FID on all 3,425 test images; class preservation on 1,000 images | re-evaluate | Small |

RTX 5060: set up, measure speed, then start pretraining DiT-S on the full ImageNet-100 set with
the current recipe (`--extra-train`, no `--extra-per-class` cap).

**Gate 1 (22 Nov).**

- Keep a lever only if it beats the baseline on LPIPS, DINOv2 similarity or retrieval with a clear
  paired interval (`scripts/gen_analysis/paired.py`).
- Target to beat: the perceptual CNN at 0 dB (DINOv2 similarity 0.38, retrieval 36%, class kept
  82%, FID 104).
- If W1-C1 matches or beats the DiT everywhere **and** no lever moved the DiT, stop the DiT track
  and choose between an architecture-independent SNR-placement study and a different generator.

### Week 2 (23 to 29 Nov): combine and scale

Kaggle, about 35 h:

| ID | Experiment | Runs |
|---|---|---|
| W2-K1 | Combined recipe (week-1 winners), blind analog and blind 16-QAM | 2 |
| W2-K2 | Fine-tunes from the 5060's pretraining checkpoints | 2 |
| W2-K3 | Best recipe at 256 px | 1 |
| W2-K4 | W1-C1 re-run at matched settings if the recipe changes data or resolution | 1 to 2 |

RTX 5060: re-pretrain with the winning recipe for 300k steps or more; then DiT-B on the same data
if DiT-S is still improving.

**Gate 2 (29 Nov).** Fix one recipe that costs at most about 4 h per run on a T4. Everything after
this uses it.

### Week 3 (30 Nov to 6 Dec): is "needs no SNR" true?

Kaggle, about 25 h, all with the fixed recipe:

| ID | Experiment | Runs | Needs code |
|---|---|---|---|
| W3-B1 | 16-QAM on Rayleigh fading: DiT blind, DiT told, SNR-aware transmitter, CNN blind | 4 | Check that `--channel rayleigh` and `--qam` combine |
| W3-B2 | Range where 16-QAM is really used: blind and told over 0 to 20 dB, specialists at 0, 10, 20 dB | 5 | No (`--snr-range`) |
| W3-B3 | CNN control for B2 | 1 | No |
| W3-B4 | Gain error: received amplitude wrong by ±1 and ±2 dB; imperfect fading estimate | evaluation only | Yes, small |
| W3-B5 | Estimated-SNR arm: the told model fed an SNR estimated from the received symbols (`semcom/snr_estimation.py`), with the estimator's error per SNR | evaluation only | Yes, small |

RTX 5060: long-training study of the best model; a second pretraining seed.

**Gate 3 (6 Dec).** Blind must stay within about 0.1 dB of told on fading and on the wide range.
If a 1 dB gain error costs more than SNR knowledge ever gained, reword the claim as "no
noise-level knowledge, with gain tracking assumed".

### Week 4 (7 to 13 Dec): retrain the arms and write

Kaggle, about 70 h:

| ID | Experiment | Runs |
|---|---|---|
| W4-A1 | 16-QAM arms with the fixed recipe: blind, told, specialists at -5 / 0 / 5 dB, SNR-aware transmitter | 6 |
| W4-A2 | Second seed for blind and told; third seed for blind | 3 |
| W4-A3 | CNN control at matched settings, 2 seeds | 2 |
| W4-A4 | 4-QAM and 64-QAM blind (optional) | 2 |
| W4-T1 | Deployment table: peak-to-average power, constellation usage, parameters, time per image for transmitter and receiver, against the CNN | evaluation only |
| W4-T2 | Final paired tables and figures from committed results | none |
| W4-L1 | Literature re-check: read CD3M, check who cites DiT-JSCC, SemQAM and DeepJSCC-Q, re-verify every quote against the PDFs | none |

RTX 5060: buffer and reruns.

## 7. Working rules for these experiments

- Evaluate on the 10-class set (`--eval-subset spread --eval-suffix spread`) at 10 sampling steps,
  and compare runs with `scripts/gen_analysis/paired.py`. Compare only runs scored on the same
  device type: CPU and GPU draw different noise.
- Extra training images go through `--extra-train`, which skips any file whose name matches an
  Imagenette image. Imagenette's validation split, the source of the test images, is 3,791 of
  3,925 ImageNet training images under their ImageNet names.
- Check a job name against `results/gen/kaggle_jobs.json` before pushing. Fetch only the run
  folders (`kaggle_jobs.py fetch <job> --only <regex>`); a full fetch downloads the whole results
  tree.
- A Kaggle job that stays queued for more than about 20 minutes while other accounts start at once
  is stuck: delete the kernel and push the same kernel folder again.
- The repository is public. Never commit Kaggle tokens.
- Record each finished experiment in section 8 with its job name, the paired result and the gate
  decision it feeds.

## 8. Log

| Date | ID | Job | Result |
|---|---|---|---|
| 16 Nov | (pre-plan) | g-x1, g-x2 | 59k images from other classes: helps at +5 dB (+0.3 dB PSNR), hurts at -5 and 0 dB (FID +18 to +28) |
| 16 Nov | (pre-plan) | g-c1 | Class preservation on GPU, 300 images: DiT-S 61%, guidance 1.5 60%, DiT-B 59% at 0 dB |
