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

**Track A, the SNR-placement study (checked 18 Nov).** A blind analog CNN JSCC is taken: CBJSCC
([Sensors 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11209452/)) works "without relying on
external SNR information", trained over -5 to 20 dB, tested on AWGN and slow Rayleigh, against
ADJSCC with perfect and fixed feedback; SIJSCC (AWGN only) reports "SNR estimation does not
significantly benefit SIJSCC". Neither constrains the symbols to a constellation (both map to
continuous complex symbols), and neither tests an estimated-SNR receiver against a true-SNR one or
a gain error. A QAM-mapped JSCC for driving ([arXiv 2604.20278](https://arxiv.org/html/2604.20278))
trains at one SNR (25 dB) with no blind/told comparison. So the open part of A is narrow: where
the SNR must be known **under a constellation constraint**, with estimated-SNR and gain-error
arms, for a CNN and a diffusion receiver. CBJSCC's Section 4.4 and SIJSCC's per-scenario results
were read through a summary, not verbatim; read them before writing related work.

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
| W1-D2 | Generator ceiling: a DiT-S conditioned on the full DINOv2 grid and clean detail features, no channel (a short training run, since the receiver expects compressed symbols) | 1 | Yes, small (bypass the encoders) |
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
  82%, FID 104). Once W1-C1 lands, the target for the 16-QAM position is the blind 16-QAM
  perceptual CNN (g-cb2) at every SNR, not the 0 dB analog specialist.
- Read W1-D1 and W1-D2 before the levers: if the noise-free DiT stays below the CNN on DINOv2
  similarity, the loss is in the transmitter or the generator, not the channel, and only W1-R1
  (generator) and W1-R3 (transmitter) address it. The transmitter already holds the image's
  DINOv2 features, yet the receiver's output reaches DINOv2 similarity 0.19 at 0 dB and 0.34 at
  +10 dB; the CNN reaches 0.38 through the same 1,024 symbols without them.
- If W1-C1 matches or beats the DiT everywhere **and** no lever moved the DiT, stop the DiT track
  and choose between an architecture-independent SNR-placement study and a different generator.

### Revised plan after Gate 1 (agreed 18 Nov)

What Gate 1 found: the blind 16-QAM perceptual CNN beats the DiT on PSNR, DINOv2 similarity
and retrieval at every SNR and on FID from -5 dB up, after both are trained to convergence; the
DiT keeps only LPIPS. One lever moved the DiT (more data, W1-F1), not enough. Two causes remain:
a weak prior from 9.5k images, and the SD-VAE, which at 128 px already costs a quarter of the
CNN's semantics on a round trip (g-s3) but much less at 256 px (`vae_resolution.py`). So the DiT is
not developed further at 128 px. Two tracks, on separate hardware:

- **A. SNR-placement study (Kaggle, from the 21 Nov quota reset; W3 brought forward).** Does the
  receiver need the SNR? Arms: blind, told, SNR-aware transmitter, and told with an estimated SNR
  (`--eval-rx est-<cue>`); 16-QAM over AWGN 0 to 20 dB and Rayleigh fading; gain error
  (`--eval-rx gain<dB>`). CNN arms first (cheap); the DiT as the second architecture, to show the
  answer does not depend on the decoder. Novelty is checked against the papers before it is
  claimed (rule in section 7).
- **B. One converged 256 px test (RTX 5060).** Blind 16-QAM DiT-S at 256 px, the g-v1 recipe
  (`configs/gen/p256.yaml`, CBR 1/48, k = 4,096; `gen_5060.py dit256`), against the blind 16-QAM
  perceptual CNN at 256 px trained at the 128 px reference's budget (40k pixel + 10k LPIPS steps;
  `cnn256`, `cnn256-ft`). Imagenette only, so resolution is the one change. Pre-registered: the
  rule of 17 Nov at 0 and +5 dB (LPIPS gain with the 95% interval below 0, or FID more than 5
  lower, and no clear loss on DINOv2 or top-1). If it passes, converge the CNN at 256 px and
  re-check, then add the 59k-image data; the DiT is again a candidate main model. If it fails, the
  DiT stays only as the second architecture of A. A CNN at the reference's budget is enough for a
  fail: a converged CNN is only stronger.
- **Replications (Kaggle, this week's quota):** seed 1 of g-v1, g-v2 and g-cb2 (g-v3, g-v4, g-cb3).
  REPA is kept only if its top-1 gain replicates with a clear paired interval.
- **5060 order:** dit256, cnn256, cnn256-ft, then the paused 128 px convergence stages (the 16-QAM
  CNN resumes from step 198k; converging it can only widen a gap that already favours the CNN).

### Week 2 (23 to 29 Nov): combine and scale (superseded by the revised plan above)

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
| 16 Nov | W1-C2 (analog blind) | g-cb1 (subhashhhh; moved from own after 30 min queued) | Launched: analog CNN, blind -5 to 5 dB, pixel 40k then +0.1 LPIPS 10k (`p128-perc/cnn_cbr48_blind-5to5_lpips0.1_ft_s0`); first-1,000, spread and spread-curve evals |
| 16 Nov | W1-C1 (s0) | g-cb2 (venkat96r) | Launched: as g-cb1 with 16-QAM (`p128-perc/cnn_cbr48_qam16_blind-5to5_lpips0.1_ft_s0`) |
| 16 Nov | W1-D1 | g-hi1 (own) | Launched: `dit-S-v_cbr48_spec30_s0` evaluated at 30/15/5/0 dB (spread); the analog and 16-QAM blind models evaluated at 15 and 30 dB (`eval-spread-high.json`) |
| 16 Nov | W1-F1 | g-ft1 (venkat96r) | Launched: 59k-image model fine-tuned 20k steps on Imagenette (`p128-ft/dit-S-v_cbr48_blind-5to5_ft-imagenette_s0`) |
| 17 Nov | W1-F1 | g-ft1 | **Lever kept (Gate 1).** 59k model + 20k steps on Imagenette vs the 9.5k blind model, paired, spread set: LPIPS -0.011 / -0.021 / -0.022 at -5 / 0 / 5 dB (all significant); top-1 +4.8 and +13.8 pts at 0 and 5 dB; DINOv2 0.111 / 0.231 / 0.355 (was 0.095 / 0.189 / 0.284); FID 175 / 121 / 92 (was 187 / 142 / 112). Not just more steps: 120k steps on Imagenette alone gains at most 0.08 dB and 0-1 pt top-1. At +5 dB it now matches the perceptual CNN on top-1 (43% vs 44%) with far better LPIPS (0.188 vs 0.303) and FID (92 vs 115); at 0 dB still well behind on semantics (DINOv2 0.23 vs 0.38) |
| 17 Nov | W1-D1 | g-hi1 | Rate ceiling: spec30 at 30 dB reaches PSNR 21.1, DINOv2 0.393, top-1 51%, LPIPS 0.158, FID 93 (best checkpoint at 14k steps; val loss rises after 18k, so it overfits). The noise-free DiT is level with the perceptual CNN at 0-10 dB on DINOv2 (0.38) and far from the VAE (0.81): the transmitter/generator caps semantics, and noise explains the drop below +10 dB |
| 17 Nov | W1-D1 | g-hi1 | Blind (trained -5 to 5 dB) tested at 15 / 30 dB: analog DINOv2 0.362 / 0.378, PSNR 20.2 / 20.3, near the 30 dB specialist without being told the SNR. 16-QAM blind: DINOv2 0.280 / 0.288, PSNR 19.2 / 19.3, so at high SNR the constellation costs about 1 dB PSNR and 0.09 DINOv2 against analog |
| 17 Nov | W1-C1 (s0) | g-cb2 | **Failed before training**: overfit sanity check, 16-QAM blind CNN loss fell only to 0.91 of its start in 150 steps (bar 0.90; the 16-QAM 0 dB CNN reached 0.50, the analog 0 dB CNN 0.83). CPU probe (8 images): plateau near 150 steps, then 0.71 at 200 and 0.65 at 300, still falling, so slow, not broken. Relaunched at 7ed59f3 with `--sanity-steps 300` |
| 17 Nov | W1-C2 (analog blind) | g-cb1 | **The DiT's low-SNR advantage was the confound.** The perceptual CNN trained blind over -5 to 5 dB beats the best DiT (59k + ft) on PSNR, DINOv2 similarity and retrieval at every SNR from -10 to +10 dB (0 dB: 20.2 vs 18.1 dB, DINOv2 0.36 vs 0.23, top-1 34% vs 19%; +10 dB: 22.6 vs 20.3 dB, 0.55 vs 0.42, 73% vs 56%) and on FID from -5 dB up (ties at 10 dB: 81 vs 82). The DiT keeps only LPIPS from 0 dB up (0.250 vs 0.277 at 0 dB, 0.165 vs 0.232 at +10 dB); LPIPS ties at -5 dB and the CNN wins below. The blind CNN also beats the 0 dB-only CNN off 0 dB by a wide margin, so graceful degradation is a property of range training, not of diffusion |
| 17 Nov | W1-C1 (s0) | g-cb2 | Also stalls on the GPU with 300 check steps (0.91). Relaunched at 739b9de with `--sanity-steps 0` (skips only the overfit check). Val loss (mean over -5/0/5 dB) 0.0176 / 0.0161 at 2k / 4k steps vs 0.0156 / 0.0141 for the analog blind CNN: learns normally, about 12% above analog. The one-batch check is too strict for the 16-QAM blind CNN, not a pipeline fault |
| 17 Nov | W1-R2 (LR decay), 16-QAM | g-a1 (subhashhhh) | Launched at f73bf2f: blind 16-QAM DiT-S, cosine LR to 0 at 40k (`p128-w1/dit-S-v_cbr48_qam16_blind-5to5_cos_s0`). Kernel name reuses g-a1 (the 8 Nov g-a1 ran on the own account; its record stays in `results/gen/kaggle_jobs.json`) |
| 17 Nov | W1-R3, 16-QAM | g-a2 (subhashhhh) | Launched: as g-a1 plus the regression loss, weight 1 (`..._cos_reg1_s0`) |
| 17 Nov | W1-R1, 16-QAM | g-a3 (venkat96r) | Launched: as g-a1 plus REPA, weight 0.5, depth 4 (`..._cos_repa0.5_s0`) |
| 17 Nov | W1-C1 (s0) | g-cb2 | **16-QAM repeats the analog result.** Blind 16-QAM perceptual CNN vs the blind 16-QAM DiT (spread set): PSNR 19.5 vs 17.3 dB, DINOv2 0.29 vs 0.16, top-1 22% vs 10%, FID 118 vs 151 at 0 dB; ahead on PSNR, DINOv2 and top-1 at every SNR from -10 to +10 dB and on FID from -5 dB up. The DiT keeps a smaller LPIPS lead from 0 dB up (0.295 vs 0.306 at 0 dB, 0.221 vs 0.257 at +10 dB). 16-QAM costs the CNN about 0.7 dB PSNR and 0.07 DINOv2 against analog at 0 dB |
| 17 Nov | W1 option B | g-b1r (own) | Launched at f73bf2f: refiner on the g-cb2 CNN (frozen), DiT-S, cosine LR, 40k steps (`p128-ref/ref-S-v_cbr48_qam16_blind-5to5_cos_s0`). Analysis spec: `scripts/gen_analysis/specs/w1-16qam.json` |
| 17 Nov | W1 A/B | g-a1..3, g-b1r | All four pass the start-up checks. The regression loss gives the transmitter about 80x the gradient (overfit check encoder_grad 0.33 vs 0.004 for g-a1), the direct signal the critique said was missing. The refiner trains at 1.3 it/s (VAE encoding of the CNN output every step) against 7.6 for the DiT: about 8.4 h for 40k steps. A lighter learned encoder would cut this if the refiner is kept |
| 17 Nov | W1-R2/R3/R1, 16-QAM | g-a1, g-a2, g-a3 | **Option A fails the rule; none of the three closes the gap.** Paired, spread set, 10 steps. Cosine decay (g-a1) is worse than the constant-LR baseline (LPIPS +0.01, top-1 -3 pts at 0/5 dB): the baseline's val loss still falls at 40k (0.327 -> 0.325) while the cosine run flattens at 0.333, so the DiT is undertrained and decay cuts learning short. Against g-a1: the regression loss (g-a2) adds PSNR (+0.21 / +0.13 / +0.05 dB at -5/0/5) but not semantics (top-1 -2.5 pts at 5 dB); REPA (g-a3) adds semantics (top-1 +1.6 / +2.7 / +3.7 pts, DINOv2 0.174 vs 0.137 at 0 dB) at -0.17 to -0.23 dB PSNR. The gap to the blind 16-QAM CNN at 0 dB is DINOv2 0.13 and top-1 12 pts; REPA's gain is about a third of the DINOv2 gap at best. Next for A, if any: REPA with constant LR and longer training |
| 17 Nov | policy | - | **Train to convergence wherever a comparison depends on it** (user, 17 Nov). Before this, runs stopped at a 40k-step budget with the best-val checkpoint: fair at equal budget but not converged (16-QAM DiT val loss -0.9%, analog pixel CNN -2.1% over the last quarter). New: `--patience 10` stops after 10 validations (20k steps) without a 0.2% val-loss gain, `--steps` is only a cap, constant LR. The long CNN and refiner runs go to the RTX 5060 (`scripts/gen_5060.py`, output `results/gen/p128-conv/`); DiT runs that fit one session go to Kaggle |
| 17 Nov | convergence | g-v1 (venkat96r), g-v2 (subhashhhh) | Launched at 8eb9570: blind 16-QAM DiT-S to convergence, constant LR, cap 150k, patience 10 (`p128-conv/dit-S-v_cbr48_qam16_blind-5to5_s0`); g-v2 the same with REPA 0.5 (`..._repa0.5_s0`). RTX 5060 queue (`scripts/gen_5060.py`): converged 16-QAM CNN (pixel, then LPIPS), refiner on it, converged analog CNN |
| 17 Nov | analysis | g-d6 (own) | **First image-level report**, `results/gen/analysis/w1-16qam/report.md` (dump: `results/gen/dump/w1-16qam`, not committed). Class kept at 0 dB: CNN 78%, DiT 59%, A1 52%, A2 47%, A3 61%; churches 62% vs 26%. What the two receivers do: the CNN keeps 20% of the originals' high-frequency power (blur plus its checkerboard texture) but keeps the object; the DiT keeps about 100% (natural sharpness) but invents it: two decodes of the same received symbols differ by 19-21 dB PSNR, about as much as a decode differs from the original (17-19 dB), and faces, hands and fine structure change from draw to draw while layout and colour stay. Even at +10 dB the spread is 11 grey levels. So the channel carries layout and colour; the DiT makes up the rest, which is why its semantics lag |
| 17 Nov | convergence | g-v1, g-v2 | **The DiT-CNN gap survives convergence.** Blind 16-QAM DiT trained to convergence: stopped at 62k, best val at 48k (0.3255 vs 0.3248 at 40k), then rises (overfits on 9.5k images). Paired against the 40k run: LPIPS -0.007 / -0.004 / -0.003 at -5/0/5 dB, top-1 +1.6 / -0.3 / +3.3 pts; DINOv2 0.178 vs 0.163 at 0 dB. With REPA: best at 34k (converges sooner, then overfits), DINOv2 0.201 and top-1 12% at 0 dB, PSNR -0.22 dB. The (not yet converged) blind 16-QAM CNN has DINOv2 0.292 and top-1 22% at 0 dB: converging closes about a tenth (DiT) to a quarter (REPA) of the semantic gap. More steps on Imagenette cannot close it; the limit is data and what the receiver invents |
| 17 Nov | RTX 5060 | gpu-box | Reachable over Tailscale (`ssh hari@gpu-box`, no password): Ubuntu 24.04 under WSL2, RTX 5060 8 GB (sm_120), 7 GB RAM, 948 GB disk; venv `~/ml` (torch 2.14.1+cu130), repo at `~/semcom`. `nvidia-smi` hangs under WSL; use torch for GPU checks. The convergence queue (`scripts/gen_5060.py`, all stages) runs in tmux session `gen`, log `~/gen_5060.log` |
| 18 Nov | W1 option B | g-b1r | **Refiner as trained (sampling from pure noise) is worse than its own CNN input**: 0 dB PSNR 17.9 vs 19.5 dB, DINOv2 0.12 vs 0.29, FID 173 vs 118; val PSNR@0 rose only 16.5 -> 16.9 dB over 40k steps, cond_gain flat at 0.22 from 2k. Not a data bug: its frozen CNN is identical to the g-cb2 weights (19.2 dB on a CPU check) and the VAE latent it conditions on decodes to 19.1 dB. A fresh sample from this DiT's prior (9.5k images) redraws the content with the DiT's own artefacts. Next: SDEdit-style decoding from the CNN's latent noised to a fraction `start` of the schedule (`--eval-start`), a dial from the CNN (0) to a full sample (1); evaluation only |
| 18 Nov | W1 option B | g-s3 | **Option B as built (a refiner in the SD-VAE latent at 128 px) cannot pass the rule.** SDEdit start 0 / 0.2 / 0.4 / 0.6 / 0.8, 0 dB: DINOv2 0.228 / 0.134 / 0.117 / 0.116 / 0.120 (CNN 0.292), top-1 13 / 5 / 4 / 5 / 5% (CNN 22%), FID 119 / 157 / 167 / 171 / 172 (CNN 118). Start 0 is only the VAE round trip of the CNN's output; paired against the CNN it gains LPIPS (-0.029 at 0 dB, -0.031 at +5) but loses top-1 (-8.4 pts at 0 dB) and DINOv2 (0.29 -> 0.23): **the SD-VAE at 128 px is itself a semantic bottleneck** (also on clean images: VAE ceiling DINOv2 0.81). Any noise above that (start 0.2 is already alpha_bar ~0.66) hands the image to this DiT's weak prior and semantics collapse. Implications: a latent-diffusion receiver at 128 px cannot beat the CNN on semantics; the options are 256 px (VAE at its native resolution), a pixel-space generator, or Gate 1's fallback (the SNR-placement study) |
| 18 Nov | seeds | g-v3 (own), g-v4 (subhashhhh), g-cb3 (venkat96r) | Launched at 9e91d29, seed 1 replications of the results the decisions rest on: 16-QAM DiT to convergence (g-v1), with REPA (g-v2), and the two-stage blind 16-QAM perceptual CNN (g-cb2). The RTX 5060 queue now skips the refiner (stages cnn16, cnn16-ft, cnn, cnn-ft) |
| 18 Nov | 256 px check | CPU (`scripts/gen_analysis/vae_resolution.py`) | **The SD-VAE bottleneck is mostly a 128 px problem.** 200 class-spread test images: clean round trip DINOv2 0.81 / PSNR 23.6 dB at 128 px vs 0.96 / 25.9 dB at 256 px. On the blind 16-QAM CNN's 0 dB outputs (upsampled for 256 px), the round trip costs DINOv2 -0.060 at 128 px but -0.017 at 256 px; class kept falls 8 / 7 pts at both (80 -> 72%, 79 -> 72%), so the class-level loss does not shrink. An 8-image first look had suggested equal losses at both sizes: too few images. Reading: 256 px removes most of the VAE's semantic cost, but not the weak-prior problem (data); a 256 px generative receiver is worth one converged test, not yet a commitment |
| 18 Nov | Gate 1 | (decision) | Revised plan agreed (section 6): stop developing the DiT at 128 px; track A, the SNR-placement study, on Kaggle from the 21 Nov reset; track B, one converged 256 px test, on the RTX 5060; seed replications continue |
| 18 Nov | 256 px test | RTX 5060 (`gen_5060.py dit256 cnn256 cnn256-ft`, then the 128 px stages) | Launched at 3091c72. The 128 px 16-QAM CNN (`p128-conv/cnn_cbr48_qam16_blind-5to5_s0`) paused at step 198k, still improving (best.pt updated at 198k; the Kaggle reference stopped at about 39k by its time limit). 256 px cache: 11.8 GB; the VAE pass now runs in sub-batches (it ran out of memory on the 8 GB GPU) |
| 18 Nov | A (code) | none | For the SNR-placement study: the CNN takes an SNR-aware transmitter (`--encoder-snr`, the encoder's AF modules); evaluation can feed the receiver an estimated SNR, a biased one or a gain error (`--eval-rx est-<cue> / offset<dB> / gain<dB>`; a blind receiver sees only the gain error, tested) |
