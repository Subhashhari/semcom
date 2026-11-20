# Generative JSCC: review of 16 Nov 2026 and the plan to 13 Dec 2026

This document records a critical review of the DiT-JSCC prototype (`semcom/gen`, results in
`results/gen`), the position the work can defend, and the four-week experiment plan that follows
from it. It is written so that a later task can pick up any experiment without the conversation
that produced it. The earlier design and status notes are in
[gen-jscc-prototype.md](gen-jscc-prototype.md); where the two disagree, this document is newer.

Every number below was recomputed on 16 Nov 2026 from the committed per-image result files
(`eval_images-*.npz`) unless marked otherwise. Literature statements come from paper text fetched
that day; section 3 says which ones could not be confirmed verbatim.

## Status on 20 Nov (read first)

Sections 1 to 5 are the review of 16 Nov and still describe the 128 px prototype. Since then:

- **128 px is closed.** Every lever was tried there:
  - REPA was kept;
  - pretraining on 59k ImageNet-100 images, then fine-tuning, was kept (the largest gain);
  - cosine decay, the regression loss, DiT-B, more steps and the bandwidth split were not kept.
  - The deciding pair (the DiT and the CNN) and REPA were trained to convergence and replicated
    with a second seed.
  - The blind 16-QAM perceptual CNN still beats the DiT on PSNR, DINOv2 and retrieval at every
    SNR. At 0 dB: DINOv2 0.29 vs 0.18, top-1 22% vs 10%, FID 118 vs 141.
  - The SD-VAE at 128 px is itself the semantic bottleneck (clean round trip DINOv2 0.81). A
    refiner on the CNN's output cannot get past it either (g-s3).
- **The SNR results hold at 128 px** for analog, 16-QAM and Rayleigh, in two seeds. Telling the
  receiver the SNR gains at most 0.04 dB. An SNR-aware transmitter gains 0.4 to 0.5 dB at -5 dB, on
  PSNR only.
- **At 256 px the DiT changed a lot.** It is converged (140k steps on the RTX 5060). Spread set at
  0 dB:
  - DINOv2 0.56, top-1 81%, FID 70, class kept 93%;
  - at +5 dB, top-1 95%.
- **B passed on 20 Nov** (`results/gen/analysis/p256/report.md`, local). Against the blind
  perceptual CNN (40k pixel + 10k LPIPS steps), paired on the 1,000 spread images, the DiT wins
  at -5, 0 and +5 dB on LPIPS (-0.08 to -0.09), DINOv2 (+0.03 at 0 and +5 dB, +0.09 at -5 dB),
  top-1 (+9.4 and +5.6 pts at 0 and +5 dB) and FID (70 vs 82, 51 vs 56), all with intervals
  clear of 0; it loses 1.2 to 1.5 dB PSNR. The verdict is not final: the pre-registered converged
  CNN re-check (`cnn256-conv`, on the 5060 since 20 Nov 08:00) decides it, about 23 Nov.
- **Study A, first result (AWGN, 256 px, 20 Nov): telling the receiver the SNR adds nothing.**
  Blind and told DiTs (0 to 20 dB, 16-QAM) both converged at 76k steps on Kaggle. Told minus
  blind, paired on the 1,000 spread images: PSNR -0.13 / -0.11 / -0.06 / -0.04 / -0.02 dB at
  0 / 5 / 10 / 15 / 20 dB (intervals below 0 except at 20 dB), LPIPS +0.001 to +0.002, top-1
  and top-5 within their intervals of 0. The told receiver is never better, so a gain from the
  SNR is ruled out at every SNR (largest upper bound +0.009 dB). Whether told is genuinely
  slightly worse is a seed question and does not affect the claim.
- **But the blind receiver depends on a correct receive gain (20 Nov, `a256-ae-awgn`).** With a
  2 dB amplitude error, the blind receiver loses 0.43 to 0.95 dB PSNR and 3 to 11 top-1 points,
  and the told receiver (given the true SNR) now beats it at every SNR: +0.21 to +0.55 dB,
  LPIPS -0.03 to -0.04, top-1 +2 to +10 pts, all clear of 0. So "needs no SNR" holds only with
  an accurate AGC/equaliser; `gain1` (after the reset) sizes the tolerance. A told receiver that
  estimates the SNR from the received energy matches the true-SNR one within 0.02 dB up to 15 dB
  (-0.09 dB at 20 dB); the M2M4 estimator costs 0.2 to 0.25 dB from 5 dB up. Still to come:
  `gain1`, Rayleigh, the specialists and the SNR-aware transmitter (from the 21 Nov reset).
- **The position is unchanged, now at 256 px:** a constellation-constrained (16-QAM) diffusion JSCC
  that needs no SNR at either end.
  - The work rests on B and on study A at 256 px (blind / told / estimated SNR / gain error /
    specialists, AWGN and Rayleigh, 0 to 20 dB).
  - Everything else is optional. Section 6, "Plan from 19 Nov evening", lists each remaining task
    with a decision.

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
| 3 | Telling the receiver the SNR adds nothing | Yes, 3 settings at 128 px; at 256 px on AWGN, 0 to 20 dB (20 Nov: told never better, -0.02 to -0.13 dB), **provided the receive gain is right**: with a 2 dB gain error told beats blind by 0.2 to 0.55 dB | Weak: SIJSCC shows it for a non-generative codec; OSD-JSCC has a partial ablation | Small |
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
| Kaggle T4, 4 accounts (own, subhashhhh, venkat96r, venkatrrr from 19 Nov) | 120 h a week in total (30 h each, reset Saturdays 00:00 UTC = 05:30 IST), 12 h per session, jobs in parallel | Short and medium runs: recipe tests, controls, seeds, the final arms |
| RTX 5060 (separate machine) | Unmetered, one run at a time; 8.5 GB, of which about 7.3 GB is free with the display on the iGPU | Long single runs: long training, 256 px |

- The development PC has CPU-only PyTorch and about 10 GB of free disk. Do not run the full test
  suite there: run targeted tests with `--basetemp` in a scratch folder and delete it.
- The RTX 5060 needs a PyTorch build for CUDA 12.8 or newer. ImageNet-100 is not on it yet; at
  256 px its cache would need about 55 GB (estimate from the Imagenette cache, 11.8 GB).
- **Kaggle at 256 px (19 Nov):**
  - The 256 px Imagenette cache (11.8 GB) is rebuilt in every session inside `/kaggle/working`,
    which has 19.5 GB free.
  - Setup counts against the 12 h session but not against `--max-hours`, which starts once
    training does. So `--max-hours 10.5` leaves about 1.5 h for setup.
  - A run longer than one session is chained: a second job with `--input-job` merges the first
    job's `results/` and resumes from its checkpoint.pt. Both jobs must be on the same account,
    and the first must end cleanly; a killed session cannot be attached.
  - At most two GPU sessions run at once per account.
  - Measured by `a256-smoke` (19 Nov):
    - Setup is 35 min, 30 of them the cache.
    - DiT-S trains at 2.2 to 2.4 it/s; validation every 2k steps adds about 9%.
    - The spread test at 5 SNRs takes about 25 min.
    - So a converged run of about 140k steps is about 83k steps in session 1, then the rest
      plus the test in session 2: about 19 h on one account.
    - REPA peaks at 7.05 GB (6.76 GB without it). On the 5060 (about 7.3 GB free) that is
      tight, so reboot Windows before `dit256-repa`.
- Measured T4 costs: DiT-S 40k steps 1.5 h, DiT-B 40k steps 3.8 h, DiT-S 100k steps on 59k images
  3.7 h, evaluation about 0.3 h. Blind 16-QAM DiT-S to convergence (patience 10): 2.1 to 2.5 h
  (stops at 52k to 62k). 128 px CNN: 3.9 to 4.5 h per 40k pixel steps, 1.8 to 2.2 h per 10k LPIPS
  steps, so its ~200k-step convergence (below) would take about 22 h, two sessions.
- Measured RTX 5060 speeds (19 Nov): 128 px CNN 3.6 it/s (1.4x the T4); 256 px DiT-S 3.1 to 3.4
  it/s (140k steps plus tests: 14.7 h); 256 px CNN 0.97 it/s for pixel loss and 0.59 it/s with
  LPIPS. The 256 px CNN needs about 12 GB at batch 64 (24 GB with LPIPS), so it runs each batch in
  chunks (`--grad-accum 2`, 8 with LPIPS; the gradient matches the whole batch's to 0.03%). The
  screen must stay on the iGPU: a desktop on the 5060 pushes training into system memory (10x
  slower).
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
  runs fit in one Kaggle week. (Given up on 19 Nov: a converged 256 px DiT-S costs about 17 to 19 h
  on a T4, so study A is spread over the weeks of 21 and 28 Nov.)

## 5. Levers for a better model, ranked by expected gain per GPU hour

| # | Lever | Cost | Why | Code |
|---|---|---|---|---|
| 1 | Two ceiling diagnostics | 4 h | Shows whether the transmitter or the generator is the limit | Small |
| 2 | REPA alignment loss | 2 h per run | Aligns the DiT's hidden states with DINOv2 patch features. The [paper](https://arxiv.org/html/2410.06940) reports FID 18.8 to 9.7 for SiT-L/2 at 400k steps with DINOv2-B, depth 8, weight 0.5. The cache already holds DINOv2-B features on the 8x8 token grid. Untested on small data or conditional decoding | New loss |
| 3 | Training recipe | 2 h per variant | Timestep weighting, learning-rate decay, and a direct regression loss so the transmitter gets a stronger signal | New options |
| 4 | Pretrain broad, then fine-tune on Imagenette | 1.5 h first test | The 59k-image model was still improving at 100k steps; `--init-from` exists | None |
| 5 | 256 px | About 6 h per run (estimate) | 4x the symbols; VAE at its native resolution; matches the literature | Check config |
| 6 | Longer training, DiT-B | 10 to 15 h | Only after 2 to 4: both overfitted on 9.5k images | None |

**Outcome at 128 px (19 Nov).** By lever number from the table above:

1. **Diagnostics:** the noise-free DiT caps at DINOv2 0.39 (W1-D1), so the limit is the
   transmitter and the generator, not the noise.
2. **REPA:** kept. FID -7 to -14 and top-1 +2 to +3 pts, at -0.2 dB PSNR and +0.01 LPIPS, in two
   seeds.
3. **Recipe:** not kept. Cosine decay hurt the undertrained 40k-step runs, and the regression loss
   adds PSNR only.
4. **Pretrain, then fine-tune:** kept, the largest gain. Top-1 +5 / +14 pts and FID -12 / -21 at
   0 / +5 dB.
5. **256 px:** run on the RTX 5060. At 0 dB, DINOv2 went from 0.18 to 0.56 and top-1 from 10% to
   81%, the largest change seen.
6. **Longer training and DiT-B:** not kept; both overfit on 9.5k images.

None of them closed the 128 px gap to the CNN. At 256 px only REPA and pretraining remain to try,
and both are optional (section 6).

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
  Superseded on 19 Nov by the plan below: three of those four stages are dropped.

### Plan from 19 Nov evening: what is still worth doing

Decided 19 Nov evening after a review of every result. The user ruled out further 128 px models
and asked that DiT runs start without waiting for CNN results. This plan supersedes the two
subsections below wherever they differ.

**What the results say, in short** (details in "Status on 19 Nov" at the top and in section 8):

1. **128 px is closed for both receivers.** The DiT loses to the CNN on every semantic measure, in
   two seeds each. The SD-VAE at 128 px is itself the bottleneck: its clean round trip reaches only
   DINOv2 0.81. No 128 px run can change a decision any more.
2. **256 px is the one change that moved the DiT a lot.** On the spread set at 0 dB, DINOv2 went
   from 0.18 to 0.56, top-1 from 10% to 81% and FID from 141 to 70. Resolution also changes the
   footing of these metrics, so this is not yet a win over the CNN.
3. **Whether the DiT beats the CNN at 256 px is open until Friday.**
   - The CNN's pixel stage reaches validation DINOv2 0.33 at 0 dB after 20k steps, against the
     DiT's 0.44 at the same step and 0.66 at its end.
   - But at 128 px, the 10k-step LPIPS stage tripled the CNN's validation DINOv2 (0.13 to 0.44 at
     0 dB) and lifted it past the DiT.
   - So validation DINOv2 cannot be compared across loss stages, and the pixel stage predicts
     nothing about the final CNN.
4. **The paper rests on two experiments.** The first is B: the DiT against the CNN at 256 px, with
   the pre-registered converged re-check. The second is A at 256 px: where the SNR must be known,
   under 16-QAM. Model levers are optional.

**The recipe for A is frozen.** A's DiT arms use the recipe that B tests (`configs/gen/p256.yaml`):
DiT-S with v-prediction, constant LR 1e-4, patience 10, a cap of 150k steps, Imagenette only and
no REPA.
- Levers found later are reported as add-ons on the blind model. A is not retrained with them.
- Blind against told is unlikely to depend on the recipe, and retraining A would cost about 190 T4
  hours.

**Each remaining task:**

| Task | Serves | Decision | Why |
|---|---|---|---|
| `cnn256-ft`, `dump256`, the analysis suite | B | **Done 20 Nov: B passes** | The pre-registered verdict |
| Converged 256 px CNN (`cnn256-conv`, `cnn256-conv-ft`, `dump256-conv`, spec `p256-conv.json`) | B re-check | **Running** on the 5060 since 20 Nov 08:00 | Pre-registered on 18 Nov. Converged by the DiT's own rule: constant LR, patience 10, the pixel stage continued from its 40k checkpoint, then the LPIPS stage. About 2 to 3 days (estimate) |
| A, AWGN blind / told (`a256-ab`, `a256-at`) | A | **Done 20 Nov**: both converged at 76k; told adds nothing (section 8) | The core comparison |
| A, Rayleigh blind / told | A | Do, with block fading (20 Nov change, `fading: block`), after the smoke test `r256-smoke` | Fading is where an SNR input is most likely to matter. One fade per image for every symbol, and the told arm is given the image's effective SNR after equalisation (SNR + 10 log10 \|h\|^2), which a coherent receiver knows. Before 20 Nov the DiT's two streams faded independently and the told arm got only the average SNR, a weak comparator |
| A, specialists at 0 and 20 dB | Claim 2 | Do | DiT-JSCC trains one model per SNR. The two ends of the range are where a single model is most likely to fall short |
| A, SNR-aware transmitter | Claim 4 (a contrast) | Do, after the rows above | Shows what SNR feedback buys. At 128 px it gained PSNR only |
| A-E evaluations: `est-energy`, `est-m2m4`, offsets of 2 and 5 dB, gain errors of 1 and 2 dB | A, the gain assumption | AWGN: **done 20 Nov** except `gain1` (`a256-ae-awgn2` after the reset); the blind receiver is fragile to a 2 dB gain error (section 8). Rayleigh: once its models finish, on the accounts that hold them | Cheap. These evaluations define what "needs no SNR" assumes |
| `dit256-s1`, seed 1 of the 256 px DiT | B and the levers | Do; after the converged CNN if B passes | Gives the seed spread at 256 px, needed to judge any 256 px lever |
| REPA at 256 px (`dit256-repa`) | Lever | Optional, after `dit256-s1` | At 128 px: FID -7 / -14, top-1 +2 to +3 pts, PSNR -0.2 dB |
| ImageNet-100 pretraining, then fine-tuning, at 256 px | Lever | Optional, 5060 only | The largest lever at 128 px (top-1 +5 / +14 pts, FID -12 / -21). Needs the data downloaded to the 5060 and a cache of about 55 GB (estimate); too large for Kaggle's 20 GB working disk |
| A specialist at 10 dB; seed 1 of A's blind and told | A | Only if needed | At 128 px blind and told differed by at most 0.04 dB, and seed noise in the mean is about 0.01 dB. Run only if the 256 px gap is borderline |
| 200k-step extension of `dit256` | B | Only on a narrow fail (the rule of 19 Nov below) | Post hoc |
| `cnn16` to its end and C1 (128 px) | A CNN recipe | **Dropped** | 128 px is closed, and the 256 px CNN is converged by the DiT's rule, so no cheaper recipe needs calibrating |
| A's CNN arms (A-C, A-S, A-P) at 128 px | A | **Dropped** | A 128 px CNN against a 256 px DiT confounds resolution, and blind CNN JSCC is published (CBJSCC, SIJSCC). If 5060 time is left over: a 256 px CNN blind / told pair at the 40k-step budget (about 11 h each) |
| A's DiT arms at 128 px (`p128-a`) | A | **Dropped** | The 128 px DiT is closed |
| 4-QAM / 64-QAM (W4-A4), the week 2 to 4 tables | - | **Dropped** / superseded | No decision needs them |

**RTX 5060 order.** B passed (20 Nov), so the first branch below is the one in force: the
converged CNN runs from 20 Nov 08:00 IST (`~/gen_256_conv.log`), then the re-check, then
`dit256-s1` and `dit256-repa`.
- After `dump256` (Fri about 06:00), `~/after_dump.sh` starts `dit256-s1` and then
  `dit256-repa`.
- When the verdict is in (about 07:00):
  - **B passes:** stop `dit256-s1`, which later resumes from its checkpoint. Run
    `gen_5060.py cnn256-conv cnn256-conv-ft dump256-conv`, then the re-check, then
    `dit256-s1`, `dit256-repa` and the ImageNet-100 lever.
  - **B fails:** keep `dit256-s1`, then try the two kept levers as possible rescues (REPA, then
    ImageNet-100). The converged CNN is not needed.

**If B fails,** A's DiT arms still run (user, 19 Nov), because they answer A for a diffusion
receiver either way. The paper then becomes an SNR-placement study of a constellation-constrained
diffusion receiver, with no quality claim against the CNN.

**Kaggle.**
- **Sessions:** a 256 px run takes two sessions on one account. The first uses `--max-hours 10.5`;
  a second job resumes it with `--input-job` and runs the test set. An account runs at most two
  sessions at once.
- **Cost:** about 19 h per run (measured T4 speed, 19 Nov).
- **Leftover hours of the week of 14 Nov (added 20 Nov, 02:45):** about 15 h would have expired at the reset, so
  each run due to start on 21 Nov got a short first session now, which next week's sessions resume:
  `a256-rb-0` (own, `--max-hours 2.0`), `a256-rt-0` (subhashhhh, 1.1) and `a256-p0-0` (venkat96r, 1.6),
  each sized to the account's remaining quota less about 1 h for setup and upload. On venkatrrr,
  `a256-ab-2` starts when `a256-ab` stops (about 06:30 on 20 Nov) with `--max-hours 6`, and `a256-ab-3`
  finishes it after the reset. A session costs about 0.6 h of setup, so each early session saves its
  training hours from next week, as long as the run still needs two more sessions.
- **Week of 21 Nov:**

  | Account | Runs |
  |---|---|
  | venkatrrr | `a256-at-3` if `a256-at-2` (20 Nov) does not finish, then spec20 (`a256-p20`, `a256-p20-2`) |
  | own | Rayleigh blind (`a256-rb`, resuming `a256-rb-0`; `a256-rb-2` if needed), then the blind Rayleigh gain-error evaluations |
  | subhashhhh | Rayleigh told (`a256-rt`, `a256-rt-2` if needed), then the Rayleigh A-E evaluations of the told model |
  | venkat96r | spec0 (`a256-p0`, `a256-p0-2` if needed) and the first session of the SNR-aware transmitter (`a256-tx`) side by side |
  | venkatrrr (also) | `a256-ae-awgn2`: `gain1` for told and blind |

  A Kaggle job can only read the outputs of jobs on its own account, so each evaluation runs on
  the account that trained the model. The AWGN runs converged at 76k steps (about 9.7 h), so a
  run that starts from a short first session may finish in one more session; the queue then
  skips its `-2` job.

- **28 Nov week:** the second session of the SNR-aware transmitter (`a256-tx-2`), anything the
  21 Nov week did not finish, the Rayleigh A-E
  evaluations, and the conditional runs.
- **Budget:** the "Do" rows cost about 6 x 19 + 19 + 9 = about 140 h, against about 270 h
  available: venkatrrr's 30 h this week plus 2 x 120 h.

**Milestones.**

- **Fri 20 Nov, about 07:00:** the B verdict, by the rules of 19 Nov morning below. **Done 08:00:
  B passes; AWGN blind / told also done the same morning.**
- **Sat 21 Nov, 05:30:** Kaggle reset; launch as in the table.
- **Tue 24 Nov, supervisor meeting:**
  - the B verdict and its report;
  - the converged re-check, running (estimate now about 23 Nov);
  - the AWGN blind-against-told result with the estimated-SNR and gain-error evaluations, and
    the first Rayleigh runs;
  - this plan.
- **Fri 27 Nov, Gate A (DiT):** blind against told on AWGN and Rayleigh across 0 to 20 dB, with the
  estimated-SNR and gain-error arms, plus the specialists if they are done. The decision rule is
  Gate 3's (below).
- **From 30 Nov:** writing, W4-T1 (deployment table) and W4-L1 (literature re-check), and the
  optional levers as the 5060 frees up.

### Plan for 20 to 29 Nov (decided 19 Nov morning; partly superseded the same evening, above)

Still in force:

- the evidence list;
- Friday steps 1 (the verdict, with the seed and extension rules), 3 (A-smoke, run on 19 Nov) and 4.

Superseded by the plan above:

- step 2 (`cnn16` and C1 are dropped);
- the 128 px track A table (A now runs at 256 px);
- the track B branches;
- the milestones.

**Evidence it rests on** (19 Nov; section 8 has the full rows):

- *Budget CNNs are undertrained on semantics.* The 128 px blind 16-QAM pixel CNN on the 5060
  (constant LR, 9 of 10 patience validations used at 198k): from 40k to 198k steps, validation
  loss 0.0125 -> 0.0117 (-6%), PSNR +0.32 / +0.39 / +0.38 dB, DINOv2 0.04 / 0.14 / 0.21 ->
  0.10 / 0.20 / 0.30 at -5 / 0 / +5 dB (32 validation images). PSNR is nearly flat after 60k, but
  DINOv2 keeps rising. Every claim against a 40k-step CNN must say so; a converged CNN needs about
  200k constant-LR steps (22 h on a T4) unless a decay schedule gets there sooner (C1 below).
- *The seed spread is small.* Seed 0 vs 1 at 128 px (g-v3, g-cb3): DINOv2 within 0.013, top-1
  within 1 pt, FID within 3, LPIPS within 0.004. Gaps of 3x that (0.04, 3 pts, 9, 0.012) are
  treated as settled by one seed.
- *The 256 px DiT converged by the rule but was still gaining semantics:* validation DINOv2 at
  0 dB rose +0.009 per 10k steps over 100k to 140k (PSNR flat, loss -0.0006 per 10k).
- *Quota:* 90 Kaggle hours a week on three accounts; 10.4 h left until the 21 Nov reset (own 4.7,
  subhashhhh 3.1, venkat96r 2.6). A fourth account (venkatrrr) joined the same evening: 120 h a week.

**Dropped, with reasons:**

| Item | Why | Saves |
|---|---|---|
| 5060 stages `cnn16-ft`, `cnn`, `cnn-ft` (converged 128 px perceptual and analog CNNs) | The 128 px DiT is closed (Gate 1); its gap to the budget CNN is already 8x the seed spread (DINOv2 0.11, top-1 12 pts at 0 dB) and converging the CNN only widens it; A uses neither the analog CNN nor the LPIPS stage | About 1.5 days of 5060 |
| `ref16` and any 128 px refiner | g-s3: the SD-VAE at 128 px is itself the semantic bottleneck | (already off) |
| ImageNet-100 pretraining of the 128 px DiT on the 5060, and the week 2 table (W2-K1 to K4) | 128 px DiT closed; the 59k-image data hurt at -5 and 0 dB (g-x2). Revisited only at 256 px, if B passes | Superseded |
| LPIPS stage in A's CNN arms | A asks where SNR knowledge is needed; an MSE-trained CNN is the standard JSCC control (ADJSCC, DeepJSCC-Q), and the DiT arms already cover perceptual quality | About 2 h per CNN arm |
| Separate week 3 (W3-B1 to B5) | Merged into A below | - |

**Kept:** `cnn16` (finishes at its next validation, then its spread test: the converged reference
for C1) and the 256 px CNN dump.

#### Friday 20 Nov

1. About 06:30 IST the 256 px CNN, its fine-tune and the dump finish. Copy back, run
   `scripts/gen_analysis/suite.py scripts/gen_analysis/specs/p256.json`, read the verdict.
   - *Seed rule:* if every margin that decides the verdict is at least 3x the seed spread above,
     one seed stands for the 24 Nov meeting; otherwise seed 1 of both 256 px systems goes first on
     the 5060 (about 30 h).
   - *Extension rule (post hoc, reported as such):* if B fails only through a DINOv2 or top-1
     loss, and the DINOv2 deficit is at most 0.04 at 0 and +5 dB, resume the DiT at constant LR to
     200k steps (about 6 h) and re-score; the pre-registered verdict stays the headline.
2. The 5060 then runs `cnn16` to its end and **C1, recipe calibration**: the same CNN with a cosine
   decay over 60k steps (`gen_5060.py cnn16-cos`, about 5 h; `~/after_dump.sh` starts both). C1
   passes if its spread test lands within the seed spread of the converged `cnn16` (PSNR 0.1 dB,
   DINOv2 0.013, top-1 1 pt). Pass: every CNN that must be converged uses cosine 60k from here on.
   Fail: try 100k (about 8 h on the 5060), else constant LR with patience on the 5060 only.
   Ready about 13:00 IST.
3. Kaggle, own account (4.7 h left): **A-smoke**, every A configuration below for 300 steps with a
   50-image test, through rayleigh x 16-QAM x told / SNR-aware transmitter and the `--eval-rx`
   est / gain / offset evaluations. Local `--dry-run` passes for all 15 configurations (19 Nov).
4. Optional, on the 5060 between jobs: a Windows reboot frees the 2.6 GB the display process held
   on the 5060 before the screen moved to the iGPU.

**Track A, the SNR-placement study** (16-QAM, 128 px Imagenette, training range 0 to 20 dB,
tests at 0 / 5 / 10 / 15 / 20 dB on the spread set, `--out-dir results/gen/p128-a`):

| ID | Arms | Runs | Where | Cost |
|---|---|---|---|---|
| A-D | DiT-S to convergence (patience 10): AWGN blind / told / SNR-aware transmitter; Rayleigh blind / told | 5 | Kaggle from the 21 Nov reset | 2.8 h each, 14 h |
| A-C | CNN, MSE, C1's recipe: AWGN and Rayleigh, each blind / told / SNR-aware transmitter | 6 | See branches | 5 h (5060) or 6.8 h (T4) each |
| A-S | CNN seed 1 of AWGN blind and told | 2 | As A-C | as A-C |
| A-P | CNN specialists at 0 / 10 / 20 dB, AWGN (the cost of one model across the range) | 3 | As A-C | as A-C |
| A-E | Evaluation only, on the blind and told models of A-D and A-C: told with an estimated SNR (`--eval-rx est-energy` and `est-m2m4`), offset +-2 / 5 dB, gain error +-1 / 2 dB | about 20 | Same device as the model | 0.3 h each |

All runs of one receiver family stay on one device type (section 7), so A-C, A-S, A-P and their
A-E evaluations go together on either the 5060 or the T4.

**Track B, by Friday's verdict:**

- **B fails** (the DiT loses to the 40k-step CNN): the DiT at 256 px stops; it stays in A as the
  second receiver at 128 px. The 5060 takes A-C, A-S, A-P (11 runs, about 2.3 days, from Saturday
  midday). Kaggle runs A-D, then DiT seed 1 of AWGN blind and told (5.6 h) and DiT specialists at
  0 / 10 / 20 dB (8.4 h): about 31 h of 90; the rest stays unused rather than spent on runs no
  decision needs.
- **B passes**: as pre-registered, the 5060 converges the 256 px CNN with C1's recipe (cosine 60k
  pixel, about 17 h, then the LPIPS stage to convergence, about 5 to 10 h) and the rule is
  re-checked, then seed 1 if the seed rule asks, then 256 px DiT + REPA (kept at 128 px, g-v3/g-v4)
  and the 59k-image data (week of 30 Nov). A-C and A-S move to Kaggle: 8 runs x 6.8 h = 54 h, plus
  A-D 14 h and A-E about 4 h = 72 h of 90; A-P waits for the 28 Nov reset.

#### Milestones

- **Tue 24 Nov, supervisor meeting:** the 256 px verdict with the full analysis report, the
  convergence evidence above, C1, A's design and A-D's first results.
- **Gate A (Fri 27 Nov):** for both receivers, blind against told with paired intervals on AWGN
  and Rayleigh across 0 to 20 dB; the estimated-SNR and gain-error arms. Decision as Gate 3 below:
  blind within about 0.1 dB of told keeps "needs no SNR"; if a 1 dB gain error costs more than
  SNR knowledge ever gains, the claim becomes "no noise-level knowledge, gain tracking assumed".
- **28 Nov reset:** A-P if not yet run, missing seeds, then W4-A4 (4-QAM / 64-QAM, optional) and
  writing.

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

### Week 3 (30 Nov to 6 Dec): is "needs no SNR" true? (merged into track A on 19 Nov)

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
- Validation metrics (32 images, logged every 2k steps) are for tracking one run.
  - Do not compare them across loss stages or architectures. The LPIPS stage tripled the 128 px
    CNN's validation DINOv2.
  - Decide only on the test sets.
- `results/gen/kaggle_jobs.json` is the committed copy of `.kaggle_jobs/*/job.json`, without token
  paths. Add new jobs to it when their results are logged.
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
| 18 Nov | seeds | g-v3, g-v4 | **Both DiT results replicate.** Seed 1 of the converged blind 16-QAM DiT matches seed 0 (spread set, 0 dB: DINOv2 0.179 vs 0.178, top-1 10.5% vs 9.7%, FID 141.5 vs 140.6, LPIPS 0.290 vs 0.291). REPA on seed 1, paired against seed 1 without it: top-1 +3.0 / +2.1 / +1.3 pts at -5 / 0 / +5 dB (the last not significant), top-5 +8.9 / +7.4 / +6.8 (all significant), FID 127.9 vs 141.5 at 0 dB and 113.9 vs 120.8 at +5 dB; LPIPS +0.010 / +0.014 and PSNR -0.2 / -0.26 dB worse at 0 / +5 dB. Seed 0 showed the same pattern (FID -7.6 / -5.6). REPA passes the 17 Nov rule on both seeds through FID, with no clear loss on DINOv2 or top-1: kept as a DiT lever. It does not close the gap to the CNN (0 dB DINOv2 0.215 vs 0.292, top-1 12.6% vs 22%) |
| 18 Nov | seeds | g-cb3 | **The CNN control replicates.** Seed 1 of the two-stage blind 16-QAM CNN (40k pixel + 10k LPIPS 0.1) vs seed 0, spread set: DINOv2 0.132 / 0.300 / 0.411 vs 0.132 / 0.292 / 0.398 at -5 / 0 / +5 dB; top-1 2.9 / 21.6 / 43.0% vs 3.3 / 21.5 / 42.4%; FID 178 / 119 / 95 vs 180 / 118 / 98. Paired (s1 - s0): top-1 within 1 pt, none significant; top-5 +0.5 / +2.7 / +3.4 (only +5 dB significant); PSNR +0.04 to +0.09 dB and LPIPS +0.002 to +0.004 (significant but tiny). Seed 0's numbers stand as the 128 px CNN reference; the DiT-vs-CNN gaps logged above are well outside the seed spread |
| 19 Nov | B (256 px) | RTX 5060 | **The 256 px DiT converged** (rule fired at 140k; last 0.2% gain at 120k; 14.7 h at 3.1-3.4 it/s once the screen moved to the iGPU). Spread set, 10 steps, -5 / 0 / +5 dB: PSNR 17.0 / 19.7 / 21.0, LPIPS 0.371 / 0.249 / 0.196, DINOv2 0.30 / 0.56 / 0.70, top-1 27 / 81 / 95%, FID 120 / 70 / 51; class kept 78 / 93 / 97% (dump, `results/gen/analysis/p256/report.md`). The 128 px DiT at 0 dB had DINOv2 0.18, top-1 10%, FID 141, but resolution changes the metrics' footing too, so only the 256 px CNN decides. Images: sharp, scene, colour and object kept at 0 dB; it invents text, faces and fine structure (two draws of one received signal differ by 14 / 9.5 / 8.3 grey levels). Training stable (loss jitter 0.8%), validation DINOv2 still +0.009 per 10k steps at the stop. The first-1,000 test set has no FID: the 5060's venv lacked torchmetrics (installed before the spread sets ran) |
| 19 Nov | B (256 px) | RTX 5060 | The 256 px CNN does not fit the 8 GB card at batch 64 (about 12 GB; 24 GB with LPIPS) and crawled at 0.009 it/s for 3 h. Relaunched at 60451b6 with `--grad-accum 2` (8 for the LPIPS stage; CNN only, no batch norm, gradient within 0.03% of the whole batch's): 0.97 it/s. Validation at 12k: PSNR 18.7 / 20.9 / 22.2 dB, DINOv2 0.09 / 0.31 / 0.44 (DiT at the end: 16.5 / 19.2 / 20.6 dB, 0.38 / 0.66 / 0.77); the verdict waits for the LPIPS stage |
| 19 Nov | convergence | RTX 5060 | The 128 px blind 16-QAM pixel CNN at 198k (constant LR, one validation from the patience limit): 40k -> 198k gains PSNR +0.32 / +0.39 / +0.38 dB and validation DINOv2 0.14 -> 0.20 at 0 dB, 0.21 -> 0.30 at +5 dB. The 40k-step budget leaves the CNN well short on semantics |
| 19 Nov | plan | (decision) | Plan for 20 to 29 Nov written into section 6: 128 px stages `cnn16-ft`, `cnn`, `cnn-ft` dropped; C1 (cosine-60k CNN against the converged one) decides whether converged CNNs are affordable; track A as A-D / A-C / A-S / A-P / A-E on 16-QAM over 0 to 20 dB, AWGN and Rayleigh; track B branches on Friday's verdict; quota corrected to 90 h a week |
| 19 Nov | plan | (decision) | All DiT runs at 256 px from now, started without waiting for CNN results (user): A's DiT arms on Kaggle (`a256-ab`, `a256-at` first, on a 4th account, venkatrrr: 120 h a week), seed 1 and REPA of the 256 px DiT on the 5060 after the dump; the 5060's 256 px CNN restarted at 16k after slowing to 0.25 it/s (cause unknown; 1.10 it/s after the restart) |
| 19 Nov | A (256 px) | a256-ab, a256-at (venkatrrr), a256-smoke (subhashhhh), a-smoke (own) | Launched at 268779d. a256-ab / a256-at: A's AWGN blind and told DiT-S at 256 px over 0 to 20 dB, first sessions (`--max-hours 10.5`, `results/gen/p256-a`). a256-smoke: 300-step checks at 256 px of told, SNR-aware transmitter, Rayleigh, `est-energy`, `gain2` and REPA, and the T4's 256 px speed. a-smoke: the same checks at 128 px (code paths only). **a-smoke passed** (COMPLETE, 28 min): every DiT and CNN arm trained, and every evaluation wrote its file (`est-energy`, `est-m2m4`, `gain2`, `offset-5`; Rayleigh, SNR-aware transmitter, specialist) |
| 19 Nov | B (256 px) | RTX 5060 | **Validation DINOv2 is not comparable across loss stages.** At 128 px the LPIPS stage took the CNN's validation DINOv2 at 0 dB from 0.13 to 0.44 in 10k steps, past the converged 128 px DiT's 0.24 at its best step. So the 256 px CNN's pixel stage (0.33 at 20k, against the DiT's 0.44 at 20k and 0.66 at 140k) says nothing about the verdict; it waits for the LPIPS stage and the test set |
| 19 Nov | review | (decision) | Every result re-read; each remaining task given a decision (section 6, "Plan from 19 Nov evening"). Dropped: `cnn16` and C1, A's CNN arms and every other 128 px run. Kept: B and its converged re-check (stages `cnn256-conv`, `cnn256-conv-ft`, `dump256-conv`, spec `p256-conv.json`: the CNN continued from its checkpoint by the DiT's own rule, constant LR and patience 10), and A at 256 px with the B recipe frozen. Optional: REPA and ImageNet-100 pretraining at 256 px. The job registry (`results/gen/kaggle_jobs.json`) gains the 21 jobs since 15 Nov |
| 20 Nov | A (code) | none | **Rayleigh fading corrected** (`GenConfig.fading`, default `block`). Two problems in every Rayleigh run so far: (1) the DiT's semantic and detail streams each drew their own fade, two independent fades per image (a diversity the one-stream CNN did not get); (2) a told receiver was given the average SNR, although it equalises with h and so knows the image's effective SNR, SNR + 10 log10 \|h\|^2. Now one fade per image for every symbol, and told receivers (DiT and CNN) get the effective SNR, clipped to [-20, 40] dB; `receiver_view` offsets and gain errors start from it; transmitters still know only the average. AWGN draws are unchanged (tested). The 128 px Rayleigh runs (g-r1, g-r2, 15 Nov: told = blind, SNR-aware transmitter +0.26 dB at -5 dB) used the old model; their configs now say `fading: legacy`, and their told-vs-blind result is a weak test. Rayleigh smoke at 256 px (`r256-smoke`, own account) first; `scripts/kaggle_queue.py` then pushes the Rayleigh runs (own, subhashhhh) at the 21 Nov reset only if it completes, with the AWGN second sessions and spec0. **`r256-smoke` passed** (COMPLETE at 062019d, 58 min): blind, told and SNR-aware-transmitter DiTs train under block fading, and the `est-energy`, `offset+3` and `gain1` evaluations each change the told receiver's output |
| 20 Nov | A (Kaggle) | `a256-rb-0`, `a256-rt-0`, `a256-p0-0` | Short first sessions on the week's leftover quota (own 3.0 h, subhashhhh 2.1 h, venkat96r 2.6 h left): Rayleigh blind, Rayleigh told and spec0 with `--max-hours` 2.0 / 1.1 / 1.6, pushed 21:14 UTC 19 Nov; the 21 Nov sessions now resume them. `a256-ab-2` moved forward to start when `a256-ab` stops, with `--max-hours 6` (venkatrrr has about 7.6 h left after its two running sessions); `a256-ab-3` added after the reset |
| 20 Nov | B | `cnn256`, `cnn256-ft`, `dump256` (5060) | **B passes** (spec `p256.json`, rule of 17 Nov). The 256 px CNN: pixel stage 40k steps (6.4 h), LPIPS 0.1 stage 10k steps (4.7 h). Spread set, DiT minus CNN, paired, 95% intervals: LPIPS -0.076 / -0.083 / -0.087 and DINOv2 +0.086 / +0.028 [+0.017, +0.038] / +0.031 [+0.023, +0.039] at -5 / 0 / +5 dB; top-1 +11.0 / +9.4 / +5.6 pts; FID 120 vs 174, 70 vs 82, 51 vs 56; PSNR -1.20 / -1.23 / -1.49 dB; class kept equal at 0 and +5 dB (93%, 97%), 78% vs 68% at -5 dB. The DiT wins on both LPIPS and DINOv2 on 55% / 61% of images at 0 / +5 dB. Its high-frequency power is 0.98 of the originals' at 0 dB (CNN 0.18): it restores texture where the CNN blurs. The pixel-only CNN (cnnpx) is far behind on every semantic measure (DINOv2 0.28 at 0 dB). *Seed rule:* 128 px seed spreads (CNN DINOv2 0.008 / 0.013 at 0 / +5 dB, LPIPS 0.004, top-1 1 pt) against the shift that would flip the verdict (an LPIPS gain lost: 0.08, 20x; a clear DINOv2 loss: 0.038 / 0.039, 4.8x / 3.0x; top-1: 13x): every margin is at least 3x, so one seed stands for 24 Nov; the thinnest is DINOv2 at +5 dB, and the 256 px spread is measured by `dit256-s1` later. *Extension rule:* not needed. Next, as pre-registered: the converged CNN (`cnn256-conv` from the 40k checkpoint, constant LR, patience 10, then the LPIPS stage), started 20 Nov 08:00 IST (resumed at step 40,100, 0.92 it/s); `dit256-s1` waits for it. The `~/after_dump.sh` hand-off stalled for 35 min (its `pgrep -f gen_5060.py` matched the tmux server's command line) and was launched by hand |
| 20 Nov | A (Kaggle) | `a256-ab`, `a256-at`, `a256-at-2`, `a256-p20-0` | **AWGN blind converged in one session:** `a256-ab` stopped on patience at 76k steps (best validation loss at 66k; 9.7 h) and ran its spread test (0 / 5 / 10 / 15 / 20 dB: DINOv2 0.40 / 0.66 / 0.75 / 0.78 / 0.79, top-1 57 / 92 / 97 / 98 / 99%, PSNR 18.8 / 20.8 / 21.8 / 22.2 / 22.3 dB), so `a256-ab-2` was skipped. `a256-at` (told) stopped on `--max-hours` at step 67,008; its second session `a256-at-2` was moved forward onto venkatrrr's 8.0 h left this week (`--max-hours 4.5`, `a256-at-3` after the reset if needed), with a short first session of the 20 dB specialist (`a256-p20-0`, 1.3 h; `a256-p20`, `a256-p20-2` after the reset). The short sessions on own / subhashhhh / venkat96r completed (`a256-rb-0` stopped at step 12,309: Rayleigh trains at 1.86 it/s against 2.38 for AWGN) |
| 20 Nov | A (Kaggle) | `a256-at-2`, `a256-ae-awgn` | **AWGN blind against told at 256 px, first look (means only, 1,000 spread images, one seed each):** `a256-at-2` converged at 76k steps (best at 64k), the same stop as blind. Blind / told at 0 / 5 / 10 / 15 / 20 dB: PSNR 18.79 / 18.66, 20.84 / 20.73, 21.80 / 21.75, 22.23 / 22.19, 22.34 / 22.32 dB; DINOv2 0.404 / 0.402, 0.658 / 0.647, 0.753 / 0.747, 0.785 / 0.777, 0.791 / 0.787; top-1 56.9 / 54.0, 91.9 / 91.4, 97.4 / 97.4, 98.2 / 97.6, 98.8 / 98.4%; FID 108 / 109, 61 / 63, 45 / 46, 39 / 41, 39 / 39. Telling the receiver the SNR buys nothing. Paired, told minus blind (`results/gen/analysis/paired_p256a_told_vs_blind_awgn.json`): PSNR -0.125 [-0.152, -0.099] / -0.109 [-0.135, -0.084] / -0.055 [-0.081, -0.029] / -0.044 [-0.066, -0.019] / -0.019 [-0.047, +0.009] dB at 0 / 5 / 10 / 15 / 20 dB; LPIPS +0.001 to +0.002 (significant, tiny); top-1 -2.9 [-6.0, 0.0] at 0 dB and within 1 pt elsewhere; top-5 not significant. The told model is slightly but significantly worse at 0 to 15 dB. That is more than the 128 px seed spread of the means (about 0.01 dB), so it may be real (an SNR input the receiver can read off the signal anyway, at the cost of capacity) or a larger seed spread at 256 px; seed 1 of blind and told would settle it but is not needed for the claim, which needs only that told is not better. Short sessions: `a256-rt-0` stopped at 8,492 steps, `a256-p0-0` at 13,274, `a256-p20-0` at 10,271. `a256-ae-awgn` (venkatrrr, 3.7 h left) evaluates the told model with `est-energy`, `est-m2m4`, `offset2`, `offset5`, `gain2` and the blind one with `gain2`; `gain1` and the Rayleigh set follow next week |
| 20 Nov | A (Kaggle) | `a256-ae-awgn` | **AWGN mismatch evaluations** (paired on the 1,000 spread images, 0 / 5 / 10 / 15 / 20 dB; JSONs `results/gen/analysis/paired_p256a_*`). *Against the blind receiver:* told + `est-energy` -0.11 / -0.11 / -0.06 / -0.06 / -0.11 dB; told + `est-m2m4` -0.08 / -0.30 / -0.28 / -0.29 / -0.27 dB (top-1 -1.4 to -3.0 pts from 5 dB up); told + `offset2` -0.00 / -0.02 / -0.01 / -0.03 / -0.08 dB (equal to blind up to 10 dB); told + `offset5` -0.16 / -0.02 / +0.02 / -0.02 / -0.16 dB. *Against the told receiver with the true SNR:* `est-energy` +0.01 / -0.00 / -0.00 / -0.01 / -0.09 dB, so the energy estimate is as good as the truth up to 15 dB; `offset2` +0.12 / +0.09 / +0.05 / +0.02 / -0.06 dB: told 2 dB more than the truth, the told model does better at 0 to 15 dB, by about its deficit to blind. This suggests (inference) that the told deficit comes from how it uses its SNR input, not from lost capacity. *Gain error of 2 dB* (received amplitude x 10^(2/20)): blind loses -0.43 / -0.63 / -0.83 / -0.91 / -0.95 dB and top-1 -10.7 / -7.6 / -4.7 / -3.1 / -2.7 pts; told loses -0.10 to -0.36 dB against its own true-SNR result. Told minus blind, both with the error: PSNR +0.21 / +0.32 / +0.47 / +0.50 / +0.55 dB, LPIPS -0.033 to -0.044, top-1 +10.1 / +7.8 / +4.8 / +2.8 / +2.2 pts, all clear of 0. **The blind receiver reads the SNR from the received amplitude, so a gain error misleads it; a told receiver keeps the true SNR.** "Needs no SNR" therefore assumes an accurate receive gain; `gain1` (after the reset) and the Rayleigh set (where equalisation sets the gain) size how accurate |
