# Generative JSCC: history of the month plan (16 to 26 Nov 2026)

Moved here unedited from [gen-jscc-month-plan.md](gen-jscc-month-plan.md) when that file was
rewritten as a status guide on 26 Nov 2026. It holds the 16 Nov review, the 20 Nov status, the
literature table of 16 to 18 Nov, the compute notes, the lever ranking, the weekly plans of
16 to 26 Nov and the working rules as they stood. References to "section 6" or "section 8" below
point to the old layout: the log is now section 8 of the main file, and the current plan is its
section 5. Results are recorded only in the main file's log.

## Status on 20 Nov (superseded by the plan of 26 Nov above)

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
- **B fails the converged re-check (22 Nov).** Against the converged CNN (pixel stage 140k, LPIPS
  stage capped at 36k), spread set, DiT minus CNN at 0 / +5 dB: LPIPS -0.039 / -0.050 (DiT better,
  intervals clear of 0) but DINOv2 -0.043 / -0.017 (CNN better, intervals clear of 0), FID 70.2 vs
  59.3 and 50.5 vs 42.9 (CNN better), top-1 ties, PSNR -1.6 / -1.8 dB. The rule of 17 Nov needs no
  clear DINOv2 loss, so B fails. The 20 Nov semantic lead came from the undertrained CNN. As the plan
  says for a fail: study A stands (it characterises the diffusion receiver either way), the paper
  makes no quality claim against the CNN beyond LPIPS, and the first rescue lever, `dit256-repa`,
  starts on the 5060 after `dump256-conv`.
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
- **Both receivers depend on the receive gain, in opposite directions (gain sweep of -2 to +2 dB,
  20-21 Nov, `a256-ae-awgn` and `a256-ae-awgn2`).** Paired PSNR change against each model's own
  true-gain result, 0 to 20 dB. *Blind:* +2 dB costs 0.43 to 0.95, +1 dB 0.19 to 0.41, +0.5 dB
  0.09 to 0.16; -1 dB is harmless (+0.09 to -0.09) and -2 dB costs 0.00 to 0.66 (from 5 dB up).
  *Told (true SNR given):* -2 dB costs 0.23 to 0.48, -1 dB 0.07 to 0.12, +1 dB 0.01 to 0.10,
  +2 dB 0.10 to 0.38 (LPIPS improves). So an amplitude too high hurts the blind receiver, and one
  too low hurts the told one. Told minus blind under the same error: +0.05 to +0.29 dB at +1 dB,
  -0.05 to -0.29 dB at -1 dB (top-1 -10 pts at 0 dB), +0.21 to +0.55 dB at +2 dB, -0.36 to +0.17
  dB at -2 dB. Worst case over +-1 dB: told 0.14 dB, blind 0.41 dB. By the Gate A rule (a 1 dB
  gain error costing more than SNR knowledge ever gains, which is 0) the AWGN result points to
  the reworded claim "no noise-level knowledge, gain tracking assumed"; Rayleigh decides it at
  Gate A. A told receiver that estimates the SNR from the received energy matches the true-SNR
  one within 0.02 dB up to 15 dB (-0.09 dB at 20 dB); the M2M4 estimator costs 0.2 to 0.25 dB
  from 5 dB up. Still to come: Rayleigh, the specialists and the SNR-aware transmitter (all
  running from the 21 Nov reset).
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
| 3 | Telling the receiver the SNR adds nothing | Yes, 3 settings at 128 px; at 256 px on AWGN, 0 to 20 dB (20 Nov: told never better, -0.02 to -0.13 dB), **provided the receive gain is right**: a +1 dB gain error costs blind 0.19 to 0.41 dB (told 0.01 to 0.10), while a -1 dB error costs told more (0.07 to 0.12) than blind; AWGN points to "gain tracking assumed" | Weak: SIJSCC shows it for a non-generative codec; OSD-JSCC has a partial ablation | Small |
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
| A-E evaluations: `est-energy`, `est-m2m4`, offsets of 2 and 5 dB, gain errors of 1 and 2 dB | A, the gain assumption | AWGN: **done 21 Nov** (gain sweep -2 to +2 dB in `a256-ae-awgn2`): too high an amplitude hurts blind, too low hurts told (section 8). Rayleigh: once its models finish, on the accounts that hold them | Cheap. These evaluations define what "needs no SNR" assumes |
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
  | venkatrrr (also) | `a256-ae-awgn2`: gain errors of +1, +0.5, -1 and -2 dB for told and blind. **Done 21 Nov** (section 8) |

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
