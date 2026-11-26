# Generative JSCC: status guide and plan to the 11 Dec 2026 experiment freeze

The single place to find where the generative JSCC work stands, what is running, what comes next
and why. It is meant to let anyone (or any later session) pick up the work without the
conversation that produced it.

- **Last rewritten 26 Nov 2026.** Experiments freeze on **11 Dec**; anything unfinished then is
  cut, not extended.
- Section 8 (Log) is the dated record of every result. Add a row there for every finished
  experiment, and update sections 1 and 5.2 when something changes.
- The review of 16 Nov, the status of 20 Nov and the superseded weekly plans are kept unedited
  in [gen-jscc-month-plan-history.md](gen-jscc-month-plan-history.md).
- The literature behind the 26 Nov decisions is in `reports/DiT JSCC path to paper.md` and
  `research_notes/` (local only, not committed).

## 1. Status at a glance (26 Nov)

**Where the work stands.** The current receiver is a blind 16-QAM DiT-S with REPA, converged on
24 Nov. Against a converged CNN deep JSCC, at the same 4,096 symbols per 256 px image:
- it beats or ties the CNN on every meaning score and is far more realistic;
- it loses 1.6 to 2.1 dB PSNR;
- the lead is large at -5 dB and thin at +5 dB.

The SNR study (study A) is complete: the receiver needs no noise-level knowledge, given gain
tracking. **The paper's headline (user, 26 Nov) is the perceptual result:** a
constellation-constrained diffusion receiver that beats a learned JSCC on most perceptual and
semantic metrics. The SNR study is the second, supporting result.

**Running now**

| Where | What | Plan ID | Expected |
|---|---|---|---|
| RTX 5060, tmux `gen` (log `~/gen_5060_queue.log`) | LR anneal of the REPA receiver (30k steps, peak 5e-5, cosine), then its tests, then the sampler sweep (guidance 1.5 / 2.0, 20 / 50 steps) | E-P2, E-P1 | Training done about 15:45 UTC 26 Nov, sweep about 18:30 UTC |
| RTX 5060, tmux `q2` (log `~/queue2.log`) | Waits for the first queue to reach the CNN stage, then: memory/speed probes (DiT-S, DiT-B, with and without checkpointing); then the stages in `~/queue2_stages.txt`: `posterior-repa`, `dit256-x100`, `dit256-x100-ft`, `cnn256-conv-ft-long` | E-A2, E-P5, E-B3 | E-P5 done about 28 Nov; DiT-B stages (E-P7) added after the probe |
| Kaggle own | `e-a7-analog`: the REPA receiver with analog symbols, session 1; `-2` queued 28 Nov 00:05 UTC | E-A7 | About 29 Nov |
| Kaggle venkatrrr | `e-p3-base`: 16-QAM REPA, 60k steps from scratch (screen baseline); `-2` queued | E-P3 | About 28 Nov |
| Kaggle subhashhhh | `e-p3-repa1`: the same with REPA weight 1.0; `-2` queued | E-P3 | About 28 Nov |
| Kaggle venkat96r | `a256-tx-2` (SNR-aware transmitter, second session) queued 28 Nov 00:05 UTC | study A | About 28 Nov |

The Kaggle queue runner (`scripts/kaggle_queue.py .kaggle_jobs/queue.json`) must be alive on the
development PC for the `-2` sessions to push.

**Next decisions**

| When | Decision | Input |
|---|---|---|
| 26-27 Nov | Default sampler (steps, guidance) for every later evaluation | E-P1 sweep |
| 27 Nov | Add DiT-B (E-P7) to the 5060 queue, and its step cap | Memory/speed probe in `~/queue2.log` |
| 28 Nov | **Gate D:** does the extra data help at 256 px? | E-P5 vs the REPA receiver |
| About 1-2 Dec | **Gate L:** which recipe levers enter the final receiver | E-P3 screen |
| About 1 Dec | **Gate M:** DiT-S or DiT-B | E-P7 vs E-P5 |
| 2 Dec | Start the final receiver (E-P6) | Gates D, L, M |
| 8 Dec | Fallback if E-P6 is not converged: report the 24 Nov receiver (or E-P2's) | |

## 2. The system

- **Data:** Imagenette (10 ImageNet classes; 9,469 training images; 3,425 test images, scored on
  the 1,000-image class-spread set), 256 px.
- **Transmitter:**
  - semantic branch: frozen DINOv2 ViT-B/14 patch features (224 px input, 16 x 16 grid) into a
    small conv encoder, 8 channels = 1,024 symbols;
  - detail branch: four stride-2 convolutions on the pixels, 24 channels = 3,072 symbols;
  - each branch is power-normalised, then snapped to 16-QAM by a soft-to-hard quantiser
    (DeepJSCC-Q style; inverse temperature 5, fixed; usage regulariser 0.05);
  - 4,096 symbols per image, bandwidth ratio 1/48; SNR measured against the energy actually sent.
- **Channel:** AWGN, SNR uniform over -5 to 5 dB in training; block Rayleigh (one fade per image,
  equalised with known h) in study A.
- **Receiver:**
  - DiT-S/2 (12 blocks, width 384, 34.9M) in the SD-VAE (ft-EMA) latent, 32 x 32 x 4;
  - semantic symbols are fused into blocks 1-3 and detail symbols into blocks 4-6;
  - v-prediction, 10 DDIM steps, no guidance;
  - **blind**: the SNR slot is a constant.
- **Training:** REPA (block 4 aligned to DINOv2, weight 0.5, training only), AdamW 1e-4, batch 64,
  EMA 0.9999, patience stop (122k steps, 38 GPU-h on the 5060). Rebuild time 0.07 s per image.
- **Baselines:**
  - **CNN:** the repo's deep JSCC with the same symbols, 16-QAM and blindness; 140k pixel-loss steps
    plus 36k LPIPS steps (69 GPU-h);
  - **separation:** AVIF + LDPC + QPSK/16/64-QAM, the best scheme per SNR, plus the 16-QAM
    capacity bound.

## 3. Results so far

### 3.1 Headline: the REPA receiver against the converged CNN

Spread set (1,000 images), same images and channel noise for both. "Clear" means a paired
bootstrap 95% interval that excludes 0 on both halves of the images. Report:
`results/gen/analysis/p256-repa-final/report`.

| Score (better) | -5 dB | 0 dB | +5 dB | Clear winner |
|---|---|---|---|---|
| CLIP-IQA realism (higher) | **0.80** / 0.32 | **0.82** / 0.43 | **0.83** / 0.46 | ours, all three |
| FID, set level, 1,000 images (lower) | **88.5** / 129.2 | **53.7** / 59.3 | 43.8 / **42.9** | ours at -5 / 0 dB |
| DreamSim (lower) | **0.334** / 0.432 | **0.211** / 0.234 | **0.162** / 0.172 | ours, all three |
| LPIPS-Alex (lower) | **0.363** / 0.402 | **0.258** / 0.288 | **0.207** / 0.246 | ours |
| Caption similarity (higher) | **0.744** / 0.657 | **0.812** / 0.782 | **0.841** / 0.826 | ours, all three |
| CLIP similarity (higher) | **0.705** / 0.623 | **0.766** / 0.729 | **0.801** / 0.785 | ours, all three |
| ConvNeXt probability of the right class, % (higher) | **46** / 23 | **62** / 47 | **66** / 57 | ours, all three |
| DINOv2 similarity (higher; circular for REPA) | **0.431** / 0.297 | **0.635** / 0.604 | **0.728** / 0.720 | ours at -5 / 0; tie at +5 |
| Retrieval top-1 among 1,000, % (higher) | **48.4** / 25.4 | **86.0** / 82.6 | **95.9** / 94.1 | tie at 0 / +5 |
| PSNR, dB (higher) | 16.89 / **18.46** | 19.38 / **21.29** | 20.70 / **22.79** | CNN, all three |

Cells are ours / CNN, with the better value in bold.

**Against us, and to be reported:**
- DISTS favours the CNN at 0 and +5 dB (0.206 vs 0.200 at 0 dB), and so does CLIP's score
  against the original's caption text.
- LPIPS-VGG favoured the CNN for the plain DiT and has not been measured for REPA.
- The object detector and CLIP zero-shot class tie at 0 and +5 dB.

**Weak points a reviewer will attack:**
- REPA was adopted after the pre-registered test B failed (22 Nov).
- The CNN's LPIPS stage was capped at 36k steps while still improving.
- DINOv2 scores are partly circular for REPA.
- One seed; Imagenette only; no out-of-distribution test.
- FID on 1,000 images is biased.

### 3.2 Study A: does the receiver need the SNR? (complete; plain DiT, 16-QAM, 0 to 20 dB, Kaggle)

- **Told minus blind:**
  - AWGN: PSNR -0.13 / -0.11 / -0.06 / -0.04 / -0.02 dB at 0 / 5 / 10 / 15 / 20 dB. Told is
    never better; the largest upper bound is +0.009 dB.
  - Rayleigh: -0.03 to -0.04 dB, top-1 within 1 pt.
- **Estimated SNR:** a told receiver given an energy-based estimate equals the true-SNR one (within
  0.02 dB up to 15 dB). M2M4 costs 0.2 to 0.25 dB (AWGN) and 0.2 to 0.5 dB (Rayleigh).
- **Gain error, worst case over +-1 dB:** blind 0.41 dB, told 0.14 dB (AWGN); blind 0.47 dB,
  told 0.13 dB (Rayleigh). Too much amplitude hurts the blind receiver; too little hurts the told
  one.
- **Verdict (Gate A, 22 Nov):** "the receiver needs no noise-level knowledge, given gain
  tracking"; being told the SNR buys robustness to gain error, not information.
- **Specialists:** a 0 dB specialist beats the blind 0-20 dB model at 0 dB by 0.86 dB and 20.6
  pts top-1 (still 0.81 dB after 50k more steps), so one model for all SNRs costs something at the
  bottom of the range. A 20 dB specialist overfits and loses everywhere.
- **Fading cost** (blind Rayleigh minus blind AWGN, same average SNR):
  - PSNR -1.1 / -1.3 / -0.8 / -0.5 / -0.2 dB;
  - top-1 -14 / -20 / -9 / -4 / -1 pts.

### 3.3 Against separation

- **AWGN, top-1 %** (ours / classical, at -5 / 0 / 5 / 10 dB): 48.4 / 0.1, 86.0 / 47.8,
  95.9 / 86.8, 97.8 / 97.3.
- **AWGN, PSNR:** classical is 3 to 6 dB sharper whenever it decodes, at 0 to 10 dB.
- **Rayleigh, top-1 %** (ours / best practical code / perfect code, at 0 / 5 / 10 / 15 / 20 dB):

  | | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB |
  |---|---|---|---|---|---|
  | Ours | 43 | 72 | 88 | 94 | 98 |
  | Best practical code | 18 | 38 | 68 | 80 | 87 |
  | Perfect code | 14 | 49 | 72 | 83 | 91 |

  Ours is ahead by 7 to 34 pts. Separation must pick its rate without knowing the fade.
- **Rayleigh, PSNR:** separation wins only from 5 dB up, by 1 to 4 dB.

### 3.4 What limits quality now

**It isn't resolution or the latent space.** The 256 px SD-VAE round trip reaches DINOv2 0.96
and 25.9 dB. The REPA receiver saturates far below that as the channel gets cleaner:

| | +5 dB | +7.5 dB | +10 dB |
|---|---|---|---|
| DINOv2 similarity | 0.728 | 0.750 | 0.763 |
| PSNR (dB) | 20.7 | 21.0 | 21.2 |

**The limit is what 4,096 symbols, a 35M model and 9.5k training images carry.** So data and
model size are the levers (E-P5, E-P7), not 320, 384 or 512 px.

## 4. Paper position and claims

### 4.1 Claims

| # | Claim | Have | Still needed by 11 Dec |
|---|---|---|---|
| C1 (headline) | A constellation-constrained (16-QAM) diffusion receiver beats a converged learned JSCC on most perceptual and semantic metrics, costs little against the same receiver with analog symbols, and degrades gracefully where digital separation falls off a cliff | In-distribution vs the CNN; separation curves | E-A7 (analog cost), E-A5 (Kodak/CLIC), the pre-registered metric list (6.2), better margins at +5 dB (E-P5/E-P7) |
| C2 | It needs no SNR at either end; it needs gain tracking, and being told the SNR buys robustness to gain error, not quality | Complete on the plain DiT | Optional: REPA bridge (E-A1) |
| C3 | Its PSNR loss is the price of realism: it behaves like a posterior sampler, and the receiver alone can move along the perception-distortion curve from one transmitted signal | none | E-A2 |
| C4 | Its failures are confident (sharp but wrong); they can be counted, and a receiver-side consistency check flags them where draw disagreement does not | Qualitative; draw disagreement fails (AUROC 0.38-0.48, plain DiT) | E-A3 |
| C5 (optional) | Mechanism: the semantic stream carries meaning, the detail stream fidelity; the posterior spread tracks the SNR without being told it | none | E-A4 |

### 4.2 Novelty (re-checked 26 Nov against fetched paper text)

**Still open:**
- No paper combines constellation-constrained (QAM) symbols with a diffusion receiver that
  generates the image and no SNR at either end.
- Every diffusion JSCC paper read sends analog symbols. SemQAM (arXiv 2610.29643) is QAM-native
  but uses a GAN/VQ decoder.
- No paper tests an estimated SNR or a receiver gain error.

**Taken:**
- **One diffusion model over a range of SNRs:** DiffJSCC, FlowSem, OSD-JSCC, all given the SNR.
- **A blind CNN JSCC:** CBJSCC (Sensors 2024, Section 4.4) and SIJSCC compare SNR at both ends,
  at the decoder only and at neither, and conclude it "does not significantly improve the
  performance". Both are analog, AWGN-only, PSNR-only, and feed the true SNR; they are probably
  two versions of one work.
- **Specialist vs wide-range training for a CNN:** CBJSCC Section 4.5. Any "cost of universality"
  claim must be limited to generative receivers.

**The architecture** is a scaled-down DiT-JSCC (arXiv 2601.03112). DiT-JSCC:
- trains one model per SNR;
- tests on 100 ImageNet-val images;
- reports no PSNR table, no Kodak/CLIC, no multi-draw or uncertainty analysis, and no latency.

C3 to C5 and the Kodak/CLIC test are where this work can say more.

**Not yet read:** CD3M, DCDDM, PODD, DiffMoECom, the DeepJSCC-Q journal version, IEEE-only
ICC/GLOBECOM 2026 papers. Re-check before writing.

**Wording:**
- Do not say "first SNR-adaptive diffusion JSCC", "SNR-independent JSCC" or "digitally
  deployable" (there are no bits or channel code).
- Say "constellation-constrained".
- "Needs no SNR" means no noise-level knowledge, with gain tracking assumed.

The 16 Nov literature table with every paper's strengths and gaps is section 3 of the history file.

## 5. Plan to 11 Dec

### 5.1 Decisions of 26 Nov and why

| Question | Decision | Why |
|---|---|---|
| 512 px, or 320 / 384 px | **No** | The 128 to 256 px jump came from the SD-VAE ceiling (0.81 to 0.96 DINOv2) and 4x the symbols. At 256 px the ceiling no longer binds (3.4). A larger size only adds symbols at a fixed bandwidth ratio and needs every baseline retrained (CNN about 110 h at 320 px, 4 x 69 h at 512 px); Imagenette's source images are 320 px |
| Large pretrained generators (SD, DiT-XL, FLUX) | **No** (limitation) | They put DiffJSCC, OSD-JSCC and JSCGC ahead, but do not fit 8 GB or two weeks |
| Newer diffusion recipes | **Cheap ones, screened** (E-P3) | Flow matching with logit-normal timesteps (SiT-S beats DiT-S, FID 57.6 vs 68.4 at 400k, arXiv 2401.08740), iREPA (arXiv 2512.10794), HASTE (REPA off late, arXiv 2505.16792), RoPE + qk-norm, EDM augmentation. Rejected on evidence: Muon, x0 target, min-SNR, REPA-E, new autoencoders |
| More training at a lower LR | **Testing** (E-P2) | Cheap; weak evidence for small models (a late LR drop helped DiT-XL only; cosine decay did not help at 128 px) |
| More data | **Main bet** (E-P5) | ImageNet-100 pretraining was the largest 128 px gain (top-1 +5 / +14 pts, FID -12 / -21) and is untried at 256 px. Its DINOv2 cache (about 39 GB per 100k images) rules out Kaggle |
| Larger DiT | **DiT-B with the extra data** (E-P7) | DiT-B gave nothing on 9.5k images at 128 px (overfitting); with 5x the data it may. Needs activation checkpointing on 8 GB (added 26 Nov). DiT-L (458M) is not feasible |
| Better baselines (SwinJSCC, adversarial CNN) | **Dropped** (user) | Effort goes into the DiT's quality and the analyses. The CNN's LPIPS stage is finished only at the end of the 5060 queue (E-B3), since it needs no code |

### 5.2 Experiment register

Status: done, running, queued, to code (needs code before it can run), optional, dropped.

**Performance (P).** Each lever is compared, paired, with the right reference: the REPA receiver
for inference-time levers, and a REPA run at the same step budget for training levers.

| ID | What | Machine, cost | Status |
|---|---|---|---|
| E-P1 | No-training sweep on the REPA receiver: (a) noise-free ceiling; (b) DDIM steps 10 / 20 / 50 and eta 0 / 0.5 / 1; (c) guidance 1.0 / 1.5 / 2.0 (trained with 10% condition dropout); (d) ft-MSE VAE decoder swap; (e) mean of K draws (with E-A2) | 5060, about 6 h | (b) steps and (c) guidance queued (`repa-sweep`); eta via E-A2; (a), (d) to code |
| E-P2 | LR anneal: REPA weights, 30k steps, peak 5e-5, cosine to 0 (`dit256-repa-anneal`) | 5060, about 4 h + tests | Running (stopped silently at 4.8k on 26 Nov, resumed from 4k) |
| E-P3 | Recipe screen, from scratch, 60k steps, REPA on, each against the 60k baseline: baseline; REPA weight 1.0; logit-normal timesteps with flow matching; iREPA projector; HASTE (REPA off from 40k); RoPE + qk-norm; EDM augmentation | Kaggle, about 9 h each | Baseline and REPA 1.0 running; the other five to code, then Kaggle from 28 Nov |
| E-P4 | Start the reverse process from an MSE estimate (the existing regression head, `--reg-weight`; the sampler already takes `x_start` / `start`); sweep the start step as a fidelity knob | 5060 or Kaggle, about 5 h + sweep | Optional; low prior (128 px SDEdit from the CNN failed, g-s3) |
| E-P5 | More data: ImageNet-100 subset (49,500 images, 99 classes, after the Imagenette-name filter; `data/imagenet100_500` on the 5060, from `scripts/gen_analysis/imagenet100_subset.py`) + Imagenette, DiT-S + REPA, 100k steps; then Imagenette fine-tuning to convergence (cap 60k) (`dit256-x100`, `dit256-x100-ft`) | 5060, about 19 h | Queued (`q2`) |
| E-P6 | Final receiver: kept levers (gate L), data (gate D), size (gate M), to convergence, then E-P2's anneal if it helped | 5060, about 25 h (DiT-S) or 50 h (DiT-B) | 2 Dec |
| E-P7 | DiT-B (width 768, about 130M) with the extra data and activation checkpointing, 80k cap; then fine-tuning (`dit256b-x100`, `dit256b-x100-ft`) | 5060, about 40 h (estimate) | Waits for the memory/speed probe |

**Analyses (A).** Each is run first on the 24 Nov REPA receiver, so every figure exists even if
no lever pays off; re-run on E-P6.

| ID | What | Cost | Claim | Status |
|---|---|---|---|---|
| E-A1 | Study A bridge on REPA: told REPA (-5 to 5 dB, AWGN) to convergence; blind vs told, estimated SNR, gain error | Kaggle, 2 sessions | C2 | Optional, week of 5 Dec |
| E-A2 | Posterior-sampler test (`scripts/gen_analysis/posterior.py`, `posterior-repa`): K = 8 draws at -5 / 0 / +5 dB, eta 0 and 1. A posterior sampler's k-draw mean gains 10 log10(2k / (k + 1)) dB over one draw, and spread / error = 1 (Blau & Michaeli 2019; Yan et al. 2021). Then: the mean vs the CNN on PSNR; the perception-distortion curve from one signal (mean/draw blend, eta, steps) | 5060, about 1-2 h | C3 | Queued (`q2`) |
| E-A3 | Failure analysis: classifier flips, DETR object additions and drops, and caption contradictions per SNR for DiT, CNN and separation, split into sharp-wrong vs blurry-wrong. Detectors: semantic-space draw disagreement, re-encoding residual per stream (re-transmit the reconstruction, compare with the received symbols), best-of-N. AUROC, risk-coverage curves, split-conformal accept rule | About 6 h GPU | C4 | To code |
| E-A4 | Mechanism: inference-time erasure of 0 to 100% of the semantic or detail symbols; posterior spread vs SNR (pixel, LPIPS, label entropy) | About 4 h | C5 | To code |
| E-A5 | Out of distribution and robustness: Kodak (24 images, centre crop to 256 px) and CLIC2020 / DIV2K-val crops, per-image metrics (no FID on Kodak); SNR outside training (existing -10 to +10 dB curve); phase offset/noise, impulsive noise, Rician K | About 6 h + loader | C1 | To code |
| E-A6 | Compute vs quality (steps 1-50 and draws vs wall-clock on the 5060); KID and CMMD with bootstrap intervals; FID on all 3,425 test images; Kendall tau between metrics | About 3 h | C1 | To code |
| E-A7 | Cost of the constellation: the REPA receiver with analog symbols, to convergence, paired with 16-QAM REPA on the full metric list | Kaggle, 2 sessions | C1 | Running (`e-a7-analog`) |

**Baselines (B).**

| ID | What | Status |
|---|---|---|
| E-B1 | SwinJSCC in our setting | Dropped 26 Nov (user) |
| E-B2 | Data-matched CNN, if gate D keeps the extra data (same pretraining and fine-tuning, CNN convergence rule; 5060, up to about 70 h) | Optional; decide at gate D (otherwise the paper states the CNN saw Imagenette only) |
| E-B3 | The CNN's LPIPS stage continued from 36k to the patience rule (`cnn256-conv-ft-long`, `results/gen/p256-conv-long`) | Last in `q2` |
| E-B4 | Adversarial (GAN-JSCC style) CNN | Dropped 26 Nov (user) |

### 5.3 Schedule

| Dates | RTX 5060 | Kaggle | Development PC |
|---|---|---|---|
| 26 Nov (Thu) | E-P2 anneal, E-P1 sweep, probes, E-A2 | `e-a7-analog`, `e-p3-base`, `e-p3-repa1` (session 1) | E-P3 lever code |
| 27 Nov (Fri) | E-P5 pretraining | (this week's quota is spent) | E-A3/E-A5 code, Kodak/CLIC loader |
| 28 Nov (Sat) | E-P5 fine-tuning; **gate D**; E-P7 starts | Reset 00:00 UTC: the `-2` sessions and `a256-tx-2` push at 00:05; then E-P3 levers (venkat96r, venkatrrr, subhashhhh) | E-A3/E-A4 on the REPA receiver |
| 29 Nov to 1 Dec | E-P7 (DiT-B) | E-P3 levers finish; **gate L** | Analyses on the REPA receiver |
| 1-2 Dec | **Gate M**; E-P6 starts | E-A5 evaluations | |
| 2-5 Dec | E-P6 to convergence (+ anneal) | Reset 5 Dec: E-A1 if wanted; A-series reruns | |
| 6-10 Dec | A-series on the final receiver; E-B3 (and E-B2 if chosen) | A-series reruns | Figures and tables |
| 11 Dec (Fri) | **Freeze** | **Freeze** | Results frozen for the draft |

**Fallback:** if E-P6 is not converged by 8 Dec, the paper reports the 24 Nov REPA receiver (or
E-P2's annealed one), and every analysis already exists for it. The performance levers are a
bonus; C1 to C4 do not depend on them.

## 6. Rules

### 6.1 Comparisons

- Every compared run is scored on the spread set (1,000 images, `--eval-subset spread
  --eval-suffix spread`) at the default sampler fixed by E-P1, and compared with
  `scripts/gen_analysis/paired.py`.
- Compare only runs scored on the same device type: CPU and GPU draw different noise.
- A difference is "clear" only if its paired 95% interval excludes 0 on both halves of the images.
- Decide only on test sets. Validation metrics (32 images every 2k steps) track one run and are
  not comparable across loss stages or recipes. The LPIPS stage tripled the 128 px CNN's
  validation DINOv2, and REPA's validation loss is above the plain DiT's although its test results
  are better.
- Screens at 60k steps compare only against the 60k baseline, never against converged runs.
- One seed per configuration (user, 22 Nov). A gain within the 128 px seed spread (about 0.01 dB
  PSNR, 0.01 DINOv2) is not a gain.
- Report every metric family, win or lose. DINOv2 is flagged as partly circular for REPA models;
  non-DINOv2 meaning scores carry the claim.

### 6.2 Pre-registered metric list for C1 (fixed 26 Nov, before any new result)

- **Perceptual (full reference):** LPIPS-Alex, LPIPS-VGG, DISTS, DreamSim.
- **Realism:**
  - per image, no reference: CLIP-IQA, BRISQUE;
  - set level: FID, KID and CMMD, with bootstrap intervals.
- **Meaning:** CLIP similarity, caption similarity (BLIP-large captions, CLIP text cosine),
  ConvNeXt agreement, DETR F1, CLIP zero-shot. DINOv2 is reported but flagged.
- **Pixel:** PSNR, MS-SSIM.

"Better on most perceptual metrics" is claimed only if ours wins (paired, both halves) a majority
of the perceptual plus realism metrics at each of -5, 0 and +5 dB against the CNN. Losses go in
the same table.

### 6.3 Practice

- Extra training images go through `--extra-train`, which skips any file whose name matches an
  Imagenette image. Imagenette's validation split, the source of the test images, is 3,791 of
  3,925 ImageNet training images under their ImageNet names. One ImageNet-100 class is an
  Imagenette class and is dropped entirely by this filter.
- **Kaggle job hygiene:**
  - Check a job name against `results/gen/kaggle_jobs.json` before pushing.
  - Fetch only run folders (`kaggle_jobs.py fetch <job> --only <regex>`).
  - A job queued for more than about 20 minutes while others start is stuck: delete it and push
    again.
  - Kaggle jobs clone the pushed HEAD, and `kaggle_jobs.py push` refuses uncommitted tracked
    changes: commit and push first, with the user's go-ahead.
- **The repository is public.**
  - Never commit Kaggle tokens.
  - `results/gen/kaggle_jobs.json` is the committed copy of `.kaggle_jobs/*/job.json` without token
    paths; add jobs to it when their results are logged.
  - `ppts/`, `research_notes/` and `reports/` stay local.
- Record each finished experiment in section 8 with its job name, the paired result and the
  decision it feeds.
- **Known test failure (pre-existing at b1e6cda):**
  `test_overfit_check_passes_a_working_model_and_catches_a_detached_one` no longer raises for a
  detached encoder. It concerns the pre-flight sanity check only.

## 7. Compute and tooling

| Machine | Budget | Use |
|---|---|---|
| RTX 5060 (`ssh gpu-box`) | Unmetered, one run at a time; 8 GB | Long runs, 256 px, anything needing the extra-data cache |
| Kaggle T4, 4 accounts (own, subhashhhh, venkat96r, venkatrrr) | 30 h each per week, reset Saturdays 00:00 UTC; 12 h sessions; at most 2 sessions per account | Screens, controls, evaluation-only jobs |
| Development PC | CPU-only PyTorch; about 19 GB free disk (C: filled up once on 26 Nov) | Code, analysis, separation baseline, decks |

**RTX 5060:**
- It is a Windows gaming PC with Ubuntu under WSL2.
- **Keep the display on the motherboard HDMI.** Close games, Epic and GPU-accelerated browsers
  before long runs: on 26 Nov a game on the 5060 cut training from 2.4 to 0.9 it/s and probably
  killed one run.
- REPA DiT-S peaks at about 7.5 GB, and host RAM pressure has crashed runs before.
- Queues: `scripts/gen_5060.py <stages>` (each stage resumes from `checkpoint.pt`; finished runs
  are skipped and only evaluated). The current wrappers are `~/run_queue_5060.sh` (tmux `gen`,
  retries 5 times) and `~/queue2.sh` (tmux `q2`, reads `~/queue2_stages.txt`).
- Code reaches the box by sftp; the code there was synced to the repository on 26 Nov (old copy
  backed up in `~/semcom_code_backup_20261126.tgz`). `nvidia-smi` hangs under WSL; use
  `/mnt/c/Windows/System32/nvidia-smi.exe`.

**Kaggle:**
- `scripts/kaggle_jobs.py push <job> --runner gen --plan "..."` (`--token-file` for the teammate
  accounts).
- A run longer than one session is chained: the second job uses `--input-job` / `resume_of` on the
  same account, after the first ends cleanly (a killed session cannot be attached). `--max-hours`
  starts when training starts; setup takes about 35 min.
- `scripts/kaggle_queue.py` pushes queued jobs unattended from `.kaggle_jobs/queue.json`.
- Check quota with `python -m kaggle quota` (with `KAGGLE_API_TOKEN` set from the teammate token
  file).

**Measured speeds:**

| Run | Speed |
|---|---|
| DiT-S 256 px, T4 | 2.2 to 2.4 it/s (Rayleigh 1.86); converged run about 17 to 19 h over two sessions |
| DiT-S 256 px, 5060, plain | 3.1 to 3.4 it/s |
| DiT-S 256 px, 5060, REPA | 2.1 to 2.6 it/s |
| CNN 256 px, 5060 | 0.97 it/s pixel loss, 0.59 it/s with LPIPS (gradient accumulation) |
| DiT-B 256 px, 5060 | Measured by the 26 Nov probe (`~/queue2.log`) |

**Caches:**
- 256 px Imagenette: 11.8 GB, rebuilt each Kaggle session in `/kaggle/working` (19.5 GB free).
- ImageNet-100 subset at 256 px: about 23 GB of DINOv2 features (59k images). It is built on the 5060 at the
  first E-P5 run.

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
| 20 Nov | B (re-check) | `cnn256-conv` (5060) | A power cut at home: the run was stopped by hand at step 66.5k (checkpoint of 66k kept, with a backup copy) and resumed from 66k at 12:06 UTC. The box itself stayed up. Cost: about 500 steps |
| 21 Nov | A (Kaggle) | `a256-rb`, `a256-rt`, `a256-p20`, `a256-p0`, `a256-tx`, `a256-ae-awgn2` | Pushed by the queue at the 00:05 UTC reset, all at `1d2c63c`; the first four resume their short sessions of 19-20 Nov. `a256-at-3` skipped (`a256-at-2` converged). The queue runner had died unseen on 20 Nov (empty error log, no job affected) and was restarted at 13:03 UTC |
| 21 Nov | A (Kaggle) | `a256-ae-awgn2` | **Gain sweep, AWGN** (paired, 1,000 spread images, 0 / 5 / 10 / 15 / 20 dB; JSONs `results/gen/analysis/paired_p256a_gainsweep_{blind,told}_vs_true_awgn.json` and `paired_p256a_gain{-2,-1,0p5,1}_told_vs_blind_awgn.json`). PSNR change against the model's own true-gain result. *Blind:* -2 dB +0.00 / -0.37 / -0.57 / -0.65 / -0.66; -1 dB +0.09 / +0.05 / -0.05 / -0.09 / -0.09; +0.5 dB -0.09 / -0.14 / -0.16 / -0.16 / -0.16; +1 dB -0.19 / -0.29 / -0.38 / -0.40 / -0.41; +2 dB -0.43 / -0.63 / -0.83 / -0.91 / -0.95. *Told:* -2 dB -0.23 / -0.31 / -0.40 / -0.45 / -0.48; -1 dB -0.07 / -0.08 / -0.10 / -0.11 / -0.12; +0.5 dB about 0 (-0.02 at worst); +1 dB -0.01 / -0.04 / -0.08 / -0.09 / -0.10; +2 dB -0.10 / -0.20 / -0.30 / -0.36 / -0.38 (LPIPS better than at the true gain). All intervals about +-0.03. *Told minus blind under the same error:* -2 dB -0.36 / -0.05 / +0.12 / +0.16 / +0.17; -1 dB -0.29 / -0.25 / -0.11 / -0.07 / -0.05; +0.5 dB -0.04 / +0.02 / +0.09 / +0.09 / +0.12; +1 dB +0.05 / +0.14 / +0.25 / +0.26 / +0.29; +2 dB +0.21 to +0.55. Top-1 at 0 dB, told minus blind: -11 pts at -2 dB, -10 at -1 dB, +5 at +1 dB, +10 at +2 dB. **Reading:** the receivers fail in opposite directions, so yesterday's "told is the robust one" (from +2 dB alone) was one-sided. Too high an amplitude makes the blind receiver read the noise as weaker than it is; too low makes the told receiver's true SNR disagree with the symbols it sees (inference, not tested). In the worst case over +-1 dB, told loses 0.14 dB and blind 0.41 dB. By the Gate A rule the AWGN result already points to "no noise-level knowledge, gain tracking assumed"; Rayleigh, where the gain varies per image, decides it |
| 21 Nov | A (Kaggle) | `a256-p20` | **The 20 dB specialist loses to the blind 0-20 dB model even at 20 dB.** `spec20` (trained only at 20 dB) converged in-session: best validation loss at 36k, then the validation loss rose (0.2139 at 34k to 0.2176 at 54k) until patience stopped it at 56k (5.9 h), so `a256-p20-2` was skipped. It overfits: the blind model's validation loss kept falling to 66k. Evaluated at its best checkpoint, spec20 minus blind, paired on the 1,000 spread images, 0 / 5 / 10 / 15 / 20 dB: PSNR -1.30 / -0.66 / -0.35 / -0.28 / **-0.24 [-0.26, -0.21]** dB; LPIPS +0.060 / +0.009 / +0.005 / +0.007 / +0.008; top-1 -37.7 / -12.9 / -2.8 / -1.3 / -1.2 pts (JSON `paired_p256a_spec20_vs_blind_awgn.json`). One seed, but the 20 dB margin is over 20 times the 128 px seed spread. Reading: training over the SNR range acts as a regulariser that a single-SNR model lacks on this data, so the specialist is no upper bound here (inference). For the paper this strengthens "one blind model for all SNRs"; it also means the specialist comparison should say that the specialist overfits, and `spec0` (venkat96r, running) is the case where a specialist is likeliest to win |
| 21 Nov | B (re-check) | `cnn256-conv` (5060) | **Capped at 140k steps** (the converged DiT's count) instead of waiting for patience: at 132k the validation loss was still falling by about 0.05% per validation (0.008468 at 126k, 0.008445 at 132k), enough to keep resetting the 0.2% rule for hours, while validation PSNR (+0.03 dB per 4k) and DINOv2 (flat, 0.456-0.463 at 0 dB since 118k) had stopped mattering for the verdict. Restarted from its 134k checkpoint. So the CNN pixel stage gets the same step count as the DiT, not convergence by the rule; the write-up says so. The LPIPS stage that follows keeps the patience rule. Also on 21 Nov: the box slept on battery for about 2.5 h after another power cut and resumed by itself |
| 21 Nov | A (Kaggle) | `a256-rt-2`, `a256-rb-2`, `a256-ae-ray-told` | **Rayleigh told converged** in its second session: patience stop at 102k steps, best at 86k (later than AWGN's 64-66k). Spread test, 0 / 5 / 10 / 15 / 20 dB: PSNR 17.62 / 19.55 / 20.96 / 21.67 / 22.13 dB, LPIPS 0.342 / 0.254 / 0.197 / 0.172 / 0.157, DINOv2 0.35 / 0.53 / 0.66 / 0.74 / 0.77, top-1 44 / 72 / 87 / 93 / 97%. `a256-rb-2` sat in Kaggle's queue for 4.2 h (the other accounts' second sessions started at once); re-pushed at 16:27 UTC as version 2 (commit `171d612`, no change to the training code) and it started at once. The Rayleigh told mismatch evaluations (`a256-ae-ray-told`, subhashhhh: est-energy, est-m2m4, offset 2 / 5, gain -2 / -1 / +0.5 / +1 / +2) pushed at the same time; the blind gain evaluations follow when `a256-rb-2` finishes |
| 21 Nov | B (re-check) | `cnn256-conv` (5060) | **Pixel stage done at the 140k cap** (validation loss 0.008427). Spread test against the 40k pixel stage used in B, at -5 / 0 / +5 dB: PSNR 19.24 to 19.59, 21.78 to 22.24, 23.11 to 23.54 dB; LPIPS 0.548 to 0.505, 0.447 to 0.411, 0.402 to 0.371; DINOv2 0.101 to 0.138, 0.277 to 0.335, 0.431 to 0.475; top-1 5.7 to 9.2, 29.6 to 41.5, 57.6 to 67.3%; FID 279 to 243, 184 to 157, 142 to 122. Longer training helps every metric, but the pixel stage stays far behind B's LPIPS-tuned CNN on the perceptual ones (LPIPS 0.332 and DINOv2 0.533 at 0 dB), so the verdict rests on the LPIPS stage now running (`cnn256-conv-ft`, patience 10, 0.66 it/s). If that stage gains as much as the pixel stage did (DINOv2 +0.04 to +0.06), the DiT's DINOv2 lead at 0 and +5 dB (+0.028, +0.031) would go; its LPIPS lead (0.08) would shrink but likely hold. `a256-ae-ray-told` re-pushed after 40 min queued (version 2, `4c682b2`); started at once |
| 22 Nov | paper | (decision) | **Target: a letter-length paper** (characterisation of a constellation-constrained diffusion JSCC without SNR: blind vs told, estimated SNR, gain errors, specialists, against a CNN JSCC and separate coding). Added: (1) a **separate source-channel coding baseline** (`semcom/gen/separation.py`, `scripts/gen_separation.py`): AVIF (libavif 1.3, AV1 intra, speed 4; payload bytes counted, the ~275-byte container fixed) fitted to the bits that M-QAM (4/16/64) with a rate 1/2-5/6 LDPC code (n = 2048 regular (3, dc) PEG, normalised min-sum, 50 iterations) carries in the same 4,096 channel uses, plus a capacity-achieving 16-QAM code as an upper bound; lost images shown as a flat training-mean image; AWGN and block Rayleigh with receiver CSI; the same 1,000 spread images and metrics (checked equal to `Metrics.per_image` to 1e-6 and to torchmetrics FID to 1e-4). Decoder check: QPSK rate 1/2 reaches BLER 1e-2 about 1.7 dB from the QPSK limit, normal for a regular code of this length. (2) Second seeds of the AWGN pair (`a256-ab-s1` on venkatrrr, `a256-at-s1` on subhashhhh after its evaluations, 9 h cap for the quota). (3) `a256-ae-ray-blind` (own, after `a256-rb-2`): the blind Rayleigh gain errors. Planning slip: venkat96r's quota (0.8 h left at 19:00 UTC) ends before `a256-p0-2`'s 10.5 h cap; unless spec0 converges or reaches 150k steps first, that session is cut and resumes from `a256-p0` after the 28 Nov reset |
| 22 Nov | paper | (decision) | **Second seeds dropped** (user): the headline claims rest on paired intervals and margins far above any plausible seed spread, and single-seed results are the norm for JSCC letters. `a256-at-s1` removed from the queue before it started; `a256-ab-s1` (venkatrrr) was already running and was stopped by deleting its notebook (the Kaggle CLI has no cancel); venkatrrr keeps about 16 h of GPU quota until 28 Nov |
| 22 Nov | A (Rayleigh) | `a256-rb` + `a256-rb-2` (own) | **Rayleigh blind converged at 100k** (best 96k). Spread test 0 / 5 / 10 / 15 / 20 dB: PSNR 17.65 / 19.59 / 21.01 / 21.69 / 22.14 dB, top-1 43 / 72 / 88 / 94 / 98%. **Told minus blind, paired:** PSNR -0.034 / -0.038 / -0.043 / -0.016 / -0.015 dB (told never better; first three intervals exclude 0), LPIPS +0.000 to +0.003, top-1 within +-1 pt (all intervals include 0). Same as AWGN: under block fading the receiver needs no SNR (`paired_p256a_told_vs_blind_rayleigh.json`) |
| 22 Nov | A (Rayleigh) | `a256-ae-ray-told` (subhashhhh) | **Rayleigh told, gain errors** (PSNR change against the true gain, 0..20 dB): +2 dB -0.13..-0.43, +1 dB -0.04..-0.13, +0.5 dB -0.01..-0.04, -1 dB -0.03..-0.11, -2 dB -0.15..-0.52. Unlike AWGN, both directions hurt about equally; the worst case over +-1 dB is 0.13 dB. The blind counterpart (`a256-ae-ray-blind`, own account) is running (`paired_p256a_gainsweep_told_vs_true_rayleigh.json`) |
| 22 Nov | A (specialists) | `a256-p0` + `a256-p0-2` (venkat96r) | **spec0 converged at 134k** (best 130k), before the quota ran out. Spec0 minus blind 0-20, paired, AWGN: at 0 dB PSNR **+0.86 dB**, LPIPS -0.024, top-1 **+20.6 pts** (77.5% against 57%); away from 0 dB it loses (5 dB -0.41, 10 dB -1.19, 20 dB -1.69 dB PSNR). So the wide-range model gives up a lot at the bottom of its range, and the told receiver does not win it back (told = blind), so the gap is not about SNR information at the receiver. spec20 lost to blind even at 20 dB (overfit), so the cost of universality sits at low SNR. The SNR-aware transmitter (`a256-tx`, `a256-tx-2` from 28 Nov) is the test of whether SNR at the transmitter recovers it (`paired_p256a_spec0_vs_blind_awgn.json`) |
| 22 Nov | baseline | (this PC) | Channel step crashed at 64-QAM (n = 2048 is not a multiple of 6 bits); fixed in 39f51dd by placing each codeword at a random bit offset in a whole-symbol frame. QPSK and 16-QAM finished: the gap to the limit at BLER 1e-2 is 1.4-1.8 dB (QPSK) and 1.5-2.4 dB (16-QAM). The source step and the queue runner were both killed at 01:40 UTC by a console close event (no reboot or sleep); both relaunched at below-normal priority |
| 22 Nov | A (specialists) | (analysis) | **Is spec0's 0 dB lead partly undertraining?** In the blind 0-20 run, after about 58k steps the 0 dB validation loss kept falling (0.2973 at 54k to 0.2946 at 76k, PSNR@0 18.14 to 18.33) while the 10 and 20 dB losses rose (high-SNR overfit, as in spec20). The average stalled and patience stopped the run at 76k with 0 dB still improving; told shows the same pattern. Spec0's 0 dB loss is 0.2750, so most of the gap is unlikely to close, but the size of the cost of universality needs a run that keeps training. **Pushed `a256-ab-x`** (venkatrrr, input `a256-ab`): blind 0-20 from its 66k best weights, 50k more steps, no patience, evaluated on the final weights (`--eval-checkpoint final`, added to run_gen) |
| 22 Nov | A (Rayleigh) | `a256-ae-ray-told` (rest) | Estimated SNR under block fading, against blind: energy estimate = told (-0.03..-0.05 dB, top-1 within 1 pt); the M2M4 estimator is worse (-0.20..-0.50 dB, -1 to -2 pts), because its moments assume a Gaussian channel. A wrong SNR: +2 dB -> +0.06 dB at 0 dB, about 0 elsewhere; +5 dB -> -0.15 dB at 0 dB, about 0 from 10 dB. **Fading cost** (blind Rayleigh minus blind AWGN, same average SNR): PSNR -1.14 / -1.25 / -0.80 / -0.54 / -0.20 dB, top-1 -14 / -20 / -9 / -4 / -1 pts. Figure `p256a_snr_placement_{psnr,top1}.png` (AWGN and Rayleigh panels; the transmitter arm fills in after `a256-tx-2`) |
| 22 Nov | A (Rayleigh) | `a256-ae-ray-blind` (own) | **Rayleigh blind, gain errors** (PSNR change against its own true-gain result, 0 / 5 / 10 / 15 / 20 dB): +2 dB -0.54 / -0.74 / -0.94 / -1.04 / -1.12; +1 dB -0.24 to -0.47; +0.5 dB -0.11 to -0.19; -1 dB +0.12 to -0.09; -2 dB +0.01 to -1.11. **Told minus blind under the same error:** +1 dB +0.17 to +0.32 dB (top-1 +0.4 to +3.9 pts), +2 dB +0.37 to +0.68 dB (+3.5 to +7.1 pts), -1 dB -0.18 to -0.03 dB, -2 dB -0.19 to +0.58 dB. Worst case over +-1 dB: blind 0.47 dB, told 0.13 dB (`paired_p256a_gainsweep_blind_vs_true_rayleigh.json`, `paired_p256a_gain{2,1,0p5,-1,-2}_told_vs_blind_rayleigh.json`) |
| 22 Nov | **Gate A** | (decision) | **Decided early: every input is in** (the SNR-aware transmitter is a separate question). Rule 1: blind is within 0.1 dB of told with the true gain on both channels (AWGN -0.13 to -0.02 dB, Rayleigh -0.04 to -0.02 dB, told never better; energy-estimated SNR = told). Rule 2: a 1 dB gain error costs the blind receiver up to 0.41 dB (AWGN) and 0.47 dB (Rayleigh), more than SNR knowledge ever gains (0). **Claim: "the receiver needs no noise-level knowledge, given gain tracking".** Corollary for the paper: what a told receiver buys is robustness to gain error, not SNR information (worst case over +-1 dB: told 0.14 / 0.13 dB against blind 0.41 / 0.47 dB) |
| 22 Nov | baseline | (this PC) | **Separation baseline done** (`results/gen/separation`, figure `p256_separation_curves.png`, paired `paired_p256*_separation_vs_*.json`). AVIF + LDPC (n = 2048, PEG (3, dc), min-sum) + QPSK/16/64-QAM on the same 4096 symbols and 1000 images; best scheme per SNR by PSNR (a genie envelope); plus AVIF at the exact 16-QAM capacity. Low-rate QPSK codes (1/8 to 2/5) were added after the first pass failed below 2 dB; regular codes are poor at low rates (gap to the limit 2.3 to 4.5 dB, against 1.4 to 3.0 dB at rate 1/2 and above), so the practical curve is pessimistic below 0 dB and the capacity curve bounds it. **AWGN:** separation wins PSNR everywhere it decodes (practical +2.5 to +7.8 dB over the DiTs; the JSCC is capped by the SD-VAE at 24.7 dB, AVIF reaches 30 dB at 2.5 KB); the DiT wins LPIPS at every SNR (ideal separation +0.29 / +0.20 / +0.11 at -5 / 0 / 5 dB against the -5..5 DiT, +0.03 at 20 dB against the 0..20 DiT); on top-1 the low-SNR DiT beats even ideal separation at -5 and 0 dB (-9.4, -9.5 pts) and practical separation by 27 to 33 pts; from 10 dB top-1 ties (within 1 pt). **Block Rayleigh, no CSI at the transmitter:** the DiT beats separation, ideal included, on LPIPS (0.14 to 0.39) and top-1 (7 to 34 pts) at every SNR; separation wins PSNR only from 5 dB up (+1 to +4 dB). The story for the paper: the diffusion JSCC trades PSNR (VAE-bound) for perceptual quality and graceful degradation, and under fading or below 5 dB it carries more meaning per symbol than any separate design with this codec |
| 22 Nov | **B (re-check)** | `cnn256-conv-ft` (5060) | **B fails the converged re-check.** LPIPS stage capped at 36k (user; validation loss still falling about 0.15% per validation). Spread set, converged CNN minus DiT (-5..5), paired 95% intervals, at -5 / 0 / +5 dB: LPIPS +0.031 / +0.039 / +0.050 (DiT better), DINOv2 -0.008 [-0.019, +0.004] / +0.043 [+0.033, +0.052] / +0.017 [+0.009, +0.024] (CNN better at 0 and +5), top-1 -1.8 / +1.7 / -1.3 pts (all intervals include 0), PSNR +1.42 / +1.57 / +1.77 dB, FID 129.2 vs 120.4 / 59.3 vs 70.2 / 42.9 vs 50.5. Against the 40k+10k CNN of 20 Nov the CNN gained DINOv2 +0.08 / +0.07 / +0.05 and top-1 +9 / +11 / +4 pts. By the rule of 17 Nov (LPIPS gain, no clear DINOv2 or top-1 loss) B fails at 0 and +5 dB. What survives: the DiT's LPIPS lead (0.03 to 0.05) and, at -5 dB, a semantic tie with better LPIPS and FID. Next: `dit256-repa` (REPA 0.5, depth 4) on the 5060 after `dump256-conv`, the first rescue lever in the plan; `dit256-s1` stays dropped (seeds dropped). `paired_p256conv_cnn_vs_dit_spread.json` |
| 22 Nov | B (exploratory) | (protocol, written before any result) | **What does the DiT do better than the converged CNN? Exploratory, after B failed.** The pre-registered verdict (fail) stands and stays the headline. To avoid metric shopping, the full list is fixed here and every metric is reported, win or lose. Systems: DiT-S (-5..5) and the converged CNN (140k + 36k), spread set, 1000 images, -5 / 0 / +5 dB, reconstructions regenerated with the evaluation's seeds (`dump.py --keep 1`). **Full-reference, per image:** DreamSim (human-aligned similarity), DISTS (structure and texture, tolerant to texture resampling), CLIP ViT-L/14 image similarity and retrieval rank among the 1000 originals, ImageNet-1k label agreement with the original (ConvNeXt-L top-1 match, top-5 overlap, Jensen-Shannon divergence of the class distributions), CLIP zero-shot ImageNet-1k agreement, BLIP-large captions of original and reconstruction (CLIP text similarity of the two captions, and CLIP score of the reconstruction against the original's caption relative to the original's own), DETR object detection agreement (label-set F1, and box-matched F1 at IoU 0.5). **No-reference:** CLIP-IQA and BRISQUE. **Distribution:** precision, recall, density and coverage (k = 5) on DINOv2 embeddings. **Content analysis:** the per-image DiT-minus-CNN differences against the original's edge density, high-frequency energy, colourfulness and class. **Rule:** the images are split by position into an exploration half (even) and a confirmation half (odd). A metric counts as a DiT advantage only if the paired 95% interval favours the DiT on both halves at 0 and +5 dB; anything else is reported as exploratory |
| 22 Nov | B (exploratory) | `semantic_metrics.py` (5060), `semantic_report.py` | **What the DiT does better than the converged CNN** (protocol above; report `results/gen/analysis/p256-semantic/report.md`). Confirmed DiT advantages (both halves, 0 and +5 dB): **no-reference quality** (CLIP-IQA 0.83 vs 0.43 at 0 dB, DiT higher on 98% of images; BRISQUE 21.6 vs 25.4), **human-aligned similarity** (LPIPS -0.039 / -0.050; DreamSim -0.005 / -0.006, and -0.041 at -5 dB), and the classifier's probability of the original's ImageNet class (+0.08 / +0.06), which is largely a confidence effect (the CNN's blurred outputs give flatter distributions: entropy 3.35 vs 2.53 at 0 dB). Confirmed CNN advantages: PSNR, DISTS (+0.007), DINOv2 cosine, CLIP zero-shot ImageNet agreement (48.5% vs 39.4% at 0 dB), CLIP score against the original's caption. Mixed or tied at 0 / +5 dB: CLIP image cosine (DiT +0.018 / +0.006), CLIP retrieval, ImageNet top-1 agreement, top-5 overlap, caption similarity, object detection F1. **At -5 dB (outside the rule, both halves):** the DiT wins most semantic measures, including CLIP cosine +0.058, CLIP retrieval 4.7% vs 1.1%, ImageNet top-5 overlap +0.048, caption similarity +0.050 and detected-object F1 +0.043. **The pattern:** the two receivers fail differently. The CNN fails softly (BLIP calls 34% of its -5 dB outputs blurry and 26% paintings, against 2% and 9% for the DiT), so its outputs keep the gist but lose detail and photographic realism. The DiT fails confidently: a realistic image whose object can be substituted (chainsaw to umbrella, gas pumps to coolers, a cassette player to a CD player). By content, the DiT gains most on simple, low-texture scenes (parachute, golf ball, tench) and loses most on complex, many-part objects (chain saw, cassette player, French horn, gas pump; DINOv2 gain falls with edge density, r = -0.14 at 0 dB), while its LPIPS gain grows with edge density (r = +0.26 / +0.43 at 0 / +5 dB): it renders texture convincingly but not always the right object. Claim this supports: at equal semantic fidelity at moderate SNR, the diffusion receiver delivers realistic, sharp images and holds semantics better near the bottom of its SNR range; it does not carry more meaning per symbol |
| 23 Nov | A (specialists) | `a256-ab-x` (venkatrrr) | **The cost of universality is real.** Blind 0-20 continued 50k steps from its 66k weights, no patience, final weights: the 0 dB validation loss plateaus at 0.2937 after about 30k more steps (spec0: 0.2750). Against the original blind 0-20 it gains everywhere (PSNR +0.05 / +0.12 / +0.13 / +0.10 / +0.09 dB, top-1 +6.7 / +2.5 / +0.3 / +0.6 / +0.0 pts at 0..20 dB), so patience on the average loss stopped every study A run a little early; the blind-vs-told comparisons used the same rule for both arms, so the relative claims stand. **Spec0 minus the continued generalist at 0 dB: PSNR +0.81 dB, top-1 +13.9 pts** (from +0.86 dB and +20.6 pts): about a third of the top-1 gap was undertraining, almost none of the PSNR gap. `paired_p256a_spec0_vs_x50k_awgn.json` |
| 23 Nov | B (rescue) | `dit256-repa` (5060) | Running since 22 Nov 10:06 UTC (paused 11:26 to 11:59 for the metric analysis). Validation at 32k against the plain DiT at 32k: DINOv2 -5 / 0 / +5 dB 0.390 / 0.542 / 0.668 against 0.171 / 0.498 / 0.639, PSNR@0 -0.13 dB; at -5 dB it already matches the plain DiT's final value. Caution for the verdict: REPA aligns the DiT with DINOv2 features, so DINOv2-based metrics are partly circular; judge it on the non-DINO semantic metrics of `semantic_metrics.py` (CLIP zero-shot, captions, ImageNet-1k agreement) as well. About 44 h to 140k steps (Wed 25 Nov) |
| 23 Nov | B (exploratory) | (protocol, before computing) | **Can the diffusion receiver flag its own errors?** The DiT fails confidently (substituted objects). From the same received symbols it can draw several reconstructions; their disagreement needs no original. `uncertainty.py`: DiT-S (-5..5), spread set, -5 / 0 / +5 dB, the evaluation's draw plus 4 more. Uncertainty = 1 - mean pairwise cosine between the draws' embeddings, in CLIP space and in DINOv2 space. Errors of the evaluation draw: DINOv2 k-NN class differs from the label; DINOv2 retrieval misses its original; DINOv2 similarity to the original. Reported: AUROC of the uncertainty for each error, cross-space (CLIP uncertainty against DINOv2 errors) as the main number so the two are not measured with the same network; error rate after flagging the most uncertain 10 / 20 %; and whether picking the medoid of the 5 draws improves semantics over the evaluation draw. Also LPIPS-VGG, the CNN's own fine-tuning loss, for both systems, as a check on the DiT's LPIPS-Alex lead |
| 23 Nov | B (exploratory) | `uncertainty.py` (5060) | **Negative on both counts.** (1) **LPIPS-VGG** (the CNN's fine-tuning loss) favours the CNN at every SNR: DiT minus CNN +0.013 / +0.024 / +0.023 (worse), the DiT better on only 34 / 17 / 16% of images. So the DiT's LPIPS lead holds only on the AlexNet variant, which neither system trained on; with DISTS also favouring the CNN, perceptual similarity is mixed (DiT: LPIPS-Alex, DreamSim; CNN: LPIPS-VGG, DISTS), not a DiT win. (2) **Self-flagging:** the pre-specified main test, CLIP-space draw disagreement against DINOv2 errors, has no signal (AUROC 0.38 / 0.40 / 0.48 for class flips at -5 / 0 / +5 dB, below 0.5); flagging the most uncertain images does not lower the error rate, and picking the medoid draw does not help. Disagreement measured with the same network is predictive (DINOv2 draws: AUROC 0.71 / 0.79 / 0.83; draws voting on the k-NN class: 0.87 / 0.88 / 0.82), but that is partly mechanical (if the draws' classes disagree, some draw is wrong) and needs the classifier at the receiver. Robust DiT advantages that remain: realism (sharpness 1.0 vs 0.18 of the originals, CLIP-IQA, BRISQUE), held-out human-aligned DreamSim (small at 0 / +5 dB, large at -5 dB), and semantics at -5 dB. Report `results/gen/analysis/p256-semantic/uncertainty.md` |
| 23 Nov | B (rescue, interim) | `dit256-repa` 96k `best.pt` (5060) | **Interim: REPA turns test B's 0 dB semantic loss into a tie and wins every non-DINOv2 semantic metric.** Copy of `best.pt` at 96k (`results/gen/p256-repa96k`), spread set, 10 steps; converged CNN and plain DiT (140k) as in the re-check. Test set, -5 / 0 / +5 dB: DINOv2 0.403 / 0.610 / 0.714 (plain DiT 0.305 / 0.561 / 0.703, CNN 0.297 / 0.604 / 0.720); top-1 43.9 / 82.6 / 94.1 % (DiT 27.2 / 80.9 / 95.4, CNN 25.4 / 82.6 / 94.1); FID 96.8 / 60.5 / 45.7 (DiT 120.4 / 70.2 / 50.5, CNN 129.2 / 59.3 / 42.9); PSNR 16.89 / 19.34 / 20.70 (-0.16 / -0.39 / -0.32 dB against the plain DiT, about 2 dB below the CNN); LPIPS 0.364 / 0.259 / 0.209 (plain DiT better at 0 / +5 dB). Paired REPA minus CNN under the 22 Nov exploratory protocol (`results/gen/analysis/p256-repa96k/report`; the merged metrics folder files REPA under the `dit` key): DINOv2 cosine moves from CNN advantage to a tie (0 dB +0.006 [-0.004, +0.016]; +5 dB -0.006 [-0.014, +0.002]); new confirmed REPA advantages on non-DINOv2 semantics: CLIP L/14 cosine (+0.029 / +0.010 at 0 / +5 dB), ImageNet-1k top-5 overlap and JS divergence (ConvNeXt-L), caption similarity; kept: LPIPS-Alex, DreamSim, probability of the original class, CLIP-IQA, BRISQUE. The CNN keeps PSNR (gap widens to 1.95 / 2.09 dB), DISTS, CLIP score against the original's caption, and CLIP zero-shot at +5 dB (-0.059 [-0.091, -0.028]); ties: retrieval, ImageNet top-1, detection F1. The FID and non-DINOv2 gains mean REPA's effect is not only alignment to the judge. Not final: 96k of a 150k run, still improving. Training crashed twice with CUDA "unknown error" (17:06 and 17:50 UTC, WSL2; no driver reset in the Windows log; Chrome at the box console was rendering on the same GPU); now in a relaunch loop (`~/repa_semantic2.sh`). |
| 24 Nov | B (rescue, final) | `dit256-repa` (5060) | **Final: REPA wins or ties every meaning metric against the converged CNN, and now also FID.** Converged by the patience rule at 122k (val loss still creeping down by < 0.2% per 10 validations; best = last). 4 CUDA crashes from host RAM pressure, relaunch loop. Test set (spread, 10 steps), REPA / CNN at -5 / 0 / +5 dB: DINOv2 0.431/0.297, 0.635/0.604, 0.728/0.720; retrieval top-1 48.4/25.4, 86.0/82.6, 95.9/94.1%; FID 88.5/129.2, 53.7/59.3, 43.8/42.9; PSNR 16.89/18.46, 19.38/21.29, 20.70/22.79 dB (-1.6 to -2.1 dB, ~0.35 dB below the plain DiT). Paired report (`results/gen/analysis/p256-repa-final/report`, both-halves rule): REPA advantage confirmed on CLIP-IQA, BRISQUE, DreamSim, LPIPS-Alex, CLIP cosine, ImageNet top-5 / JS / p(orig class), caption similarity; CNN advantage on PSNR, DISTS and the CLIP score against the original's caption; DINOv2, CLIP zero-shot, top-1 agreement, detections tie at 0/+5 and favour REPA at -5. Distribution: REPA has higher precision and recall (DINOv2 and CLIP features) at every SNR. Per-image meaning score (z-mean of CLIP, caption, DreamSim, ImageNet JS; not DINOv2): REPA better on 89 / 73 / 66% of images. Worst classes for REPA: gas pump, chain saw, cassette player (confident substitutions); best: tench, parachute, springer. Not redone with REPA: LPIPS-VGG, self-flagging. Deck: `SemCom_Progress_v12_final.pptx` (REPA the only generative receiver; examples by `pick_examples.py`) |
| 26 Nov | plan | (decision) | **Plan from 26 Nov to the 11 Dec freeze written** (top of this document), after the progress meeting and a literature review (`reports/DiT JSCC path to paper.md`, local). Decisions: no 512 px (the 256 px SD-VAE ceiling, DINOv2 0.96 / 25.9 dB, is far above the receiver's 0.728 / 20.7 dB at +5 dB, and neither a 512 px DiT nor a 512 px CNN baseline converges before 11 Dec); more data (ImageNet-100 pretraining, E-P5) is the main performance bet; recipe levers screened at 60k steps on Kaggle (E-P3); analyses for claims C3 to C5 run first on the 24 Nov REPA model. The 5060's code was synced to the repository (backup `~/semcom_code_backup_20261126.tgz` on the box); `run_gen.py` gained `--lr` |
| 26 Nov | E-P2 | `dit256-repa-anneal` (5060) | Started 09:16 UTC: the converged REPA weights, 30k steps, peak LR 5e-5, cosine to 0, then the three test sets. 2.1 it/s, about 4 h of training |
| 26 Nov | E-P2 | `dit256-repa-anneal` (5060) | **Stopped at step 4,800 (about 10:00 UTC), no error in the log** (tmux server gone, box not rebooted). Checkpoint at about 4k steps kept; resume with `python scripts/gen_5060.py dit256-repa-anneal`. **User: the 5060 stays paused until they give the go-ahead.** |
| 26 Nov | E-A7, E-P3 | `e-a7-analog` (own), `e-p3-base` (venkatrrr), `e-p3-repa1` (subhashhhh), at 6a7098b | Pushed to use this week's remaining quota (own 10.2 h, venkatrrr 7.4 h, subhashhhh 6.2 h; venkat96r 0): the analog REPA receiver to convergence (session 1, `--max-hours 8.5`, `results/gen/p256-analog`); the 60k-step 16-QAM REPA screen baseline and its REPA-weight-1.0 variant (`--tag s60k`, `results/gen/p256-screen`, max-hours 5.8 and 4.8). Second sessions `-2` queued in `.kaggle_jobs/queue.json` for 28 Nov 00:05 UTC on the same accounts |
