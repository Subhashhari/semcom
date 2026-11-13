# Quantised / digital-modulation deep JSCC for images, and the perceptual/generative subset (2021 to Nov 2026)

Verification labels used throughout:
- **[FULL TEXT]**: claims checked against fetched paper text (arXiv HTML). Quotes are verbatim as returned by the fetch tool, with section names.
- **[ABSTRACT ONLY, UNVERIFIED]**: only the arXiv abstract page was read.
- **[SNIPPET ONLY, UNVERIFIED]**: only search-engine snippets were seen; treat as leads.

Notes were compiled 2026-11-08.

## Q1. Main quantisation techniques for digital deep JSCC, and which work best per their own comparisons

### Takeaway
Digital deep JSCC for images falls into a few families:
1. **Deterministic symbol quantisation onto a fixed M-QAM grid.** The soft-to-hard softmax of DeepJSCC-Q, and the noise-then-STE scheme of MDJCM.
2. **Probabilistic constellation-symbol sampling.** JCM uses a VAE with Gumbel-Softmax.
3. **Vector quantisation with codeword indices mapped to bits or QAM.** VQ-DeepSC, MOC-RVQ, uJSCC, VQ-DSC-R and SemQAM.
4. **Bit-level schemes with an explicit channel code.** D²-JSCC, and bit-quantisation schemes such as Park et al. 2024.

Head-to-head evidence is thin and comes only from authors comparing against their own baselines. Two such results: MDJCM reports "STE holds a higher performance than JCM at most modulation orders". The Zhejiang perspective's CIFAR-10 case study reports that fine-tuning with irregular (learned) constellations beat all other digital schemes. Both papers say higher modulation order narrows the gap to analog.

### Map of digital-modulation deep JSCC for images (and close relatives)

| Work | arXiv / venue, date | How symbols are made digital | Constellation order | SNR adaptivity | Metrics | Diffusion? | Verification |
|---|---|---|---|---|---|---|---|
| DeepJSCC-Q, conference version (Tung, Kurka, Jankowski, Gündüz) | arXiv 2111.13042, 25 Nov 2021 | Soft-to-hard: softmax-weighted sum of constellation points, hardness σq=100 | QPSK, 16-QAM, 64-QAM; also 1024 and 4096 | **SNR-conditioned single model**: trained on SNR ~ U[0,10] dB with an attention module fed the SNR; mismatch study in Fig. 3 | PSNR, SSIM (MSE or 1−SSIM loss) | No | [FULL TEXT] |
| DeepJSCC-Q, journal version | arXiv 2206.08100, IEEE JSAIT 2022 | Same family; per the Imperial abstract, a large enough constellation comes close to unconstrained DeepJSCC and beats BPG+LDPC with no cliff effect | — | — | — | No | [ABSTRACT/SNIPPET ONLY]; the PDF would not parse |
| JCM, VAE joint coding-modulation (Bo, Duan, Shao, Tao) | arXiv 2310.06690, IEEE TCOM 2024; BPSK precursor arXiv 2208.05704 | Probabilistic: the encoder outputs probabilities over constellation points; Gumbel-Softmax relaxation; "matching loss" | Several orders | Not determined | Not determined | No | [SNIPPET ONLY] (mechanism confirmed second-hand in MDJCM's full text) |
| D²-JSCC (Jianhao Huang, Kai Yuan, Chuan Huang, Kaibin Huang) | arXiv 2403.07338, 2024 (IEEE conference listing) | Deep source coding with an adaptive density model, then a real channel code, jointly optimised | Not determined | Not determined | Claims no cliff or levelling-off effect | No | [SNIPPET ONLY] |
| uJSCC, modulation-agnostic (Huh, Seo, Choi) | arXiv 2405.10749, May 2024 (v2 Feb 2025) | Trained VQ codebooks; one encoder/decoder; BN layers selected by modulation order | Several orders, chosen by SNR boundaries (a snippet says BPSK to 256-QAM) | **SNR-switched modulation order** | "model complexity, communication efficiency, and task effectiveness" | No | [ABSTRACT ONLY, UNVERIFIED] |
| MDJCM (Guangyi Zhang, Pujing Yang, Yunlong Cai, Qiyu Hu, Guanding Yu) | arXiv 2406.05437, IEEE TCOM vol. 73 no. 6, Jun 2025 | Modulation treated as constrained quantisation: scaled uniform noise in training, then STE fine-tuning of the receiver | 4 to 1024-QAM in one model (conditional response network) | **SNR-conditioned** through an SNR conditional module; SNR sampled uniformly in training | PSNR main; no LPIPS | No | [FULL TEXT] |
| MOC-RVQ (Zhou, Sun, Chen, Xu, … Cui, Zhang) | arXiv 2401.01272, GLOBECOM 2024 | Multi-head octonary codebook with RVQ; "modulation favors few discrete states"; Swin-based noise-reduction block in stage 2 used as a "semantic knowledge base for generative feature restoration" | Octonary heads (exact QAM mapping not stated) | Not stated | Not stated (abstract compares with BPG/JPEG) | No | [ABSTRACT ONLY, UNVERIFIED] |
| Perspective: "Towards Compatible Semantic Communication: A Perspective on Digital Coding and Modulation" (G. Zhang, K. Zhou, Y. Cai, Q. Hu, G. Yu) | arXiv 2412.18876, 25 Dec 2024 | Taxonomy (see below) | Case study on CIFAR-10 | — | — | No | [FULL TEXT] |
| VQ-DeepISC | arXiv 2508.03740, 2025 | VQ-assisted encoder, QPSK over OFDM | QPSK | "Channel adaptive" (title) | Not determined | No | [SNIPPET ONLY] |
| Rate-adaptive multi-stage VQ SemCom | arXiv 2510.02646, Oct 2025 | Quantise to bits, then QAM | QAM | Rate-adaptive | Not determined | No | [SNIPPET ONLY] |
| ESC-MVQ | arXiv 2504.11709, 2025 | Multi-codebook VQ | Not determined | Not determined | Not determined | No | [SNIPPET ONLY] |
| VQ-DSC-R (Jianqiao Chen, Nan Ma, Xiaodong Xu, … Ping Zhang) | arXiv 2602.15045, 5 Feb 2026 | VQ with a shared codebook, index transmission over OFDM, KNN-adjusted quantisation | Not stated | Attention-based adaptation to channel noise | Not stated | **Yes, but for CSI refinement**: "a conditional diffusion model (CDM) to refine channel state information" (abstract). It is not an image prior. | [ABSTRACT ONLY, UNVERIFIED] |
| SemQAM (Hao Cao, Jingkai Ying, Yinhuan Huang, Zhijin Qin, Yuan Shen, Tsinghua) | arXiv 2610.29643, 21 Oct 2026 | RVQ (1024-codeword codebooks); each index maps to a Gray-labelled 1024-QAM symbol; no channel code; codebook updated using the detected symbol | 1024-QAM only | **Per-operating-point codebooks** (10, 16 and 24 dB plus noiseless, shared transforms). SNR must be known; no mismatch tests | LPIPS, DISTS, FID; not PSNR or DINO | **No.** VQ, autoregressive learned codec, LPIPS + PatchGAN | [FULL TEXT] |
| Joint semantic-channel coding and modulation for token communications (Ying, Qin, Feng, Wang, Tao) | arXiv 2511.15699, 19 Nov 2025 | Token encoder outputs "modulated tokens (constellation points)" | Not determined | Channel adapter | PSNR | No; point clouds, not images | [SNIPPET ONLY] |

### Cited findings
- **DeepJSCC-Q quantiser (conference version).** "each quantized symbol is generated as the softmax weighted sum of the symbols" of the constellation (Sec. III-A). "The softmax hardness assignment parameter is chosen as σq=100" (Sec. IV). Orders tested were QPSK, 16-QAM and 64-QAM, plus "very high order modulations M=1024,4096". The paper reports no experimental comparison with other quantisers; benchmarks are BPG+LDPC and continuous DeepJSCC. [FULL TEXT] — [arXiv 2111.13042](https://arxiv.org/html/2111.13042)
- **DeepJSCC-Q journal claims.** A large enough constellation brings performance close to unconstrained DeepJSCC, and the scheme beats separation-based BPG+LDPC without the cliff effect. [ABSTRACT/SNIPPET ONLY] — [Imperial PDF](https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/ipc-lab/deepjsccq.pdf); [arXiv 2206.08100](https://arxiv.org/pdf/2206.08100)
- **MDJCM training.** "we approximate the modulation by using scaled uniform noise" (Sec. III-A2), followed by STE-based fine-tuning of the receiver with the transmitter frozen (Sec. III-C). Constellation set "M = [4, 16, 64, 256, 1024]" (Sec. V-A1). [FULL TEXT] — [arXiv 2406.05437](https://arxiv.org/html/2406.05437v2)
- **MDJCM's critique of other quantisers.** STE and soft-to-hard annealing "encounter training issues such as biased or unstable gradients" (Sec. III-C). Fixed-order schemes are a weakness: "most current methods only support a fixed type of modulation, necessitating model updates when the modulation order changes" (Sec. I-B). [FULL TEXT] — [arXiv 2406.05437](https://arxiv.org/html/2406.05437v2)
- **MDJCM's comparison results.** "STE holds a higher performance than JCM at most modulation orders" (Sec. V-C). BESC (Park et al. 2024) "fails to achieve high performance in the whole SNR range". Liu et al. 2024 "does not consider the modulation process in the training phase". [FULL TEXT] — [arXiv 2406.05437](https://arxiv.org/html/2406.05437v2)
- **JCM mechanism, as described by MDJCM.** "the encoder outputs the probabilities associated with each constellation point"; "the Gumbel-Softmax strategy is employed as a continuous approximation of sampling" (Sec. I-A). [FULL TEXT of MDJCM] — [arXiv 2406.05437](https://arxiv.org/html/2406.05437v2)
- **JCM's own claim.** It outperforms quantisation-based digital semantic coding-modulation methods, and its gap to analog shrinks as modulation order grows. [SNIPPET ONLY] — [arXiv 2310.06690](https://arxiv.org/pdf/2310.06690)
- **Perspective taxonomy (2412.18876).**
  - Probabilistic methods, which use score-function, reparameterisation, Gumbel-Softmax or multisample bounds.
  - Deterministic methods: scalar quantisation, symbol quantisation onto M-QAM, and vector quantisation.
  - Differentiable approximations: NN approximation, STE, soft-to-hard annealing, and additive uniform noise.
  - Constellation designs: uniform, non-uniform, and irregular (K-means or GMM).
  - The paper does not cover BICM or channel-code integration.
  - [FULL TEXT] — [arXiv 2412.18876](https://arxiv.org/html/2412.18876v1)
- **Perspective quotes.**
  - On Gumbel-Softmax: "they may lead to performance degradation due to mismatches between training and testing" (Sec. III-A).
  - "The performance of digital SC improves with the modulation orders, and thus higher modulation order is generally preferable." (Sec. IV-C)
  - "The scheme fine-tuned with irregular constellations outperforms all other digital schemes" (Sec. IV-C).
  - "the VQ-based methods have proven to be the dominant techniques" at low bit rates (Sec. V-A3).
  - [FULL TEXT] — [arXiv 2412.18876](https://arxiv.org/html/2412.18876v1)
- **SemQAM on digital versus continuous.** The finite-constellation "constraint reduces the available signal space and can lower the achievable performance relative to continuous JSCC" (Sec. 2). "DeepJSCC-Q constrains channel symbols to finite constellations" (Sec. 5). [FULL TEXT] — [arXiv 2610.29643](https://arxiv.org/html/2610.29643v1)
- **D²-JSCC.** Combines deep source coding with an adaptive density model and a channel code, jointly optimised, and is reported "free from the undesirable cliff effect and leveling-off effect". [SNIPPET ONLY] — [arXiv 2403.07338](https://arxiv.org/pdf/2403.07338)
- **PAPR in OFDM JSCC (analog symbols).**
  - Clipping during training "significantly reduce[s] the PAPR with graceful performance degradation" ([arXiv 2109.05194](https://arxiv.org/pdf/2109.05194)).
  - Among distortion-type PAPR techniques, "clipping with retraining achieves the best performance" ([arXiv 2309.11803](https://arxiv.org/pdf/2309.11803)).
  - Shao and Gündüz report a trade-off between low PAPR and reconstruction quality ([arXiv 2208.08342](https://www.arxiv.org/pdf/2208.08342)).
  - DeepStream: "direct integration of DeepJSCC and OFDM significantly increases the PAPR" ([arXiv 2509.05971](https://arxiv.org/pdf/2509.05971)).
  - All four are [SNIPPET ONLY, UNVERIFIED], and all are analog-symbol works.

### Inferences
- No paper found runs a controlled, same-architecture comparison of the main quantisers (soft-to-hard, uniform-noise then STE, Gumbel/probabilistic, VQ-index-to-QAM). The "which is best" evidence is author-reported and confounded by architecture differences. MDJCM (noise then STE) beating JCM (Gumbel), and the perspective paper's preference for learned irregular constellations, are the only explicit rankings found.
- DeepJSCC-Q's soft-to-hard method, which the repo uses, is the original Gündüz-group approach. MDJCM later argues soft-to-hard annealing has "biased or unstable gradients". A noise-plus-STE variant is a reasonable ablation to include.
- The ADJSCC-Q style of SNR-conditioned digital JSCC has direct prior art. The DeepJSCC-Q conference paper already trains one SNR-attention model over 0 to 10 dB, and MDJCM has an SNR conditional module.

### Gaps
- The DeepJSCC-Q journal (2206.08100) full text could not be parsed: the PDF was binary and no HTML version exists. Its extra experiments, such as additional constellations and comparisons, are not quote-verified.
- These were not read in full text: JCM (2310.06690), D²-JSCC (2403.07338), uJSCC (2405.10749), MOC-RVQ (2401.01272), VQ-DeepISC, ESC-MVQ, and Wang et al. 2022 "constellation design for deep JSCC".
- No paper was found that uses BICM with soft-demapper LLRs as the interface of a learned image JSCC decoder (query: "modulation-agnostic deep joint source-channel coding image digital BICM soft demapping LLR").
- No work was found on probabilistic shaping specifically for image JSCC, beyond JCM's learned symbol probabilities.

## Q2. Which digital JSCC works use GAN or diffusion priors at the receiver and report LPIPS/FID/DINO? How are symbols made digital?

### Takeaway
After genuine search, the only verified image work that combines a finite QAM constellation with a generatively trained receiver and perceptual metrics is **SemQAM** (arXiv 2610.29643, 21 Oct 2026). It uses RVQ indices mapped to Gray-labelled 1024-QAM, LPIPS + PatchGAN training, and reports LPIPS, DISTS and FID. **It uses no diffusion model and reports no DINO.**

The diffusion-receiver JSCC works found all transmit continuous (analog) symbols: DiT-JSCC, DiffJSCC, SING/CommIN, CDM-JSCC and GenerativeJSCC/InverseJSCC (StyleGAN). VQ-DSC-R is the one digital work found with a diffusion model, and it uses that diffusion model for CSI refinement, not as an image prior.

### Cited findings
- **SemQAM digitisation.** Latents are quantised with RVQ, 1024 codewords per codebook (16-dim main latent, 8-dim hyperlatent). Each index maps to a Gray-labelled 1024-QAM symbol with no channel code in between.
- **SemQAM training loss.** λrec‖x−x̂‖² + λper·LPIPS + λadv·GAN (PatchGAN), plus a receiver codebook loss on the detected symbol and a transmitter commitment loss (Sec. 3.3, Eq. 6).
- **SemQAM receiver.** An EF-LIC-based VQ autoregressive codec, "not a diffusion model".
- **SemQAM metrics and baselines.** Metrics are LPIPS, DISTS, FID and BD-rate; DINO and PSNR are not reported. Baselines are NTSCC, NTSCC+, MDJCM-A and Quad-JSCC, plus separated generative codecs with LDPC and QAM: MS-ILLM, OSCAR, OneDC, DiT-IC, EF-LIC and CoD-Lite.
- [FULL TEXT] for all four SemQAM items above — [arXiv 2610.29643](https://arxiv.org/html/2610.29643v1)
- **SemQAM quotes.**
  - "Learned image JSCC do not use digital modulation, but directly transmit continuous symbols over the wireless channel" (Sec. 5).
  - "most evaluated JSCC baselines are optimized primarily for distortion rather than perceptual quality" (Sec. 4.2).
  - "SemQAM nevertheless achieves stronger perceptual performance while operating directly with finite 1024-QAM" (Sec. 4.2).
  - Quad-JSCC "does not publicly release the corresponding perceptual-quality model", so the authors obtained LPIPS-optimised weights from its authors. This implies a perceptual Quad-JSCC variant exists.
  - [FULL TEXT] — [arXiv 2610.29643](https://arxiv.org/html/2610.29643v1)
- **DiT-JSCC (Tan, Dai, Wang, Lu, Shao, Niu, W. Zhang, P. Zhang).** arXiv 2601.03112, v1 Jan 2026, v2 18 Jun 2026. The encoder outputs "a vector of complex-valued channel input symbols", transmitted "After power normalization which ensures an average power constraint". These are analog, with no quantisation.
  - Digital modulation appears only in the baselines: "5G LDPC … with 4-ary quadrature amplitude modulation (4QAM)".
  - SNR handling: "we train separate models for each SNR" (per-SNR models).
  - Metrics: LPIPS, DISTS, CLIP, DreamSim, DINOv2 and FID.
  - Its future work does not mention digital or QAM transmission.
  - [FULL TEXT] — [arXiv 2601.03112](https://arxiv.org/html/2601.03112)
- **GenerativeJSCC / InverseJSCC (Erdemir, Tung, Dragotti, Gündüz).** arXiv 2211.13772, IEEE JSAC 41(8) 2023. Uses a StyleGAN prior with an MSE + LPIPS objective. The analog channel was not quote-verified. [SNIPPET ONLY] — [arXiv 2211.13772](https://arxiv.org/pdf/2211.13772)
- **DiffJSCC.** arXiv 2404.17736, 2024. Uses Stable Diffusion refinement of a deep JSCC output with continuous symbols. [SNIPPET ONLY] — [arXiv 2404.17736](https://arxiv.org/html/2404.17736)
- **SING and CommIN.** SING (arXiv 2503.12484) and CommIN (arXiv 2310.01130) use diffusion priors at the receiver as inverse-problem solvers over DeepJSCC (analog). [SNIPPET ONLY] — [SING](https://arxiv.org/html/2503.12484); [CommIN](https://arxiv.org/pdf/2310.01130)
- **SC-CDM.** arXiv 2410.02121 pairs a Swin JSCC encoder with a compact diffusion decoder. A snippet says experiments use "a modulation order of 4"; whether its symbols are actually quantised to QPSK is unverified. **This is the main uncertain item to check.** [SNIPPET ONLY, UNVERIFIED] — [arXiv 2410.02121](https://arxiv.org/html/2410.02121)
- **VQ-DSC-R.** Digital (VQ index transmission over OFDM) with "a conditional diffusion model (CDM) to refine channel state information". The diffusion model is not used as an image prior. [ABSTRACT ONLY, UNVERIFIED] — [arXiv 2602.15045](https://arxiv.org/abs/2602.15045)
- **MOC-RVQ.** A "Digital Generative Semantic Communication" scheme whose "generative" component is a Swin noise-reduction block acting as a codebook knowledge base. It is not a GAN or diffusion model, and the abstract names no perceptual metrics. [ABSTRACT ONLY, UNVERIFIED] — [arXiv 2401.01272](https://arxiv.org/abs/2401.01272)
- **Pei et al., latent-diffusion low-latency SemCom.** IEEE TWC 2025, arXiv 2406.06644. Reports MS-SSIM and LPIPS; its abstract does not mention quantisation or QAM. [ABSTRACT ONLY, UNVERIFIED] — [arXiv 2406.06644](https://arxiv.org/abs/2406.06644)
- **Lokumarambage et al., VQ end-to-end generative SemCom.** IEEE TMLCN 2025. Reports LPIPS and FID (5.496 on COCO). The modulation details were not checked. [SNIPPET ONLY] — [Strathprints PDF](https://strathprints.strath.ac.uk/94180/7/Lokumarambage-etal-IEEE-TMLCN-2025-Generative-AI-based-vector-quantized-end-to-end-semantic-communication-system.pdf)
- **PLSCC (arXiv 2205.13120).** Trains JSCC with LPIPS plus an adversarial loss; analog. [SNIPPET ONLY] — [arXiv 2205.13120](https://arxiv.org/pdf/2205.13120)

### Inferences
- **Novelty boundary.** "Digital (finite-QAM) image JSCC with a *diffusion* receiver evaluated on perceptual/semantic metrics (LPIPS/FID/DINO)" appears to be open as of Nov 2026, based on the searches listed below. However, "digital QAM JSCC with a *generative (GAN/LPIPS)* receiver and LPIPS/FID" is **taken by SemQAM**, which is three weeks old as of these notes.
- **Contrasts with SemQAM.** A quantised DiT-JSCC at 16-QAM would differ from SemQAM in four ways: diffusion versus GAN-trained VQ-AR decoder; soft-to-hard symbol quantisation of analog-style features versus VQ-index-to-QAM; low versus high order (16 versus 1024); and DINO semantic metrics, which SemQAM does not report.
- **What a novelty claim must address.** SemQAM, the possible perceptual Quad-JSCC variant, and any digital SC-CDM setting should be cited and argued against.

### Gaps
- SC-CDM (2410.02121) full text was not read. It must be checked whether "modulation order of 4" means real QPSK quantisation, which would make it a prior digital-plus-diffusion work.
- The perceptual Quad-JSCC model mentioned by SemQAM was not located. "Quad-DeepSC" (arXiv 2512.05395, Huang and Qin, 5 Dec 2025) uses continuous symbols and reports PSNR and MS-SSIM only [FULL TEXT] ([arXiv 2512.05395](https://arxiv.org/html/2512.05395v1)). The LPIPS-optimised weights SemQAM obtained appear unpublished.
- No digital JSCC paper reporting DINO metrics was found.
- **Queries run for this question** (all searches, not exhaustive):
  - "digital semantic communication generative adversarial JSCC QAM perceptual LPIPS image"
  - "diffusion model receiver digital modulation deep JSCC image QAM arXiv"
  - "digital generative semantic communication diffusion quantized constellation image LPIPS FID 2025"
  - "diffusion semantic communication 'digital modulation' vector quantization QAM receiver diffusion prior image arXiv 2025 2026"
  - "perceptual digital deep JSCC adversarial loss LPIPS quantized constellation DeepJSCC-Q OR digital JSCC generative"
  - "Quad-JSCC generative reconstruction digital image transmission"
  - "'Quad-JSCC' quadtree JSCC"
- Google Scholar "cited by" for DeepJSCC-Q was not accessible, so citation tracing was done through related-work sections (SemQAM, MDJCM, the perspective paper) instead.

## Q3. Which digital JSCC works are SNR-adaptive, and which test SNR mismatch or blind operation?

### Takeaway
SNR-conditioned digital JSCC is established:
- DeepJSCC-Q (2021) has an SNR-attention model trained over 0 to 10 dB.
- MDJCM (2024/25) has an SNR conditional module.
- uJSCC switches modulation order by SNR boundaries.
- SemQAM uses per-operating-point codebooks, so the SNR must be known.

Only DeepJSCC-Q reports a mismatch study, in its Fig. 3 with a fixed SNR estimate. No digital JSCC paper found evaluates truly blind (no-CSI) operation, and none of the generative or diffusion ones test SNR mismatch.

### Cited findings
- **DeepJSCC-Q (conference version), single model.** It contrasts with earlier work's "need to train multiple networks, one for each channel condition". Its single model is trained with SNR uniform on 0 to 10 dB and an attention module given the SNR.
- **DeepJSCC-Q mismatch result.** Fig. 3 fixes the noise estimate and varies the true noise. The paper attributes weaker low-SNR 16-QAM results to "the SNREst used for the 16-QAM model being much greater than the channel SNR in that region".
- [FULL TEXT] for both DeepJSCC-Q items — [arXiv 2111.13042](https://arxiv.org/html/2111.13042)
- **MDJCM.** Uses an SNR conditional module that "takes SNR as input" (Sec. IV-C), with SNR sampled uniformly in training. No dedicated train/test mismatch experiment is reported. There is an internal inconsistency: Sec. V-A1 says "we train the model at an SNR of 10 dB". [FULL TEXT] — [arXiv 2406.05437](https://arxiv.org/html/2406.05437v2)
- **SemQAM.** Trained at 10, 16 and 24 dB with codebooks chosen per operating point. "All ablation configurations are trained only on the four 10 dB operating points." Baselines are incomplete at other SNRs: "model availability at other SNRs is incomplete across the compared methods" (Sec. C.5). No mismatch tests. [FULL TEXT] — [arXiv 2610.29643](https://arxiv.org/html/2610.29643v1)
- **uJSCC.** Modulation order is selected "according to signal-to-noise ratio (SNR) boundaries". [ABSTRACT ONLY, UNVERIFIED] — [arXiv 2405.10749](https://arxiv.org/abs/2405.10749)
- **DiT-JSCC (analog).** Uses per-SNR models. For the Rayleigh baseline it uses "the nearest available 1 dB and 7 dB backbones", which is a backbone mismatch rather than a formal test. [FULL TEXT] — [arXiv 2601.03112](https://arxiv.org/html/2601.03112)
- **Perspective paper.** "relying on a fixed coding-modulation scheme can result in performance degradation as channel conditions fluctuate" (Sec. V-B3). [FULL TEXT] — [arXiv 2412.18876](https://arxiv.org/html/2412.18876v1)

### Inferences
- **Possible gap.** A systematic SNR-mismatch or blind (no SNR side-information) evaluation of a digital, generative or diffusion JSCC receiver appears to be missing. This fits the repo's planned "SNR adaptivity" direction: per-SNR versus SNR-conditioned versus blind, with mismatch curves for a 16-QAM DiT-JSCC.
- The repo's ADJSCC-Q is not novel by itself. DeepJSCC-Q's conference version already has SNR attention plus quantisation.

### Gaps
- It is unverified whether the DeepJSCC-Q journal adds blind or mismatch experiments, because the PDF was unparseable.
- VQ-DeepISC (channel adaptive) and VQ-DSC-R (attention-based noise adaptation) were not checked for mismatch tests.

## Q4. Does any work argue digital modulation is incompatible with, or harmful for, generative/diffusion receivers, or the opposite?

### Takeaway
No source found argues that digital modulation is *incompatible* with generative or diffusion receivers.

SemQAM argues the opposite in practice: generative quality is achievable "directly with finite 1024-QAM". It concedes only the general point that a finite constellation "can lower the achievable performance relative to continuous JSCC".

The perspective paper lists GANs, VAEs and diffusion models only as compression tools and does not discuss them as digital receivers.

### Cited findings
- **SemQAM.**
  - "this constraint reduces the available signal space and can lower the achievable performance relative to continuous JSCC" (Sec. 2).
  - It "achieves stronger perceptual performance while operating directly with finite 1024-QAM" (Sec. 4.2).
  - It "continues to provide competitive perceptual reconstruction through the same standard QAM interface" at 16 and 24 dB (appendix).
  - The fetch summary noted that "The paper does not say digital modulation is harmful to generative receivers."
  - [FULL TEXT] — [arXiv 2610.29643](https://arxiv.org/html/2610.29643v1)
- **Perspective paper.** "generative models, such as generative adversarial networks (GANs), VAEs, and diffusion models" are listed as compression tools (Sec. V-A3). On multiround transmission: "Generally, digital systems demonstrate stronger resistance to interference compared to analog systems" (Sec. V-A1). [FULL TEXT] — [arXiv 2412.18876](https://arxiv.org/html/2412.18876v1)
- **DeepJSCC-Q (conference version).** Motivates digital symbols by hardware: "where modulators are hard-coded, limiting the output space available for the encoder" (Sec. I). [FULL TEXT] — [arXiv 2111.13042](https://arxiv.org/html/2111.13042)
- **MDJCM.** Analog transmission "poses challenges for power amplifiers with non-ideal characteristics" (Sec. I-A). [FULL TEXT] — [arXiv 2406.05437](https://arxiv.org/html/2406.05437v2)

### Inferences
- **An open empirical question.** For a diffusion receiver conditioned on received symbols, quantisation noise and the residual error after hard decisions both change the conditioning distribution. No paper found measures this. "Does 16-QAM quantisation hurt diffusion-receiver perceptual/semantic quality relative to analog, and how much across SNR?" is an unanswered question that a quantised DiT-JSCC could answer.
- **Prior expectation.** The papers report a shrinking digital-to-analog gap as modulation order grows (DeepJSCC-Q, JCM, the perspective paper). So a gap should be expected at 16-QAM, especially at high SNR where the constellation caps the information rate.

### Gaps
- No direct quote was found claiming digital modulation is harmful to diffusion priors, nor any claim that it helps them, for example as regularisation. The absence is itself a finding, bounded by the queries listed in Q2.

## Q5. Standardisation-facing discussion (3GPP, O-RAN, 6G white papers) on learned JSCC with analog versus digital symbols, PAPR, and 5G NR PHY compatibility

### Takeaway
The standardisation-facing material found is the 6G SemCom roadmap (arXiv 2509.12758, BUPT, Ping Zhang et al.). It says SemCom is "actively discussed within the 3GPP working groups" but "still lacks the standard foundations". Its compatibility design keeps the existing protocol stack, and its case study uses "JSCC+QPSK". It does not discuss PAPR or analog versus digital explicitly.

Research papers repeatedly cite NR compatibility as the motivation for going digital: 5G LDPC plus QAM baselines, and MCS tables from TS 38.214. Nothing found indicates any 3GPP decision on analog-symbol JSCC.

### Cited findings
- **6G SemCom roadmap (2509.12758).**
  - SemCom is "actively discussed within the 3rd generation partnership project (3GPP) working groups" (Abstract).
  - "SemCom still lacks the standard foundations needed for practical deployment in native-AI 6G networks" (Sec. I).
  - The design retains "the existing mobile communication protocol stack structure" (Sec. IV).
  - "semantic-aware channel coding and modulation techniques dynamically adjust coding rates and modulation schemes" (Sec. IV-A).
  - "'JSCC+QPSK' applies deep JSCC for compression and quadrature phase shift keying (QPSK) for modulation" (Fig. 4 caption).
  - The case study uses the "3GPP TR 38.901 clustered delay line (CDL) model" (Sec. V).
  - On ITU: "In SG13, SemCom has been identified as a key enabler for future networks beyond IMT-2020" (Sec. I-B).
  - "the standard should specify baseline requirements for error-tolerance settings" (Sec. III-B1).
  - No PAPR discussion; ETSI is not mentioned.
  - [FULL TEXT] — [arXiv 2509.12758](https://arxiv.org/html/2509.12758)
- **Perspective paper.** Analog designs show "low compatibility with existing digital communication system regarding hardware and protocols" (Intro). "To ensure compatibility with existing digital systems, the coding and modulation steps can be technically separated." No PAPR or 3GPP specifics. [FULL TEXT] — [arXiv 2412.18876](https://arxiv.org/html/2412.18876v1)
- **Quad-DeepSC.** Its digital baseline picks "the optimal modulation order and channel code rate … based on the MCS Table 5.1.3.1-1 for the PDSCH". It claims to be the first DeepSC to beat VTM with 3GPP-optimal MCS, while its own symbols are continuous. [FULL TEXT] — [arXiv 2512.05395](https://arxiv.org/html/2512.05395v1)
- **SemQAM.** Mentions 5G only for its separated baselines (3GPP TS 38.214 MCS tables, 5G-LENA BLER curves). It leaves OFDM and fading to future work (Sec. 6). [FULL TEXT] — [arXiv 2610.29643](https://arxiv.org/html/2610.29643v1)
- **IMT-2030 proposals.** A survey table lists company proposals for China's IMT-2030 SemCom work (OPPO and Xiaomi on JSCC and CSI feedback; SJTU on "digital SemCom … bit-level JSCC"). A secondary blog claims 3GPP has no dedicated SemCom track and that formal discussion may start around 2027 to 2028. [SNIPPET ONLY, UNVERIFIED; the blog is low-quality] — [arXiv 2512.20917](https://arxiv.org/pdf/2512.20917); [dev.to](https://dev.to/linou518/beyond-bits-how-semantic-communication-is-rewriting-the-foundation-of-6g-3oep)
- **PAPR literature.** Only analog OFDM-JSCC works were found (2109.05194, 2208.08342, 2309.11803, 2509.05971; see Q1). DeepJSCC-Q's conference version reports no PAPR. [mixed; see Q1 labels]

### Inferences
- The case for "16-QAM DiT-JSCC" can lean on two things: the stated NR-compatibility motivation (fixed modulators, existing protocol stacks, QPSK/QAM interfaces) and the absence of PAPR analysis for digital generative JSCC. Measured PAPR of quantised versus analog symbols, single-carrier or OFDM, would be a cheap and uncontested addition.

### Gaps
- No 3GPP RAN1 contributions, O-RAN documents, or operator or vendor 6G white papers (Ericsson, Nokia, Qualcomm) were found that explicitly discuss analog versus digital learned-JSCC symbols. The one Qualcomm hit was a 2021-era generic 6G deck with no JSCC content.
- Query run: "3GPP 6G semantic communication deep JSCC analog vs digital symbols standardization compatibility 5G NR".
- 3GPP Rel-20 or Rel-21 study-item text was not checked.
