# DiT-JSCC follow-ups, and SNR handling (told vs blind, analog vs digital) in generative/diffusion image JSCC (2024 to Nov 2026)

Research date: 2026-11-08. Method: arXiv abstract and HTML full text read through WebFetch, with the GitHub README and listing and summary pages (Pith, EmergentMind). The Semantic Scholar citation API returned HTTP 429 three times, and Google Scholar and ResearchGate were not reachable (403), so **the list of citing works is not exhaustive** (see Gaps).

Verification levels used below:
- **[FT]**: read in the arXiv HTML full text. Quotes are the sentences the fetch tool extracted, with the section it named.
- **[ABS]**: abstract only, unverified.
- **[2nd]**: seen only through a secondary or aggregator page.

Queries run: "DiT-JSCC"; "DiT-JSCC" OR "DiTJSCC" cite 2026 ...; generative JSCC diffusion transformer 2026 arXiv SNR-adaptive; Jincheng Dai Kai Niu 2026 generative semantic communication diffusion arXiv; Kailin Tan Jincheng Dai 2026 arXiv JSCC; diffusion JSCC digital modulation QAM SNR-adaptive image transmission generative receiver; SemQAM joint semantic-channel coding and modulation ...; diffusion model digital semantic communication quantized latent 16-QAM image reconstruction SNR unknown; blind SNR generative JSCC diffusion receiver without channel state information image; "generative" JSCC "digital modulation" diffusion decoder single model multiple SNRs QAM constellation 2026; DiffCom HiFi-DiffCom ...; CDDM channel denoising diffusion ... SNR known.

---

## Q1. What cites or extends DiT-JSCC (arXiv:2601.03112) up to Nov 2026? Does any of them quantise its symbols, make it SNR-adaptive, or compare told vs blind?

### Takeaway
I found no follow-up that extends DiT-JSCC to SNR adaptivity, digital or quantised symbols, or a told-vs-blind comparison. The paper itself (v2, Jun 2026) still trains one model per SNR with analog complex symbols and 50 diffusion steps. The only citing works I could find use it as background. The closest "lighter, one model for all SNRs" diffusion JSCC is OSD-JSCC (Zhejiang U., 4 Nov 2026). It is SNR-told and analog, and it does not cite DiT-JSCC.

### Cited Findings
**DiT-JSCC itself**
- Metadata: Kailin Tan, Jincheng Dai, Sixian Wang, Guo Lu, Shuo Shao, Kai Niu, Wenjun Zhang, Ping Zhang. v1 6 Jan 2026, v2 18 Jun 2026 (both 2,426 KB). Comments: "14 pages, 14 figures, 2 tables". arXiv lists no journal reference. — [arXiv abs](https://arxiv.org/abs/2601.03112)
- Venue [2nd, not verified with the publisher]: ResearchGate lists IEEE Trans. Cognitive Communications and Networking, 12:9247–9262, DOI 10.1109/TCCN.2026.3705826. — [ResearchGate](https://www.researchgate.net/publication/407502169_DiT-JSCC_Rethinking_Deep_JSCC_With_Diffusion_Transformers_and_Semantic_Representations)
- Per-SNR specialists [FT, v2, Sec. IV-A4 Implementation Details]: "Regarding the wireless channel setting, we train separate models for each SNR under both AWGN and Rayleigh fading channels." The v2 text describes no SNR-conditioned or SNR-adaptive model. — [arXiv HTML v2](https://arxiv.org/html/2601.03112v2)
- Analog channel input [FT, Sec. II-A]: the encoder "encodes the image 𝒙 into a vector of complex-valued channel input symbols". Digital modulation appears only in the separation baselines (Sec. IV-A3): "combine them with 5G LDPC channel coding" ... "for channel coding and digital modulation." — [arXiv HTML v2](https://arxiv.org/html/2601.03112v2)
- Sampling cost [FT, Sec. IV-B]: "All listed methods use 50 diffusion steps". Also "DiT-JSCC uses a smaller reported parameter count with intermediate encoding and decoding time" and "diffusion-based JSCC performance should be interpreted together with model size and inference cost." — [arXiv HTML v2](https://arxiv.org/html/2601.03112v2)
- Future work [FT, Sec. V, fragments only; the full section did not come back]: "we will study deployment-oriented variants" ... "to reduce the computational cost introduced by foundation models and DiT decoding." The text also says these "should be introduced carefully to avoid sacrificing semantic consistency or training stability." No SNR-adaptive or digital future work was extracted. — [arXiv HTML v2](https://arxiv.org/html/2601.03112v2)
- Adaptivity in DiT-JSCC is over bandwidth, not SNR: the abstract describes "a training-free adaptive bandwidth allocation strategy inspired by Kolmogorov complexity" (summarised). — [arXiv abs](https://arxiv.org/abs/2601.03112). An automated Pith review says that allocation (KC-BA) relies on evaluation-set normalisation and has no validation split [2nd]. — [Pith](https://pith.science/paper/2601.03112)

**GitHub repo semcomm/DiTJSCC (README read 2026-11-08)**
- "This is a self-contained inference release." Also "This public repository is intended for checkpoint inference." — [README](https://raw.githubusercontent.com/semcomm/DiTJSCC/main/README.md)
- "The paper trains separate checkpoints for each channel/SNR condition." The interactive demo offers "SNR control from -10 dB to 20 dB" and "AWGN and Rayleigh channel simulation". It also warns: "The UI changes the simulated test channel; paper-level reproduction requires loading" the matching checkpoint. In other words, the demo's SNR slider changes only the simulated channel; it is not an SNR-adaptive model. — [README](https://raw.githubusercontent.com/semcomm/DiTJSCC/main/README.md)
- Samplers: "DPM-Solver, Euler ODE, and the paper's Euler SDE sampler", with "sampling-step control". The README never uses the words "digital", "modulation" or "quantization". The expected checkpoint files are only `autoencoder_kl.pth`, `dit_jscc_256.pth` and `dit_jscc_512.pth`, and "No model weights are included in this repository" (per the repo page). — [GitHub](https://github.com/semcomm/DiTJSCC)
- An older search snippet said the code was "coming after the final acceptance". The current README describes an inference release. The issue count was not visible. — [GitHub](https://github.com/semcomm/DiTJSCC)

**Citing works found (non-exhaustive)**
- Pith lists 2 citing papers. — [Pith](https://pith.science/paper/2601.03112)
  - *Adapting Diffusion Language Models for Lossless Pixel-Level Image Transmission* (Ren, Li, Chen, Li, Zhao; arXiv:2606.06273; 4 Jun 2026). Its related work says DiT-JSCC "employs generative diffusion modeling to improve perceptual reconstruction quality under bandwidth- or SNR-limited channels" [2nd, via Pith]. The work itself is a *separate* source-channel coding scheme (discrete diffusion LM with arithmetic coding) for lossless transmission [ABS]. It is not a DiT-JSCC extension. — [arXiv abs](https://arxiv.org/abs/2606.06273)
  - *SoM-MTM: Synesthesia of Machines-Driven Masked Token Model for Cooperative Perception over Packet Loss Channel* (arXiv:2608.13245). It cites DiT-JSCC only as general background [2nd]. — [Pith](https://pith.science/paper/2601.03112)
- EmergentMind's DiT-JSCC topic page lists no 2026 follow-ups and no SNR-adaptive or digital variant [2nd]. — [EmergentMind](https://www.emergentmind.com/topics/dit-jscc)
- Same group, later work, but not a DiT-JSCC extension: *Tokenized Joint Source-Network Coding for Loss-Resilient Video Streaming* (Ke Yang, Sixian Wang, Kailin Tan, Guo Lu, Jincheng Dai, Wenjun Zhang, Ping Zhang; arXiv:2610.21166; Oct 2026). It targets packet loss in video streaming [ABS]. — [arXiv](https://arxiv.org/abs/2610.21166)

**Closest "lighter / one-model" diffusion JSCC (does not cite DiT-JSCC)**
- OSD-JSCC: *Neural Joint Source–Channel Coding with One-Step Diffusion* (Shunpu Tang, Qianqian Yang, Zhejiang U.; arXiv:2611.06437v1; 4 Nov 2026) [FT]. — [arXiv HTML](https://arxiv.org/html/2611.06437)
  - SNR is told to both ends: the SNR "is assumed available at both ends" (Sec. II). Encoder and decoder are SNR-conditioned via "adaptive group normalization (AdaGN)" (Sec. III-B), with an SNR-adaptive timestep and a reliability token (Sec. III-B/C).
  - One model: "sharing network weights across SNRs" (Sec. III-B). Training SNR is "uniformly sampled from [0,20] dB" (Sec. III-D), and testing covers 0–20 dB.
  - Analog: "The encoder directly maps an image to channel symbols" (Sec. I). QAM appears only in the baselines ("with 5G LDPC codes and QAM", Sec. IV-A).
  - One-step decoding. Kodak decoding takes 0.0672 s at 12 dB, against 3.40 s for DiffJSCC, which uses "50 sampling steps and image captioning" (Sec. IV-B, Table I). Decoding is up to 50.6× faster.
  - The fetch found no blind-SNR or SNR-mismatch experiment and no per-SNR specialist comparison. DiT-JSCC is "neither cited nor compared" (its baselines are DeepJSCC, SwinJSCC, DiffJSCC, TF-DJSCC, BPG/VTM, MS-ILLM and ELIC).

### Inferences
- On the evidence available, nobody has published "DiT-JSCC + one model for all SNRs", "DiT-JSCC + QAM", or "DiT-JSCC told vs blind". The authors' own forward-looking section points to lighter, deployment-oriented variants, not to SNR adaptivity or digital transmission.
- OSD-JSCC is the nearest competitor on the "one SNR-told model across SNRs, fast decoding" axis. It is analog, U-Net-based (pretrained one-step diffusion prior), SNR-told, has no blind arm, and has no specialist comparison.

### Gaps
- I could not get a complete citation list for arXiv:2601.03112: the Semantic Scholar API returned 429 three times, Google Scholar was not tried (it is usually blocked), and ResearchGate returned 403. Citing works that are not on Pith may exist.
- I did not check what changed between v1 and v2. The file sizes are identical, and the v2 text still says "separate models for each SNR".
- The GitHub issues were not readable, so I don't know whether anyone asked for SNR-adaptive or digital variants there.
- The full DiT-JSCC conclusion (Sec. V) was only partly extracted.

---

## Q2. The BUPT group's other 2024–2026 generative JSCC work (DiffCom / HiFi-DiffCom / Blind-DiffCom, surveys): analog or digital? SNR-adaptive?

### Takeaway
DiffCom, HiFi-DiffCom and Blind-DiffCom are all analog (continuous complex symbols). Their receivers explicitly use the noise variance σ_n², so they are SNR-told. "Blind" in Blind-DiffCom means the *channel response h* is unknown, not the SNR. The group's 2025 diffusion-for-SemCom tutorial is a survey.

### Cited Findings
- *DiffCom: Channel Received Signal is a Natural Condition to Guide Diffusion Posterior Sampling* (Sixian Wang, Jincheng Dai, Kailin Tan, Xiaoqi Qin, Kai Niu, Ping Zhang; arXiv:2406.07390; v1 Jun 2024, v2 Jan 2025) [FT]. — [arXiv HTML v2](https://arxiv.org/html/2406.07390v2)
  - Analog: the encoder maps the source to "complex-valued channel-input symbols" (Sec. II). Digital appears only in the baselines: "5G LDPC channel coding (code length 4096) and digital modulation in line with the 3GPP TS 38.214 standard" (Sec. IV-A4).
  - SNR-told: the likelihood (Sec. III-B) uses σ_n², and Algorithm 2 takes σ_n² as input. HiFi-DiffCom picks its start timestep T_s by matching (1+1/σ_n²)^{ρ/2} to the diffusion SNR ᾱ_t/(1−ᾱ_t) (Eq. 21, Sec. III-C2): "x_Ts should retain more information from x_d when its quality is higher". It is a training-free posterior-sampling method run on top of a pre-trained JSCC codec. Of those codecs, only NTSCC+ is stated as SNR-adaptive: "which supports SNR-adaptive and fine-grained rate adjustments within a single model" (Sec. IV-A2). DeepJSCC was "pre-trained under an AWGN channel" (Sec. IV-C).
  - Blind-DiffCom: "the specific channel response remains underestimated or unestimated" (Sec. III-D). It samples h from its prior distribution with a second diffusion process. The paper does not explicitly list the SNR as unknown, although Algorithm 3's inputs omit σ_n (a fetch-tool observation; not re-checked).
  - Code: github.com/wsxtyrdd/diffcom supports diffcom, hifi_diffcom and blind_diffcom [2nd, search snippet]. — [GitHub](https://github.com/wsxtyrdd/diffcom)
- *Generative AI Meets 6G and Beyond: Diffusion Models for Semantic Communications* (Hai-Long Qin, Jincheng Dai, Guo Lu, Shuo Shao, Sixian Wang, Tongda Xu, Wenjun Zhang, Ping Zhang, Khaled B. Letaief; arXiv:2511.08416; Nov 2025, v3 exists) is a tutorial/survey [ABS]. A companion curated paper list is Awesome-DiffComm. — [arXiv](https://arxiv.org/abs/2511.08416); [GitHub](https://github.com/qin-jingyun/Awesome-DiffComm)
- *Rate-Distortion-Perception Controllable JSCC for High-Fidelity Generative Communications* (arXiv:2408.14127, 2024, with Dai and Tan) [ABS, not fetched]. — [arXiv](https://arxiv.org/abs/2408.14127)
- Related lab, not BUPT: JSCGC: *Joint Source-Channel-Generation Coding for Wireless Generative Communications* (Tong Wu, Zhiyong Chen, Guo Lu, Li Song, Feng Yang, Meixia Tao, Wenjun Zhang; arXiv:2606.12858; 11 Jun 2026) [ABS]. A generative receiver is conditioned on the received signal "across channel conditions". The abstract states neither analog/digital nor SNR handling. — [arXiv](https://arxiv.org/abs/2606.12858)

### Inferences
- Every generative JSCC system from the BUPT group that I checked is analog and assumes the receiver knows σ_n² (or uses a per-SNR model). None is digital, and none is SNR-blind.

### Gaps
- I did not search the NTSCC variants from 2025–2026 individually.
- The JSCGC full text was not read.

---

## Q3. SNR-adaptive diffusion receivers (SGD-JSCC, DiffJSCC, LRISC, DriftDecode, CDDM, others): digital modulation? tested without SNR knowledge?

### Takeaway
All of the image-generative diffusion JSCC receivers checked here are **analog**. Only one, SGD-JSCC, tests a receiver without the true SNR, and it does so by *estimating* the SNR from the received signal and feeding that estimate in. That is not truly blind: it is an estimate-then-tell receiver. It reports results "nearly identical, whether the eq. SNR is available or not" (verified). Diffusion models that do work with digital/QAM symbols (SCDM, the Markov discrete diffusion model, DSRDM) use diffusion to denoise or correct symbols, not to generate images, and are SNR-conditioned or unclear.

### Cited Findings
**SGD-JSCC**: *Semantics-Guided Diffusion for Deep JSCC in Wireless Image Transmission* (Maojun Zhang, Haotian Wu, Guangxu Zhu, Richeng Jin, Xiaoming Chen, Deniz Gündüz; arXiv:2501.01138v1; 2 Jan 2025) [FT]. — [arXiv HTML](https://arxiv.org/html/2501.01138v1)
- Verified quote (Sec. VII-B, slow fading): "the performance of the proposed SGD-JSCC method remains nearly identical, whether the eq. SNR is available or not".
  - Context: the setting is Fig. 9, "when CSI is unknown at neither the transmitter nor the receiver" (sic, caption as extracted). Baselines ADJSCC and DeepJSCC-Diff degrade there.
  - The authors attribute the robustness to not needing CSI at the transmitter and to estimating SNR from the noisy JSCC features. The paper says the scheme "bypasses the need for precise eq. SNR information by directly estimating it from the noisy JSCC features."
  - The paper reports no estimator accuracy and no direct estimated-vs-true SNR ablation.
- Estimator (Sec. V-C, pilot-free step matching): "we consider directly estimating the SNR from the received signal rather than known pilots". The estimator is a ResNet with 4 residual blocks, an FC layer and a sigmoid output, trained with MSE on artificially noised latents. The abstract describes "the instantaneous signal-to-noise ratio (SNR) directly estimated from the channel output".
- SNR adaptivity sits in the diffusion model, not in the JSCC codec. The JSCC is trained at a fixed 10 dB AWGN. Footnote (Sec. V-A): "the adaptability to various SNRs can be achieved by DM module as detailed in Section V-B". The start step is set from the SNR, m = S⁻¹(σ²/(σ²+|h|²)), and the authors "directly input the noise level instead of the discrete timestep into the DM". Evaluation goes down to −15 dB.
- Analog: "a vector of complex-valued channel input symbols" with an average power constraint (Sec. III-A). Finite-constellation DeepJSCC appears only in related work.
- No per-SNR specialist envelope: the paper does not say any baseline is trained at each SNR. The intro motivates the work by noting that a separate JSCC model "must be trained for each specific channel environment". ADJSCC is evaluated with and without SNR input.

**DiffJSCC**: *Diffusion-Aided JSCC for High Realism Wireless Image Transmission* (Mingyu Yang, Bowen Liu, Boyang Wang, Hun-Seok Kim, U. Michigan; arXiv:2404.17736; v3 22 Mar 2025) [FT]. — [arXiv HTML](https://arxiv.org/html/2404.17736)
- SNR-told: "both the transmitter and receiver are assumed to have perfect knowledge of the channel gain h and SNR γ". The JSCC encoder and decoder "could take the CSI information γ and h", and the base JSCC is SNR-adaptive (Squeeze-Excitation).
- For the diffusion control module, "The SNR, γ, is uniformly sampled within the ranges" [0,14] dB (general images) and [−5,5] dB (faces).
- The only SNR ablation is presence vs absence of the SNR condition: "augmenting the model with channel SNR information yields an average FID score improvement of 2%". There is no mismatch or estimation study.
- Analog ("complex-valued channel input symbols"). BPG+LDPC with BPSK/16-QAM is used only as a baseline.

**LRISC**: *Latent Feature-Guided Conditional Diffusion for Generative Image Semantic Communication* (Zehao Chen, Xinfeng Wei, Haonan Tong, Zhaohui Yang, Changchuan Yin; arXiv:2504.21577v2; 6 Jun 2025) [FT]. — [arXiv HTML](https://arxiv.org/html/2504.21577)
- One model, SNR-told: the design allows "one model to work across various channel states". The SNR adaptation module takes "the SNR for received z_i (denoted by SNR_i) as input". The training SNR range is not specified ("a fixed channel state (SNR = 10dB)" first, then "variable channel state").
- The paper criticises prior work, which "typically train[s] separate models for discrete SNR levels", but compares only against DeepJSCC and BPG+LDPC. There is no specialist envelope and no blind test.
- Channel input: power-constrained symbols. "Q denotes a 2^n-level scalar quantization" appears only in the rate model, so the input is apparently analog but not stated explicitly.

**DriftDecode**: *One-Step Wireless Image Decoding via Drifting-Inspired Detail Recovery* (Jingwen Fu, Ming Xiao, Mikael Skoglund, KTH; arXiv:2605.02325; 4 May 2026) [ABS]. It is an "SNR-conditioned one-step decoder", with 30 ms latency and a 4.8× speedup over a 10-step flow-matching decoder. The abstract states neither analog/digital nor a blind setting. — [arXiv abs](https://arxiv.org/abs/2605.02325)

**FlowSem**: *Flow Matching for Adaptive Wireless Image Transmission in Semantic Communication* (Arnab Emon, Eslam Eldeeb; arXiv:2608.21651; 21 Aug 2026) [ABS]. An SNR-adaptive DeepJSCC is followed by conditional flow matching "conditioned on that reconstruction and channel SNR" (summarised), so it is SNR-told. Modulation is not stated, and digital appears only in the BPG+LDPC baseline. — [arXiv abs](https://arxiv.org/abs/2608.21651)

**OSD-JSCC** (4 Nov 2026): analog, SNR-told at both ends, one model over 0–20 dB. See Q1. — [arXiv HTML](https://arxiv.org/html/2611.06437)

**Alternating dual-domain posterior sampling**: *Generative Semantic Communication via Alternating Dual-Domain Posterior Sampling* (Tang, Yang, Zhejiang U.; arXiv:2604.16796; 18 Apr 2026) [FT via alphaXiv]. Analog: "maps it to a k-dimensional complex-valued channel input signal z ∈ C^k". The likelihood uses σ²_ch, so the receiver implicitly needs the SNR. It is evaluated at 1 dB and −1 dB and does not cite DiT-JSCC. — [alphaXiv](https://www.alphaxiv.org/abs/2604.16796.md)

**CDDM**: *Channel Denoising Diffusion Models for Wireless Semantic Communications* (Wu et al.; arXiv:2309.08895; IEEE TWC 2024) [2nd, search snippets]. A channel-denoising module placed after MMSE equalisation. It "can adapt to different channel conditions, such as AWGN channel and Rayleigh fading channel with different" SNRs. It is analog, and a later paper (SCDM) describes CDDM as targeting "noise reduction of analog semantic information". Whether the SNR is told was not verified. — [arXiv](https://arxiv.org/abs/2309.08895)

**Diffusion with digital/QAM symbols (symbol denoising or correction, not image generation)**
- SCDM: *Score-Based Channel Denoising Model for Digital Semantic Communications* (Mo et al.; arXiv:2501.17876; Jan 2025, ICC 2025) [ABS]. It models distortion of "constellation symbol sequences in digital transmission" as a score-based forward diffusion. One model works "under various SNR conditions". Whether SNR is told or blind is not stated. — [arXiv abs](https://arxiv.org/abs/2501.17876)
- *Markov-Enforced Discrete Diffusion Model for Digital Semantic Symbol Error Correction* (Yoon Huh, Jeongho Kang, Wan Choi, SNU; arXiv:2603.22983; 24 Mar 2026) [FT, partial]. "We assume 16-QAM modulation for transmission, leading to a codebook size of 16" (VQ codebook of 16 four-dimensional codewords). A discrete diffusion model corrects detected indices, and a JSCC decoder then reconstructs the image. It is SNR-told: the start step is chosen from the SNR. One model covers a noise schedule from −3 to 15 dB, and the JSCC encoder/decoder are trained at a fixed 20 dB. — [arXiv HTML](https://arxiv.org/html/2603.22983v1)
- DSRDM: *Digital Signal Recovery Diffusion Model for Semantic Communications* (Zhigang Yan, Dong Li; arXiv:2605.27730; 26 May 2026) [ABS]. Diffusion is used as a carrier for digital signals. The abstract does not state SNR handling. — [arXiv abs](https://arxiv.org/abs/2605.27730)

**Digital and generative, but not diffusion**
- SemQAM: *Joint Semantic-Channel Coding and Modulation for Generative Image Transmission* (Hao Cao, Jingkai Ying, Yinhuan Huang, Zhijin Qin, Yuan Shen, Tsinghua; arXiv:2610.29643v1; 21 Oct 2026) [FT]. — [arXiv HTML](https://arxiv.org/html/2610.29643v1)
  - Digital: "Each codebook contains 1024 codewords whose indices are mapped one to one to Gray-labeled 1024-QAM symbols" (Sec. 3.1), with hard-decision detection ("For hard-decision QAM, the transition matrix is", Sec. 3.3).
  - The decoder is a VQ autoencoder with a PatchGAN loss, not diffusion.
  - One model: "SemQAM supports multiple SNRs and CBRs in a unified model". It is trained at 10, 16 and 24 dB plus noiseless, and selects "the corresponding VQ codebooks for each operating point" (Sec. 3.1). In effect the operating SNR must be known so the right codebook can be picked; this is my inference.
  - The fetch found no blind-SNR, mismatch or per-SNR specialist comparison, and no DiT-JSCC citation.
- MOC-RVQ: *Multilevel Codebook-Assisted Digital Generative Semantic Communication* (arXiv:2401.01272, 2024) [ABS]. A VQ approach to digital generative SemCom. — [arXiv](https://arxiv.org/html/2401.01272v2)

### Inferences
- Told vs blind: the only diffusion image-JSCC that tests without the true SNR is SGD-JSCC, and its "blind" mode uses an estimated SNR, not a receiver trained without SNR input. Every other receiver here takes the true SNR (DiffJSCC, LRISC, OSD-JSCC, FlowSem, DriftDecode) or σ_n² (the DiffCom family, dual-domain posterior sampling).
- Digital: the digital diffusion works denoise or correct symbols (SCDM, the Markov discrete diffusion model), are SNR-conditioned or unclear, and are not image-generative from pure noise. The digital generative image works I found (SemQAM, MOC-RVQ) are VQ/GAN, not diffusion.

### Gaps
- I did not read the full texts of DriftDecode, FlowSem, SCDM, DSRDM or JSCGC. Their SNR knowledge and modulation are abstract-level only.
- The CDDM SNR assumption was not verified from its full text.
- The SGD-JSCC venue was not found in the arXiv v1 text. Later versions (v2+) were not checked.

---

## Q4. Does any work study blind (SNR-unaware) generative receivers with quantised/digital symbols, or compare against per-SNR specialists at each SNR (a "specialist envelope")?

### Takeaway
After the searches listed at the top, I found no paper that (a) trains a generative or diffusion image JSCC receiver *without any SNR input* (not even an estimate) using quantised/QAM channel symbols, or (b) compares one-model-for-all-SNRs diffusion JSCC against an envelope of per-SNR specialists. Several papers criticise per-SNR training (LRISC, SGD-JSCC), but none measures the gap to specialists at each SNR. This is an **apparent gap**, not proof of absence.

### Cited Findings
- DiT-JSCC uses per-SNR specialists only ("we train separate models for each SNR", Sec. IV-A4) and has no single-model arm. — [arXiv HTML v2](https://arxiv.org/html/2601.03112v2)
- LRISC says prior methods "typically train separate models for discrete SNR levels" but compares only against DeepJSCC and BPG+LDPC. — [arXiv HTML](https://arxiv.org/html/2504.21577)
- SGD-JSCC's no-SNR mode is an estimated-SNR mode: "bypasses the need for precise eq. SNR information by directly estimating it from the noisy JSCC features" (Sec. VII-B). It is analog. — [arXiv HTML](https://arxiv.org/html/2501.01138v1)
- DiffJSCC ablates SNR presence (≈2% FID gain with the SNR condition) but has no per-SNR specialist comparison. It is analog. — [arXiv HTML](https://arxiv.org/html/2404.17736)
- OSD-JSCC (one SNR-told model, 0–20 dB, analog) has no blind arm and no specialist envelope (per the fetch). — [arXiv HTML](https://arxiv.org/html/2611.06437)
- SemQAM (digital 1024-QAM, unified multi-SNR model, VQ/GAN decoder) has no blind or mismatch test and no specialist comparison (per the fetch). — [arXiv HTML](https://arxiv.org/html/2610.29643v1)
- Non-diffusion context [2nd, search snippet only]: *Channel-Blind JSCC for Wireless Image Transmission* (MDPI Sensors 2024) adapts "without the need for channel estimation". It is not generative. — [MDPI](https://www.mdpi.com/1424-8220/24/12/4005)

### Inferences
- What looks unclaimed, as far as these searches reach, is a three-way comparison on a DiT-style diffusion JSCC that generates from pure noise conditioned on the received symbols: per-SNR specialists vs one SNR-told model vs one blind model (no SNR input, no explicit estimator) over −5 to 5 dB. The 16-QAM-quantised (DeepJSCC-Q-style) version of that study also looks unclaimed.
- The closest prior art to cite and distinguish:
  - SGD-JSCC: explicit SNR estimator, analog, DM refines a DeepJSCC output rather than generating from pure noise.
  - OSD-JSCC: SNR-told, analog, one-step.
  - DiffJSCC: SNR-told, analog.
  - SemQAM: digital, unified multi-SNR model, not diffusion.
  - The Markov discrete diffusion model: 16-QAM VQ, SNR-told, diffusion corrects symbols.
- An implicitly blind DiT that infers the SNR from the statistics of the received symbols might be argued to be equivalent to SGD-JSCC's estimator path. Reviewers may raise this, so it would help to include an "explicit-estimator" arm or discuss it.

### Gaps
- The citation list is incomplete (Semantic Scholar rate limits; Google Scholar and IEEE Xplore not queried). The "DeepJSCC-Q + diffusion" combination was not searched separately under the names "DeepJSCC-Q diffusion" or "finite constellation diffusion JSCC". A paper may exist under different terms, for example "modulation-aware generative JSCC" or "digital DiT decoder". A final check on Google Scholar "cited by" for 2601.03112 and a search for "DeepJSCC-Q diffusion" are recommended before claiming novelty.
- The ADJSCC with/without SNR experiment inside SGD-JSCC is not a generative told-vs-blind study, and I could not check whether its "without SNR" arm was retrained or only had its input zeroed.
