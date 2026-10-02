# Semantic Communication: State of the Art by Subfield, and What Is Open

*Surveyed September 2026. For each subfield: what currently leads, what was displaced, and
where the open problems actually are.*

---

## 0. How to read this, and how much to trust it

**On the word "SOTA."** In a field publishing this fast, "state of the art" has a short
half-life and a narrow scope. A method is SOTA *on a benchmark, under a metric, at a rate and
channel model* — not in general. Where a leader is named below, the qualifier is included,
because the unqualified version is almost always wrong.

**Verification method.** Each claim here was checked against the primary source or a 2026
survey, and cross-checked against at least one work that *cites* it, specifically looking for
papers that supersede it. Where a claim rests on a single source, or where the most recent
comparison predates 2026, that is flagged inline. Claims are dated.

**Three failure modes this document tries to avoid:**

1. **Citing a 2022–2023 leader that the field passed.** The DeepJSCC literature in particular
   turns over roughly annually; several widely-repeated "SOTA" claims are now two
   architectures out of date.
2. **Accepting a self-declared "digital" claim at face value.** Several recent papers
   describe themselves as digital semantic communication while quantising *feature bit-width*
   and still transmitting unconstrained baseband symbols — a different operation from
   constraining the *channel input alphabet*. Where this distinction matters it is called out.
3. **Treating an absence of results as an open problem.** Some gaps are open because they are
   hard; others are quiet because they were solved and the result was unglamorous.

**Standing caveat:** absence of evidence is weak evidence here. Anything below stated as
"not done" means "not found in a September 2026 search," which is a materially weaker claim.

---

## 1. Joint source-channel coding — the core engine

### Where it stands

The architectural lineage runs CNN → Transformer → state-space model. **SwinJSCC** (Swin
Transformer backbone) displaced the CNN generation and became the standard comparison point.
**MambaJSCC** then reported beating SwinJSCC by 0.52 dB PSNR using 72% of the MACs and 51% of
the parameters, via a generalised state-space model
([arXiv:2409.16592](https://arxiv.org/pdf/2409.16592)). On the entropy-model branch,
**NTSCC++** reported being the first end-to-end system to outperform VTM + 5G LDPC on
straight PSNR ([project page](https://semcomm.github.io/ntscc_plus/)), and **MNTSCC**
(VMamba-based) reports a further 1.72 dB on Kodak24 at 32% lower cost.

**Practical reading:** if you need a strong modern backbone, the state-space architectures
(MambaJSCC, MNTSCC) are the current efficiency-per-dB leaders, and NTSCC++ is the leader on
content-adaptive rate control. If you need a *comparison baseline that reviewers recognise*,
SwinJSCC remains the common denominator.

**Caution:** these numbers come from the proposing papers under their own conditions, and the
head-to-head comparisons are not all mutually consistent. Nobody has published a neutral
reproduction across all four.

### Open

- **A neutral benchmark.** There is no shared, independently-run leaderboard across
  DeepJSCC variants — every comparison is run by a proponent. This is the single most useful
  unglamorous contribution available in this subfield.
- **Perception–distortion trade-off.** PSNR-optimal and perceptually-optimal models are
  different models; the field reports both but has no agreed way to trade them off.
- **Complexity at the receiver.** An LDPC decoder is extraordinarily efficient in silicon.
  A decoder DNN is not obviously cheaper, and few papers report energy-per-bit honestly.

---

## 2. Digital and constellation-constrained SemCom

### Where it stands

This is the subfield this repository contributes to, so the survey here is the most careful.

**DeepJSCC-Q** (2022) established that a finite alphabet costs little at high modulation
order and — crucially — that **graceful degradation survives quantisation**. The cliff comes
from the separation architecture (all-or-nothing code blocks gated on a hard CRC), not from
digital modulation. **JCM-VAE** established the probabilistic alternative, learning a
transition distribution over constellation points.

**The 2026 direction is learned constellations.**
[Distribution-Aware Constellation Learning (arXiv:2605.30988, June 2026)](https://arxiv.org/html/2605.30988)
represents the constellation as a trainable codebook so its geometry adapts to the empirical
distribution of semantic features — explicitly motivated by the observation that conventional
QAM *mismatches* that distribution. The VQ-based line (**VQ-DeepISC**, **VQ-DSC-R**) has
converged on the same answer from a different direction.

**An important taxonomy point, frequently blurred.** "Digital semantic communication" now
covers three genuinely different things:

| What is discretised | Standards-legal? | Examples |
|---|---|---|
| The channel input, to a **fixed standard** constellation | **Yes** — a real modulator can emit it | DeepJSCC-Q, JCM-VAE |
| The channel input, to a **learned** codebook | No — reintroduces custom hardware, one level up | VQ-DeepISC, VQ-DSC-R, Distribution-Aware CL |
| Feature **bit-width**, with unconstrained symbols still transmitted | No — the air interface is still analog | Several 2026 "adaptive quantization" papers |

Rows two and three deliver better numbers. Row one is the only one that survives contact with
a conformance test, because EVM is *defined* relative to a reference constellation. This is a
real tension in the field and it is rarely stated plainly: **the research frontier is moving
away from the property that made the subfield deployable in the first place.**

### Open

- **The deployability regression.** If learned constellations win on performance, either
  standards must admit learnable alphabets, or the gains must be given up. Standardising
  *learnable* constellations — with a conformance story — is an unclaimed and genuinely
  important problem.
- **5G NR's actual orders.** DeepJSCC-Q's best results need 4096-QAM. NR tops out at 256-QAM
  uplink / 1024-QAM downlink. At the orders NR really offers, the advantage over separation
  is real but much narrower, and at 16/64-QAM it does not uniformly win.
- **SNR-conditioned + fixed-alphabet.** As of this survey, not published — see
  [ADJSCC-Q-EXPLAINED.md §4](ADJSCC-Q-EXPLAINED.md) for the checked comparison table. This is
  what this repository measures.
- **Still no bit interface.** Constellation indices carry no bit-level semantics: no CRC, no
  HARQ soft-combining, no standard ciphering, no rate matching.

---

## 3. SNR- and rate-adaptivity

### Where it stands

ADJSCC's attention-feature module remains the most-reused primitive in the field — adopted
far beyond its original paper, at ~0.6% parameter cost. The 2025–2026 work refines *how* the
conditioning is injected rather than whether to do it:

- **SNR-EQ-JSCC** ([arXiv:2501.04732](https://arxiv.org/pdf/2501.04732)) embeds SNR into
  attention blocks and adjusts attention scores via channel embedding and query.
- **SA-RA-JSCC** ([arXiv:2606.17940](https://arxiv.org/html/2606.17940v1), June 2026) maps SNR
  *and* a semantic-rate factor into one vector and applies a single global reweighting,
  arguing against layer-wise injection. It beats SNR-EQ-JSCC, SwinJSCC and ADJSCC on its own
  benchmark — **and is analog**, using a continuous fading model with no quantisation.

**NTSCC++** is the leader for *rate* adaptivity: one model across bandwidth ratios and channel
states via a response network, plus online latent editing.

### Open

- **Conditioning on more than SNR.** The AF context vector is (SNR, pooled features). Not
  delay spread, Doppler, interference, or antenna configuration. Generalising the context is
  obvious, cheap, and still largely undone.
- **The signalling cost.** These methods consume a *continuous* SNR scalar. Real systems
  report quantised CQI on a slow, delayed schedule. ADJSCC's own mismatch experiments show
  graceful degradation, but reconciling "continuous μ" with "4-bit CQI every few slots" is
  real integration work.
- **Adaptivity and digital, jointly.** Note that the strongest 2026 adaptive schemes are
  analog. The adaptivity and digital-modulation branches have largely *not* merged.

---

## 4. Fading, OFDM, and MIMO

### Where it stands

The durable lesson is architectural, and it is one of the most transferable findings in the
whole field: **the most practical systems are hybrids** — learned components wrapped around
classical, well-understood physical-layer machinery. OFDM-guided DeepJSCC inserted explicit
OFDM layers plus channel estimation and equalisation as *differentiable modules* rather than
hoping a black box would learn to equalise, and beat the black-box alternative on both
convergence and performance. It also trained with deliberate clipping to control PAPR.

**DeepJSCC-MIMO** (ViT-based, open- and closed-loop) jointly learns feature mapping and power
allocation, and is robust to channel-estimation error and varying antenna counts without
retraining — which matters because perfect CSI is a fiction.

### Open

- **Carrier frequency offset, packet detection, pilot design** — explicitly left as future
  work by the OFDM-guided authors, and still thin.
- **Massive MIMO and mmWave/sub-THz** at realistic array sizes.
- **Hybrid-design theory.** "Insert classical DSP as differentiable layers" works, but there
  is no principled account of *which* blocks should stay classical and which should be
  learned. It is currently craft, not theory.

---

## 5. Task-oriented and goal-oriented communication

### Where it stands

The **Information Bottleneck** formulation (Shao, Mao, Zhang, JSAC 2021) remains the
theoretical anchor: optimise a rate–relevance trade-off against the downstream task, with
the variational bound (VIB) making it tractable. The 2026 refinements target multi-task and
bandwidth-varying settings — e.g. **Conditional Rate-Utility**, generalising IB with a sparse
mixture-of-experts for task specialisation, orthogonal projection for feature
disentanglement, and a bandwidth-adaptive bottleneck.

This subfield delivers the most extreme compression, because everything task-irrelevant is
discarded by construction.

### Open

- **Multi-task without retraining.** Most systems are trained for a known task. A feature
  stream useful for *tasks not known at encode time* is the actual deployment requirement and
  is largely unsolved.
- **Task drift.** When the downstream model is updated, the encoder's learned relevance is
  silently stale. Nobody has a good detection or renegotiation story.
- **Verification.** For a safety-relevant task, "the receiver inferred correctly" needs a
  confidence account. Task-oriented systems are worse at saying "I don't know" than a CRC is.

---

## 6. Multi-user and multiple access

### Where it stands

Two branches, and they are at very different maturity levels.

**Grafting onto existing schemes** (OFDMA / NOMA / RSMA) is the pragmatic branch and where
most 2026 activity sits. **RSMA** is reported to outperform NOMA, with particular advantage in
high semantic-rate regimes, and there is active work on RSMA for the *coexistence* of semantic
and bit communications ([arXiv:2409.10314](https://arxiv.org/abs/2409.10314)) — which is
probably the more realistic deployment question than pure-semantic networks.

**New semantic-domain access** is the ambitious branch. **MDMA** introduces a semantic/model
domain beyond time/frequency/space, exploiting approximate orthogonality of different users'
embeddings, and reports at least a 5 dB advantage over NOMA at low SNR on AWGN.

### Open

- **Interference in the semantic domain** is not well characterised. Approximate
  orthogonality of learned embeddings is an empirical property, not a designed guarantee, and
  it is unclear how it degrades as users scale.
- **Scheduling and QoS.** A scheduler needs a notion of "how much semantic value does this
  user gain from one more resource block." That utility function does not exist in agreed form.
- **Fairness** across users running different models, tasks, and knowledge bases.

---

## 7. Control-plane SemCom: CSI feedback and HARQ

### Where it stands

**This is the subfield closest to real standardisation, and by a wide margin.** The strategic
argument is risk-based: CSI feedback is internal signalling, already AI-friendly, and does not
require rewriting how user data is carried.

The concrete status matters more than the research here. At **RAN#108 (June 2025), 3GPP
approved WI `NR_AIML_air_Ph2`**, focused on CSI compression, including lifecycle management
covering model pairing, activation, fallback, and version synchronisation between vendors.
**Release 20 is the 6G study phase; Release 21 begins normative 6G work.** Broader semantic
communication is reported as expanding from roughly **2029** onward — so SemCom proper is not
a Release 20 deliverable.

Anyone writing a motivation section should use these dates rather than implying SemCom is
being standardised now. It is the *two-sided model machinery* that is being standardised now,
and that machinery is the prerequisite.

### Open

- **Two-sided model lifecycle** is the crux for the entire paradigm, not just CSI: training
  data alignment, version sync, fallback behaviour, and multi-vendor interoperability.
- **Semantic HARQ.** What does a retransmission *mean* when there is no CRC and no bit-exact
  target? Sibling problems: semantic-aware ACK/NACK, semantic CRC.
- **Overhead accounting.** Feedback schemes are often evaluated on compression ratio while
  ignoring the signalling and model-sync overhead they add.

---

## 8. Semantic knowledge bases

### Where it stands

The KB is what moves SemCom from "compress the message" to "send only the surprising part of
the message." Implementations span learned VQ/VAE codebooks, cross-modal feature stores, and
triple-based knowledge graphs, with LLM-backed KBs appearing in 2026
([arXiv:2604.05504](https://arxiv.org/pdf/2604.05504)).

Standardisation is live: ITU-T has a technical report on the **Architectural Framework for
Knowledge-Based Semantic Communication over Public IMT Network**, led by Pengcheng Laboratory
with BUPT and China Telecom, alongside earlier semantic-aware networking work (TR.Reqts-SAN,
Y.RA-SAN).

### Open

This subfield has the widest gap between conceptual appeal and engineering reality. Largely
unsolved at scale:

- **Synchronisation and versioning.** Transmitter and receiver must agree on KB contents. What
  happens mid-update, or on divergence, is not specified anywhere.
- **Format standardisation** across vendors.
- **Security isolation.** A shared KB is a shared attack surface and a cross-tenant leak path.
- **Capacity and eviction.** What gets *removed* from a KB, and who decides.

---

## 9. Generative, LLM, and token communication

### Where it stands

The fastest-moving subfield, and the one where "SOTA" decays quickest.

**Diffusion at the receiver** is the dominant generative approach — chosen for generation
quality, stable training, and a rigorous theoretical footing. Latent-diffusion denoising
receivers are an active 2025–2026 line
([arXiv:2506.05710](https://arxiv.org/html/2506.05710)), and at least one framework reports
SOTA on both pixel-level accuracy and perceptual quality across a wide SNR range *and under
transmission-distribution shift without fine-tuning* — robustness to distribution shift being
the more interesting claim of the two.

**Token communication (TokCom)** is the emerging unifying abstraction: make the *token* the
communication unit, exploiting cross-modal context with transformer processing at both ends
([arXiv:2502.12096](https://arxiv.org/abs/2502.12096), and the 2026 position paper
[arXiv:2609.10714](https://arxiv.org/abs/2609.10714)). The argument for it is that tokens are
the missing abstraction unifying semantic representation across modalities and systems — the
same role bits play in the classical stack.

**Assessment:** TokCom is currently a compelling *framing* with early results, not a
consolidated technique. Treat "tokens are the new bits" as a research bet.

### Open

- **Determinism and verifiability.** A generative receiver *synthesises* plausible content.
  For anything consequential, "plausible" is a liability, and there is no accepted way to
  bound or flag hallucinated reconstruction.
- **Standardisability.** Hard to write a conformance test against a generative model.
- **Compute at the edge.** A diffusion receiver on a battery-powered device is not currently
  realistic; latency and energy are under-reported.
- **Model distribution.** Both endpoints need matched large models — the two-sided problem
  at its worst.

---

## 10. Evaluation metrics

### Where it stands

**This remains a genuine blocker for standardisation, not a cosmetic complaint**, and it is
the subfield where the 2026 literature is most candid. Evaluation is fragmented across
telecommunications, NLP, CV, and ML, and **no single metric characterises semantic quality
across modalities, tasks, and channel conditions**
([Rethinking Communication Metrics, arXiv:2608.21626](https://arxiv.org/abs/2608.21626)).

The current stack: reconstruction metrics (PSNR/SSIM/MS-SSIM; BLEU/BERTScore; PESQ), task
metrics (accuracy, mIoU, mAP), embedding-based semantic similarity (CLIP-style cosine; SGCS
for CSI), perceptual metrics, and efficiency metrics (CBR, semantic spectral efficiency).

### Open

Named explicitly in the 2026 metric literature: absence of universal semantic success
criteria; no standardised semantic ground truth; **semantic drift**; limited reference-free
evaluation; weak integration of ML metrics with communication constraints; missing
relation-level and multimodal KPIs.

Add one that the field states less often: **metrics are chosen after the fact.** A method that
wins on LPIPS and loses on PSNR can be presented either way, and with no agreed primary
metric, both are defensible. This is a structural incentive problem, not an oversight.

---

## 11. Security and privacy

### Where it stands

Recognised as a first-class problem — semantic features can leak *more* than raw bits, because
they are by design the meaningful part. Distinct threat classes: semantic eavesdropping and
reconstruction of source data from exposed features; **semantic noise** (adversarial
perturbation, and also ambiguity arising from context or culture, which has no physical-layer
analogue); poisoning of two-sided codecs; and jamming. Defences are early: adversarial
training, multi-task learning with adversarial perturbations
([arXiv:2512.24452](https://arxiv.org/abs/2512.24452)), and physical-layer approaches.

### Open

- **No ciphering story.** Standard ciphers operate on bitstrings. Without a bit interface
  there is no standard way to encrypt a latent without serialising it — which reintroduces
  the interface you removed.
- **Semantic noise is not physical noise** and is not addressed by anything in §1–§4. A model
  robust to 0 dB AWGN can be defeated by a semantically adversarial input at 30 dB.
- **Privacy quantification.** No accepted measure of how much a semantic feature leaks.
- **KB as attack surface** (§8).

---

## 12. Hardware and over-the-air validation

### Where it stands

Real but thin. The recurring platform pairs **USRP SDRs with embedded GPUs** (Jetson Xavier
NX), e.g. the **ICP** prototype behind ASCViT-JSCC, and GNU Radio-based cognitive SemCom
testbeds using USRP 2954R. These close the credibility loop — the gains are not purely a
simulation artifact.

### Open

- **Volume.** A handful of prototypes for a field this size. Most published gains have never
  touched an antenna.
- **Real-time inference** at realistic rates, with honest latency and energy numbers.
- **Nothing integrated.** No prototype is simultaneously digital, MIMO, OFDM-robust,
  variable-rate, SNR-adaptive *and* over-the-air. Integration remains the frontier, exactly as
  the roadmap argues.

---

## 13. Cross-cutting: the honest summary

**What is genuinely solved, in isolation:** graceful degradation and low-SNR robustness;
single-model SNR adaptivity; fading via OFDM with hybrid learned+classical design; MIMO with
imperfect CSI; content-aware variable rate; finite-constellation digital transmission; and at
least proof-of-concept over-the-air validation.

**What is not:**

1. **Integration.** Still the frontier. Each paper removes one idealisation while re-assuming
   others, and this repository is itself a small instance of attacking that.
2. **Metrics and conformance.** A real blocker, and the least glamorous work available.
3. **The protocol stack.** Constellation-constrained JSCC makes the *modulator* compatible.
   CRC, HARQ, ciphering, segmentation and QoS remain unaddressed because there are no bits to
   attach them to. **The physical-layer problem is substantially solved; the
   protocol-integration problem is not.**
4. **Two-sided model lifecycle.** The operational problem 3GPP is working through now, and
   the gate on everything else.
5. **The deployability regression** (§2). The frontier is drifting toward learned alphabets
   and generative receivers — both of which give back the standards-compatibility that made
   the digital branch matter.

**And the framing worth keeping:** the honest claim for SemCom is *regime-dependent
advantage*, not universal superiority. It wins at low SNR, tight bandwidth, fast-varying
channels, latency budgets that cannot absorb a retry, and perceptual or task-oriented
payloads. That is a real and growing slice — XR, machine-vision offload, sensing, V2X, IoT
imaging — and it is not all traffic. A 6G stack plausibly needs both paths, with the semantic
one attachable per-bearer.

---

## Appendix: source notes and confidence

| Claim | Source | Confidence |
|---|---|---|
| MambaJSCC beats SwinJSCC (0.52 dB, 72% MACs) | [arXiv:2409.16592](https://arxiv.org/pdf/2409.16592) | Proposer-reported, not independently reproduced |
| NTSCC++ first to beat VTM+5G LDPC on PSNR | [project page](https://semcomm.github.io/ntscc_plus/) | Proposer-reported |
| 2026 constellation work is *learned*, not fixed QAM | [arXiv:2605.30988](https://arxiv.org/html/2605.30988) | High — verified against the paper |
| SA-RA-JSCC is analog, no quantisation | [arXiv:2606.17940](https://arxiv.org/html/2606.17940v1) | High — verified against the paper text |
| Sensors 2026 quantises bit-width, not channel alphabet | [PMC13517896](https://pmc.ncbi.nlm.nih.gov/articles/PMC13517896/) | High — verified against the paper |
| 3GPP RAN#108 (Jun 2025) approved `NR_AIML_air_Ph2` | Multiple 2026 standardisation summaries | Medium-high — secondary sources; check the 3GPP work-item record before citing |
| SemCom proper expands from ~2029 | 6G standardisation briefs | Medium — a projection, not a commitment |
| MDMA ≥5 dB over NOMA at low SNR, AWGN | MDMA literature | Proposer-reported, narrow conditions |
| RSMA outperforms NOMA at high semantic rate | 2026 RSMA-SemCom work | Medium — condition-dependent |
| ITU-T KB-SemCom architectural framework TR | ITU-T SG13 liaison documents | High |
| No SNR-conditioned + fixed-QAM work published | Negative result from a Sept 2026 search | **Low-medium — absence of evidence; re-verify** |

**If you cite one thing from this document, cite the last row's caveat.** A negative
literature result is the weakest claim here and the one most likely to be overtaken.
