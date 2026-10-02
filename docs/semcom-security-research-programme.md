# Security in Semantic Communication: Landscape, Verified Gaps, and a Research Programme

*Surveyed 30 September 2026. Written to be actionable from the ADJSCC-Q repository as it
stands. Companion to [the SOTA survey](semcom-sota-and-future-work.md) and
[the build explanation](ADJSCC-Q-EXPLAINED.md).*

---

## 0. How to read this, and how far to trust it

Every novelty claim below was checked against primary sources, not summaries. Three
practices are worth stating because they changed conclusions:

1. **Primary sources, not fetched summaries.** An automated summary of
   [arXiv:2603.24082](https://arxiv.org/abs/2603.24082) confidently reported that the paper
   *"compares quantized (digital) and analog"* and found *"quantization enhances robustness"*
   via boundary *"barriers."* Pulling the actual PDF showed **all of that was fabricated**:
   the paper compares analog DeepJSCC against classical LDPC separation, and the word
   "constellation" appears **zero times** in 17 pages. That single check is the difference
   between §6 being dead and being the strongest idea here.
2. **Absence of evidence is weak evidence.** Where a gap is claimed, it means "not found in
   a September 2026 search," which is materially weaker than "does not exist."
3. **Confidence is stated per claim.** See the table in §10.

**Standing risk:** security-in-SemCom is accelerating. Several gaps below were open when
checked and may not be in six months. Re-verify before committing.

---

## 1. Why security is different here

Three structural facts make SemCom security its own problem rather than a re-run of
classical wireless security.

**Semantic features leak more than bits do.** They are, by construction, the *meaningful*
part of the source. An eavesdropper intercepting a latent does not need to break a cipher to
learn something — the representation was optimised to be informative.

**Semantic noise is not physical noise.** A model robust to 0 dB AWGN can be defeated by a
semantically adversarial input at 30 dB. Nothing in the SNR-adaptivity, OFDM, or MIMO
literature addresses this, because those mechanisms defend against the wrong thing.

**There is no bit interface to hang classical security on.** In analog JSCC there is no
bitstring to cipher, no CRC to authenticate, no sequence number to anti-replay. This is the
same protocol-integration debt the roadmap identifies, viewed through a security lens — and
it is the reason digital SemCom matters for security specifically, not just deployability.

---

## 2. Threat taxonomy

Adapted from [Secure Digital Semantic Communications (arXiv:2512.24602)](https://arxiv.org/html/2512.24602v7),
which is the best-organised treatment available and is explicitly structured around the
digital pipeline.

### 2.1 Threats common to analog and digital

| Threat | Mechanism |
|---|---|
| **Semantic leakage / inference** | Intercepted features directly reveal sensitive attributes |
| **Semantic manipulation** | Perturbations distort recovered *meaning* without tripping link-level metrics |
| **Knowledge-base attacks** | Poisoning, manipulation, transmitter–receiver KB inconsistency |
| **Model attacks** | Training-data poisoning, backdoors, model extraction, model inversion |
| **Authenticity / availability** | Jamming, flooding, computational denial-of-service |

### 2.2 Threats unique to *digital* SemCom

This is the part that matters for this repo, because these threats **do not exist in the
analog schemes most of the security literature studies**.

**Deterministic modulation (our case — quantiser + fixed constellation):**
- **Quantisation boundary crossing** — small perturbations near a decision boundary are
  *"amplified into discrete changes"* that propagate through decoding.
- **Constellation decision-region exploitation** — nudging received points across region
  boundaries.
- **Vector-quantisation attacks** — codebook index corruption, nearest-codeword confusion,
  codebook poisoning and desynchronisation.

**Probabilistic modulation (JCM-VAE style):**
- **Probability biasing** — shifting the learned distribution toward error-inducing patterns.
- **Posterior manipulation** — concentrating mass on wrong hypotheses, or inducing
  overconfidence.

**Packet and protocol level:**
- **Traffic analysis** — packet sizes, timing, sequencing, retransmission patterns.
- **Metadata leakage** — task identifiers, model/codebook version tags, and — critically for
  §7 — **importance indicators**.
- **Packet manipulation** — injection, replay, reordering, selective dropping.
- **Control-plane attacks** — forged ACKs, manipulated rate adaptation and configuration.

---

## 3. What defences already exist

Stated so that no one re-invents them.

| Layer | Existing defences |
|---|---|
| Physical | Friendly jamming with asymmetric power allocation; constellation obfuscation ([ECM](https://arxiv.org/pdf/2505.14153), CD-PHY); physical-layer key-based interleaving |
| Cryptographic | [DeepJSCEC](https://arxiv.org/pdf/2208.09245) — encryption *preserving graceful degradation*, defeating the avalanche problem; [homomorphic SemCom](https://arxiv.org/html/2501.10182v1) |
| Training-time | Adversarial training; [layer-wise adversarial training for transformers](https://arxiv.org/pdf/2609.13128); [paired adversarial residual networks](https://arxiv.org/pdf/2407.02053) |
| Receiver-side | [SecDiff](https://arxiv.org/pdf/2511.01466) (diffusion-aided purification); [ROME](https://arxiv.org/pdf/2501.01172) (model ensembling vs semantic jamming); semantic clustering + adversarial purification (SCAPJSCC) |
| Quantiser design | Safety margins around thresholds; boundary-aware objectives; Gray-style mappings; codebook encryption and version validation |
| Privacy | [Multi-task learning + adversarial perturbations](https://arxiv.org/html/2512.24452); superposition coding with [quantifiable security](https://arxiv.org/pdf/2401.13980) |

**Note the asymmetry:** the quantiser-design and codebook-protection rows are, in the survey,
*proposals* rather than evaluated systems. That is where §6 lives.

---

## 4. Adversarial audit of the SOTA — who has actually been attacked?

You asked specifically for this. Below is what the SemCom SOTA looks like when you ask, of
each leading architecture, *has anyone ever attacked it?*

| Architecture | Role | Adversarially evaluated? | Evidence |
|---|---|---|---|
| **DeepSC** (text) | Foundational | ✅ Yes | Multi-domain adversarial attacks ([2212.10438](https://arxiv.org/pdf/2212.10438)); repeatedly used as attack target |
| **DeepJSCC** (analog image) | Foundational | ✅ Yes | [Physical-layer adversarial robustness (2305.07220)](https://arxiv.org/pdf/2305.07220); [2603.24082](https://arxiv.org/abs/2603.24082) |
| **SwinJSCC** | Common baseline | ✅ Yes | SCAPJSCC reports beating it under attack; layer-wise AT targets transformer SemCom |
| **Any multimodal ML radio** | — | ✅ Yes | [Magmaw, NDSS 2025](https://arxiv.org/abs/2311.00207) — universal, modality-agnostic, **real SDR**, defeats defences, −5.88 dB PSNR; [code public](https://github.com/juc023/Magmaw) |
| **NTSCC / NTSCC++** | Rate-adaptivity SOTA | ❓ **Not found** | Entropy-model attack surface unexamined |
| **MambaJSCC** | Efficiency SOTA | ❌ **Not found** | State-space backbone never attacked |
| **DeepJSCC-Q / constellation-constrained** | Deployability SOTA | ❌ **Not found** | ← **the gap §6 exploits** |
| **VQ-based digital** (VQ-DeepISC, VQ-DSC-R) | Digital SOTA | ⚠️ Conceptual only | Survey names index-corruption and codeword-confusion; [Channel-Aware VQ (2510.18604)](https://arxiv.org/abs/2510.18604) addresses *channel* robustness, not adversarial |
| **TokCom** | Emerging | ⚠️ Partial | Conformal risk control for robustness exists |

**Three things fall out of this table.**

1. **The adversarial literature studies analog SemCom almost exclusively.** The digital
   branch — the one that is standards-legal and therefore the one that might actually deploy
   — is essentially unexamined empirically.
2. **No cross-architecture comparison exists.** Nobody has asked *which* SemCom architecture
   is most robust under a common attack budget. Every study attacks one system.
   [arXiv:2603.24082](https://arxiv.org/abs/2603.24082) compares semantic *versus classical*,
   not architecture versus architecture.
3. **Magmaw is the strongest available attack baseline and it has public code.** Any
   credible security paper here should include it rather than only hand-rolled PGD.

---

## 5. Verified open gaps

Ordered by how confident I am that they are genuinely unoccupied.

1. **Adversarial robustness of constellation-constrained (digital) SemCom.** Not measured by
   anyone. Three competing theoretical predictions (§6.2). *High confidence.*
2. **Security cost of importance-aware protection (UEP).** The flagship importance-ordering
   paper contains *no* security discussion at all. *High confidence.*
3. **Security metrics for digital SemCom.** Survey §V-A calls this unsolved and "a key future
   direction": metrics for what can be *inferred* from intercepted semantic traffic, reported
   *jointly with utility and communication cost*. *High confidence (the survey says so).*
4. **Leakage as a function of the quantisation choice.** Analog vs fixed public QAM vs learned
   private codebook, matched conditions. *Medium-high.*
5. **Joint cryptography + modulation** (survey §V-B, "largely unaddressed"). **But** partially
   occupied by ECM, CD-PHY, and DeepJSCEC. *Low-medium — do not lead with this.*
6. **Secure multiple access / scheduling** for digital SemCom (survey §V-C): configuration
   desynchronisation, cross-user semantic confusion. *Medium — but far from this repo.*

---

## 6. P1 in detail — Does going digital cost you the adversarial robustness?

### 6.1 The one-sentence version

Analog DeepJSCC was recently *proved* to be far more adversarially robust than classical
separation, via a mechanism that a hard quantiser should destroy — so the field's
deployability fix may have silently undone its security advantage, and nobody has checked.

### 6.2 Why this is a real question and not a fishing trip

[arXiv:2603.24082](https://arxiv.org/abs/2603.24082) (Zhang, Shao, An, Qin, Huang) establishes,
theoretically and empirically, that semantic communication needs **14–16× more attack power**
than classical systems for the same distortion. Their explanation is specific:

> *the implicit regularization from noisy training forces decoder smoothness, a property that
> inherently provides built-in protection against adversarial attacks*

formalised through **Lipschitz smoothness** of the decoder. They close by asking:

> *can we characterize the fundamental trade-off between semantic efficiency and adversarial
> resilience, and design encoders that optimally balance both?*

They never go digital. That leaves a sharp question with **three mutually incompatible
predictions**:

| # | Prediction | Reasoning | Source |
|---|---|---|---|
| **H1** | Digital is **less** robust | Their robustness derives from decoder Lipschitz smoothness. A hard quantiser is a step function — **not Lipschitz**. The constellation constraint should break the mechanism | 2603.24082's own theory, extended |
| **H2** | Digital is **less** robust | "Quantisation boundary crossing": perturbations near decision boundaries are *amplified into discrete changes* | [2512.24602 §III](https://arxiv.org/html/2512.24602v7) |
| **H3** | Digital is **more** robust | Quantisation is *feature squeezing* — an established adversarial defence. Sub-threshold perturbations snap back to the same constellation point and vanish | adversarial ML literature |

H1/H2 and H3 predict opposite signs. H2 and H3 can even both be true at different
perturbation magnitudes — which would give a **non-monotonic robustness curve in attack
power**, a genuinely interesting result: a regime where the quantiser protects you, and a
regime where it betrays you, separated by the half-lattice spacing.

That structure — a crisp disagreement between named mechanisms, resolvable by a controlled
experiment — is what makes this a paper rather than a benchmark.

### 6.3 Why this repository specifically

The experiment needs an analog and a digital system that differ in *nothing else*. That is
hard to retrofit and this repo already has it:

- The 2×2 shares encoder, decoder, backbone, channel, power normalisation, and training
  protocol; **the digital arms add zero trainable parameters** (fixed constellation), so
  capacity is controlled by construction and already asserted by a test.
- The constellation is **fixed and standards-legal**, so the finding is about *deployable*
  systems, not a bespoke learned codebook.
- `M` is a clean knob: as M→∞ the quantiser approaches identity, so digital must converge to
  analog. That gives a **falsifiable curve**, not a single point.
- The invariant testing discipline (bit-exact constellation membership, matched power) means
  an attack cannot succeed by accidentally pushing symbols off-alphabet.

### 6.4 Hypotheses, stated before running

- **H-A.** At matched capacity and matched symbol budget, digital (ADJSCC-Q) requires
  *different* attack power than analog (ADJSCC) for equal semantic distortion. Sign not
  pre-committed.
- **H-B.** The gap closes monotonically as M increases, vanishing as the quantiser approaches
  identity. *This is the sanity check: if it fails, something is wrong with the setup, not
  with the field.*
- **H-C.** Whatever the sign, it is explained by the empirical decoder Lipschitz constant —
  i.e. 2603.24082's mechanism extends to the digital case.
- **H-D.** There exists a perturbation magnitude near half the lattice spacing where the
  ordering flips (the H2/H3 boundary).

Pre-registering these matters. With four arms and a sweep, a post-hoc story is always
available; the value is in committing first.

### 6.5 Experimental design

**Arms.** The existing 2×2 — BDJSCC, ADJSCC, DeepJSCC-Q, ADJSCC-Q — at matched capacity, plus
the existing separation baseline (capacity bound and fixed-MCS) as the external reference,
since 2603.24082's headline is precisely semantic-vs-classical and reproducing it validates
the harness.

**Attacks.** Three, in increasing credibility:
1. **PGD / progressive gradient ascent on the input image** — white-box, matches
   2603.24082's methodology so results are comparable to theirs.
2. **Perturbation on the transmitted symbols** — over-the-air-style, the realistic threat, and
   the one where the quantiser sits *between* attacker and decoder.
3. **[Magmaw](https://github.com/juc023/Magmaw)** — universal, modality-agnostic,
   black-box-ish, published at NDSS with code. Including it pre-empts the obvious reviewer
   objection that hand-rolled attacks are too weak.

For the digital arms, add a **boundary-aware attack**: instead of maximising distortion
directly, minimise the perturbation needed to flip each symbol's nearest-neighbour decision.
This is the concrete instantiation of the survey's "quantisation boundary crossing" threat and,
as far as I can tell, has never been implemented.

**The primary metric is attack power, not distortion.** Report **minimum perturbation power
to induce a target semantic distortion** — the same axis 2603.24082 uses, which makes the
comparison interpretable and lets you state results as "N× more/less attack power."

**Sweeps.** Attack power × test SNR × modulation order M ∈ {4, 16, 64, 256, 1024} × arm.
Add the empirical decoder Lipschitz estimate per arm as the mechanistic covariate for H-C.

**Controls that decide whether reviewers believe it.**
- Identical channel realisations across arms (already the pattern in `erasure_importance`).
- Attack budget matched in *power*, not L∞ pixel norm, or the comparison is meaningless.
- Verify transmitted symbols remain on-constellation *under attack* — the repo already
  refuses to report otherwise.
- Report the unattacked operating point alongside, so robustness is not confused with a
  model that was simply worse to begin with.

### 6.6 What each outcome means

| Outcome | Interpretation | Publishable? |
|---|---|---|
| Digital **less** robust | **Standards compliance carries a hidden security cost.** Strong, uncomfortable, policy-relevant — it means the deployable branch is the attackable one | Strongly |
| Digital **more** robust | Quantisation is a free defence; argues for digital on security grounds *as well as* deployability. Pleasant and useful | Yes |
| **Non-monotonic** (H-D) | Best outcome: a regime boundary at the lattice spacing, with a design rule for choosing M under an adversarial threat model | Strongly |
| No difference | Null result, but it *refutes* the survey's stated boundary-crossing threat — worth reporting | Weakly, as part of a larger paper |

Note there is no losing branch here. That is a property worth having in a first security paper.

### 6.7 Threats to validity — state these yourself before a reviewer does

- **CIFAR-10 at 32×32.** Adversarial results are resolution- and dataset-sensitive. Kodak or
  ImageNet replication strengthens it considerably.
- **The mechanism claim is the risky part.** H-C needs a defensible empirical Lipschitz
  estimator; a sloppy one invites attack. Consider reporting it as corroborating evidence
  rather than as proof.
- **Attack strength is a confound.** "Digital is more robust" is indistinguishable from "our
  attack is weak against digital." This is exactly why Magmaw and the boundary-aware attack
  belong in the design.
- **Gradient obfuscation.** A non-differentiable quantiser can produce *fake* robustness by
  breaking gradients — a well-documented failure mode in adversarial ML. **You must test for
  it** using BPDA (backward pass differentiable approximation) — conveniently, the repo's
  soft-to-hard surrogate *is* the natural BPDA. If straight-through gradients attack better
  than hard ones, the robustness was an artefact. **Getting this wrong is the single most
  likely way this paper gets rejected.**
- **Under-trained arms look spuriously robust.** Discovered empirically while building
  `semcom/importance.py`: on an *untrained* model, erasing 0% versus 100% of the transmitted
  symbols moved PSNR from 9.0633 dB to 9.0628 dB. The decoder simply does not depend on its
  latent yet, so nothing you do to the channel input matters. The same trap applies directly
  here — an attack against such a model registers as near-zero damage, which is
  indistinguishable from genuine robustness.

  This has three consequences for the protocol. **(i)** Every arm must be trained to
  convergence *and* to comparable quality before being attacked; an arm that trained worse
  will look more robust for the wrong reason. **(ii)** Always report the unattacked operating
  point beside the robustness number — §6.5 says this for interpretability, but it is also
  the detector for this failure. **(iii)** Include a sensitivity floor check: verify the
  reconstruction actually responds to its latent (erasure moves PSNR by more than ~1 dB)
  before any attack number from that arm is admissible. This is now enforced as a test
  precondition in `tests/test_importance.py`.

  Note this is a close cousin of gradient obfuscation: both produce a model that *appears*
  robust because the attack signal never reaches the output. It is worth checking for both,
  because they have different causes and different fixes.

### 6.8 Minimum viable result

Two arms (ADJSCC vs ADJSCC-Q), one attack (PGD on input), one SNR, M ∈ {16, 256}, on CIFAR-10.
If a gap appears and shrinks with M, the paper exists. Everything else is strengthening.

---

## 7. P2 — Importance-aware protection leaks importance

**The claim.** UEP improves robustness to *random* noise while making the system **cheaper to
attack intelligently**, because differential protection publicly signals where the valuable
symbols are.

**Why it is open.** I checked the flagship importance-ordering paper,
[ISFR/SI-UEP (arXiv:2604.00595)](https://arxiv.org/html/2604.00595), which reports 23%
robustness improvement. It contains **no discussion of privacy or security** — not of whether
revealing importance ordering enables selective corruption, not of importance metadata as a
side channel, not of differential protection as an observable. Meanwhile
[2512.24602](https://arxiv.org/html/2512.24602v7) lists "importance indicators" under metadata
leakage, unmeasured.

**Experiment.** Adversary observes only the protection pattern (or infers it from power/order
statistics). Compare targeted attack efficiency against UEP vs equal protection at matched
total protection budget. Metric: attack power for equal distortion. The predicted result —
**UEP is more robust to noise and less robust to adversaries** — negates the premise of a
growing subfield, which is what makes it worth writing.

**Fit to this repo.** Your `semcom/importance.py` already measures ground-truth per-group
importance by erasure. An attacker allocating perturbation by that ranking is a small addition.

---

## 8. P3 — Leakage as a third axis on the rate–distortion plane

Survey §V-A, stated as unsolved: metrics for what can be inferred from intercepted traffic,
**reported jointly with utility and communication cost.**

**The experiment.** Train an eavesdropper decoder on intercepted symbols for each arm —
analog, fixed public QAM, learned private codebook — at matched rate and SNR. Report
eavesdropper reconstruction quality (and MINE-estimated leakage) *as a surface over rate and
distortion*, not as a scalar.

**The sharp tension.** A public standardised alphabet is exactly what an eavesdropper needs.
Learned codebooks are not *security* — obscurity never is — but they are measurably harder to
invert without the codebook. So standards-legality may cost privacy **as well as** (per §6)
possibly costing robustness. Two independent security costs of deployability is a coherent
thesis for a full paper.

---

## 9. Suggested programme

A defensible arc, smallest useful unit first:

1. **Run the existing ablation.** Not merely sequencing — a **validity precondition**. P1's
   digital arm assumes ADJSCC-Q works, and per §6.7 an under-trained arm registers attacks as
   near-zero damage, which is indistinguishable from robustness. Trained, converged,
   quality-comparable arms are a prerequisite for *any* attack number being meaningful.
2. **P1 minimum viable** (§6.8) — one focused result.
3. **P1 full**, with Magmaw, the boundary-aware attack, and the BPDA gradient-obfuscation check.
4. **P2** — reuses the attack harness; a second, independent finding.
5. **P3** if scope allows — completes "the security cost of deployability" as a unifying thesis.

The three together say one thing: **the field made SemCom deployable, and nobody checked what
that cost in security.** That is a paper-shaped claim, and every piece of it is measurable
with what you already have.

---

## 10. Source and confidence table

| Claim | Source | Confidence |
|---|---|---|
| Analog DeepJSCC needs 14–16× attack power vs classical; mechanism is decoder Lipschitz smoothness | [2603.24082](https://arxiv.org/abs/2603.24082), **PDF read directly** | High — verified in primary text |
| That paper never studies digital/constellation-constrained SemCom | Same, **"constellation" = 0 occurrences in 17 pages** | High |
| Quantisation-boundary crossing named as a digital-unique threat, unmeasured | [2512.24602](https://arxiv.org/html/2512.24602v7) | High |
| Security metrics for digital SemCom are unsolved (§V-A) | Same, authors' own wording | High |
| Joint crypto+modulation "largely unaddressed" (§V-B) | Same | Medium — contradicted in part by ECM/CD-PHY/DeepJSCEC |
| ISFR/SI-UEP has no security discussion | [2604.00595](https://arxiv.org/html/2604.00595) | High |
| Magmaw: universal, SDR-validated, −5.88 dB PSNR, public code | [2311.00207](https://arxiv.org/abs/2311.00207), NDSS 2025 | High |
| DeepJSCC-Q has never been adversarially evaluated | Negative search result | **Medium — absence of evidence; re-verify** |
| MambaJSCC / NTSCC never adversarially evaluated | Negative search result | **Medium — same caveat** |
| No cross-architecture adversarial comparison exists | Negative search result | **Medium** |
| Encryption preserving graceful degradation is solved | [DeepJSCEC 2208.09245](https://arxiv.org/pdf/2208.09245) | High — do not re-invent |

**If you cite one line from this document, cite the caveat on the last four rows.** The
strongest ideas here rest on negative results, which are the weakest evidence available and
the most likely to be overtaken.
