# The Roadmap of Semantic Communication
### Development, Subfields, and How Practical It Really Is

*Complete edition. This merges the original roadmap with the expansion pack: the full
treatments of ADJSCC (§3.1), the analog-symbol debt (§2.3) and DeepJSCC-Q (§3.6) are folded
into their proper places rather than left in a companion file, and the worked two-pipeline
example now sits at §6.5 where it belongs.*

*A note on provenance, carried over from both source documents: everything attributed to
Xu et al., Tung et al., or any other cited author is a reported result under those authors'
own experimental conditions, not independently reproduced here. The worked example in §6.5
is **constructed for illustration** — the geometry, symbol budgets and code-block arithmetic
are computed from published 5G NR and DeepJSCC-Q system parameters, but the end-to-end
quality figures are indicative of the shapes reported in the literature, not reproductions
of a specific measured experiment.*

*A conceptual-but-rigorous survey, tracing the field from its foundational deep joint source–channel coding (DeepJSCC) works to its current standardization push in 6G. Emphasis is placed on a question that is easy to gloss over: do these systems actually handle real channel properties, modulation, and the transmission bottleneck between encoder and decoder — or do they quietly assume idealized conditions?*

---

## 0. How to read this document

Semantic communication (SemCom) is often presented as a clean break from Shannon-era "bit-centric" communication. That framing is useful but can be misleading, because it hides the fact that the field's real progress has been a *staircase of practicality*: each generation of work relaxed one unrealistic assumption of the previous one. Understanding the roadmap means understanding which assumption each step removed.

So this document is organized around two intertwined threads:

1. **The intellectual lineage** — how we got from Shannon/Weaver's three-level model to DeepSC, DeepJSCC, and today's rate-adaptive, digital, channel-aware systems.
2. **The practicality audit** — for each idea, whether it genuinely closes the gap to a deployable radio (real fading, finite constellations, hardware, bandwidth constraints) or whether it works only in an idealized sandbox.

The uploaded standardization roadmap paper (Zhang et al., *"Towards Native AI in 6G Standardization: The Roadmap of Semantic Communication"*) is used as one input — mainly for the standardization and architecture framing — but the technical backbone comes from the primary research literature, cited throughout.

---

## 1. What "semantic communication" actually means (and the three-level framing)

The conceptual root is Weaver's introduction to Shannon's 1949 monograph, which split communication into three levels: the **technical** problem (how accurately can symbols be transmitted?), the **semantic** problem (how precisely do the transmitted symbols convey the desired meaning?), and the **effectiveness** problem (how well does the received meaning drive the desired conduct?). Classical digital communication solved the technical problem almost perfectly and deliberately ignored the other two. SemCom is the attempt to engineer the semantic and effectiveness levels directly.

In practice this means a shift in the optimization target:

- **Bit-centric systems** minimize bit error rate (BER) or symbol error rate (SER), treating every bit as equally valuable.
- **Semantic systems** minimize a *task-relevant distortion* — reconstruction fidelity of what matters, a downstream task's accuracy, or the alignment of recovered intent — and are free to discard bits that don't serve that goal.

This is not merely philosophical. It changes what the encoder is allowed to do: it can throw away information that is irrelevant to the task, and it can allocate protection unequally according to semantic importance. That single freedom is the engine behind most of the gains reported in the literature.

Two clarifications that prevent confusion later:

- **"Semantic" does not necessarily mean linguistic meaning.** For an image classifier, the "semantics" are whatever features determine the class. The term is about *task-relevance*, not about language.
- **SemCom and JSCC are overlapping but not identical.** JSCC is a *technique* (jointly designing source and channel coding). SemCom is a *goal* (transmitting meaning). Deep-learning JSCC turned out to be the most successful vehicle for SemCom, which is why the two are often used almost interchangeably — but one can do task-oriented SemCom without JSCC, and one can do JSCC purely for reconstruction without any "semantic" task.

---

## 2. The foundational works: where the modern field starts

Two papers, both around 2019–2021, define the modern starting point. They attacked the problem from different modalities and with different philosophies.

### 2.1 DeepSC (2020–2021): the task/meaning branch, for text

Xie, Qin, Li, and Juang proposed **DeepSC**, a Transformer-based end-to-end system for text transmission. Instead of minimizing bit or symbol errors, DeepSC is trained to recover the *meaning of sentences*, and it uses a mutual-information term in its loss to make the learned representation channel-robust. The reported headline result — and the one that recurs throughout the field — is that it is **more robust to channel variation and performs especially well in the low-SNR regime**, where conventional bit-pipelines fail catastrophically. DeepSC also used transfer learning so the model could adapt across channel environments without full retraining. This established the template: an autoencoder trained end-to-end with the physical channel modeled as a (differentiable) layer in the middle.

### 2.2 DeepJSCC (2019): the signal/reconstruction branch, for images

Bourtsoulatze, Burth Kurka, and Gündüz proposed **deep joint source–channel coding for wireless image transmission**. The scheme "departs from the conventional use of explicit source and channel codes ... and directly maps the image pixel values to the complex-valued channel input signal." Architecturally it is an autoencoder with a *non-trainable middle layer that represents the noisy channel*, so gradients flow through a differentiable model of the channel during training.

The single most important property this work demonstrated is the **absence of the "cliff effect."** In a conventional separated design (e.g., BPG image compression + LDPC channel code), performance is excellent above the code's design SNR and then collapses abruptly below it — the "cliff." DeepJSCC instead degrades *gracefully*: as SNR drops, image quality declines smoothly rather than falling off a cliff. This graceful degradation is repeatedly cited as the killer property motivating the whole line of work, and it maps directly onto the standardization paper (Zhang et al.)'s claim that JSCC "mitigates the cliff effect ... and enables smooth performance degradation."

### 2.3 What these foundational works assumed — the practicality debt

Here is the crux of your question, stated plainly. The original DeepJSCC made an assumption that is elegant for training but problematic for deployment: **the encoder outputs continuous complex-valued channel symbols** — "any complex value can be transmitted over the channel." This is called *analog* JSCC. It is fundamental to the design because end-to-end training by backpropagation requires a *differentiable* channel-input representation, and mapping to a discrete constellation (QAM, PSK) is a non-differentiable step.

The problem: real radios do not transmit arbitrary complex numbers. Cellular hardware, RF chains, and standards operate on **finite digital constellations** (QPSK, 16-QAM, 64-QAM, ...). So the beautiful original DeepJSCC, as published, could not be dropped into a 5G/6G physical layer. Almost the entire subsequent history of "making SemCom practical" is the story of paying down this debt and several sibling debts (SNR-adaptivity, real fading, bandwidth flexibility, hardware validation). The next sections walk through each.

---


> *The full treatment below is folded in from the companion expansion pack. Everything
> attributed to the cited authors is a reported result under their own experimental
> conditions.*

#### §2.3-E — "The encoder outputs continuous complex-valued channel symbols": what this actually costs

The parent document identifies this as the field's foundational practicality debt. This section takes it apart properly, because the reasons it is a problem are more numerous and more structural than "the hardware can't do it."

##### 2.3-E.1 The assumption, stated formally

In DeepJSCC and its immediate descendants, the encoder is a map

$$f_\theta : \mathbb{R}^n \to \mathbb{C}^k$$

subject only to an **average** power constraint, typically \(\tfrac{1}{k}\|z\|_2^2 \le \bar{P}\). The image of \(f_\theta\) is a \(2k\)-real-dimensional continuum. Any point in \(\mathbb{C}^k\) inside the power ball is a legal transmission. There is no constellation, no bit, no codeword index, no discrete object anywhere in the transmitter.

Tung et al. state the consequence precisely: DeepJSCC does not merely merge source and channel coding — **it removes the constellation diagram entirely.**

##### 2.3-E.2 Why the assumption exists

It is not laziness; it is forced by the training method. End-to-end optimisation requires gradients to flow from the reconstruction loss, through the decoder, through the channel model, and back into \(\theta\). This requires the channel-input representation to be differentiable in \(\theta\). Mapping a real-valued network output to the nearest point of a finite constellation is a step function: its derivative is zero almost everywhere and undefined on the decision boundaries. Insert that operation and training stops.

So the analog assumption is a **training-time convenience that leaks into the system design.** Almost everything in §3.6-E is about paying it back without losing trainability.

##### 2.3-E.3 The obstacles, itemised

It is easy to state the objection as "commercial radios use QAM." That is the headline, but it undersells the problem by about six items.

**1. Standards enumerate constellations, and modulators are hard-coded.** This is the direct form. Tung et al. give the concrete cases: IEEE 802.11ad's PHY specifies BPSK, QPSK, 16-QAM and 64-QAM; IEEE 802.15.4's PHY specifies OQPSK and no channel coding at all. 5G NR is the same in kind — QPSK, 16-, 64-, 256-QAM (with 1024-QAM downlink in later releases, subject to UE capability). Modulator hardware is fixed-function for efficiency: a lookup table and a mapper, not a general-purpose I/Q source. An encoder that wants to emit \(0.3172 - 0.8814j\) has nowhere to put it. The observation that follows is theirs and is the crux: **DeepJSCC as published requires custom hardware.**

**2. There is no bit interface, so the entire stack above the PHY has nothing to attach to.** This is the largest and least-discussed cost. Digital systems are built on a bit pipe, and an enormous amount of machinery assumes one exists:
   - **CRC** — computed over bits. No bits, no error detection, no way to know whether what arrived is what was sent.
   - **HARQ** — requires (a) detecting failure via CRC and (b) soft-combining retransmissions via LLRs over coded bits. Neither is defined for a continuous latent.
   - **Ciphering and integrity protection** — block and stream ciphers operate on bitstrings. There is no standard way to encrypt a point in \(\mathbb{C}^k\) without first serialising it, which reintroduces the bit interface you removed.
   - **Scrambling, interleaving, rate matching** — all bit-domain operations in NR, all undefined here.
   - **MCS-based link adaptation** — the entire mechanism by which a base station adapts is "pick an index from a table of (modulation order, code rate) pairs." An analog encoder has neither.
   - **Segmentation and layering** — MAC/RLC/PDCP hand down transport blocks measured in bits.

   The point is that "make it digital" is not one fix but the precondition for the system being able to participate in a protocol stack at all.

**3. RF front-end economics: PAPR, DAC resolution, and power-amplifier backoff.** An unconstrained encoder output trained under an average power constraint tends toward an approximately Gaussian amplitude distribution, which has a long tail and therefore a high peak-to-average power ratio. Power amplifiers are efficient only near saturation; a high-PAPR signal forces backoff, which wastes battery, or drives the PA into nonlinearity, which distorts the signal in ways the decoder was never trained on. Standardised constellations are designed with bounded, well-characterised amplitude sets partly for this reason. (Note that the OFDM-guided work of §3.3 addresses PAPR by training with deliberate clipping — the same debt attacked from a different direction.)

**4. Conformance testing has no reference to test against.** This one is fatal for standardisation specifically. RF transmitter conformance in 3GPP is measured largely by **error vector magnitude** — the distance between the transmitted symbol and the ideal constellation point it was supposed to be, with limits that tighten as constellation order rises. EVM is *defined* relative to a reference constellation. An encoder that emits arbitrary complex values has no reference constellation, so EVM is undefined, so there is no way to write a conformance test, so there is no way to certify a device. Standardisation is not merely inconvenienced by analog output; it is structurally unable to describe it.

**5. Interoperability and the two-sided model problem become unbounded.** With a fixed constellation, the transmitter's output alphabet is a public, finite, specified object; only the *mapping* is proprietary. With analog output, the transmitted waveform is entirely determined by a specific set of trained weights. A receiver from a different vendor cannot decode it, cannot partially decode it, and cannot fall back to a standard mode. The two-sided model lifecycle problem that 3GPP is wrestling with for CSI feedback becomes strictly harder when there is no discrete interface to anchor the contract.

**6. Spectral and regulatory characterisation.** Emission masks, occupied bandwidth, and spurious emission limits are all characterised for known signal statistics. A learned continuous signal with data-dependent, model-dependent statistics is harder to bound analytically and harder to argue through certification.

**7. Silicon economics.** Even if all of the above were solved, "custom hardware" means a modem that cannot amortise across the existing volume of standards-compliant parts. This is not a technical obstacle but it is a decisive commercial one, and it is the real reason the DeepJSCC-Q line of work exists.

##### 2.3-E.4 The steelman, and why it does not save the assumption

The obvious rebuttal: *transmission is ultimately analog anyway.* A QAM symbol is not fundamentally more physical than an arbitrary complex number; both are realised as voltages through a DAC and a mixer. Finite-resolution DACs quantise the waveform regardless. So why is the constellation constraint real?

Tung et al. concede the premise and answer the question in one move: transmission is indeed ultimately analog whether the scheme is digital or DeepJSCC, **but commercially available hardware implements a particular communication standard**, and that standard includes predefined constellation order and design. The constraint is not a law of physics; it is a **property of the deployed ecosystem** — hardware, protocol, test regime, certification, and installed base. That makes it, if anything, *more* binding than a physical constraint, because you cannot engineer your way around it with a better circuit. You have to comply with it or build a parallel ecosystem.

There is also a narrower technical point worth keeping: a DAC's uniform quantisation grid is not the same object as a constellation. A 12-bit DAC quantises a *waveform sample*; a constellation constrains the *symbol alphabet* at the symbol rate, after pulse shaping. Satisfying the former does not satisfy the latter.

##### 2.3-E.5 The solution space

Four families of answer exist, and it is useful to see them as a spectrum rather than as competitors:

| Approach | Mechanism | Trainability recovered by | Representative |
|---|---|---|---|
| **Deterministic quantisation** | Encoder emits continuous latent; a quantiser snaps to the nearest constellation point | Soft-to-hard relaxation: hard in forward pass, soft (softmax-weighted) gradient in backward pass | **DeepJSCC-Q** |
| **Probabilistic mapping** | Encoder emits a *distribution* over constellation points; a point is sampled | Reparameterisation / Gumbel-softmax; the transition probability is what is learned | **JCM-VAE** |
| **Learned codebook / VQ** | Latent is vector-quantised against a learned codebook; the index is then modulated conventionally | Straight-through estimator plus commitment loss (VQ-VAE family) | KB/codebook SemCom (§4.5) |
| **Accept analog on a dedicated waveform** | Keep continuous symbols, deploy on SDR or a non-standard air interface | Nothing needs recovering | ASCViT-JSCC prototypes (§3.7) |

The rest of this document takes the first path apart in detail.


## 3. The practicality staircase: relaxing one idealization at a time

This is the heart of the roadmap, and directly answers "how practical are current works, especially regarding channel properties, modulation, and the transmission bottleneck between encoder and decoder." Each subsection is one step up the staircase.

### 3.1 Step 1 — Adapting to *SNR* with a single model (ADJSCC, 2020–2021)

The first embarrassing limitation of early DeepJSCC: a model trained at one SNR performs poorly if the deployment SNR differs, so you would need a separate network per SNR — computationally wasteful and storage-heavy.

Xu, Ai, Chen et al. solved this with **Attention DL-based JSCC (ADJSCC)**. The idea borrows the *resource-allocation* intuition from classical JSCC: when the channel is good, spend fewer resources on protection; when it is bad, spend more. ADJSCC realizes this with **channel-wise soft attention** that rescales encoded features according to the current SNR, letting a *single* network operate across a wide SNR range. The SNR-adaptive "attention feature" module became a standard building block that later works (including the CSI-feedback schemes) reuse.

**Practicality verdict:** genuine progress. The encoder now reacts to a real channel property (SNR). But note the hidden assumption — the SNR must be *known/fed back* to the transmitter, which is itself a signaling cost later works try to reduce (e.g., via coarse CQI indices).


> *The full treatment below is folded in from the companion expansion pack. Everything
> attributed to the cited authors is a reported result under their own experimental
> conditions.*

#### §3.1-E — Attention DL-based JSCC (ADJSCC), in full

**J. Xu, B. Ai, W. Chen, A. Yang, P. Sun, M. Rodrigues, "Wireless Image Transmission Using Deep Source Channel Coding With Attention Modules," *IEEE Trans. Circuits Syst. Video Technol.*, 32(4):2315–2328, Apr. 2022. (arXiv:2012.00533; code at github.com/alexxu1988/ADJSCC.)**

##### 3.1-E.1 The precise problem

The parent document states the limitation loosely as "a model trained at one SNR performs poorly if the deployment SNR differs." It is worth being exact about *why* this is worse than it sounds, because the shape of the problem determines the shape of the fix.

A DeepJSCC encoder is a function \(f_\theta : \mathbb{R}^n \to \mathbb{C}^k\) trained by minimising \(\mathbb{E}[d(x,\hat{x})]\) where the expectation is over the source *and over the channel noise*. The noise variance \(\sigma^2\) is a hyperparameter of the training distribution, not an input to the network. So the learned encoder is implicitly optimised for one noise level. What does it optimise *for* that level? It is trading off two things:

- **How much source detail to pack into the \(k\) available complex dimensions.** More detail per dimension means the latent is less redundant, so each dimension carries more distinct information and noise on it is more damaging.
- **How much redundancy/margin to keep.** A latent whose components are spread out and mutually redundant survives noise better but represents the image more coarsely.

This is exactly the classical source-rate versus channel-rate trade-off, except that the network has learned an unnamed, continuous version of it and baked one operating point into its weights. Deploy that network 10 dB away from where it was trained and the operating point is simply wrong — in one direction it wasted capacity on redundancy it did not need, in the other it packed detail that the noise destroys.

The two obvious workarounds are both bad:

1. **Train one model per SNR and switch.** Correct, but the storage and training cost scale linearly in the number of operating points, which is fatal for the resource-constrained devices (IoT, sensors, embedded modems) that are supposed to be the beneficiaries. The paper quantifies this precisely; see §3.1-E.6.
2. **Train one model over a random SNR range with no conditioning.** Cheap, but the model can only learn a compromise representation that is optimal nowhere. This is the naive baseline that ADJSCC must beat to justify itself.

ADJSCC's claim is that there is a third option: **one set of weights, conditioned at inference time on the current SNR**, which beats both. The strong version of the claim — and the one the paper actually demonstrates — is that it beats a per-SNR-trained model *even when that model is tested at exactly the SNR it was trained for*.

##### 3.1-E.2 The classical intuition being formalised

The authors are explicit that the design is a translation of the resource-assignment strategy from traditional concatenated source–channel coders into the neural setting. In the classical picture, with a fixed total bit budget:

- **Low SNR:** allocate more bits to the channel encoder (stronger code, lower rate), fewer to the source encoder (coarser image). Redundancy buys survival.
- **High SNR:** allocate fewer bits to the channel encoder (weaker code, higher rate), more to the source encoder (finer image). Detail buys quality.

The insight in ADJSCC is that in a monolithic neural encoder there is no bit-allocation knob to turn — but there *is* a set of intermediate feature channels, and turning some of them up while turning others down is a continuous analogue of the same reallocation. If the network has learned some feature channels that carry robust, coarse, low-frequency structure and others that carry fragile high-frequency detail, then scaling the fragile ones down at low SNR and up at high SNR *is* the classical strategy, expressed in feature space rather than in bits.

Whether the network actually organises itself that way is an empirical question. §3.1-E.7 shows the evidence that it does.

##### 3.1-E.3 System model

The system is a point-to-point image transmission link **with an SNR feedback path**. This is the key structural difference from the original DeepJSCC and it must not be glossed over, because it is the assumption ADJSCC buys its gains with.

An image of size \(H \times W \times C\) is flattened to \(x \in \mathbb{R}^n\), \(n = H \times W \times C\). The encoder takes **two** inputs:

$$z = f_\theta(x, \mu) \in \mathbb{C}^k$$

where \(\mu \in \mathbb{R}\) is the channel SNR in dB, estimated at the receiver and fed back to the transmitter. An average power constraint \(\tfrac{1}{k}\mathbb{E}(zz^*) \le 1\) is imposed by a power-normalisation layer.

The channel is AWGN, \(\hat{z} = z + \omega\), \(\omega \sim \mathcal{CN}(0,\sigma^2 I)\); the paper notes that a fading channel \(\hat{z} = hz + \omega\) reduces to the same form after receiver equalisation, with a modified noise distribution. The decoder is likewise SNR-conditioned:

$$\hat{x} = g_\phi(\hat{z}, \mu) \in \mathbb{R}^n$$

Distortion is MSE; the reported metric is PSNR. The bandwidth ratio is \(R = k/n\).

Two things to notice in this model, both of which matter for §3.1-E.8:

- **Both ends are conditioned on \(\mu\).** The receiver has it natively (it estimated it); the transmitter has it only via feedback. So ADJSCC is not a drop-in replacement for a feedback-free link.
- **The channel input is still \(\mathbb{C}^k\) — arbitrary complex values.** ADJSCC removes the fixed-SNR idealisation and leaves the analog-symbol idealisation entirely untouched. This is the canonical illustration of this document's observation that the staircase steps do not compose automatically.

##### 3.1-E.4 Architecture: FL modules and AF modules

The network is built from two alternating module types.

**Feature Learning (FL) modules** are the conventional part, inherited from prior DeepJSCC work rather than invented here. In the baseline architecture — which the paper calls **BDJSCC** ("basic" DL-based JSCC), taken from Kurka and Gündüz's DeepJSCC-F — the encoder is five modules: the first four are `Conv → GDN → PReLU`, and the fifth is `Conv → GDN`. The decoder mirrors this with five modules of `TransposedConv → GDN → PReLU`, the last substituting a sigmoid for the PReLU. Convolutions are parameterised as \(F \times F \times K | S\) (filter size, filter count, stride). Around this sit a normalisation layer, a reshape layer, and the power-normalisation layer.

**Attention Feature (AF) modules** are the contribution. Each FL module is followed by an AF module *except the last FL module* in both encoder and decoder — so four AF modules per side. The output of FL module \(i\) is one input to AF module \(i\); the other input is the scalar SNR \(\mu\). The AF module's output feeds FL module \(i+1\).

The bandwidth ratio \(R\) is set by the output channel count of the encoder's final convolution, exactly as in the baseline. Nothing about the AF mechanism is entangled with the rate — which is why later work reuses the AF module freely.

A terminological trap, flagged in the paper's own footnote and worth repeating: **"channel-wise" refers to the feature channels of the tensor, not the communications channel.** ADJSCC does not attend over the wireless channel; it attends over convolutional feature maps, using the wireless channel's SNR as a conditioning scalar.

##### 3.1-E.5 The AF module, step by step

Let \(F^G = [F_1^G, \dots, F_c^G] \in \mathbb{R}^{h \times w \times c}\) be the FL module's output: \(c\) feature maps of spatial size \(h \times w\).

**(a) Context extraction.** Convolutional features are local — a small kernel cannot see beyond its receptive field. Global average pooling collapses each feature map to a single scalar summarising it globally:

$$I_i^G = \frac{1}{h \times w}\sum_{j=1}^{h}\sum_{k=1}^{w} u_{jk} \in \mathbb{R}$$

These \(c\) scalars are then **concatenated with the SNR** to form the context vector:

$$I = (\mu,\, I_1^G,\, I_2^G,\, \dots,\, I_c^G) \in \mathbb{R}^{c+1}$$

This single concatenation is the entire mechanism by which channel state enters the network. It is worth pausing on how minimal it is: one extra scalar appended to a pooled descriptor.

**(b) Factor prediction.** A small two-layer MLP maps context to per-channel scaling factors:

$$S = P_\omega(I) = \sigma\!\left(W_2\,\delta(W_1 I + b_1) + b_2\right) \in \mathbb{R}^c$$

with \(\delta\) = ReLU and \(\sigma\) = sigmoid. The sigmoid is load-bearing: it confines every factor to \((0,1)\), making the module a *gate* that can only attenuate, never amplify. Deliberately kept to two FC layers to avoid inflating complexity — see the parameter count in §3.1-E.6.

**(c) Feature recalibration.** Channel-wise multiplication:

$$F_i^A = S_i \cdot F_i^G, \qquad i = 1, \dots, c$$

That is the whole module. Structurally it is a Squeeze-and-Excitation block with the SNR spliced into the squeeze vector — but the *interpretation* is what makes it a communications contribution rather than a vision one: the gate is a learned, continuous, per-feature analogue of the classical source/channel bit split.

**A concrete trace.** Take the first AF module of the encoder with \(c = 256\) feature maps at \(h \times w = 16 \times 16\). Global average pooling produces 256 scalars. Concatenating \(\mu = 4\) dB gives a 257-vector. The MLP emits 256 gates; suppose feature map #23 (which, per the paper's own visualisation, responds to fine texture detail) receives \(S_{23} = 0.21\). Its activations are scaled to roughly a fifth. Now hold the image fixed and re-run with \(\mu = 19\) dB: the same map receives, say, \(S_{23} = 0.88\), and its detail passes through nearly intact. Downstream layers therefore see a materially different tensor, and the final latent allocates its \(k\) complex dimensions differently — *from the same weights*. This is the mechanism in one sentence.

##### 3.1-E.6 Training protocol and cost

| Item | Setting |
|---|---|
| Framework | TensorFlow / Keras |
| Optimiser | Adam, learning rate \(10^{-4}\) |
| Batch size | 128 (CIFAR-10); 16 (ImageNet) |
| Epochs | 1280 (CIFAR-10); 2 (ImageNet, sufficient for convergence) |
| Primary dataset | CIFAR-10, 50k train / 10k test, \(32\times32\times3\) |
| Large-image dataset | ImageNet, ~5.8M crops of \(128\times128\); evaluated on Kodak (24 images, \(768\times512\)) |
| ADJSCC \(\text{SNR}_\text{train}\) | \(\mathcal{U}[0, 20]\) dB, resampled per example |
| BDJSCC \(\text{SNR}_\text{train}\) | Fixed at 1, 4, 7, 13, 19 dB (separate models) |
| Bandwidth ratios | \(R = 1/12,\ 1/6\) |
| Test averaging | Each test image transmitted 10× (CIFAR) / 100× (Kodak) |

The training loop is the only place where sampling \(\mu\) matters: the SNR is drawn fresh per example, the channel layer is instantiated at that noise power, and \(\mu\) is fed to every AF module at both ends. The network therefore never sees a fixed channel and cannot collapse to a single operating point.

**Cost of the mechanism.** BDJSCC has 10,690,351 parameters (40.78 MB at float32); ADJSCC has 10,758,191 (41.04 MB). That is **+0.6% parameters**. Training time per batch rises from ~110 ms to ~114 ms (+3.6%); inference from ~49 ms to ~53 ms (+8.1%). The AF modules are, in engineering terms, nearly free — a fact that goes a long way toward explaining why the module was adopted so widely by later work.

##### 3.1-E.7 Reported results

**Adaptability (CIFAR-10, AWGN).** At \(R = 1/12\), ADJSCC outperforms *every* BDJSCC model at *every* test SNR in \([0,20]\) dB. Against the model trained at 1 dB, the margin at high test SNR reaches about 6 dB PSNR. Against the models trained at 13 or 19 dB, ADJSCC wins at low test SNR. Critically, ADJSCC also wins **at the matched point** — where \(\text{SNR}_\text{test} = \text{SNR}_\text{train}\) for the specialist model. A single generalist beating each specialist on the specialist's home turf is the paper's most striking claim, and the natural reading is that exposure to a range of noise levels acts as a regulariser and forces a better-organised feature hierarchy.

At \(R = 1/6\) the picture is similar, except the gap against the high-SNR specialists largely closes in the high-SNR regime. The conclusion drawn: **ADJSCC's advantage is largest at low bandwidth ratio**, i.e. precisely where the encoder is most starved of dimensions and allocation decisions matter most.

**Storage.** The comparison that most directly kills the "just train several models" workaround, on CIFAR-10 at \(R=1/6\), averaged over the SNR range:

| Strategy | Models | Storage | PSNR |
|---|---|---|---|
| **ADJSCC** | 1 (SNR-conditioned) | **41.04 MB** | **29.831 dB** |
| BDJSCC-1 | 1 @ 10 dB | 40.78 MB | 24.474 dB |
| BDJSCC-2 | 2 @ 5, 15 dB | 81.56 MB | 28.495 dB |
| BDJSCC-5 | 5 @ 2, 6, 10, 14, 18 dB | 203.9 MB | 29.694 dB |
| BDJSCC-10 | 10 @ 1, 3, …, 19 dB | 407.8 MB | 29.826 dB |

Read the first and last rows together. Ten specialist models — a 10× storage bill and a 10× training bill — buy 29.826 dB. One ADJSCC buys 29.831 dB. The authors put it as: ADJSCC needs **10.06% of the storage and 10.36% of the training time** of BDJSCC-10 for the same quality. And a single fixed model at a sensible midpoint (BDJSCC-1) is 5.3 dB worse, which is a large gap in image terms.

**Robustness to SNR mismatch.** Because ADJSCC depends on fed-back SNR, the obvious attack on it is: what if the feedback is wrong? The paper tests this and finds ADJSCC does lose some performance under mismatch, but still beats BDJSCC under the same mismatch — with the advantage largest at high feedback SNR. The mechanism degrades gracefully rather than becoming a liability, which matters because feedback SNR in a real system is always stale and always noisy.

**Versatility (large images).** Trained on ImageNet crops, evaluated on Kodak at \(R=1/6\): ADJSCC matches the *ensemble* of BDJSCC models for \(\text{SNR}_\text{test} \le 17\) dB with negligible difference, and falls 0.3 dB behind above 17 dB — a regime where PSNR already exceeds 35 dB and differences are visually indistinguishable. The fully convolutional architecture is what permits training at \(128\times128\) and testing at \(768\times512\); a cross-experiment confirms the asymmetry, with ImageNet→CIFAR transfer costing only 1–3 dB while CIFAR→Kodak transfer is severely degraded. Train on the complex, high-resolution distribution.

**Against a classical JSCC baseline.** ADJSCC is also compared against a traditional JSCC (TJSCC) scheme: SPIHT source coding + CRC + rate-compatible punctured convolutional codes with fast unequal error protection, BPSK modulation, at twice the bandwidth ratio to compensate for TJSCC's real-valued output. TJSCC improves quickly from 0–2 dB, slows from 3–15 dB, and **saturates above 15 dB** — the classical signature of a scheme whose source rate is fixed once the channel is good enough. ADJSCC's reported advantage runs from about 7 dB PSNR at \(\text{SNR}_\text{test}=3\) dB to about 13 dB at 20 dB, and it does so without TJSCC's per-image rate-distortion computation at the transmitter.

##### 3.1-E.8 What the attention actually learned — the interpretability result

This is the part of the paper most often skipped in citations and the part most useful for the roadmap's thesis, because it is rare evidence that a learned communications system organised itself the way the designers hoped.

The authors log the mean and standard deviation of the scaling factors \(S\) produced by each of the encoder's four AF modules, across the test set, at a range of test SNRs. Two patterns emerge.

**Pattern 1 — the gates get more selective as SNR rises.** At low SNR the scaling factors across feature channels are relatively flat; at high SNR they fluctuate much more sharply between channels. The interpretation given is the natural one: when noise is severe, *every* feature is compromised roughly equally, so there is little to be gained by discriminating between them. When the channel is clean, some features contribute much more to reconstruction quality than others, and it pays to boost those and suppress the rest to avoid spending power on low-value content. The network discovered that discrimination is worth more in good conditions.

**Pattern 2 — SNR-dependence concentrates in the early layers.** The spread between the scaling-factor curves at different SNRs *shrinks* with depth. By the fourth AF module, the curves at different SNRs are nearly identical. The reading offered: channel noise damages low-level features (pixel-level relationships, texture, edges) far more than high-level ones (the semantic content of the scene). High-level features are intrinsically noise-robust, so there is little reason to modulate them by SNR.

Pattern 2 is a genuinely important result for semantic communication as a field, and it is worth stating its implication bluntly: **the network independently rediscovered that meaning is more robust to channel noise than pixels are.** The whole premise of SemCom — that transmitting task-relevant abstraction is more channel-efficient than transmitting samples — appears here not as an assumption but as an emergent property of a network that was only ever asked to minimise MSE.

The accompanying visualisation reinforces it: a heatmap of one particular encoder feature (map #23, on a Kodak image of caps) shows fine detail strongly enhanced at 19 dB and suppressed at 1 dB. Detail is what you spend power on when you can afford it.

##### 3.1-E.9 Practicality verdict, expanded

**Genuine progress, with an itemised bill.**

*What it fixed:* the fixed-SNR idealisation, at essentially zero parameter cost, with a module that composes cleanly into other architectures. The storage argument is decisive against the model-ensemble alternative. The interpretability result is a bonus that later work (importance-aware allocation, semantic UEP) builds on directly.

*What it did not fix, and what it added:*

1. **It still emits arbitrary complex symbols.** \(f_\theta: \mathbb{R}^n \times \mathbb{R} \to \mathbb{C}^k\). The analog-symbol debt of §2.3 is untouched. ADJSCC is not deployable on standard hardware for exactly the reasons DeepJSCC is not.
2. **It requires an SNR feedback channel to the transmitter,** and the SNR is consumed as a continuous scalar. Real systems report quantised CQI on a slow, delayed schedule. The paper's mismatch experiments show the degradation is graceful, but the signalling requirement is a new dependency, not a free lunch — and reconciling "continuous \(\mu\)" with "4-bit CQI reported every few slots" is real integration work that later CSI-oriented schemes have to do.
3. **AWGN only.** Fading is handled by the standard argument that receiver equalisation reduces it to an equivalent AWGN model — true for flat fading, and silent on frequency-selective multipath, which is why the OFDM-guided work of §3.3 was needed as a separate step.
4. **Fixed bandwidth ratio per model.** \(R\) is set by the encoder's final channel count; changing it means retraining. NTSCC (§3.5) is the answer, not ADJSCC.
5. **The conditioning is on SNR only.** Not on delay spread, Doppler, interference, or antenna configuration. The AF module's context vector is \((\mu, I^G)\) — one channel scalar. Generalising the context is an obvious extension and a good part of why the module proved so reusable.

*Where it sits in the staircase:* ADJSCC is the cleanest single demonstration that **conditioning beats specialisation** in learned physical-layer systems, and the AF module is the field's most-reused building block. But it is one step, and it explicitly does not compose with the others for free.


### 3.2 Step 2 — Flexible *bandwidth* / rate (bandwidth-agile and successive-refinement JSCC, 2019–2021)

Early models were trained for one fixed *bandwidth ratio* (channel symbols per source sample). Kurka and Gündüz introduced **bandwidth-agile** transmission and **successive refinement**, where the image is sent in layers so that more channel uses progressively improve quality, and the system can stop at whatever bandwidth is available. This decouples the model from a single hard-coded rate.

**Practicality verdict:** important for deployment, because real networks allocate variable bandwidth. But early rate flexibility was coarse; fine-grained, content-aware rate control arrives with NTSCC (Step 5).

### 3.3 Step 3 — Handling real *fading* and *frequency selectivity* via OFDM (2021–2022)

AWGN (additive white Gaussian noise) is the textbook channel, but real wireless is **multipath fading** and **frequency-selective**, which causes inter-symbol interference. Yang, Bian, and Kim's **OFDM-guided DeepJSCC** was, by their account, the first implementation of an OFDM-assisted deep JSCC for multipath fading. The key design choice is telling: rather than hoping a black-box network learns to equalize the channel, they **inserted explicit OFDM layers plus classical signal-processing steps (channel estimation, equalization) as differentiable modules**, combined with refinement neural networks. Injecting this "expert domain knowledge" improved convergence and performance versus a naive black-box net, and even beat strong separated baselines (image codec + channel code + OFDM) — with the largest margins, again, at low SNR and low rate. They also trained with deliberate **signal clipping** to control the peak-to-average power ratio (PAPR), a very real hardware constraint, with graceful degradation.

This is a pivotal conceptual lesson for the whole field: **the most practical SemCom systems are hybrids** — learned components wrapped around classical, well-understood physical-layer machinery (OFDM, channel estimation, equalization), not end-to-end black boxes that pretend the physical layer doesn't exist.

**Practicality verdict:** major step toward realism. Real fading, real ISI mitigation, real PAPR handling. Caveats the authors themselves flag: carrier frequency offset, packet detection, and pilot design were left for future work.

### 3.4 Step 4 — Multi-antenna (*MIMO*) channels (2023–2024)

Wu, Shao, Bian, Mikolajczyk, and Gündüz extended the line to **DeepJSCC-MIMO**, a Vision-Transformer-based scheme for both open-loop and closed-loop MIMO. Using self-attention, it jointly learns feature mapping and power allocation adapted to the source and the channel. It reportedly surpasses separation-based baselines while being **robust to channel-estimation errors** and flexible across channel conditions and antenna counts **without retraining**.

**Practicality verdict:** MIMO is the backbone of modern cellular, so this is essential. The robustness-to-estimation-error result matters because perfect CSI is a fiction in real systems.

### 3.5 Step 5 — Content-aware *variable-rate* coding (NTSCC, 2022–2023)

Dai, Wang, Tan, Si, Qin, Niu, and Zhang introduced **Nonlinear Transform Source-Channel Coding (NTSCC)**. Its innovation: alongside the learned nonlinear transform that maps a source into a latent space, it **learns an entropy model as a prior over the latent features**. That entropy model estimates how much information (and thus how many channel resources) each latent element deserves — enabling **variable-rate** JSCC where "important" (high-entropy) features get more bandwidth and protection, and predictable features get compressed harder. NTSCC was reported to outperform *both* standard analog DeepJSCC and classical separation-based digital transmission.

The follow-up, **improved/versatile NTSCC** (Wang, Dai, et al.), added a contextual entropy model, a "response network" so a **single trained model supports various bandwidth ratios and channel states** (avoiding per-configuration retraining), and online latent-feature editing for semantic-guided rate control — explicitly aimed at "deployment-friendly" operation for large-data applications like XR.

**Practicality verdict:** this is where the transmission bottleneck between encoder and decoder becomes genuinely *content-adaptive and standard-friendly*. The entropy-model idea is exactly the "adaptive control of coding length" mechanism the standardization paper (Zhang et al.) describes in its JSCC section, and it is one of the most standardization-relevant primitives in the field.

### 3.6 Step 6 — Closing the *digital modulation* gap (DeepJSCC-Q and JCM-VAE, 2022–2024)

This directly resolves the core debt from Section 2.3. Two complementary approaches emerged.

**DeepJSCC-Q (Tung, Kurka, Jankowski, Gündüz)** constrains the channel input to a **finite constellation** (an M-QAM alphabet) instead of arbitrary complex values. It uses soft-to-hard quantization so training stays differentiable. Findings: it preserves the graceful-degradation property despite the finite alphabet, and its performance **asymptotically approaches unconstrained analog DeepJSCC as the modulation order increases** — a 4096-QAM model performs nearly as well as the analog original. So the practicality cost of going digital can be made small if the hardware supports high-order constellations.

**Joint Coding-Modulation via VAE (JCM)** takes a probabilistic route: rather than mapping deterministically to a constellation point, the network outputs a **transition probability from source data to discrete constellation symbols**, sidestepping the non-differentiability of hard modulation (often via the reparameterization/Gumbel-softmax trick). Because coding and modulation are learned jointly, the modulation strategy can be **matched to the operating channel condition**. It reportedly beats quantization-based digital baselines across a range of channel conditions, rates, and modulation orders.

**Practicality verdict:** this is arguably *the* practicality milestone. After this line of work, "SemCom cannot run on real digital radios" stops being true. The uploaded paper's own case study (Section 6 below) is squarely in this tradition, comparing **JSCCM** (joint source-channel coding *and modulation*) against JSCC+QPSK and fully separated digital baselines.


> *The full treatment below is folded in from the companion expansion pack. Everything
> attributed to the cited authors is a reported result under their own experimental
> conditions.*

#### §3.6-E — DeepJSCC-Q, in full

**T.-Y. Tung, D. B. Kurka, M. Jankowski, D. Gündüz, "DeepJSCC-Q: Constellation Constrained Deep Joint Source-Channel Coding," arXiv:2206.08100, 2022 (IEEE JSAIT).**

##### 3.6-E.1 Problem statement

The encoder becomes a map into a *finite alphabet's* \(k\)-fold product:

$$f : \{0,\dots,255\}^{H\times W\times C} \to \mathcal{C}^k, \qquad \mathcal{C} = \{c_1,\dots,c_M\} \subset \mathbb{C},\ |\mathcal{C}| = M$$

with average power constraint \(\tfrac{1}{k}\sum_i |\bar{z}_i|^2 \le \bar{P}\). The channel is \(y = h\bar{z} + n\), \(n \sim \mathcal{CN}(0,\sigma^2 I)\). Bandwidth compression ratio \(\rho = k/(H\times W\times C)\) in channel symbols per pixel-component. Metrics: PSNR and MS-SSIM.

The authors frame this precisely as **JSCC over a discrete-input AWGN channel**: for a given blocklength there is now a finite set of transmittable codewords, and the task is to find the map from images to codewords, plus a matching decoder, minimising expected end-to-end distortion. Finding those maps directly is intractable, so it is posed as an autoencoder with a **non-differentiable quantisation layer followed by a non-trainable channel layer** in the bottleneck.

Two CSI scenarios are handled:

- **Scenario 1 — full CSI at both ends** (static channel, \(|h|=1\), effectively AWGN). The transmitter can precode: \(\bar{z} \leftarrow \tfrac{h^*}{|h|}\bar{z}\).
- **Scenario 2 — receiver-only CSI** (slow fading, \(h \sim \mathcal{CN}(0,1)\), one realisation per image). The receiver equalises: \(y \leftarrow \tfrac{h^*}{|h|^2} y\), and the transmitter knows only the noise power.

Note what precoding does to the constellation constraint in Scenario 1: the phase rotation is applied *after* quantisation, so the alphabet constraint is satisfied at the mapper and the rotation is a front-end operation — consistent with how a real transmitter works.

##### 3.6-E.2 Architecture

Fully convolutional, and notably heavier than the ADJSCC/BDJSCC lineage:

- **Encoder:** attention module → residual blocks with stride-2 downsampling stages interleaved with plain residual blocks (256 channels throughout) → attention module → quantiser. Residual blocks use `3×3 Conv → LeakyReLU → 3×3 Conv → GDN`. GDN (generalised divisive normalisation) is inherited from the learned-compression literature for its effectiveness in density modelling.
- **Decoder:** mirror structure with `Residual block upsample` stages using **pixel shuffle** (sub-pixel convolution) rather than transposed convolution, for cheaper upsampling with fewer parameters.
- **\(C_{out}\)**, the channel count of the encoder's final tensor, controls \(k\) and hence \(\rho\).

**An important clarification about the attention modules here.** These are the *simplified attention modules* from the learned-image-compression literature (Cheng et al.), whose purpose is to focus the network on image regions needing higher rate. They are **content attention, not SNR conditioning** — they are not ADJSCC AF modules and take no channel-state input. This is why DeepJSCC-Q models are still trained per \(\text{SNR}_\text{train}\) (7, 10, 12, 16 dB across the reported experiments). The two innovations are orthogonal and, in this paper, uncombined. It is a clean instance of this document's §3.8 point: the staircase steps were climbed in separate papers.

##### 3.6-E.3 The soft-to-hard quantiser

This is the mechanism that makes the whole thing trainable. The encoder DNN \(f_\theta\) is *not* constrained to discrete outputs — that would demand an impossibly large output space. Instead it produces an unconstrained latent \(z \in \mathbb{C}^k\), and a separate quantiser \(q_\mathcal{C}\) maps it to the alphabet.

**Hard quantisation (forward pass).** Each element \(z_i\) maps to its nearest constellation point in \(\ell_2\):

$$\bar{z}_i = \arg\min_{c_j \in \mathcal{C}} \|z_i - c_j\|_2^2$$

This is what is actually transmitted. It is a legal QAM symbol. It is also non-differentiable.

**Soft quantisation (backward pass).** A differentiable surrogate: the softmax-weighted convex combination of *all* constellation points, weighted by negative squared distance:

$$\tilde{z}_i = \sum_{j=1}^{M} \frac{e^{-\sigma_q d_{ij}}}{\sum_{n=1}^{M} e^{-\sigma_q d_{in}}}\, c_j, \qquad d_{ij} = \|z_i - c_j\|_2^2$$

where \(\sigma_q\) controls **hardness**. Small \(\sigma_q\): \(\tilde{z}_i\) sits near the centroid of the constellation and gradients flow to every point. Large \(\sigma_q\): the softmax collapses onto the nearest point and \(\tilde{z}_i \to \bar{z}_i\).

**The splice.** Forward uses hard, backward uses soft:

$$\frac{\partial \bar{z}}{\partial z} \;\triangleq\; \frac{\partial \tilde{z}}{\partial z}$$

This is a straight-through-style estimator with a distance-weighted, temperature-controlled relaxation rather than an identity pass-through. The practical consequence is that **the network is always trained on the true transmitted signal**, with only the gradient approximated — which is why the learned encoder does not develop a mismatch between what it thinks it sent and what actually went over the air.

**Annealing schedule.** \(\sigma_q\) is raised during training according to

$$\sigma_q^{(t)} = \min\!\left(100,\ \sigma_q^{(t-1)} + 5\left\lfloor \tfrac{t}{10000}\right\rfloor\right), \qquad \sigma_q^{(0)} = 5$$

Early in training the relaxation is soft, the loss surface is smooth, and the encoder can explore. Late in training the relaxation is nearly hard, and the gradient the encoder receives closely matches the discrete reality it faces at inference. The gap between surrogate and truth is closed gradually rather than assumed away.

**Constellation geometry.** Symbols sit on a uniform square lattice in the complex plane, QAM-style, with maximum amplitude and inter-symbol spacing set so that the average power under a uniform distribution over the alphabet equals \(\bar{P}\). Experiments use \(\bar{P} = 1\), varying \(\sigma^2\) to set SNR.

##### 3.6-E.4 The learned constellation variant (L-\(M\))

An extension: make \(\mathcal{C}\) itself trainable at fixed order \(M\). Gradients reach the constellation points through the same softmax:

$$\frac{\partial \tilde{z}_i}{\partial z_i} = \frac{\partial \mathcal{C}}{\partial z_i}\frac{\partial \tilde{z}_i}{\partial \mathcal{C}}$$

Points are initialised as standard QAM and updated during training, with power renormalised after each update. Because the transmit power depends on how *often* each point is used, the renormalisation needs \(P(c_j)\), which is unavailable — so it is estimated empirically from the softmax weights across a batch:

$$\hat{P}(c_j) = \frac{1}{Bk}\sum_{v=1}^{B}\sum_{i=1}^{k} \frac{e^{-\sigma_q d_{ij}^v}}{\sum_{n=1}^{M} e^{-\sigma_q d_{in}^v}}$$

The softmax weights sum to one, so they can be read as probabilities directly. Renormalisation then enforces \(\sum_j \hat{P}(c_j)|c_j|^2 = \bar{P}\).

The authors contrast this with VQ-EMA (the VQ-VAE family), which needs a separate embedding loss balanced against the reconstruction loss and careful hyperparameter tuning. By treating constellation points as ordinary encoder parameters updated through the softmax, DeepJSCC-Q avoids that balancing act — and the authors report it worked better in their setting.

##### 3.6-E.5 The KL regulariser

A failure mode of any learned discrete bottleneck is **codebook collapse**: the encoder finds a handful of points convenient and never uses the rest, wasting the alphabet. The countermeasure is a KL term pulling the empirical symbol distribution toward uniform over the alphabet:

$$\ell(x,\hat{x}) = d(x,\hat{x}) + \lambda\, D_{KL}\!\left(\hat{P}(\mathcal{C})\,\|\,U(\mathcal{C})\right)$$

with \(d\) = MSE when optimising PSNR, or \(1-\text{MS-SSIM}\) when optimising MS-SSIM.

The empirical tuning of \(\lambda\) is itself informative:

- **AWGN, \(M < 4096\):** \(\lambda = 0.05\). Encouraging uniform use of a small alphabet helps.
- **AWGN, \(M \ge 4096\):** \(\lambda = 0\) worked better. With a very large alphabet, it is *better* to use a favoured subset of points more often than to spread uniformly. This is probabilistic shaping arriving unbidden.
- **Slow fading, any \(M\):** \(\lambda = 0\).

##### 3.6-E.6 Experimental setup

| Item | Setting |
|---|---|
| Framework / optimiser | PyTorch; Adam (\(\beta_1=0.9,\ \beta_2=0.99\)) |
| Learning rate | \(10^{-4}\) (AWGN); \(5\times10^{-5}\) (slow fading); ×0.8 on 4-epoch plateau |
| Batch size / stopping | 32; early stopping patience 8, max 1000 epochs |
| Training data | ImageNet (~1.2M images), random \(128\times128\) crops, 9:1 train/val |
| Evaluation data | Kodak, 24 images at \(768\times512\) |
| \(\text{SNR}_\text{train}\) | 5, 7, 10, 12, 16 dB across experiments |
| Alphabets | BPSK, 16-, 64-, 4096-QAM; learned L-4, L-16, L-64, L-4096 |
| Power | \(\bar{P} = 1\), \(\sigma^2\) varied to set SNR |
| **Baseline** | **BPG source coding + LDPC channel coding**, IEEE 802.11ad codes, blocklength 672 bits, rates 1/2 and 3/4, with BPSK / QPSK / 16-QAM |

##### 3.6-E.7 Results, and the conceptual asymmetry at their centre

**Graceful degradation survives quantisation.** Across every M-QAM alphabet tested, DeepJSCC-Q degrades smoothly as SNR falls. This is the headline: the cliff-free property was not an artifact of continuous channel input. It survives the discrete alphabet, *even when the DeepJSCC-Q scheme and the separation baseline use the same constellation.* The cliff comes from the separation architecture, not from the modulation.

**Approaching analog.** With \(\text{SNR}_\text{train}=16\) dB on AWGN, increasing \(M\) monotonically improves performance, and 4096-QAM performs **nearly identically to unquantised DeepJSCC**. The constellation constraint is therefore not a fundamental barrier — it is a resolution knob whose cost vanishes as resolution grows.

**Against separation.** The 4096-QAM model sits essentially **on the envelope of all the separation-based schemes** — meaning it matches, with one model, the best that BPG+LDPC achieves only by selecting the right (rate, constellation) pair for each SNR. And when 4096-QAM is trained at 7 or 10 dB, it beats the separation baselines convincingly. The 16- and 64-QAM models do *not* uniformly beat separation, which is an honest limitation the authors report rather than bury: at those orders, quantisation error is a real cost. There is also a note that for the modulation orders in their analog-comparison figure, the LDPC codes considered could not decode successfully at all over the tested SNR range.

**The rate–distortion sweep.** Varying \(\rho\), 64-QAM DeepJSCC-Q beats BPG+LDPC-1/2+QPSK in all but one instance; the exception is at \(\rho = 1/6\) with 64-QAM, attributed to the encoder having more degrees of freedom at high bandwidth and thus suffering more from uniform quantisation. Visually, at aggressive compression (\(\rho = 1/48\)) BPG produces conspicuously blurry reconstructions where DeepJSCC-Q does not.

**Now the conceptual point, which is the single most useful thing in the paper.** In a digital separation system, modulation order and rate are the same knob:

> Increasing \(M\) increases the transmission rate, which improves the compressed image quality but *also increases the error probability*. So there is an optimal \(M\) for each SNR, and choosing wrongly is costly. This is what MCS tables exist to manage.

In DeepJSCC-Q, they are different knobs:

> \(k\) is fixed by the architecture. Increasing \(M\) does **not** increase the transmission rate — it increases the number of quantisation levels available for representing the encoder output \(z\), so \(\bar{z}\) approximates \(z\) more faithfully. Higher \(M\) is therefore *monotonically better* at any SNR.

This is why 4096-QAM trained at 16 dB outperforms 64-QAM trained at 10 dB even at SNRs where the 64-QAM model's training point was better matched. And it explains the asymptote: as \(M \to \infty\), \(\bar{z} \to z\) and DeepJSCC-Q → DeepJSCC. **Constellation order means something categorically different in the two paradigms** — rate in one, fidelity of representation in the other.

##### 3.6-E.8 What the learned constellations reveal

The L-\(M\) results are where the paper stops being an engineering fix and starts being interesting.

- **L-4 ≈ 4-QAM.** At \(M=4\) the learned points barely move and the usage distribution is near-uniform. QPSK is already close to optimal; nothing to gain.
- **L-16 on AWGN is strikingly non-standard.** Clearly non-square, and **not centred on the origin**: points in the first quadrant carry much higher power than those in the third. The authors' reading is that since the noise is zero-mean and symmetric, high-power points raise instantaneous received SNR for whatever is mapped to them, and the average power constraint is satisfied by selecting the low-power third-quadrant points far more frequently. They then name the thing it is: **a form of unequal error protection**, with a few important features assigned high-power symbols and less important features assigned low-power ones.

  That deserves emphasis in a semantic-communication roadmap. UEP by semantic importance is usually presented as a design principle one imposes. Here it *emerged* from end-to-end optimisation, and it emerged in the geometry of the constellation rather than in a coding-rate allocation.

- **L-16 under slow fading is different again** — much more circular, arranged as two rings around a centre point, with outer points exceeding the highest-power 16-QAM points and the probability distribution decaying radially to hold average power. Compared to the AWGN case it is far more centred. Different channel, different learned geometry.
- **Fading is where learned constellations pay.** On AWGN the margins over M-QAM are modest (and L-4096 actually underperforms 4096-QAM, plausibly because square 4096-QAM is already near-optimal and the search space is huge). Under slow fading the margins are large, and **L-64 outperforms even 4096-QAM on PSNR** — a 64-point learned alphabet beating a 4096-point standard one. The conclusion the authors draw, and it is the right one: optimising channel-input geometry and distribution matters most for non-Gaussian channels.

##### 3.6-E.9 Practicality verdict, expanded

*What it fixed:* the analog-symbol debt, decisively, without losing graceful degradation and without much loss versus analog at high \(M\). "SemCom cannot run on standards-compliant radios" stops being true.

*What remains open:*

1. **The best results need 4096-QAM, which is outside 5G NR.** NR tops out at 256-QAM uplink and 1024-QAM downlink (release- and UE-capability-dependent). At the orders NR actually offers, DeepJSCC-Q's advantage over separation is real but much narrower, and at 16/64-QAM it does not uniformly win. This is a genuine gap between "works on a finite constellation" and "works on *this* finite constellation." Note also that operating at very high order requires the SNR to support it *for the modulation to be physically reasonable*, which couples back to hardware EVM limits.
2. **No SNR conditioning.** Models are trained per \(\text{SNR}_\text{train}\). Composing the ADJSCC AF module with the soft-to-hard quantiser is the obvious next step and is not done here.
3. **Still no bit interface.** The transmitted objects are constellation *indices* with no bit-level semantics: they are not protected by CRC, cannot be soft-combined by HARQ, cannot be ciphered by standard means, and cannot be rate-matched. DeepJSCC-Q makes the *modulator* compatible. It does not make the *stack* compatible. That work is §7 of this document.
4. **AWGN and flat slow fading only** — no OFDM, no frequency selectivity, no MIMO, no channel-estimation pipeline (though channel-estimation-error sensitivity is examined).
5. **Quantisation error is a real cost at low order**, and the one case where separation won (\(\rho=1/6\), 64-QAM) points at the mechanism: more encoder freedom means more to lose to a uniform grid. Non-uniform or learned alphabets are the indicated fix, and they work — which is a good argument for standardising *learnable* constellations rather than only fixed ones.


### 3.7 Step 7 — *Hardware / over-the-air* validation (2022–2024)

Simulations can hide a lot. A smaller but crucial body of work builds actual prototypes. Xu et al.'s **ASCViT-JSCC** and the associated **ICP prototype** ran over-the-air tests on a platform combining **software-defined radios (USRP) and embedded GPUs (Jetson)**, using LabVIEW for the physical layer and Python for the neural inference, validating that adaptive semantic image transmission survives contact with real RF hardware. Other groups have demonstrated DeepJSCC proof-of-concepts inside 5G-style setups, in some cases recovering images even below the receiver's rated sensitivity.

**Practicality verdict:** these are still research prototypes, not products, and they are relatively few. But they close the credibility loop: the gains are not purely a simulation artifact.

### 3.8 Summary table — the practicality staircase

| Step | Idealization removed | Representative work(s) | What it now handles |
|---|---|---|---|
| Foundation | (baseline) | DeepSC; DeepJSCC (2019) | End-to-end learned coding; graceful degradation; **but analog symbols, fixed SNR/rate, AWGN** |
| 1 | Fixed SNR | ADJSCC | One model across an SNR range (needs SNR feedback) |
| 2 | Fixed bandwidth | Bandwidth-agile / successive refinement | Variable channel-symbol budget |
| 3 | AWGN only | OFDM-guided DeepJSCC | Multipath fading, ISI, PAPR (hybrid with classical DSP) |
| 4 | Single antenna | DeepJSCC-MIMO | MIMO, imperfect CSI, varying antenna counts |
| 5 | Fixed/uniform rate | NTSCC / versatile NTSCC | Content-aware variable rate via learned entropy model |
| 6 | Analog channel input | DeepJSCC-Q; JCM-VAE | **Finite digital constellations (real modulation)** |
| 7 | Simulation only | ASCViT-JSCC / ICP prototype | Over-the-air on SDR + embedded GPU |

The honest reading of this table: **the field has, collectively, addressed every one of the major practicality gaps** — but usually in *separate* papers, each removing one idealization while often re-assuming others. A single system that is simultaneously digital, MIMO, OFDM-fading-robust, variable-rate, SNR-adaptive, *and* hardware-validated is still rare. Integration, not any single missing capability, is the current frontier.

---

## 4. The major subfields of semantic communication

With the lineage in place, the field can be mapped into subfields. The uploaded standardization paper groups the enabling technologies into four (JSCC, SemCom-based multiple access, SemCom-based CSI feedback, semantic knowledge base); the broader research literature supports a somewhat wider taxonomy. Both are given here.

### 4.1 Joint source–channel coding (the core engine)
Everything in Section 3 lives here. This is the most mature subfield and the one closest to standardization. Sub-branches include analog vs. digital JSCC, SNR-/rate-adaptive JSCC, entropy-model-based variable-rate JSCC (NTSCC), and modality-specific variants for image, video, speech, and text.

### 4.2 Task-oriented / effectiveness-level communication
Rather than reconstructing the source, the system transmits only what a downstream task needs — classification, detection, segmentation, retrieval. The distortion metric becomes task accuracy, not PSNR. This is the "effectiveness level" of Weaver's model, and it delivers the most extreme compression because everything task-irrelevant is discarded.

### 4.3 Multi-user / SemCom-based multiple access
How do multiple semantic transmitters share a channel? Two branches, per the literature and the standardization paper (Zhang et al.): (a) graft semantic encoders onto existing schemes — OFDMA, NOMA, rate-splitting multiple access (RSMA); and (b) invent new semantic-domain multiple access. The flagship of (b) is **Model Division Multiple Access (MDMA)** (Zhang, Xu, et al.), which introduces a *semantic/model domain* beyond time/frequency/space: different users' encoder-decoder models produce outputs that are approximately orthogonal in the semantic embedding space, so they can share the same physical resources and still be separated at the receiver. There is also work on channel-transferable multi-user SemCom over OFDM-NOMA (train on AWGN, transfer to fading).

### 4.4 SemCom for control-plane signaling: CSI feedback and HARQ
A pragmatic and standardization-friendly subfield, because it targets *control* information rather than user data. **CSI feedback** is compressed with JSCC/JSCCM (transforming CSI into spatial-frequency or delay-Doppler domain, sparsifying, then deeply compressing), sharply reducing uplink overhead; LSTM layers exploit CSI's temporal correlation, and Doppler features can be embedded for high-mobility (e.g., high-speed rail). This is exactly the subfield the standardization paper (Zhang et al.)'s case study targets, and it is under active discussion in 3GPP RAN1/RAN2. Semantic HARQ and semantic-aware ACK/NACK coding are the sibling control-plane topics.

### 4.5 Semantic knowledge bases (KB)
A shared knowledge base at transmitter and receiver lets the system send only what the receiver *cannot already infer* — an index into a shared codebook, or a residual against a KB prediction. Implementations range from learnable vector-quantized/VAE codebooks to cross-modal feature stores and triple-based knowledge graphs. The KB is what lets SemCom move from "compress the message" to "send only the surprising part of the message," and it introduces its own hard problems: KB synchronization, versioning, format standardization, and security isolation across vendors.

### 4.6 Generative and LLM/foundation-model SemCom (the current wave)
The newest subfield. Generative models (diffusion, GANs) act as powerful priors at the receiver: the transmitter sends a compact semantic description and the generator *reconstructs* or even *synthesizes* plausible content, pushing compression to extremes. Related industrial threads (visible in the standardization paper (Zhang et al.)'s survey of company positions) include **token communication** and **split inference**, where large-model inference is split across device and network and only intermediate tokens/features are transmitted. This subfield is the least mature and the hardest to standardize, but it is where much 2024–2026 energy is going.

### 4.7 Cross-cutting concerns
Several issues span all subfields: **unified evaluation metrics** (see Section 5), **semantic noise and robustness** (adversarial or ambiguity-induced errors, distinct from physical noise), **security and privacy** (semantic features can leak more than bits), and **compatibility with the legacy protocol stack** (the dominant theme of the uploaded standardization paper).

---

## 5. The evaluation-metric problem (why "how good is it?" is genuinely hard)

A recurring, under-appreciated obstacle: SemCom breaks the traditional yardsticks. BER and throughput say nothing about whether *meaning* survived. The field has responded with a layered metric stack:

- **Reconstruction metrics** per modality: PSNR, SSIM, MS-SSIM (image/video); WER, BLEU, BERTScore (text); PESQ, SDR, MCD (speech).
- **Task metrics**: accuracy, precision/recall (classification/detection); IoU/mIoU (segmentation); mAP, nDCG, Recall@k (retrieval).
- **Cross-modal semantic similarity**: embed source and reconstruction into a shared space (e.g., via CLIP-style models) and take cosine similarity. The uploaded paper proposes a unified **Semantic Similarity (SS)** metric of exactly this form, and its CSI case study uses a squared generalized cosine similarity (SGCS).
- **Transmission-efficiency metrics** blending old and new: channel bandwidth ratio (CBR), semantic spectral efficiency.

The open problem — and a genuine blocker for standardization — is that there is **no single agreed-upon, cross-vendor, cross-modality metric**, which makes fair benchmarking and conformance testing difficult. This is why standardization bodies emphasize *unified evaluation and validation frameworks* as a prerequisite, not an afterthought.

---

## 6. A concrete practicality check: the CSI-feedback case study

The uploaded paper's own experiment is a good sanity check on "does this handle real channel and modulation." It is a CSI-feedback study under a **3GPP TR 38.901 clustered-delay-line (CDL) channel** — an industry-standard fading model, not toy AWGN — with **least-squares channel estimation to emulate imperfect CSI**, at 3.5 GHz with realistic OFDM numerology (15 kHz subcarrier spacing, 64 subcarriers, 100 ns delay spread). It compares:

- **JSCCM** — joint source-channel coding *and* modulation (fully learned, digital);
- **JSCC+QPSK** — learned compression, classical QPSK modulation;
- **Separated digital baselines** — e.g., "32bit + 1/4 LDPC + QPSK" and higher-order variants, where an AI model does source coding and then a standard LDPC code + QAM handles the channel.

All schemes are constrained to transmit the same number of symbols over the air (a fair, bandwidth-matched comparison), and the learned schemes use a two-layer Transformer backbone trained end-to-end. The reported result: **JSCCM achieves the highest semantic similarity (SGCS) across the entire SNR range** under the CDL fading channel.

Why this matters for your question: this is not an idealized-channel demo. It uses a standardized fading model, imperfect CSI, finite modulation, and a symbol-budget-matched comparison against real LDPC+QAM baselines. It is a credible datapoint that digital, channel-aware SemCom can beat the separated baseline on the metric that matters — within the scope of one control-plane task.

---


---

### 6.5 Worked example: one image, two pipelines

*This section is constructed for illustration. The geometry, symbol budgets, and code-block arithmetic below are computed from published 5G NR and DeepJSCC-Q system parameters. Reconstruction-quality figures are indicative of the behaviour reported in the literature under the cited papers' own conditions; they are not reproductions of a specific measured experiment. The purpose is to make the architectural difference concrete, not to assert a benchmark.*

#### 6.5.1 The setup and the fairness contract

**Source.** One Kodak image, \(768 \times 512\) RGB.

- Pixels: \(768 \times 512 = 393{,}216\)
- Source dimension: \(n = 768 \times 512 \times 3 = 1{,}179{,}648\) values

**Air-interface budget (identical for both tracks).** We fix the number of complex channel symbols — resource elements — because that is the resource an operator actually allocates:

- Bandwidth ratio \(\rho = 1/12\)
- **\(k = 1{,}179{,}648 / 12 = 98{,}304\) complex channel symbols**

**Physical layer (identical for both).** FR1, 20 MHz, 15 kHz subcarrier spacing (numerology \(\mu=0\)):

- 106 PRBs × 12 subcarriers = 1,272 subcarriers
- 14 OFDM symbols per 1 ms slot; assume 1 symbol of DMRS → 13 data symbols
- **16,536 data REs per slot** → \(98{,}304 / 16{,}536 \approx 5.9\), so **~6 slots ≈ 6 ms** of air time either way

**Modulation (identical for both).** 16-QAM. Both tracks put one 16-QAM symbol on each of the 98,304 REs. Neither track gets an easier alphabet than the other. This is the fairness contract that makes the comparison meaningful — and it is only *possible* because of DeepJSCC-Q. Against original analog DeepJSCC there is no fair comparison to run, because the analog scheme cannot be placed on this resource grid at all.

---

#### 6.5.2 Track A — the 5G NR pipeline (non-SemCom)

**A1. Source compression (BPG / VVC-intra class codec).**
Budget backwards from the air interface. 98,304 REs × 4 bits/symbol = **393,216 coded bits**. At an MCS with target code rate ≈ 1/2, the transport block carries ≈ **196,608 information bits**. Over 393,216 pixels that is **0.5 bits/pixel**. The codec runs its rate-distortion optimisation — DCT-like transform, quantisation, entropy coding — and emits a bitstream of that size. Its target is a source-domain metric; it knows nothing about the channel.

> **Where the semantic information goes:** into a bitstream where every bit is now *equally important and equally fragile*. Entropy coding has deliberately removed the redundancy that made the image robust. A bit flip early in the stream can desynchronise the arithmetic decoder and destroy everything after it. This is the cost of doing compression well: the output is maximally brittle.

**A2. Transport block CRC.** A 24-bit CRC is appended: \(B = 196{,}608 + 24 = 196{,}632\) bits.

**A3. Code block segmentation.** NR LDPC base graph 1 has maximum information block size \(K_{cb} = 8448\) bits; each code block carries its own 24-bit CRC, leaving 8,424 usable bits.

$$\left\lceil \frac{196{,}632}{8{,}424} \right\rceil = \lceil 23.34 \rceil = \mathbf{24 \text{ code blocks}}$$

> **This is where the cliff is manufactured.** The image is now 24 independent all-or-nothing units. The transport block succeeds only if all 24 decode. Even at a benign 1% per-code-block error rate, TB failure probability is \(1 - 0.99^{24} \approx 21\%\). The failure mode is combinatorially amplified by segmentation.

**A4. LDPC encoding.** Each code block is encoded with the NR quasi-cyclic LDPC base graph at mother rate 1/3, then written to a circular buffer.

**A5. Rate matching and redundancy version selection.** Bits are read out of the circular buffer to hit the target rate for this MCS, starting at the redundancy version for this transmission. Total output: 393,216 coded bits.

**A6. Scrambling and modulation mapping.** Bits are scrambled with a cell/UE-specific sequence, then mapped in groups of 4 to 16-QAM points via a fixed Gray-coded lookup table. The mapper is the same fixed-function block for every service the radio carries — voice, video, web, this image. **It has no idea what it is carrying.**

**A7. Layer mapping, precoding, RE mapping, OFDM.** Symbols are placed on the resource grid alongside DMRS, then IFFT and CP insertion.

**A8. Receiver.** Channel estimation from DMRS → equalisation → 16-QAM demapping to **log-likelihood ratios** → descrambling → LDPC belief-propagation decoding, per code block → per-CB CRC check → TB CRC check.

**A9. The decision point.** The receiver now makes a **binary** determination.

- **All CRCs pass** → the exact bitstream is delivered → the codec decodes → the image is exactly the 0.5 bpp reconstruction, **PSNR ≈ 33–34 dB, and this number does not change no matter how good the channel is.** At 30 dB SNR the image is identical to the one at 10 dB SNR. All the surplus channel quality is thrown away.
- **Any CRC fails** → NACK → HARQ retransmission with a different redundancy version → soft-combine → retry. Each retransmission consumes another ~6 ms of air time and another 98,304 REs.
- **HARQ exhausted** → RLC/TCP-level recovery or outright failure. **The receiver has no image at all.** Not a degraded image. Nothing.

---

#### 6.5.3 Track B — the DeepJSCC-Q pipeline

**B1. Direct encoding — there is no compression stage.** The image tensor enters \(f_\theta\). Residual blocks with GDN normalisation and stride-2 downsampling stages progressively reduce spatial resolution while expanding channel depth; content attention modules let the network concentrate representational capacity on regions that need it.

For \(k = 98{,}304\) with 16× spatial downsampling: the latent is \(48 \times 32\) spatially (1,536 positions) with \(C_{out} = 128\) real channels — \(48 \times 32 \times 128 = 196{,}608\) real values = **98,304 complex latents**. \(C_{out}\) is precisely the knob that set \(\rho\).

> **Where the semantic information goes:** into a latent tensor with *graded* importance. Some dimensions carry global structure and scene layout; others carry fine texture. Nothing has forced them to be equally important, and end-to-end training has actively arranged for them not to be. There is no entropy coder stripping out the redundancy that provides robustness.

**B2. Soft-to-hard quantisation.** Each of the 98,304 complex latents is snapped to the nearest of the 16 QAM points. At inference this is a hard nearest-neighbour lookup — architecturally identical to a conventional mapper's table lookup, and just as cheap.

*The quantisation error here is the entire cost of being digital.* Each latent \(z_i\) is displaced by up to half the lattice spacing. With 16-QAM there are 4 levels per dimension, which is coarse; with 4096-QAM there are 64 per dimension and the displacement is small — this is the mechanism behind the asymptotic approach to analog performance.

**B3. What is *not* in this pipeline.** No CRC. No code block segmentation. No LDPC encoder. No rate matching. No circular buffer. No redundancy versions. No scrambling. Stages A2–A5 have no counterpart. The encoder output goes straight to the RE mapper.

**B4. RE mapping and OFDM.** Identical to A7. The waveform on the air is a standard 16-QAM OFDM signal. A spectrum analyser cannot distinguish it from Track A. **A conformance test measuring EVM against the 16-QAM reference works normally** — which was impossible for analog DeepJSCC and is the entire practical payoff.

**B5. Receiver front end.** Identical channel estimation and equalisation as A8.

**B6. Decoding — and here the architectures diverge completely.** There is **no demapping to LLRs and no channel decoding.** The equalised complex symbols — noisy points that no longer sit exactly on constellation coordinates — are fed **directly into the decoder network** \(g_\phi\). The residual blocks and pixel-shuffle upsampling stages reconstruct the image from noisy observations.

> The decoder was trained on exactly this: latents corrupted by channel noise. It has learned an implicit joint denoiser-and-reconstructor. It never asks "which constellation point was intended?" — the question that dominates the entire receive chain in Track A. It asks only "what image best explains these observations?"

**B7. Output.** An image. **Always an image.** There is no failure branch. At high SNR the symbols land near their intended points and reconstruction is sharp. At low SNR they scatter, and reconstruction is softer, losing fine detail first — precisely the behaviour the ADJSCC feature analysis in §3.1-E.8 predicts, since low-level detail features are the ones noise damages most.

---

#### 6.5.4 Stage-by-stage comparison

| Stage | 5G NR (Track A) | DeepJSCC-Q (Track B) |
|---|---|---|
| Source coding | BPG/VVC-intra → 196,608 bits @ 0.5 bpp | **None** — image → latent directly |
| Error detection | TB CRC-24 + 24× CB CRC-24 | **None** |
| Segmentation | 24 code blocks (all-or-nothing each) | **None** — one continuous latent |
| Channel coding | NR LDPC BG1, mother rate 1/3 | **None** — robustness is inside \(f_\theta\)'s weights |
| Rate matching | Circular buffer, RV0–3 | **None** — rate set by \(C_{out}\) at design time |
| Modulation | Fixed Gray-mapped 16-QAM LUT, content-blind | Nearest-neighbour to same 16-QAM alphabet, **content-aware and end-to-end trained** |
| Air interface | 98,304 REs, ~6 slots | **Identical** |
| Receiver | Demap → LLR → LDPC BP → CRC | Equalised symbols → decoder DNN, **one pass** |
| Adaptation | MCS table + HARQ retransmission | Model conditioning (ADJSCC-style, if composed) |
| Failure mode | Binary: perfect image or **nothing** | Continuous: image degrades in quality |
| Effect of one corrupted symbol | Wrong LLRs → possible CB failure → possible TB failure → **whole image lost** | Slight local softening; **no propagation** |
| Meaning of higher-order QAM | **Higher rate**, higher error probability — needs an optimal choice per SNR | **Finer quantisation** of the latent — monotonically better |
| Testability | EVM vs 16-QAM reference ✓ | EVM vs 16-QAM reference ✓ |
| Stack compatibility | Full (CRC, HARQ, ciphering, MAC) | **Modulator-compatible only** |

#### 6.5.5 Behaviour across the SNR range

The same image, the same 98,304 REs, swept over channel SNR. Indicative behaviour, not measured values:

| Channel SNR | 5G NR, 16-QAM @ R≈1/2 | DeepJSCC-Q, 16-QAM |
|---|---|---|
| 20 dB | Perfect delivery. **33–34 dB PSNR** — capped by the 0.5 bpp source decision made before transmission | Sharp reconstruction; **surplus SNR converts into quality** |
| 14 dB | Perfect delivery. **33–34 dB PSNR** (unchanged) | Slightly better than at 10 dB; still improving |
| 10 dB | Occasional CB failures; HARQ recovers. **33–34 dB PSNR** at the cost of extra air time | Good reconstruction, minor texture loss |
| 8 dB | Near the operating threshold. Frequent HARQ. Latency spikes | Visible softening of fine detail; structure fully intact |
| 6 dB | **Cliff.** Most TBs fail. HARQ exhausts. **No image** | Noticeably degraded but entirely recognisable |
| 2 dB | **No image** | Blurred, low detail; scene content still readable |
| 0 dB | **No image** | Heavily degraded; coarse structure still present |

The two columns are not two points on the same curve. They are **two different shapes**. Track A is a step function whose height was fixed at stage A1 and whose edge is at ~7–8 dB. Track B is a monotone curve with no edge.

Two consequences that follow directly and are easy to miss:

- **Above threshold, Track A wastes channel quality.** Going from 10 dB to 25 dB buys nothing, because the source rate was chosen before transmission. Track B converts every dB into quality. This is the same waste the ADJSCC results identified from the other direction, and the same saturation observed in the TJSCC classical baseline above 15 dB.
- **Below threshold, Track A wastes everything.** Six milliseconds of spectrum for zero delivered information, repeated per HARQ attempt. Track B delivers a usable image on the first attempt in conditions where Track A delivers nothing at all.

#### 6.5.6 Where the 5G pipeline still wins

An honest accounting, in keeping with this document's §A.1 verdict of *regime-dependent advantage*:

1. **Exactness.** When Track A succeeds, the delivered bitstream is bit-exact. For a firmware image, a financial transaction, or a signed document, "gracefully degraded" is not a category that exists. Most traffic in a real network is not perceptual.
2. **The ceiling is a floor.** Track A's flatness above threshold is also a *guarantee*: a known, contracted quality level. Track B's quality is channel-dependent and therefore harder to write an SLA against.
3. **Universality and separability.** One LDPC/QAM chain carries every service. Track B needs a trained encoder–decoder pair per source domain, per rate, and (absent ADJSCC-style conditioning) per SNR — with both endpoints holding matched weights.
4. **Maturity.** Decades of optimisation, silicon, test equipment, conformance suites, and field experience. LDPC decoders are extraordinarily efficient in hardware; a decoder DNN inference is not obviously cheaper.
5. **The stack.** Ciphering, integrity protection, HARQ, segmentation, QoS. Track B has none of these and cannot simply acquire them.
6. **High SNR with generous bandwidth.** At \(\rho\) large and SNR high, a good neural codec plus a good channel code is genuinely competitive, and the joint design's advantages compress.

The regimes where Track B wins are: **low SNR, low bandwidth ratio, unpredictable or rapidly varying channels, latency-constrained links that cannot afford HARQ round trips, and perceptual or task-oriented payloads.** That is a real and growing slice — XR, machine vision offload, sensing, V2X, IoT imaging — and it is not all traffic. The correct conclusion is not that Track B replaces Track A but that a 6G stack plausibly needs both, with the semantic path attachable per-bearer. This is exactly the compatibility-overlay argument in §7 of this document.

#### 6.5.7 What the example demonstrates about the roadmap's thesis

Three things the stage-by-stage view makes visible that the summary table in §3.8 cannot:

1. **The cliff is architectural, not modulational.** Both tracks used the same 16-QAM alphabet, the same REs, the same OFDM grid. One has a cliff; the other does not. The cliff comes from *segmentation into all-or-nothing code blocks with a hard CRC decision* — from the separation architecture. Removing the analog assumption did not reintroduce the cliff, which is precisely the result DeepJSCC-Q needed to demonstrate.

2. **DeepJSCC-Q converts an *incomparable* claim into a *comparable* one.** The original DeepJSCC could not be placed on this resource grid at all, so every comparison with a 5G baseline was, at some level, apples to oranges. After DeepJSCC-Q, the semantic scheme and the 5G scheme differ *only* in what happens between the source and the modulator. That is what makes the comparison — and eventually the standardisation — possible.

3. **The remaining gap is the stack, not the physical layer.** Track B's pipeline is compatible with the modulator, the resource grid, the waveform, and the conformance test. What it lacks is CRC, HARQ, ciphering, and MAC-layer integration. This is exactly why this document's §7 sequencing argument — control plane first, then user plane, then a native data plane, with compatibility overlays throughout — is the right shape for the remaining work. The physical-layer problem is substantially solved. The protocol-integration problem is not.


## 7. The standardization roadmap (where the field is trying to go)

The uploaded paper's central contribution is a *sequencing* argument: **standardize SemCom from the control plane outward.** The logic is risk-based — start where the payoff is high and the disruption is low, then expand.

1. **Control plane first.** CSI feedback and HARQ-ACK/NACK are low-risk entry points: they are internal signaling, already AI-friendly (3GPP is already studying two-sided models for CSI in RAN1/RAN2), and don't require rewriting how user data is carried. This is the beachhead.
2. **Then the user plane.** Semantic-aware coding/modulation, semantic-importance-driven resource allocation, semantic HARQ and semantic CRC for actual payload data.
3. **Then a native "data plane."** A unified, data-centric plane (embraced by 3GPP SA2 discussions) where semantic/intent-aware extraction governs how sensing data, AI/ML model data, and IoT data move through the network.

Wrapped around this sequence are three enablers the paper argues must come first: a **unified E2E architecture** with clearly defined modules (semantic encoder/decoder, inference, feedback, QoS scheduling); a **compatibility design** that overlays semantic-aware blocks onto the existing protocol stack so SemCom is an *attachable enhancement* to legacy systems rather than a replacement (semantic identifiers at the data link layer, semantic-aware CRC/HARQ, semantic-QoS mapping); and a **unified platform / protocol ecosystem** — open-source reference models, standardized datasets, common APIs, and interoperable KB formats.

On the institutional side: **ITU** has moved SemCom into formal study (SG13 requirements and reference architectures, SG16 multimedia semantic processing, SG20 IoT framework, plus a recent knowledge-based-SemCom technical report), **3GPP** is discussing it in SA1 use cases and RAN working groups, and **IEEE** is engaged as well. The paper frames the recent ITU-T technical report on knowledge-based SemCom as the symbolic transition "from conceptual research to international standardization."

---

## 8. Honest assessment: how practical is the field, really?

Pulling the threads together, here is a candid scorecard on the practicality question you asked about.

**What is genuinely solved (in isolation):**
- Graceful degradation and low-SNR robustness — the original and most durable win.
- SNR-adaptivity with a single model (ADJSCC and successors).
- Real fading via OFDM, including ISI and PAPR handling, using hybrid learned+classical designs.
- MIMO operation with robustness to imperfect CSI.
- Content-aware variable-rate coding via learned entropy models (NTSCC).
- **Digital, finite-constellation modulation** (DeepJSCC-Q, JCM-VAE) — the single biggest deployability unlock, because it removes the "analog symbols" showstopper.
- At least proof-of-concept over-the-air hardware validation.

**What is still weak or open:**
- **Integration.** Most systems solve one practicality dimension while re-idealizing others. A unified system that is digital + MIMO + OFDM-fading-robust + variable-rate + SNR-adaptive + hardware-validated + task-oriented, all at once, is still uncommon.
- **Standardized metrics and conformance testing.** Without an agreed cross-vendor, cross-modality metric, benchmarking and certification are hard — a real blocker, not a cosmetic one.
- **Knowledge-base logistics.** KB synchronization, versioning, format standards, and security across vendors are largely unsolved at scale.
- **Two-sided model lifecycle management.** Deploying paired encoder/decoder models across a multi-vendor network (training data, model alignment, updates, fallback) is an operational problem 3GPP is only beginning to work through.
- **Generalization and retraining cost.** Many gains assume the deployment distribution matches training; robustness to genuine distribution shift (new content, new channels) is uneven.
- **Backward compatibility.** Retrofitting semantics-aware processing into a fixed protocol stack (signaling, HARQ, layering) is exactly the hard integration work the standardization papers foreground.
- **Generative/LLM-era SemCom** is exciting but the least mature and the hardest to make deterministic, verifiable, and standardizable.

**Bottom line:** The field has progressed from "elegant but un-deployable" (analog DeepJSCC on AWGN) to "credibly deployable in narrow, well-defined slices" (digital JSCCM for CSI feedback under standardized fading, validated in prototypes). The next phase is less about inventing new tricks and more about *integration, evaluation standardization, and lifecycle/operational engineering* — which is precisely why the current center of gravity has shifted from algorithm papers to standardization roadmaps like the uploaded one.

---

## Appendix A: Additional questions worth asking (and short answers)

*You asked me to add related questions that should also be answered. Here are the ones a careful reader of this roadmap tends to reach next.*

**A.1 Is SemCom actually better than "just use a great neural codec + a great channel code"?**
Sometimes, especially at low SNR, low rate, and under fading — the regimes where the separated design's cliff effect bites and where joint design's graceful degradation and unequal protection pay off. At high SNR with generous bandwidth, the separated design (which benefits from decades of optimization and clean modularity) is often competitive. The honest claim is *regime-dependent advantage*, not universal superiority.

**A.2 Why not just make everything a black box end-to-end?**
Because the most practical systems (OFDM-guided JSCC, DeepJSCC-MIMO) deliberately keep classical, interpretable DSP blocks (OFDM, channel estimation, equalization) and learn only around them. Domain knowledge improves convergence, performance, and — critically for standards — *interpretability and testability*. Pure black boxes are harder to certify and to make interoperable.

**A.3 What breaks when you move from analog to digital symbols?**
You lose exact differentiability and take a performance hit that shrinks as modulation order grows (DeepJSCC-Q approaches analog performance at high-order QAM). You gain hardware compatibility and standards compatibility. The probabilistic JCM-VAE route recovers trainability by learning a distribution over constellation points rather than a hard mapping.

**A.4 Who has to agree for two-sided (encoder-at-transmitter, decoder-at-receiver) models to work across vendors?**
This is the model-alignment / lifecycle-management problem. It requires standardized model exchange or alignment procedures, agreed KB formats, and fallback behavior — which is why 3GPP's RAN discussions on CSI two-sided models are considered the practical proving ground for the whole paradigm.

**A.5 Where does semantic *security* fit?**
Semantic features can leak more about the source than raw bits (they are, by design, the *meaningful* part), and shared knowledge bases create new attack surfaces. Security isolation of KBs and privacy of semantic representations are recognized open problems, called out in survey and standardization literature alike.

---

## Appendix B: Key sources

*Foundational*
- E. Bourtsoulatze, D. Burth Kurka, D. Gündüz, "Deep Joint Source-Channel Coding for Wireless Image Transmission," *IEEE Trans. Cognitive Communications and Networking*, 5(3):567–579, 2019. (ICASSP 2019 version also.)
- H. Xie, Z. Qin, G. Y. Li, B.-H. Juang, "Deep Learning Enabled Semantic Communication Systems (DeepSC)," *IEEE Trans. Signal Processing*, 69:2663–2675, 2021. (arXiv:2006.10685.)

*SNR- and bandwidth-adaptivity*
- J. Xu, B. Ai, W. Chen, A. Yang, P. Sun, M. Rodrigues, "Wireless Image Transmission Using Deep Source Channel Coding With Attention Modules (ADJSCC)," *IEEE Trans. Circuits Syst. Video Technol.*, 32(4):2315–2328, 2022. (arXiv:2012.00533.)
- D. B. Kurka, D. Gündüz, "Bandwidth-Agile Image Transmission With Deep Joint Source-Channel Coding," *IEEE Trans. Wireless Communications*, 20(12):8081–8095, 2021.

*Fading / OFDM / MIMO*
- M. Yang, C. Bian, H.-S. Kim, "OFDM-guided Deep Joint Source Channel Coding for Wireless Multipath Fading Channels," *IEEE Trans. Cognitive Communications and Networking*, 8(2):584–599, 2022. (arXiv:2109.05194; ICC 2021 version arXiv:2101.03909.)
- H. Wu, Y. Shao, C. Bian, K. Mikolajczyk, D. Gündüz, "Deep Joint Source-Channel Coding for Adaptive Image Transmission Over MIMO Channels," *IEEE Trans. Wireless Communications*, 23(10):15002–15017, 2024. (arXiv:2309.00470.)

*Variable-rate / entropy-model*
- J. Dai, S. Wang, K. Tan, Z. Si, X. Qin, K. Niu, P. Zhang, "Nonlinear Transform Source-Channel Coding for Semantic Communications (NTSCC)," *IEEE J. Sel. Areas Commun.*, 40(8):2300–2316, 2022. (arXiv:2112.10961.)
- S. Wang, J. Dai, X. Qin, Z. Si, K. Niu, P. Zhang, "Improved Nonlinear Transform Source-Channel Coding to Catalyze Semantic Communications," *IEEE J. Sel. Topics Signal Process.*, 17(5):1022–…, 2023. (arXiv:2303.14637.)

*Digital modulation*
- T.-Y. Tung, D. B. Kurka, M. Jankowski, D. Gündüz, "DeepJSCC-Q: Constellation Constrained Deep Joint Source-Channel Coding," 2022. (arXiv:2206.08100.)
- Y. Bo, Y. Duan, S. Shao, M. Tao, "Joint Coding-Modulation for Digital Semantic Communications via Variational Autoencoder (JCM)," *IEEE Trans. Communications*, 2024. (arXiv:2310.06690; earlier BPSK version, GLOBECOM 2022.)

*Prototype / hardware validation*
- J. Xu et al., "Adaptive Wireless Image Semantic Transmission (ASCViT-JSCC) with ICP prototype," 2024. (arXiv:2410.17536.)

*Standardization roadmap (uploaded)*
- P. Zhang, X. Xu, M. Sun, H. Gao, N. Ma, X. Wang, R. Zhang, J. Wang, D. Niyato, "Towards Native AI in 6G Standardization: The Roadmap of Semantic Communication," arXiv:2509.12758v2, 2026. (Also its cited surveys: Guo et al. 2024; Zhang et al. 2025; Yang et al. 2022 — IEEE Communications Surveys & Tutorials.)

---

*Note: This document synthesizes published research and one standardization paper. It reflects the state of the literature as surveyed; specific numerical claims (e.g., "highest SGCS across SNR range," "approaches analog performance at high modulation order") are the reported results of the cited papers under their own experimental conditions, not independently reproduced here.*

### Appendix B.1 — sources for the expanded sections

*Primary sources for this expansion (both fetched in full)*

- J. Xu, B. Ai, W. Chen, A. Yang, P. Sun, M. Rodrigues, "Wireless Image Transmission Using Deep Source Channel Coding With Attention Modules," *IEEE Trans. Circuits Syst. Video Technol.*, 32(4):2315–2328, 2022. arXiv:2012.00533v3. DOI 10.1109/TCSVT.2021.3082521. Code: github.com/alexxu1988/ADJSCC.
- T.-Y. Tung, D. B. Kurka, M. Jankowski, D. Gündüz, "DeepJSCC-Q: Constellation Constrained Deep Joint Source-Channel Coding," arXiv:2206.08100, 2022; IEEE J. Sel. Areas Inf. Theory.

*Methodological antecedents cited by those works and relevant to the mechanisms described above*

- J. Hu, L. Shen, G. Sun, "Squeeze-and-Excitation Networks," CVPR 2018. — structural antecedent of the AF module.
- E. Agustsson, F. Mentzer, M. Tschannen, L. Cavigelli, R. Timofte, L. Benini, L. Van Gool, "Soft-to-Hard Vector Quantization for End-to-End Learning Compressible Representations," NeurIPS 2017. — origin of the soft-to-hard quantiser.
- J. Ballé, V. Laparra, E. P. Simoncelli, "Density Modeling of Images Using a Generalized Normalization Transformation," arXiv:1511.06281, 2015. — GDN, used in both architectures.
- Z. Cheng, H. Sun, M. Takeuchi, J. Katto, "Learned Image Compression with Discretized Gaussian Mixture Likelihoods and Attention Modules," CVPR 2020. — source of DeepJSCC-Q's content attention modules (distinct from ADJSCC's AF modules).
- W. Shi et al., "Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network," CVPR 2016. — pixel shuffle upsampling.
- F. A. Aoudia, J. Hoydis, "Joint Learning of Probabilistic and Geometric Shaping for Coded Modulation Systems," GLOBECOM 2020. — constellation learning antecedent; DeepJSCC-Q's learned-constellation results are the JSCC analogue.
- D. B. Kurka, D. Gündüz, "DeepJSCC-f: Deep Joint Source-Channel Coding of Images with Feedback," *IEEE J. Sel. Areas Inf. Theory*, 1(1):178–193, 2020. — source of the BDJSCC baseline architecture used by ADJSCC.

*5G NR parameters used in the §6.5 construction* — 3GPP TS 38.212 (LDPC base graphs, code block segmentation, rate matching), TS 38.211 (resource grid, numerology, modulation mapping), TS 38.214 (MCS tables, TBS determination), TS 38.101 (UE EVM requirements). Figures are used as published system parameters; no measured results from these specifications are claimed.
