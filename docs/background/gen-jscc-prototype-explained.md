# The Generative JSCC Prototype: What We Are Building, and Why It Is Built This Way

**Scope.** This document explains the small DiT-JSCC prototype in `semcom/gen/`. It covers:

- the problem it addresses;
- the two models it descends from: DiT (Peebles & Xie, 2023) and DiT-JSCC (Tan et al., arXiv:2601.03112);
- every change we made and why;
- how it is trained and evaluated;
- what the first runs showed.

For the DiT-JSCC paper on its own, see [dit-jscc-explained.md](dit-jscc-explained.md). For the
research plan and novelty position, see [../proposal/gen-jscc-prototype.md](../proposal/gen-jscc-prototype.md).

**Sources.** Every number below falls into one of three kinds:

- read from the code or config (file named);
- computed (the computation is shown);
- measured in a run (run named).

Where something is a hypothesis rather than a result, the text says so.

**Status (8 Nov 2026).**

- **Done:** the 0 dB specialist runs, g-w1 and g-w2.
- **Running:** the SNR study, g-w3 (four runs).

---

## Contents

1. [The goal in one paragraph](#1-the-goal-in-one-paragraph)
2. [The communication problem, stated precisely](#2-the-communication-problem-stated-precisely)
3. [Why a generative decoder: the MMSE blur and the perception–distortion trade-off](#3-why-a-generative-decoder)
4. [Diffusion models: the minimum needed to follow the rest](#4-diffusion-models)
5. [The original DiT](#5-the-original-dit)
6. [DiT-JSCC: what the paper built](#6-dit-jscc-what-the-paper-built)
7. [Our prototype, component by component](#7-our-prototype-component-by-component)
8. [The SNR question: specialists, told, blind](#8-the-snr-question-specialists-told-blind)
9. [Training](#9-training)
10. [What went wrong first, and the two fixes](#10-what-went-wrong-first-and-the-two-fixes)
11. [Evaluation protocol](#11-evaluation-protocol)
12. [Results so far](#12-results-so-far)
13. [The week-2 experiment and how to read its outcome](#13-the-week-2-experiment)
14. [Limitations and what we must not claim](#14-limitations-and-what-we-must-not-claim)
15. [Symbol glossary](#15-symbol-glossary)

---

## 1. The goal in one paragraph

DiT-JSCC showed something new. At extreme compression (1/48 to 1/384 channel uses per pixel
value) and low SNR (−5 to 5 dB), a diffusion-transformer receiver reconstructs images that are
semantically and perceptually far better than deep JSCC or classical separation. But it trains
**a separate model for every SNR** ("we train separate models for each SNR under both AWGN and
Rayleigh fading channels").

Our question is:

> **Can one generative receiver serve the whole SNR range (−5 to 5 dB) as well as a set of per-SNR
> specialists, and does it need to be told the SNR?**

The second question, receiver told the SNR vs receiver blind, continues this repo's blind-JSCC
study, now asked of a generative decoder. A miscalibrated generative decoder does not merely
blur: it can produce a sharp image of the wrong thing.

To afford the many runs this needs, we built a **much smaller** DiT-JSCC:

- 128 px images, 10 ImageNet classes;
- a 36M-parameter model trained from scratch;
- it trains in about 1.5 h on a Kaggle T4.

It is a prototype for comparing **arms against each other**, not for matching the paper's
absolute numbers.

---

## 2. The communication problem, stated precisely

**Source.** An RGB image $x \in [0,1]^{3 \times H \times W}$, with $H = W = 128$, so
$n = 3HW = 49{,}152$ source values.

**Transmitter.** An encoder $f_\phi$ maps the image to $k$ complex channel symbols
$z = f_\phi(x) \in \mathbb{C}^k$, normalised to unit average power:

$$\frac{1}{k}\sum_{i=1}^{k} |z_i|^2 = 1 .$$

**Channel.** AWGN, $y = z + w$ with $w \sim \mathcal{CN}(0, \sigma^2 I_k)$. Since the signal power
is 1, the channel SNR in dB is $\gamma = -10\log_{10}\sigma^2$, that is $\sigma^2 = 10^{-\gamma/10}$
([semcom/channel.py](../../semcom/channel.py), `snr_to_noise_power`).

| SNR $\gamma$ | −5 dB | 0 dB | 5 dB |
|---|---|---|---|
| noise power $\sigma^2$ | 3.162 | 1.000 | 0.316 |

At −5 dB the noise has about three times the power of the signal.

**Bandwidth.** The channel bandwidth ratio (CBR) is $\rho = k/n$. We use $k = 1024$, so
$\rho = 1024 / 49152 = 1/48$, the bandwidth of DiT-JSCC's headline table.

**Receiver.** A decoder $g_\theta$ produces $\hat x = g_\theta(y)$. "Joint" source-channel coding
means $f_\phi$ and $g_\theta$ are trained together through the (differentiable) channel, rather
than a compressor followed by a separately designed channel code.

**The extremity of the regime.** 1024 complex symbols is 2048 real numbers for 49,152 pixel
values, each corrupted by noise of power comparable to the signal. Exact reconstruction is
impossible. The design question is *which* information about the image is worth those 2048
noisy numbers.

---

## 3. Why a generative decoder

### 3.1 An MSE-trained decoder outputs a posterior mean, which is blurry

Train $g_\theta$ to minimise $\mathbb{E}\,\|x - g_\theta(y)\|^2$. For any fixed encoder, the
optimum is the conditional mean (the MMSE estimator):

$$g^\star(y) = \mathbb{E}[x \mid y].$$

When $y$ carries little information, $p(x \mid y)$ is broad: many different images are consistent
with it. Their average is a blur. The edges, textures and exact positions that differ between
the plausible images cancel. This is the "clear blurry artifacts" failure DiT-JSCC describes for
deep JSCC. Our CNN baseline is exactly this kind of decoder.

### 3.2 A generative decoder samples from the posterior instead

A generative decoder outputs a *sample* $\hat x \sim p(x \mid y)$. Every sample is a sharp,
realistic image consistent with what was received. The price is pixel distortion, and it can be
computed exactly for an ideal sampler.

Let $\hat x$ be drawn from $p(x \mid y)$ independently of the true $x$, given $y$, and let
$\mu(y) = \mathbb{E}[x \mid y]$. Then

$$
\mathbb{E}\|x - \hat x\|^2
= \mathbb{E}\|(x - \mu) - (\hat x - \mu)\|^2
= \underbrace{\mathbb{E}\|x-\mu\|^2}_{\text{MMSE}} + \underbrace{\mathbb{E}\|\hat x-\mu\|^2}_{\text{MMSE}} - 2\,\underbrace{\mathbb{E}\langle x-\mu,\hat x-\mu\rangle}_{=\,0\ \text{(conditional independence)}}
= 2\,\text{MMSE}.
$$

**A perfect posterior sampler has exactly twice the MSE of the MMSE decoder, a PSNR loss of
$10\log_{10}2 = 3.01$ dB.** In return its outputs are distributed like real images. This is
the perception–distortion trade-off (Blau & Michaeli, 2018): one cannot have minimum distortion
and realistic outputs at once.

So when the DiT has lower PSNR than the CNN (§12), that alone is not a failure. Two things
decide whether it works:

1. **Do its samples stay close in meaning and appearance to the true image?** We measure this with
   LPIPS and DINOv2 similarity (§11).
2. **Is the PSNR gap roughly the ~3 dB the theory predicts, or much larger?** A much larger gap
   would mean the samples come from a *broader* distribution than the true posterior: the
   receiver is not using $y$ well. That was the g-w1 failure (§10).

The 2·MMSE identity is for an ideal sampler and an MMSE decoder. Neither of our models is
exactly either. In particular, the DiT decodes through a VAE that alone caps PSNR at about
22.8 dB (§7.2). Treat the 3 dB as a reference point, not a prediction.

---

## 4. Diffusion models

### 4.1 Forward process

Take a clean sample $x_0$. For us this is the image's VAE latent, see §7.2. The forward process
adds Gaussian noise over $T = 1000$ steps. With a variance schedule $\beta_1,\dots,\beta_T$ and
$\bar\alpha_t = \prod_{s \le t}(1 - \beta_s)$, the noised sample at step $t$ has the closed form

$$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon, \qquad \epsilon \sim \mathcal{N}(0, I).$$

We use the DDPM/DiT **linear schedule**, $\beta_t$ linear from $10^{-4}$ to $0.02$
([semcom/gen/diffusion.py](../../semcom/gen/diffusion.py)). The *diffusion* signal-to-noise
ratio at step $t$ is

$$\lambda_t = \frac{\bar\alpha_t}{1-\bar\alpha_t}.$$

This is not the channel SNR $\gamma$. The two are unrelated in this design (§6.3).

Computed from the schedule:

| $t$ | 0 | 100 | 250 | 500 | 750 | 900 | 999 |
|---|---|---|---|---|---|---|---|
| $\bar\alpha_t$ | 0.9999 | 0.895 | 0.521 | 0.0778 | 3.3e-3 | 2.7e-4 | 4.0e-5 |
| $\lambda_t$ | ~1e4 | 8.5 | 1.09 | 0.084 | 3.3e-3 | 2.7e-4 | 4.0e-5 |

Signal and noise have equal power ($\lambda_t = 1$) at about $t = 258$. Roughly three quarters of
the timesteps are noise-dominated.

### 4.2 The reverse process and what the network learns

Generation runs the process backwards: start from $x_T \sim \mathcal{N}(0, I)$ and denoise step by
step. A network $F_\theta(x_t, t, c)$, given the noisy sample, the step and any conditioning $c$,
supplies the quantity needed at each step.

There are three equivalent choices of what $F_\theta$ outputs (its **parameterisation**). Each
determines the others given $x_t$:

| Target | Definition | Recover $\hat x_0$ from it |
|---|---|---|
| noise $\epsilon$ | $\epsilon$ | $\hat x_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\hat\epsilon)/\sqrt{\bar\alpha_t}$ |
| clean sample $x_0$ | $x_0$ | directly |
| velocity $v$ (Salimans & Ho, 2022) | $v = \sqrt{\bar\alpha_t}\,\epsilon - \sqrt{1-\bar\alpha_t}\,x_0$ | $\hat x_0 = \sqrt{\bar\alpha_t}\,x_t - \sqrt{1-\bar\alpha_t}\,\hat v$ |

The training loss is a plain MSE to the chosen target, with $t$ uniform on $\{0,\dots,T-1\}$:

$$\mathcal{L} = \mathbb{E}_{x_0, \epsilon, t}\,\big\|F_\theta(x_t, t, c) - \text{target}\big\|^2.$$

The parameterisations are *equivalent at the optimum* but **not equivalent as training
objectives**. They weight the timesteps very differently, and this turned out to matter for us
(§10.2).

### 4.3 DDIM sampling

We sample with DDIM (Song et al., 2021) on an evenly spaced subset of $S$ timesteps
$t_1 > t_2 > \dots > t_S$. With $\eta = 0$, which is deterministic and what we use, each step is

$$x_{t_{i+1}} = \sqrt{\bar\alpha_{t_{i+1}}}\,\hat x_0 + \sqrt{1-\bar\alpha_{t_{i+1}}}\,\hat\epsilon,$$

with $\hat x_0, \hat\epsilon$ from the network output at $(x_{t_i}, t_i)$ (`Diffusion.split`).

One trained model can be decoded with any $S$, and the receiver's cost is $S$ network
evaluations. That is why we evaluate at $S \in \{50, 20, 10\}$.

### 4.4 Conditioning and classifier-free guidance

A *conditional* diffusion model learns $p(x_0 \mid c)$ by giving the network $c$. Here $c$ is the
received symbols. **Classifier-free guidance** (CFG; Ho & Salimans, 2022) trains the same network
also *without* $c$: the condition is replaced by a learned "null" input with probability 0.1. At
sampling time the conditional and unconditional outputs can then be extrapolated:

$$F = F_\text{uncond} + s\,(F_\text{cond} - F_\text{uncond}).$$

We train with condition dropout 0.1, so guidance is available. **We evaluate at $s = 1$**, which
is plain conditional sampling. We also use the unconditional pass as a diagnostic (§9.3).

---

## 5. The original DiT

DiT ("Scalable Diffusion Models with Transformers", Peebles & Xie, ICCV 2023) replaced the U-Net
of latent diffusion with a plain Vision Transformer. Its design:

1. **Latent space.** It works on the 4-channel latent of Stable Diffusion's VAE, which is 8× smaller
   per side than the image, not on pixels.
2. **Patchify.** The latent is cut into $p \times p$ patches; each becomes a token with a fixed 2-D
   sine-cosine position embedding.
3. **adaLN-Zero conditioning.** The timestep and the class label are embedded and summed into a
   vector $c$. In every block, a linear map of $c$ produces a shift, a scale and a gate for the
   attention and MLP branches:

   $$h \leftarrow h + g_1 \odot \text{Attn}\big(\text{LN}(h)(1+s_1) + b_1\big), \qquad h \leftarrow h + g_2 \odot \text{MLP}\big(\text{LN}(h)(1+s_2) + b_2\big).$$

   The map is **zero-initialised**, so every block starts as the identity. This was the best of
   the conditioning schemes the DiT paper compared.
4. **Output.** A final adaLN-modulated linear layer predicts, per token, the noise (and a learned
   variance) for its patch.
5. **Sizes.** S, B, L, XL. **DiT-S: 12 blocks, hidden 384, 6 heads.**
6. **Training.** ImageNet class-conditional. Linear schedule $10^{-4} \to 0.02$, $T = 1000$; AdamW,
   learning rate $10^{-4}$, batch 256; EMA of the weights with decay 0.9999.

DiT is an image *generator*: given a class label, it makes a new image of that class. It has no
notion of a channel, an encoder or reconstruction.

---

## 6. DiT-JSCC: what the paper built

Details are in [dit-jscc-explained.md](dit-jscc-explained.md), checked against the paper's text.
The essentials:

### 6.1 What it transmits

Two branches share the channel:

- **Semantic branch.** A frozen **DINOv2 ViT-B** feature map, followed by a learned JSCC encoder.
  DINOv2 features encode *what is in the image and where* (identity, layout), which is exactly
  what a generator cannot invent.
- **Detail branch.** A pixel-domain JSCC encoder (SwinJSCC) for what the semantics do not pin
  down: edges, texture, colour.

The total $k = k_s + k_d$ is fixed. The split is chosen per image from the complexity of a BLIP-2
caption.

### 6.2 How the receiver uses it

The receiver is a large conditional DiT in the SD-VAE latent space, initialised from a
pretrained class-conditional DiT, with U-Net-style long skips. The received symbols enter
through **channel fusion**: projected by an adapter and concatenated with the token
embeddings, rather than through cross-attention. This happens in the **first few blocks only**,
semantics first, then detail (**coarse-to-fine**).

### 6.3 An important structural fact

In DiT-JSCC the diffusion process **starts from pure Gaussian noise**. The received symbols are
only *conditions*. The channel noise is not the diffusion noise.

This separates DiT-JSCC from "denoise the received features" designs such as SGD-JSCC, where
the channel SNR maps naturally onto a diffusion timestep. In DiT-JSCC no such mapping exists.
How a receiver should handle an unknown or varying SNR is therefore an open question, and it is
our question.

---

## 7. Our prototype, component by component

```text
 TRANSMITTER                                               CHANNEL               RECEIVER
 image x (3×128×128)
   │
   ├─► DINOv2 ViT-B/14 (frozen) ─► 768×16×16 ─pool─► 768×8×8 ─► semantic enc ─► 8×8×8  ─► 256 symbols ─┐
   │                                                                                                     │  y = z + w
   └─► detail enc (4× stride-2 conv) ─────────────────────────────────────────► 24×8×8 ─► 768 symbols ─┤  same σ²
                                                                                                         │
                       (each branch power-normalised per image: mean |z_i|² = 1)                         ▼
                                                                                      received grids 8×8×8, 8×8×24
                                                                                                         │
   noise x_T (4×16×16) ─► conditional DiT-S, S DDIM steps, fused with received grids ─► latent ─► SD-VAE decoder ─► x̂
```

### 7.1 Data

**Imagenette.** fast.ai's 10-class ImageNet subset, 320 px release
([semcom/gen/data.py](../../semcom/gen/data.py)).

- **Images:** short side resized to 128 (bicubic), then centre-cropped to 128 × 128.
- **Training set:** 9,469 images, with horizontal-flip augmentation.
- **Validation and test:** the official val split (3,925 images) is split by a fixed permutation
  into **500 validation** images (model selection, diagnostics) and **3,425 test** images
  (evaluation only; we evaluate the first 1,000).

Imagenette keeps ImageNet's image statistics, which DINOv2 and the VAE were trained on, at 1/100
of the size.

**Frozen models run once.** The VAE latent and the DINOv2 features of every image, and of its
flipped copy, are computed once and cached as memory-mapped arrays. Training then runs only the
JSCC encoders and the DiT. This is the main reason a run takes about 1.5 h rather than many.

### 7.2 Latent space: the Stable Diffusion VAE

We use `stabilityai/sd-vae-ft-ema`, frozen ([semcom/gen/frozen.py](../../semcom/gen/frozen.py)).

- **Encoding:** the posterior mean (not a sample, so the cache is deterministic), times the
  standard scale 0.18215. This gives latents of shape $4 \times 16 \times 16$ with standard
  deviation ≈ 0.80 (`SANITY data`, g-w2).
- **Decoding:** the receiver's final latent goes through the VAE decoder.

**The VAE ceiling.** Decoding the *true* latent gives PSNR 22.8 dB and DINOv2 similarity 0.81 on
the 32 fixed validation images (`SANITY vae_ceiling`, g-w2). No DiT receiver can beat this,
however good, so it is reported next to every result.

### 7.3 The semantic branch

**DINOv2.** DINOv2 ViT-B/14, frozen. The 128 px image is upsampled (bicubic) to 224 px, giving
$16 \times 16$ patch tokens of dimension 768. These are **average-pooled to an $8\times 8$ grid**,
one cell per 16 × 16 image pixels.

**Encoder.** A small CNN: 1×1 conv 768→256, GELU, 3×3 conv 256→256, GELU, 1×1 conv 256→8
([semcom/gen/model.py](../../semcom/gen/model.py), `SemanticEncoder`). The output is
$8 \times 8 \times 8 = 512$ real values, paired into **$k_s = 256$ complex symbols**.

### 7.4 The detail branch

**Encoder.** Four 4×4 stride-2 convolutions (width 128, GELU), taking 128 → 8 per side, then a 3×3
conv and a 1×1 conv to 24 channels (`DetailEncoder`). This gives $8 \times 8 \times 24 = 1536$
real values, or **$k_d = 768$ complex symbols**.

**Total.** $k = 1024$, CBR 1/48, with a fixed 25% / 75% semantic/detail split.

### 7.5 Power normalisation and the channel

Each branch is power-normalised **separately, per image**, to unit average symbol power. The
split therefore allocates *bandwidth* between semantics and detail, never *power*. Both branches
pass through the same AWGN channel at the same $\sigma^2$.

The channel is a differentiable addition of noise, so gradients flow from the receiver's loss
back through it into both encoders. The pre-flight check verifies this (`encoder_grad > 0`, §9.3).

### 7.6 The receiver: a conditional DiT-S

Based on DiT-S ([semcom/gen/dit.py](../../semcom/gen/dit.py)):

- **Tokens.** The $4 \times 16 \times 16$ noisy latent is patchified with $p = 2$ into
  $8 \times 8 = 64$ tokens of width 384, plus sine-cosine positions. Token $(i,j)$, symbol cell
  $(i,j)$ and image patch $(16i..16i{+}15,\ 16j..16j{+}15)$ all refer to the **same image region**,
  so the received symbols need no resampling to line up with the tokens.
- **Blocks.** 12 transformer blocks, hidden 384, 6 heads, MLP ratio 4, **adaLN-Zero** as in DiT.
- **Conditioning vector $c$.** The timestep embedding, plus an SNR embedding in the arms that have
  an SNR slot (§8).
- **Condition adapters.** Each received grid ($8{\times}8{\times}8$ semantic,
  $8{\times}8{\times}24$ detail) passes through a 3×3 conv (to width 384), GELU and a 1×1 conv,
  giving one 384-d feature $a_{ij}$ per token. The 3×3 kernel lets each token see its neighbours'
  symbols.
- **Channel fusion.** Blocks 1–3 receive the semantic features, blocks 4–6 the detail features,
  and blocks 7–12 neither. In a conditional block, before attention:

  $$h \leftarrow h + W\,[\,h \,;\, a\,], \qquad W \in \mathbb{R}^{384 \times 768},\ W \text{ zero-initialised}.$$

  Zero initialisation means training starts from an *unconditional* DiT and learns to use the
  channel. This mirrors adaLN-Zero's start-from-identity design.
- **Null conditions.** Two learned 384-d vectors replace the adapter outputs when the condition
  is dropped, which happens with probability 0.1 per image in training (CFG, §4.4).
- **Output.** An adaLN-modulated final layer outputs $2 \times 2 \times 4$ values per token,
  unpatchified to $4 \times 16 \times 16$: the **velocity** $\hat v$ (§10.2).

**Parameter count.** 36,386,864 for the whole trainable model: both encoders, the adapters and
the DiT. The CNN baseline has 10,422,815.

### 7.7 What we changed, relative to each parent

| | DiT (Peebles & Xie) | DiT-JSCC (paper) | **Prototype** | Why |
|---|---|---|---|---|
| Task | class-conditional generation | reconstruction through a channel | reconstruction through a channel | — |
| Data, resolution | ImageNet 256 px | ImageNet 256 / 512 px | **Imagenette, 128 px** | ~100× less data and 4× fewer pixels than 256 px, so many runs fit on a T4 |
| Condition | class label (adaLN) | received symbols (fusion, early blocks) | received symbols (fusion, blocks 1–6) | same idea as the paper |
| Backbone | DiT-S … XL | large DiT, long skips, pretrained init | **DiT-S, no long skips, from scratch** | compute; no pretrained 128 px conditional DiT exists |
| Tokens | 256 (DiT-S/2, 256 px) | — | **64** | the 16 × 16 latent at patch 2 |
| Semantic branch | — | DINOv2 ViT-B + JSCC encoder | same, pooled to 8 × 8 | same idea |
| Detail branch | — | SwinJSCC | **4-stage strided CNN** | simpler; same downsampling factor (16) |
| Bandwidth split | — | per image, from a BLIP-2 caption | **fixed 256 / 768** | removes a confound from the SNR study |
| Target | noise (+ learned variance) | noise prediction | **velocity $v$** | the noise target failed here (§10.2) |
| EMA | 0.9999 | — | **ramped decay, cap 0.9999** | the fixed decay failed here (§10.1) |
| SNR handling | — | one model per SNR | **specialists, plus *told* and *blind* single models** | **the research question** |

The scale-down means our absolute numbers are not comparable to the paper's. The study compares
**arms trained identically on the same data**, evaluated on the same images with the same noise.

---

## 8. The SNR question: specialists, told, blind

### 8.1 The arms

All arms share the architecture of §7. They differ only in the training SNR distribution and in
what fills the SNR slot ([semcom/gen/config.py](../../semcom/gen/config.py), `snr_mode`;
[semcom/gen/model.py](../../semcom/gen/model.py), `snr_input`).

| Arm | Training SNR | SNR slot | What it learns |
|---|---|---|---|
| `spec−5`, `spec0`, `spec5` | fixed at one $\gamma$ | none | $p(x \mid y)$ at that one $\gamma$: DiT-JSCC's practice |
| `told-5to5` | $\gamma \sim U[-5, 5]$ dB per image | embedding of $\gamma$ | $p(x \mid y, \gamma)$: one model, informed |
| `blind-5to5` | $\gamma \sim U[-5, 5]$ dB per image | embedding of the constant 0 | $p(x \mid y) = \int p(x \mid y, \gamma)\,p(\gamma \mid y)\,d\gamma$: one model, uninformed |

**The SNR slot.** In `told`, $\gamma$ is passed as $10\gamma$ through the same sinusoidal embedder
and MLP as the timestep, then added to $c$. It modulates every block through adaLN, exactly like
the timestep.

**Matched blind control.** `blind` has *identical* parameters, but the slot always holds 0. The
told-vs-blind difference therefore isolates the *information* in $\gamma$, not a difference in
architecture or capacity. This is the same matched-control design as the repo's blind-JSCC study
(arms B and C-att).

### 8.2 Why "blind" may not be blind: SNR is visible in the received power

Because each branch is normalised to exactly unit power per image, the received power reveals
the noise power:

$$\frac{1}{k}\|y\|^2 = 1 + \frac{2}{k}\,\mathrm{Re}\langle z, w\rangle + \frac{1}{k}\|w\|^2,
\qquad \hat\sigma^2 = \frac{1}{k}\|y\|^2 - 1 .$$

$\hat\sigma^2$ is unbiased. Since $\|w\|^2/\sigma^2$ is a sum of $k$ unit-mean exponentials
(variance $k\sigma^4$) and the cross term has variance $2k\sigma^2$,

$$\mathrm{Var}(\hat\sigma^2) = \frac{\sigma^4 + 2\sigma^2}{k}.$$

With $k = 1024$ (both branches together):

| SNR | $\sigma^2$ | std of $\hat\sigma^2$ (formula) | std (simulated, 4,000 images) | ≈ in dB |
|---|---|---|---|---|
| −5 dB | 3.162 | 0.126 | 0.126 | 0.17 dB |
| 0 dB | 1.000 | 0.054 | 0.054 | 0.23 dB |
| 5 dB | 0.316 | 0.027 | 0.026 | 0.35 dB |

The simulation used the repo's own `power_normalise` and `channel`.

**So the received symbols carry the SNR to within a fraction of a dB**, far finer than the 10 dB
training range. A blind receiver *could* infer the SNR. Nothing in the architecture prevents it:
the adapters see per-cell magnitudes, and attention can aggregate them over the image.

**Whether the DiT learns to do this is an empirical question, and it is what told-vs-blind
measures.** If blind ≈ told, the SNR slot is redundant for this receiver, and this is the likely
reason. If blind is clearly worse, the network does not extract the information unaided, which is
worth knowing for generative receivers.

### 8.3 What "one model for all SNRs" would buy

- **Storage.** One 36.4M-parameter receiver instead of one per operating SNR: 3× here, more
  with a finer SNR grid.
- **Operation.** No switching between models as the channel changes.
- **Training.** One run instead of three.

Decode cost is the same per image, set by the DDIM step count (§12.3).

---

## 9. Training

### 9.1 The objective

For a training image with cached latent $x_0$, DINOv2 features and pixels:

1. Draw the channel SNR $\gamma$: fixed, or $U[-5,5]$ dB.
2. Encode both branches, power-normalise, add channel noise: received grids $y_s, y_d$.
3. Draw $t \sim U\{0,\dots,999\}$ and $\epsilon \sim \mathcal{N}(0, I)$; form $x_t$.
4. With probability 0.1, replace both received conditions by the null embeddings.
5. Minimise
   $$\mathcal{L}(\phi,\theta) = \big\|F_\theta(x_t, t, y_s, y_d, \text{slot}) - v_t\big\|^2,\qquad v_t = \sqrt{\bar\alpha_t}\,\epsilon - \sqrt{1-\bar\alpha_t}\,x_0 .$$

The loss is a function of the encoders' parameters $\phi$ through $y_s, y_d$. One backward pass
trains **the encoders and the receiver jointly, end to end**; that is the "J" in JSCC. DINOv2 and
the VAE stay frozen.

**The encoders are never told to reconstruct anything directly.** They learn to send whatever
most helps the receiver's denoising, which at high noise means whatever best predicts the clean
latent.

**The CNN baseline** is the repo's deep JSCC (`semcom.models.JSCC`). Its 32 × 32 × 2 output gives
the same $k = 1024$. It is trained on plain pixel MSE with the same data, steps and optimiser.

### 9.2 Optimisation

From [configs/gen/p128.yaml](../../configs/gen/p128.yaml) and
[semcom/gen/train.py](../../semcom/gen/train.py):

| Setting | Value |
|---|---|
| Optimiser | AdamW, lr $10^{-4}$, no weight decay; linear warm-up over 1,000 steps, then constant |
| Batch, steps | 64, **40,000 steps** (= 2.56 M samples ≈ **270 epochs** of 9,469 images) |
| Precision | fp16 autocast with dynamic loss scaling; gradient-norm clip 1.0 |
| EMA | $\beta_t = \min(0.9999,\ (1+t)/(10+t))$; the EMA weights are the ones evaluated (§10.1) |
| Validation | every 2,000 steps on the 500 validation images |
| Model selection | `best.pt` = EMA weights at the lowest validation loss |
| Throughput | ≈ 8 it/s on a T4, so ≈ 1.45 h per 40k-step run (g-w2) |

270 epochs on a small dataset invites overfitting. Selecting `best.pt` by validation loss guards
against it. In g-w2 the validation loss was still falling at 40k, so best = last.

### 9.3 Built-in correctness checks

A generative receiver that ignores the channel still produces realistic images, so plausible
samples prove nothing. The pipeline therefore checks itself
([semcom/gen/checks.py](../../semcom/gen/checks.py)), and the watcher script
(`scripts/watch_job.py`) turns these lines into alerts.

**Before step 1:**
- **Data.** The cache is finite, images lie in [0, 1], the latent spread shows the VAE scale was
  applied, and the DINOv2 features are not constant.
- **VAE ceiling.** See §7.2.
- **One-batch overfit.** A copy of the model, with all randomness frozen, must drive its loss on
  one batch down by at least half, and gradient must reach both encoders. In g-w2 the loss fell
  from 0.725 to 1e-6, and the encoders received gradient. This catches a detached channel path, a
  wrong target or a sign error in minutes.

**At every validation (2,000 steps):**
- **`cond_gain`.** The validation loss with every received symbol dropped, minus the loss with
  them. This is the direct measure of how much the receiver uses the channel. If it is ≈ 0, the
  model is an unconditional generator.
- **Real reconstructions** of 32 fixed validation images (20 DDIM steps), scored by PSNR and
  DINOv2 similarity.
- **`zstd`**, the spread of the sampled latents relative to the true ones. If it is far from 1,
  the sampler is producing latents the VAE never saw.
- **All of the above for both the EMA and the raw weights**, so that an EMA problem cannot pass
  for a model problem. This check was added after g-w1 (§10.1).

---

## 10. What went wrong first, and the two fixes

The first job, **g-w1**, used the noise target and the conventional fixed EMA. The losses
looked healthy, but the DiT's reconstructions were nearly unrelated to the inputs:

| g-w1 DiT at 0 dB, 20 steps | Value | Reference |
|---|---|---|
| PSNR | 13.0 dB | 9.4 dB for an *unrelated* real photo |
| DINOv2 similarity | 0.07 | 0.34 for an unrelated real photo |

Both are references measured on our images. The diagnosis, on the downloaded weights, found two
independent problems.

### 10.1 The EMA was dominated by almost-untrained weights

The original rule was DiT's fixed decay $\beta = 0.9999$, applied from step 1,000, with the EMA
copied from the weights at that step. The EMA then holds $\beta^{\,s}$ of the step-1,000 weights
after $s$ more steps:

$$0.9999^{9000} = e^{-0.90} = 0.41 \ \text{at step 10k}, \qquad 0.9999^{39000} = 0.020 \ \text{at step 40k}.$$

Decay 0.9999 averages over about $1/(1-\beta) = 10{,}000$ steps. That suits DiT's 400k+-step
runs, but at 40k steps it is a quarter of the run, and in the early part the evaluated model
was mostly a barely-trained one. Evaluating g-w1's checkpoint offline, the raw weights
reconstructed about 2.4 dB better than the EMA. The CNN's mid-run rise in validation loss was
the same artefact.

**Fix: the ramped decay.** $\beta_t = \min(0.9999,\ (1+t)/(10+t))$. Then $1 - \beta_t = 9/(10+t)$,
and the weight of step $s$ in the average at step $T$ is approximately proportional to
$(s/T)^8$. Computed exactly for $T = 40{,}000$:

- the average's mean age is $0.90\,T$;
- 61% of its weight comes from the last 10% of training, 92% from the last 25%.

The EMA tracks recent training at any run length. The cap 0.9999 is only reached at $t \approx
90{,}000$; at 40k, $\beta = 0.99978$. In g-w2, raw and EMA weights agree within 0.3 dB throughout.

### 10.2 The noise target leaves the image layout unconstrained

Write each loss in terms of the error of the implied clean-latent estimate, using the identities
in §4.2:

$$
\|\hat\epsilon - \epsilon\|^2 = \lambda_t\,\|\hat x_0 - x_0\|^2,
\qquad
\|\hat v - v\|^2 = \frac{1}{1-\bar\alpha_t}\,\|\hat x_0 - x_0\|^2 = (1 + \lambda_t)\,\|\hat x_0 - x_0\|^2 .
$$

Derivation for $v$: from $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$ one checks
$x_0 = \sqrt{\bar\alpha_t}\,x_t - \sqrt{1-\bar\alpha_t}\,v$, so
$\hat x_0 - x_0 = -\sqrt{1-\bar\alpha_t}\,(\hat v - v)$.

So the **noise loss weights clean-latent errors by $\lambda_t$**, which is
$3\times10^{-3}$ at $t = 750$ and $4\times10^{-5}$ at $t = 999$. The **velocity loss weights them by
$1 + \lambda_t \ge 1$** everywhere. Equivalently, a small noise-prediction error at high $t$ becomes
a huge clean-latent error: it is amplified by $\sqrt{(1-\bar\alpha_t)/\bar\alpha_t}$, which is 17
at $t = 750$ and 157 at $t = 999$.

**Why this is fatal for JSCC specifically.** At high $t$, $x_t$ is almost pure noise, so the best
estimate of the clean latent is $\mathbb{E}[x_0 \mid x_t, y] \approx \mathbb{E}[x_0 \mid y]$. *All*
information about the image's layout must come from the received symbols. These are also the
steps where DDIM commits to the global layout: its first steps move $x_t$ towards $\hat x_0$.

Under the noise loss, the network is barely penalised for an $\hat x_0$ that ignores $y$ at
exactly these steps. Two things follow:

- **The encoders get almost no training signal to make $y$ informative about layout.**
- **Sampling starts from a layout that is essentially random.** Later, low-noise steps, which the
  noise loss does weight, then add plausible texture to the wrong arrangement.

This is the "sharp but wrong" failure in its purest form.

**Measured on g-w1's checkpoint.** At $t = 999$, the conditioned model's clean-latent error was
11.7, *worse than with the conditions dropped*.

**Toy model.** A toy with a known answer reproduced the effect: sampled-latent error to the
true target of 0.80 to 1.11 with the noise target, against 0.12 with velocity.

**Fix:** the velocity target (`prediction: v`). The clean latent appears in the target at every
noise level, so conditioning is rewarded where it matters.

### 10.3 Which fix mattered how much: the ablation

g-w2 trained the velocity model *and* a noise-target model with the fixed EMA. g-w1 had trained
the noise-target model with the old EMA. All three ran 40k steps on the same data and were
evaluated identically:

| at 0 dB, 20 DDIM steps | Target | EMA | PSNR | LPIPS ↓ | DINOv2 ↑ | FID ↓ |
|---|---|---|---|---|---|---|
| g-w1 DiT | noise | fixed 0.9999 | 13.01 | 0.490 | 0.070 | 240.5 |
| g-w2 ablation | noise | ramped | 16.14 | 0.288 | 0.142 | 167.6 |
| **g-w2 main** | **velocity** | **ramped** | **17.49** | **0.266** | **0.170** | **149.3** |

- **The EMA fix accounts for most of the recovery:** +3.1 dB, and LPIPS from 0.49 to 0.29.
- **The velocity target adds a consistent further gain:** +1.35 dB, with better LPIPS, DINOv2 and
  FID, at all three SNRs and all three step counts.
- **Velocity also learns to use the channel faster.** At 12k steps the noise-target model's
  validation DINOv2 was 0.084 against 0.135 for velocity, and it only caught up partially by 40k.
  This fits §10.2: the noise loss gives the conditioning a weaker learning signal.

The two models' loss values (0.139 vs 0.293) are **not comparable**: they are MSEs to different
targets.

---

## 11. Evaluation protocol

From [semcom/gen/evaluate.py](../../semcom/gen/evaluate.py) and
[semcom/gen/metrics.py](../../semcom/gen/metrics.py):

- **Images.** The first 1,000 test images. These are never used for training or model selection.
- **Conditions.** Channel SNR −5, 0 and 5 dB for every model, including specialists away from
  their training SNR, which is the mismatch test. DDIM with 50, 20 and 10 steps, $\eta = 0$,
  guidance scale 1.
- **Weights.** `best.pt` (EMA).
- **Paired randomness.** The channel noise and the sampling noise are seeded per (SNR, batch), so
  every model sees *the same* noise on *the same* image. Differences between arms are then paired
  differences, not differences of independent noisy estimates.

### 11.1 Metrics

| Metric | Definition | Measures | Better |
|---|---|---|---|
| PSNR | $10\log_{10}(1/\mathrm{MSE})$ per image, averaged | pixel fidelity | higher |
| LPIPS | AlexNet-feature distance (Zhang et al., 2018) | perceptual similarity to the original | lower |
| DINOv2 similarity | cosine between DINOv2 ViT-B/14 CLS embeddings of $x$ and $\hat x$ | semantic agreement (the paper's "DINOv2" column) | higher |
| FID | Fréchet distance of Inception-v3 pool features, 1,000 reconstructions vs 1,000 originals | realism of the set | lower |
| Decode time | seconds per image, batch 50, T4 | receiver cost | lower |

**FID with 1,000 samples is biased upward** relative to the usual 50k-sample FID. Compare FIDs
only within this protocol, never with the paper's.

### 11.2 Calibration: what the numbers mean on these images

Measured on our validation images:

| Reconstruction | PSNR | DINOv2 sim. |
|---|---|---|
| the VAE alone (the DiT's ceiling) | 22.6–22.8 dB | 0.77–0.81 |
| Gaussian blur, σ = 4 px | 20.0 dB | 0.35 |
| down/up-sampling ×8 | 19.5 dB | 0.24 |
| flat grey | 11.3 dB | ≈ 0 |
| a different, unrelated real photo | 9.4 dB | 0.34 |

Two lessons:

1. **High PSNR does not mean recognisable.** A heavy blur scores 20 dB.
2. **DINOv2 similarity penalises artefacts heavily.** An unrelated *clean* photo scores 0.34,
   higher than a blurred copy of the right image. So DINOv2 similarity is read alongside LPIPS
   and the sample grids, never alone.

---

## 12. Results so far

### 12.1 The 0 dB specialist vs the CNN baseline

Both models were trained at 0 dB and evaluated on 1,000 test images. DiT figures are at 20 DDIM
steps.

| Test SNR | Model | PSNR | LPIPS ↓ | DINOv2 ↑ | FID ↓ |
|---|---|---|---|---|---|
| −5 dB | CNN | **14.31** | 0.648 | 0.023 | 302.7 |
| | DiT (v) | 13.48 | **0.421** | **0.042** | **253.5** |
| 0 dB | CNN | **21.21** | 0.426 | 0.161 | 207.5 |
| | DiT (v) | 17.49 | **0.266** | **0.170** | **149.3** |
| 5 dB | CNN | **21.80** | 0.448 | 0.198 | 216.4 |
| | DiT (v) | 18.45 | **0.253** | **0.229** | **126.5** |

**The week-1 gate** was fixed in advance: beat the CNN on LPIPS *and* DINOv2 at 0 dB, train
stably, and keep the right objects and layout in the sample grids. **It is passed.**

- **Perceptual quality.** LPIPS is 0.266 vs 0.426, FID 149 vs 207, DINOv2 0.170 vs 0.161. At
  5 dB the margins widen.
- **Sample grids.** At 0 and 5 dB they keep the scene layout, colours and objects. Fine
  structure such as faces is still distorted, which is expected at this size and bandwidth.
- **PSNR.** The DiT is 3.7 dB below the CNN at 0 dB and 3.4 dB at 5 dB. That is the size of gap
  §3.2 predicts for posterior sampling against MMSE (3.0 dB), with the caveats stated there.
- **Away from the training SNR (−5 dB),** both models degrade sharply. This motivates the
  week-2 study.

### 12.2 Training health (g-w2, velocity)

- **Validation loss:** fell monotonically from 0.386 (2k) to 0.293 (40k), still decreasing, so
  longer training would likely help.
- **Conditioning gain:** `cond_gain` 0.30. The loss roughly halves when the receiver sees the
  channel.
- **Sample spread:** latent spread 0.94 of the true spread.
- **Mixed precision:** 13 of 40,000 steps skipped.
- **Watcher:** no alerts.

### 12.3 Receiver cost

Measured on a T4, batch 50, per image:

| DDIM steps | 50 | 20 | 10 |
|---|---|---|---|
| Decode time | 0.101 s | 0.063 s | 0.050 s |

The time does not scale in proportion to the steps. Most likely this is because it also includes
fixed per-image costs: the VAE decode, the encoders and the channel. These were not timed
separately.

**Fewer steps scored slightly *better* here** on PSNR, DINOv2 and FID; LPIPS was roughly flat. At
0 dB, 10 steps gave 18.09 dB against 17.05 dB for 50.

A plausible explanation, not yet tested: coarser deterministic DDIM integration lands nearer the
conditional mean, which raises PSNR. It may also indicate an under-trained model whose
fine-step trajectory drifts. Either way, a 10-step receiver loses nothing at this scale. This is
relevant to the "lightweight receiver" direction.

---

## 13. The week-2 experiment

**Job g-w3, running.** Four further velocity-target DiT-S runs, identical to the 0 dB specialist
except for the SNR handling: `spec−5`, `spec5`, `told-5to5`, `blind-5to5`.

### 13.1 Comparisons and what each would mean

`scripts/gen_report.py` builds these tables and figures from the run folders.

1. **Specialist envelope vs `told`.** The envelope is each specialist scored *at its own* SNR,
   the best a bank of specialists can do. At each test SNR the report gives `told` minus envelope,
   signed so that positive is better for `told`.
   - **Small or no gap:** one informed model replaces three, at 3× less storage and one training
     run.
   - **Clear gap:** a single model pays a measurable price at this capacity. A larger DiT would
     be the natural next test.
2. **`told` vs `blind`.** This isolates the value of telling the receiver the SNR.
   - **blind ≈ told:** consistent with the receiver inferring the SNR from received power
     (§8.2). The SNR slot is then unnecessary.
   - **blind < told:** the network does not extract the SNR unaided. Explicit SNR side
     information then matters for generative receivers.
3. **Mismatch.** Each specialist away from its SNR, for example `spec0` at −5 dB, against
   `told`/`blind` there. This quantifies what a fixed-SNR model loses when the channel changes.
4. **All of the above at 50, 20 and 10 steps,** and against the CNN.

### 13.2 Reading the differences honestly

- **Evaluation noise.** Differences between arms are paired (same images, same noise), so their
  uncertainty is much smaller than the spread of any one metric. A bootstrap over images would
  give confidence intervals.
- **Seeds.** Each arm is a single training seed. A small gap between arms may be within
  seed-to-seed variation, which we have not yet measured.
- **Wording.** Conclusions should be framed as "at this scale and training budget".

---

## 14. Limitations and what we must not claim

**Scale.**
- Imagenette at 128 px, a 36M-parameter receiver trained from scratch, 40k steps.
- Absolute numbers are not comparable to DiT-JSCC's: ImageNet at 256/512 px, a pretrained large
  DiT, 300k iterations.
- The study's claims are about **differences between arms under identical conditions**.

**Simplifications.**
- A fixed bandwidth split (no BLIP-2 allocation).
- A CNN detail encoder instead of SwinJSCC; no long skips.
- AWGN only; Rayleigh fading is implemented but not yet run.
- Analog (unquantised) channel symbols, as in the paper.

**Prior work (see the proposal's §2).**
- SNR-adaptive diffusion receivers exist: SGD-JSCC, DiffJSCC, LRISC, DriftDecode.
- SGD-JSCC already reports near-identical performance with and without the SNR.
- Our contribution is narrower: a matched comparison of one adaptive receiver, told vs blind,
  against per-SNR specialists, *in the DiT-JSCC design*, where the channel noise is not the
  diffusion noise (§6.3).
- **We must not claim** SNR-adaptive generative JSCC in general as new.

**Known open technical points.**
- **Terminal noise level.** The linear schedule's last step has $\bar\alpha_{999} = 4\times10^{-5}$,
  not 0. Sampling therefore starts from pure noise that was never quite seen in training. A
  zero-terminal-SNR schedule is a known remedy (Lin et al., 2024); it is not used here.
- **Number of sampling steps.** Fewer steps scored better (§12.3). This needs explaining before
  any decode-cost claims.
- **Training length.** Validation loss was still falling at 40k steps.

---

## 15. Symbol glossary

| Symbol | Meaning |
|---|---|
| $x$, $\hat x$ | original image, reconstruction (3 × 128 × 128, values in [0, 1]) |
| $n = 3HW$ | number of source values, 49,152 |
| $z$, $y$, $w$ | channel input, channel output, channel noise (complex) |
| $k = k_s + k_d$ | complex channel symbols per image: 1024 = 256 semantic + 768 detail |
| $\rho = k/n$ | channel bandwidth ratio, 1/48 |
| $\gamma$, $\sigma^2$ | **channel** SNR (dB) and noise power, $\sigma^2 = 10^{-\gamma/10}$ |
| $x_0$, $x_t$ | clean VAE latent (4 × 16 × 16), noised latent at diffusion step $t$ |
| $\beta_t$, $\bar\alpha_t$ | diffusion variance schedule and its cumulative product |
| $\lambda_t = \bar\alpha_t/(1-\bar\alpha_t)$ | **diffusion** signal-to-noise ratio (unrelated to $\gamma$) |
| $\epsilon$, $v$ | diffusion noise; velocity target $\sqrt{\bar\alpha_t}\epsilon - \sqrt{1-\bar\alpha_t}x_0$ |
| $S$ | number of DDIM sampling steps (50, 20, 10) |
| $s$ | classifier-free guidance scale (1 in evaluation) |
| $c$ | adaLN conditioning vector: timestep embedding (+ SNR embedding) |
| $a$ | per-token features of the received symbols, fused into blocks 1–6 |
| $\phi$, $\theta$ | encoder and receiver parameters, trained jointly |
| MMSE | minimum mean squared error, the error of $\mathbb{E}[x \mid y]$ |
| EMA | exponential moving average of the weights, decay $\beta_t$ |

**References.**
- **DiT:** W. Peebles, S. Xie, "Scalable Diffusion Models with Transformers," ICCV 2023.
- **DiT-JSCC:** K. Tan et al., arXiv:2601.03112, 2026.
- **DDPM:** J. Ho, A. Jain, P. Abbeel, NeurIPS 2020.
- **DDIM:** J. Song, C. Meng, S. Ermon, ICLR 2021.
- **Velocity target:** T. Salimans, J. Ho, "Progressive Distillation for Fast Sampling of Diffusion
  Models," ICLR 2022.
- **Classifier-free guidance:** J. Ho, T. Salimans, 2022.
- **Perception–distortion trade-off:** Y. Blau, T. Michaeli, CVPR 2018.
- **Zero terminal SNR:** S. Lin et al., "Common Diffusion Noise Schedules and Sample Steps are
  Flawed," WACV 2024.
- **LPIPS:** R. Zhang et al., CVPR 2018.
- **DINOv2:** M. Oquab et al., 2023.
