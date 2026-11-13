# DiT-JSCC, Explained with Intuition

**Paper:** K. Tan, J. Dai, S. Wang, G. Lu, S. Shao, K. Niu, W. Zhang, P. Zhang,
*"DiT-JSCC: Rethinking Deep JSCC with Diffusion Transformers and Semantic Representations,"*
[arXiv:2601.03112](https://arxiv.org/abs/2601.03112), January 2026.
Code: [github.com/semcomm/DiTJSCC](https://github.com/semcomm/DiTJSCC).

Every number and design detail below was checked against the paper's text. Where the paper does not
say something (for example, decoding time), this document says so instead of guessing.

---

## 1. The idea in one sentence

> When the channel is too poor to carry the pixels, **send what the receiver cannot guess (what is in
> the image, and where) and let a generative model fill in what it can guess (texture and fine
> detail).**

---

## 2. The setting: a channel far worse than usual

| | Typical deep JSCC (our CIFAR runs) | DiT-JSCC |
|---|---|---|
| Channel uses per pixel value | 1/12 | **1/384 to 1/48** |
| SNR range | 0 to 20 dB | **−5 to 5 dB** |
| Images | CIFAR-10, 32×32 | **ImageNet, 256×256 and 512×512** |
| Channels | AWGN | AWGN and Rayleigh fading |

At a bandwidth ratio of 1/384, a 512×512 colour image (786,432 values) gets about **2,048 channel
symbols**, all heavily corrupted. Exact reconstruction is impossible. The real question is what is
worth sending at all.

---

## 3. Why the existing approaches fail here

### 3.1 Deep JSCC blurs: averaging plausible answers

A deep JSCC decoder trained on pixel MSE learns to output the **average** of every image that is
consistent with what it received. When the channel says very little, many images are consistent,
and their average is a blur.

```
   what was received is consistent with...         MSE-optimal output
   ┌──────────┐   ┌──────────┐   ┌──────────┐      ┌──────────┐
   │ dog,     │   │ dog,     │   │ dog,     │      │ ░░▒▒░░   │
   │ head left│ + │ head up  │ + │ head right│  →  │ ▒▓▓▓▒ ?  │   a blurry "average dog"
   └──────────┘   └──────────┘   └──────────┘      └──────────┘
```

The paper calls these "clear blurry artifacts", caused by an objective that "biases the model toward
learning the pixel-wise average of plausible solutions across the dataset."

### 3.2 Diffusion decoders look sharp, but can drift to the wrong image

The fix tried before DiT-JSCC was a **generative decoder** (a diffusion model). It always produces a
realistic, sharp image, because it has learned what real images look like. But if the received signal
guides it only weakly, it produces a *realistic image of the wrong thing*. The paper reports
reconstructions that deviate "in terms of local structures or even object categories."

```
   sent: black dog, red collar, pebbly ground
   received (weak, noisy guidance)
          │
          ▼
   diffusion decoder:  "a dog on a beach, white collar"   ← sharp, realistic, WRONG
```

The paper identifies two causes:

1. **The encoder sends the wrong thing.** Earlier generative JSCC reused encoders trained for pixel
   reconstruction, which carry no explicit "what is this" information.
2. **The decoder is guided poorly.** The received signal is not wired into the generator in a way
   that constrains it strongly.

So there is a dilemma: **blurry but faithful, or sharp but possibly wrong.** DiT-JSCC aims for
**sharp and semantically right**.

---

## 4. The key intuition: describing a picture over a bad phone line

Imagine describing a photo to an artist over a crackling phone line, with only a few seconds:

1. First you say **what matters most**: *"A black dog with a red collar, running left, on pebbles."*
   With only that, the artist already paints the right scene. The details are invented, but the
   picture is "correct."
2. If you have time left, you add **details**: *"the collar has a buckle, the pebbles are wet."*
3. The artist's own skill (the **generative prior**) fills in everything you didn't say.

DiT-JSCC builds this literally:

| Phone-line analogy | DiT-JSCC component |
|---|---|
| "What matters most" | **Semantic branch:** features from DINOv2, a vision foundation model |
| "Details if time allows" | **Detail branch:** an ordinary pixel-domain JSCC encoder (SwinJSCC) |
| How long to spend on each | **Bandwidth allocation:** split by how complex the image is to describe |
| The artist's skill | **Diffusion transformer (DiT) decoder:** a generative model guided by both |

---

## 5. The full picture

```
 TRANSMITTER                                    CHANNEL          RECEIVER
 ───────────                                    ───────          ────────

                ┌──────────────┐  ┌───────────┐                  ┌───────────────────────────┐
           ┌──▶ │ DINOv2 ViT-B │─▶│ latent    │─ k_s symbols ─┐  │  Conditional DiT          │
           │    │ (frozen)     │  │ JSCC enc. │  "semantics"  │  │  (latent diffusion)       │
           │    └──────────────┘  └───────────┘               │  │                           │    ┌────────────┐
 image x ──┤                                                  ├─▶│  start from pure noise,   │──▶ │ VAE decoder│──▶ x̂
           │    ┌───────────────────────────────┐             │  │  denoise step by step,    │    │ (Stable    │
           └──▶ │ pixel JSCC encoder (SwinJSCC) │─ k_d ──────┘  │  guided by the received   │    │ Diffusion) │
                │                               │  "detail"      │  semantics and detail     │    └────────────┘
                └───────────────────────────────┘                └───────────────────────────┘
                        ▲                                  ŝ = h·s + n
                        │                            (AWGN: h = 1; Rayleigh: random h)
     ┌──────────────────┴─────────────────────┐
     │ bandwidth split  k = k_s + k_d          │
     │ chosen per image from a BLIP-2 caption  │
     └────────────────────────────────────────┘
```

Total symbols are fixed (k = k_s + k_d). Each image decides how to split them between semantics and
detail.

---

## 6. Each component, with intuition

### 6.1 Semantic branch: why DINOv2?

DINOv2 is a large vision model trained by self-supervised learning. Its features encode **what is in
an image and where**: object identity, layout and structure. They are *not* tuned to reproduce exact
pixels. The paper turns the image into a compact map of size H/16 × W/16 × 256 and describes it as a
latent space "with clear semantic separation and strong discriminative structure."

```
 512×512×3 image  ──DINOv2──▶  32×32×256 feature map  ──JSCC encoder──▶  k_s channel symbols
 (786k values)                 (layout + identity,                        (what gets sent)
                                not pixel detail)
```

**Why it helps:** these features are the information the generator *cannot* make up. Once the
receiver knows "black dog, red collar, pebbles, here", the generator can draw everything else
plausibly. DINOv2 stays frozen; only the small JSCC encoder after it is trained.

### 6.2 Detail branch

A standard pixel-domain JSCC encoder (SwinJSCC architecture, four downsampling stages) sends
whatever bandwidth is left as **detail**: edges, textures, colours that the semantics don't pin down.
The paper's Fig. 5 shows the division of labour:

```
 semantic only   →  right scene and objects, but blurry
 detail only     →  some texture, but structure can drift
 both            →  right scene AND sharp detail
```

### 6.3 The decoder: latent diffusion in one picture

A diffusion model generates an image by starting from **pure noise** and removing noise step by
step. At each step a network predicts the noise to subtract. **Conditioning** means that network
also sees extra information (here, the received symbols), which steers where the denoising ends up.

```
 step T (noise)                                                  step 0 (clean latent)
 ▓▒░▓▒░▓▒  ──denoise──▶  ▒░ ▒░▒  ──denoise──▶  ░ ▄▀░  ──...──▶  dog image (latent)
              ▲                      ▲                ▲
              └──────────────────────┴────────────────┘
                guided at every step by the received semantics + detail
```

Two standard choices keep this tractable:

- **Latent diffusion:** denoising happens in the compressed latent space of Stable Diffusion's VAE,
  not on raw pixels. The VAE decoder turns the final latent into the image.
- **DiT backbone:** the denoising network is a *transformer* (a "diffusion transformer"), the
  mainstream architecture for high-quality image generation. Here it has **43 blocks** (2N+1 with
  N = 21), 16 attention heads and hidden size 1024, with **U-Net-style long skip connections**
  between matching early and late blocks.

### 6.4 Coarse-to-fine conditioning

Not every block receives the conditions. The first **six** blocks are "conditional" blocks: the
earliest ones receive the **semantic** symbols, the next few the **detail** symbols. The rest are
ordinary transformer blocks.

```
 block:   1   2   3 │ 4   5   6 │ 7 ... 21 │ 22 (middle) │ 23 ... 43
          ───────── │ ───────── │ ──────── │             │ ──────────
          semantics │  detail   │  plain   │    plain    │  plain, with skips
          condition │ condition │          │             │  from blocks 21...1
          ◀── shallow half (all conditioning lives here) ──▶
```

*(The paper fixes six conditional blocks; the exact semantic/detail split among them is a tuned
hyperparameter, so the 3 + 3 above is illustrative.)*

**Intuition:** early layers decide **global structure** (what goes where), so they get the
semantics; the following layers refine with detail. The authors report that putting all the
conditioning in the shallow half worked best in their ablations.

The received symbols enter through **channel fusion**: they are projected by a small adapter and
concatenated to the transformer's token embeddings. The paper says this "better preserves spatial
consistency" than cross-attention, at negligible cost. It also uses **classifier-free guidance**, a
standard diffusion technique that strengthens how closely the output follows the conditions.

### 6.5 Training

- **Loss:** the standard latent-diffusion noise-prediction loss, trained **end to end** through both
  encoders (except the frozen DINOv2) and the DiT.
- **Starting point:** the DiT is initialised from a pretrained class-conditional DiT.
- **Data:** ImageNet training set; test on the same **100** ImageNet validation images used by
  DiffCom.
- **Compute:** **4× NVIDIA RTX 4090**, 300,000 iterations per model, learning rate 10⁻⁴, batch 96
  (256×256) or 32 (512×512).
- **One model per SNR:** "we train separate models for each SNR under both AWGN and Rayleigh fading
  channels." There is no SNR input; adaptivity across SNRs is not attempted.

---

## 7. Bandwidth allocation: spend more on images that are harder to describe

### 7.1 First finding: the best semantic budget is almost fixed

The authors searched exhaustively over semantic/detail splits at fixed total bandwidth (Fig. 7).
Across a wide range of total bandwidths, the best split always used **a similar amount of semantic
bandwidth**; the remainder goes to detail. Too little semantics breaks the image; beyond that point,
extra symbols are better spent on detail.

```
 quality │        ╭──── best: a similar semantic budget at every total bandwidth
         │      ╭─╯╲
         │    ╭─╯   ╲___
         │  ╭─╯         ╲____
         │ ╱                  ╲___
         └──────────────────────────▶ share of symbols given to semantics
```

### 7.2 Second idea: adjust that budget per image by "description length"

The intuition comes from **Kolmogorov complexity**: the information in an object is the length of
its shortest description. A plain blue sky is short to describe; a busy street is long. That quantity
cannot be computed, so the paper uses a **caption** as a stand-in:

```
 image ─▶ BLIP-2 caption ─▶ 3 measures ─────────────────▶ score I ─▶ split
                            • word count
                            • lexical diversity (unique words / words)
                            • syntactic complexity (tokens per sentence)
```

Example from the paper: *"A wooden barn stands in a green field with a bison grazing in front, set
against snow-covered mountains under a clear sky."* This is a long, varied caption, so the image
gets more semantic symbols.

The rule shifts the fixed-best semantic budget k̄_s up or down around the dataset average Ī:

```
 k_s = k̄_s + ⌈ η · (I − Ī) · k̄_s ⌉          k_d = k − k_s
```

It is **training-free**: it only changes the split at inference, so no retraining is needed. The
authors themselves call it a practical direction rather than a principled measure of "semantic
value."

---

## 8. Results

### 8.1 How to read the metrics

| Metric | What it measures | Better |
|---|---|---|
| LPIPS, DISTS, DreamSim | perceptual distance from the original (learned features) | lower |
| CLIP, DINOv2 similarity | how close the meaning / features are to the original | higher |
| FID | how realistic the outputs look as a set | lower |
| PSNR | pixel-level fidelity | higher, but **deliberately not targeted** here |

### 8.2 Headline table (ImageNet 256×256, bandwidth ratio 1/48, AWGN, SNR 0 dB, Table I)

| Method | LPIPS ↓ | DISTS ↓ | CLIP ↑ | DINOv2 ↑ | FID ↓ |
|---|---|---|---|---|---|
| BPG + 5G LDPC (separation) | 0.444 | 0.314 | 0.828 | 0.717 | 252 |
| VTM + 5G LDPC (separation) | 0.427 | 0.313 | 0.844 | 0.720 | 244 |
| SwinJSCC (deep JSCC, at 1 dB) | 0.261 | 0.233 | 0.915 | 0.878 | 138 |
| HiFi-DiffCom (NTSCC), best prior generative | 0.211 | 0.191 | 0.936 | 0.900 | 107 |
| **DiT-JSCC** | **0.166** | **0.151** | **0.963** | **0.958** | **60** |

- **Rayleigh fading, 0 dB (Table II):** DiT-JSCC reaches LPIPS 0.185 and FID 77, against SwinJSCC's
  0.297 and 169. BPG + LDPC cannot run at all ("the available bandwidth is insufficient to support
  BPG encoding").
- **Across SNR (Fig. 10):** ahead of every baseline over the whole −5 to 5 dB range (512×512,
  bandwidth ratio ≈ 1/96).
- **The trade-off, stated by the authors (Fig. 1):** best on every semantic metric "while its PSNR
  is much lower than the original deep JSCC."

---

## 9. Strengths and weaknesses

### Strengths

1. **Works where classical radio cannot.** It still produces recognisable, semantically right images
   at 1/384 channel uses per value and at −5 dB, where a separation scheme cannot even fit the
   compressed bitstream.
2. **A clear design principle.** Transmit what the generator cannot invent; let the prior supply the
   rest. This goes beyond one architecture.
3. **Reuses pretrained models** (DINOv2, a pretrained DiT, the Stable Diffusion VAE) instead of
   learning everything from scratch.
4. **Content-adaptive bandwidth with no retraining**, and the code is public.

### Weaknesses and open questions

1. **Not faithful pixel by pixel.** PSNR is much lower than deep JSCC. A generated detail can look
   right and still be wrong (a number, a face, a road sign), which rules it out where exactness
   matters.
2. **One model per SNR.** No SNR conditioning and no blind operation across SNRs: the problem ADJSCC
   solved comes back.
3. **Analog channel symbols.** The transmitted values are continuous, with no constellation. This is
   the deployment barrier DeepJSCC-Q removed.
4. **Heavy at both ends.** The transmitter runs DINOv2 *and* BLIP-2 (the caption is computed at the
   sender); the receiver runs an iterative diffusion sampler. The paper reports **no decoding time**
   and does not state the number of sampling steps.
5. **Thin evaluation.** 100 test images; the caption-based complexity score is a heuristic, as the
   authors acknowledge.

---

## 10. Where it sits relative to our work

```
                         legal QAM symbols?   one model for all SNRs?   pixel-faithful?
 DeepJSCC-Q                     yes                  no                     yes
 ADJSCC                         no                   yes                    yes
 ADJSCC-Q (ours)                yes                  yes                    yes
 DiT-JSCC                       no                   no                     no — semantic instead
```

DiT-JSCC optimises a different axis, semantic and perceptual quality at extreme settings, and gives
up the two practicality properties our work focuses on. Two links to the blind-decoder study:

- Its decoders get **no SNR input**, but each is trained at one SNR, so they are blind only in the
  trivial sense. Whether a single generative decoder could run blind across SNRs, as our CNN
  decoders do, is untested.
- A generative decoder that misjudges the noise level could make **plausible but wrong**
  reconstructions rather than merely blurrier ones. That makes the "what does a blind decoder rely
  on?" question more consequential for generative JSCC.

---

## 11. Questions a professor is likely to ask

**Q: Isn't this just image generation with a hint?**
Partly, and deliberately so. The contribution is choosing *which* hint survives a terrible channel
(foundation-model semantics), and wiring it into the generator so that it constrains the output
(early-layer conditioning, trained end to end), instead of reusing a pixel encoder.

**Q: Why not just send a caption?**
A caption fixes the category but not the layout or appearance. DINOv2 features keep spatial
structure, so the scene is drawn in the right arrangement. The caption is used only to decide the
bandwidth split.

**Q: How fair is the comparison with separation schemes?**
The separation baselines use VTM or BPG with 5G LDPC and 4-QAM, the code rate chosen per SNR. At
these bandwidths they often cannot deliver an image at all, so the comparison mostly shows that
separation breaks down in this regime.

**Q: Is the semantic gain real or just a metric artefact?**
The gain is consistent across five different semantic metrics (LPIPS, DISTS, CLIP, DreamSim, DINOv2)
and in visual examples, including text in images. But the test set is only 100 images, and no seed
variance is reported.

**Q: What would it take to deploy?**
Quantising the channel symbols to a standard constellation, one model across SNRs, and a much
cheaper receiver. None of these are addressed in the paper.

---

## 12. Glossary

| Term | Meaning |
|---|---|
| JSCC | Joint source–channel coding: one learned network maps an image straight to channel symbols |
| Generative JSCC (GJSCC) | JSCC with a generative (here, diffusion) decoder |
| Bandwidth ratio (CBR) | channel symbols per source value (k / 3HW) |
| DINOv2 | self-supervised vision foundation model whose features encode semantics and layout |
| Diffusion model | generates by iteratively denoising from pure noise |
| Latent diffusion | diffusion performed in a VAE's compressed latent space |
| DiT | Diffusion Transformer: a transformer used as the denoising network |
| Classifier-free guidance | sampling trick that strengthens adherence to the conditions |
| Kolmogorov complexity | length of the shortest description of an object; uncomputable, so approximated here by caption complexity |
| BLIP-2 | image-captioning vision–language model |
| LPIPS / DISTS / DreamSim | learned perceptual distances |
| FID | Fréchet Inception Distance: distribution-level realism |
