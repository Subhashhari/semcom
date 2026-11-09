"""Configuration for the generative-JSCC prototype (a small DiT-JSCC).

One dataclass covers both model families trained on the cached data:

    model="dit"   semantic + detail JSCC encoders, a conditional DiT receiver (the prototype)
    model="cnn"   the repo's CNN deep JSCC at the same bandwidth (the MSE baseline)

The SNR switch mirrors the blind-JSCC arms, so the same matched contrasts apply:

    snr_mode="fixed"   a specialist: trained at one SNR, the receiver is not told it
    snr_mode="snr"     one model for U[min, max], the receiver is told the SNR    (like B)
    snr_mode="blank"   identical architecture, the SNR slot holds a constant      (like C-att)

Digital transmission (`digital=True`): every channel symbol is an exact point of a square M-QAM
constellation (unit average power under uniform use), as in DeepJSCC-Q (semcom.constellation):
nearest-point quantisation going forward, a temperature-annealed softmax relaxation for the
gradient, and a KL-to-uniform penalty against codebook collapse. The receiver sees the noisy
received symbols, exactly what an equaliser hands a demapper; there is no channel code.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

#: DiT sizes from Peebles & Xie (2023), table 1: (depth, hidden, heads).
DIT_SIZES = {"S": (12, 384, 6), "B": (12, 768, 12), "L": (24, 1024, 16)}
SNR_MODES = ("fixed", "snr", "blank")


@dataclass
class GenConfig:
    # --- identity
    name: str = "gen-p128"
    seed: int = 0
    out_dir: str = "results/gen/p128"
    model: str = "dit"  # "dit" | "cnn"

    # --- data (built once by semcom.gen.data.prepare and cached)
    dataset: str = "imagenette"
    data_root: str = "data"
    image_size: int = 128
    dino_model: str = "dinov2_vitb14"
    dino_input: int = 224  # DINOv2 sees the image at this size; 224 / 14 = 16 patches a side
    vae_model: str = "stabilityai/sd-vae-ft-ema"
    vae_scale: float = 0.18215  # Stable Diffusion's latent scaling factor
    # Tiny fixed random stand-ins for the VAE and DINOv2 (semcom.gen.frozen): offline tests only.
    fake_frozen: bool = False

    # --- bandwidth. Both branches emit an (image_size/16)^2 grid; channels pair into I/Q.
    c_sem: int = 8  # semantic channels per grid cell -> k_s = grid^2 * c_sem / 2
    c_det: int = 24  # detail channels per grid cell  -> k_d = grid^2 * c_det / 2
    det_hidden: int = 128  # width of the detail encoder
    cnn_c_out: int = 2  # CNN baseline: k = (image_size/4)^2 * c_out / 2
    cnn_hidden: int = 256

    # --- channel and SNR
    channel: str = "awgn"
    avg_power: float = 1.0
    snr_mode: str = "fixed"
    snr_train_fixed: float | None = 0.0
    snr_train_min: float = -5.0
    snr_train_max: float = 5.0
    # Digital transmission: M-QAM channel symbols (see the module docstring). SNR is Es/N0 with
    # Es the constellation's average energy under uniform use (1).
    digital: bool = False
    modulation_order: int = 16
    # Softmax inverse temperature of the gradient relaxation. None: 2 / d_min^2 (5 for 16-QAM, 1 for
    # QPSK, 21 for 64-QAM), held fixed. Measured on unit-power Gaussian inputs, this leaves ~91% of
    # symbols with a surrogate gradient above 0.1 for every M; DeepJSCC-Q's rise to 100 (from
    # `anneal_period`) leaves 14% for 16-QAM, so the encoders would barely train after it.
    sigma_q_init: float | None = None
    anneal_period: int | None = None  # None: no annealing; 10_000 reproduces DeepJSCC-Q's schedule
    kl_weight: float = 0.05  # weight of KL(symbol usage || uniform), training only
    # What the SNR is measured against, digital only. "actual": Es/N0 with Es the energy the symbols
    # really carry (the batch mean in training; at evaluation one constant measured on the test
    # set, i.e. a fixed constellation scaled by one amplifier gain). "nominal": Es = 1, the
    # constellation's energy under uniform use, DeepJSCC-Q's convention; clipping onto the outer
    # points then makes the encoder send less energy (0.83 in g-w4), a hidden SNR handicap.
    # Default "nominal" so configs saved before this option (g-w4) load unchanged.
    power_ref: str = "nominal"

    # --- DiT receiver
    dit_size: str = "S"
    patch_size: int = 2
    cond_blocks_sem: int = 3  # blocks that take the semantic symbols (coarse)
    cond_blocks_det: int = 3  # the following blocks take the detail symbols (fine)
    cfg_dropout: float = 0.1  # drop all received symbols with this probability (for CFG)

    # --- diffusion
    num_timesteps: int = 1000
    prediction: str = "eps"  # network target: "eps" (noise, as DiT) or "v" (velocity)
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # --- optimisation (iterations, not epochs: diffusion training is step-budgeted)
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_steps: int = 1000
    batch_size: int = 64
    steps: int = 50_000
    ema_decay: float = 0.9999  # cap on the EMA decay
    # Ramp the EMA decay as min(ema_decay, (1 + step) / (10 + step)), so the average spans about
    # the last tenth of training. A fixed 0.9999 from step 1000 keeps 41% of the barely-trained
    # step-1000 weights at step 10k; in the first 40k-step run this broke sampling (g-w1).
    ema_warmup: bool = True
    grad_clip: float = 1.0
    num_workers: int = 2
    amp: bool = True
    # Stop cleanly (checkpoint saved, no history.json) after this many hours, so a run that
    # would outlive a Kaggle session resumes in the next job instead of being killed. None: off.
    max_hours: float | None = None

    # --- fine-tuning diagnostics: start from another run's weights and train only some parts.
    # Used to locate where one SNR-adaptive model loses to a specialist (transmitter or receiver).
    init_from: str | None = None  # run folder whose best.pt (else final.pt) EMA weights start training
    train_parts: str = "all"  # "all", "transmitter" (the encoders) or "receiver" (the DiT); DiT only
    tag: str = ""  # appended to the run name, so fine-tunes never collide with trained-from-scratch runs

    # --- correctness checks (semcom.gen.checks)
    sanity: bool = True  # pre-flight data / VAE-ceiling / overfit checks before step 1
    sanity_steps: int = 150  # optimiser steps of the one-batch overfit check
    val_quality_images: int = 32  # val images reconstructed at every validation pass
    max_bad_steps: int = 100  # abort after this many consecutive non-finite losses

    # --- evaluation
    eval_snrs: list[float] = field(default_factory=lambda: [-5.0, 0.0, 5.0])
    sample_steps: list[int] = field(default_factory=lambda: [50, 20, 10])
    cfg_scale: float = 1.0
    # Weights evaluated: "best" (lowest validation loss; early stopping, since Imagenette is small
    # and the g-w1 CNN's validation loss rose from step 6k) or "final" (end of training).
    eval_checkpoint: str = "best"
    eval_images: int = 1000  # test images per SNR point; None-like 0 means all
    eval_batch: int = 50
    val_every: int = 2_000  # steps between validation-loss passes
    sample_every: int = 10_000  # steps between sample grids
    ckpt_every: int = 2_000

    # --- logging
    wandb: bool = True
    wandb_project: str = "gen-jscc"
    wandb_entity: str | None = None
    wandb_mode: str = "online"
    log_every: int = 100

    def __post_init__(self):
        if self.model not in ("dit", "cnn"):
            raise ValueError(f"model must be 'dit' or 'cnn', got {self.model!r}")
        if self.eval_checkpoint not in ("best", "final"):
            raise ValueError(f"eval_checkpoint must be 'best' or 'final', got {self.eval_checkpoint!r}")
        if self.prediction not in ("eps", "v"):
            raise ValueError(f"prediction must be 'eps' or 'v', got {self.prediction!r}")
        if self.train_parts not in ("all", "transmitter", "receiver"):
            raise ValueError(f"train_parts must be 'all', 'transmitter' or 'receiver', got {self.train_parts!r}")
        if self.train_parts != "all" and self.model != "dit":
            raise ValueError("train_parts applies to the DiT model only")
        if self.tag and not all(c.isalnum() or c == "-" for c in self.tag):
            raise ValueError(f"tag must be letters, digits and '-', got {self.tag!r}")
        if self.snr_mode not in SNR_MODES:
            raise ValueError(f"snr_mode must be one of {SNR_MODES}, got {self.snr_mode!r}")
        if self.snr_mode == "fixed" and self.snr_train_fixed is None:
            raise ValueError("snr_mode='fixed' needs snr_train_fixed")
        if self.snr_mode != "fixed" and self.snr_train_min > self.snr_train_max:
            raise ValueError("snr_train_min must not exceed snr_train_max")
        if self.power_ref not in ("nominal", "actual"):
            raise ValueError(f"power_ref must be 'nominal' or 'actual', got {self.power_ref!r}")
        if self.digital:
            m, side = self.modulation_order, round(self.modulation_order ** 0.5)
            if m < 4 or m & (m - 1) or side * side != m:
                raise ValueError(f"modulation_order must be square QAM (4, 16, 64, ...), got {m}")
        if self.image_size % 16 != 0:
            raise ValueError(f"image_size must be divisible by 16, got {self.image_size}")
        if self.c_sem % 2 or self.c_det % 2 or self.cnn_c_out % 2:
            raise ValueError("channel counts must be even to pair into (I, Q)")
        if self.dit_size not in DIT_SIZES:
            raise ValueError(f"dit_size must be one of {sorted(DIT_SIZES)}")
        depth = DIT_SIZES[self.dit_size][0]
        if self.cond_blocks_sem + self.cond_blocks_det > depth:
            raise ValueError("more conditional blocks than DiT blocks")
        if (self.image_size // 8) % self.patch_size:
            raise ValueError("VAE latent size must be divisible by patch_size")

    # --- derived quantities
    @property
    def grid(self) -> int:
        """Side of the symbol grid both DiT branches emit (one cell per 16x16 pixels)."""
        return self.image_size // 16

    @property
    def sigma_q(self) -> float:
        """The quantiser's starting inverse temperature (see sigma_q_init)."""
        if self.sigma_q_init is not None:
            return float(self.sigma_q_init)
        side = round(self.modulation_order ** 0.5)
        d_min_sq = 4 / (2 * (side * side - 1) / 3)  # odd-integer lattice at unit average power
        return 2 / d_min_sq

    @property
    def k_sem(self) -> int:
        return self.grid**2 * self.c_sem // 2

    @property
    def k_det(self) -> int:
        return self.grid**2 * self.c_det // 2

    @property
    def k(self) -> int:
        """Complex channel symbols per image."""
        if self.model == "cnn":
            return (self.image_size // 4) ** 2 * self.cnn_c_out // 2
        return self.k_sem + self.k_det

    @property
    def source_dim(self) -> int:
        return self.image_size**2 * 3

    @property
    def bandwidth_ratio(self) -> float:
        return self.k / self.source_dim

    @property
    def arm(self) -> str:
        if self.snr_mode == "fixed":
            return f"spec{self.snr_train_fixed:g}"
        return f"{'told' if self.snr_mode == 'snr' else 'blind'}{self.snr_train_min:g}to{self.snr_train_max:g}"

    @property
    def run_name(self) -> str:
        """e.g. dit-S_cbr48_spec0_s0, dit-S-v_cbr48_blind-5to5_s0 (velocity target), cnn_cbr48_spec0_s0,
        dit-S-v_cbr48_qam16_spec0_s0 (16-QAM symbols)."""
        family = f"dit-{self.dit_size}" if self.model == "dit" else "cnn"
        if self.model == "dit" and self.prediction != "eps":
            family += f"-{self.prediction}"
        parts = [family, f"cbr{round(1 / self.bandwidth_ratio)}"]
        if self.digital:
            parts.append(f"qam{self.modulation_order}")
        parts.append(self.arm)
        if self.channel != "awgn":
            parts.append(self.channel)
        if self.tag:
            parts.append(self.tag)
        parts.append(f"s{self.seed}")
        return "_".join(parts)

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.run_name

    @property
    def cache_dir(self) -> Path:
        dino = "fake" if self.fake_frozen else self.dino_model.replace("dinov2_", "")
        return Path(self.data_root) / "gen_cache" / f"{self.dataset}_{self.image_size}_{dino}"

    DERIVED_KEYS = ("arm", "k", "k_sem", "k_det", "bandwidth_ratio", "run_name")

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d.update(
            arm=self.arm, k=self.k, k_sem=self.k_sem, k_det=self.k_det,
            bandwidth_ratio=self.bandwidth_ratio, run_name=self.run_name,
        )
        return d

    def replace(self, **kw) -> "GenConfig":
        return dataclasses.replace(self, **kw)

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> "GenConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for key in cls.DERIVED_KEYS:
            data.pop(key, None)
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=True)
