"""Experiment configuration."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    # --- experiment identity
    name: str = "adjscc-q"
    seed: int = 0
    out_dir: str = "results"

    # --- architecture
    c_out: int = 8  # sets the rate: k = (image_size/4)^2 * c_out / 2
    hidden: int = 256
    image_size: int = 32
    in_channels: int = 3

    # Per-side widths for the capacity sweeps; None means `hidden`.
    hidden_enc: int | None = None
    hidden_dec: int | None = None

    # --- arm selection (the 2x2)
    snr_adaptive: bool = False
    digital: bool = False
    modulation_order: int = 16

    # --- blind-JSCC arms (see semcom/models.py). Defaults follow snr_adaptive, so legacy
    # configs are unchanged. B: encoder_snr=false, decoder_input=snr; C-att: blank; C: none.
    encoder_snr: bool | None = None
    decoder_input: str = "auto"

    # --- channel
    channel: str = "awgn"
    avg_power: float = 1.0

    # --- SNR
    snr_train_min: float = 0.0  # adaptive arms sample U[min, max] per example
    snr_train_max: float = 20.0
    snr_train_fixed: float | None = None  # fixed-SNR specialists set this instead

    # --- quantiser
    sigma_q_init: float = 5.0
    anneal_period: int = 10_000
    kl_weight: float = 0.05  # lambda; DeepJSCC-Q uses 0 for M >= 4096

    # --- optimisation
    lr: float = 1e-4
    lr_schedule: str = "constant"  # "constant" | "cosine" (cosine decays over `epochs`)
    lr_min_factor: float = 0.01  # cosine floor, as a fraction of lr
    batch_size: int = 128
    epochs: int = 1280
    # Early-stopping patience in epochs; None disables it. The blind study trains every arm
    # in a comparison for the same shared epoch budget, so it sets None.
    patience: int | None = 50
    num_workers: int = 4
    amp: bool = True

    # --- evaluation
    eval_snrs: list[float] = field(default_factory=lambda: [float(s) for s in range(0, 21)])
    eval_repeats: int = 10  # each test image transmitted this many times
    eval_every: int = 10  # epochs between validation passes

    # --- data
    data_root: str = "data"

    # --- logging
    wandb: bool = True
    wandb_project: str = "adjscc-q"
    wandb_entity: str | None = None
    wandb_mode: str = "online"  # "online" | "offline" | "disabled"
    log_every: int = 50  # training steps between wandb scalar logs
    # Upload best.pt, config.yaml and history.json to the wandb run when training ends, so
    # a deleted local results folder no longer loses the weights.
    wandb_save_checkpoint: bool = True

    def __post_init__(self):
        if self.c_out % 2 != 0:
            raise ValueError(f"c_out must be even, got {self.c_out}")
        if self.image_size % 4 != 0:
            raise ValueError(f"image_size must be divisible by 4, got {self.image_size}")
        if self.snr_train_fixed is None and self.snr_train_min > self.snr_train_max:
            raise ValueError("snr_train_min must not exceed snr_train_max")
        if self.digital and self.modulation_order >= 4096 and self.kl_weight != 0.0:
            # DeepJSCC-Q found a favoured subset beats uniform usage at very large M.
            raise ValueError("set kl_weight=0 for modulation_order >= 4096")
        if self.lr_schedule not in ("constant", "cosine"):
            raise ValueError(f"lr_schedule must be 'constant' or 'cosine', got {self.lr_schedule!r}")
        # Validates the arm selection; raises on contradictory or meaningless combinations.
        self.conditioning  # noqa: B018

    @property
    def conditioning(self) -> tuple[bool, str]:
        """(encoder_snr, decoder_input), with snr_adaptive folded in.

        Resolved on demand rather than written back into the fields, so `replace()` with a
        different snr_adaptive keeps working on a config built with the defaults.
        """
        from .models import resolve_conditioning

        return resolve_conditioning(
            self.snr_adaptive, self.encoder_snr, self.decoder_input, self.digital
        )

    @property
    def is_legacy_arm(self) -> bool:
        """True for the four 2x2 arms, whose run names predate the blind study."""
        enc, dec = self.conditioning
        return (enc and dec == "snr") or (not enc and dec == "none")

    @property
    def k(self) -> int:
        return (self.image_size // 4) ** 2 * self.c_out // 2

    @property
    def bandwidth_ratio(self) -> float:
        return self.k / (self.image_size**2 * self.in_channels)

    @property
    def arm(self) -> str:
        from .models import arm_label

        return arm_label(*self.conditioning, self.digital)

    @property
    def run_name(self) -> str:
        """Filesystem- and wandb-safe identifier that encodes the arm and its settings.

        Legacy 2x2 runs keep their original names at seed 0, so existing run records still
        resolve. Every other run carries its seed, since the blind study trains several
        seeds of each arm into the same results folder.
        """
        slug = self.arm.lower().replace("-", "").replace("+", "p")
        parts = [slug, f"r{self.bandwidth_ratio:.4f}"]
        if self.digital:
            parts.append(f"m{self.modulation_order}")
        if self.snr_train_fixed is not None:
            parts.append(f"snr{self.snr_train_fixed:g}")
        else:
            parts.append(f"snr{self.snr_train_min:g}-{self.snr_train_max:g}")
        if self.channel != "awgn":
            parts.append(self.channel)
        if self.hidden_enc is not None or self.hidden_dec is not None:
            parts.append(f"w{self.hidden_enc or self.hidden}x{self.hidden_dec or self.hidden}")
        if not self.is_legacy_arm or self.seed != 0:
            parts.append(f"s{self.seed}")
        return "_".join(parts)

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.run_name

    #: Computed properties written into saved configs for readability. They are not
    #: constructor arguments, so `from_yaml` strips them rather than rejecting them -
    #: without this, a config saved into a run directory could never be loaded back.
    DERIVED_KEYS = ("arm", "k", "bandwidth_ratio", "run_name")

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d.update(
            arm=self.arm, k=self.k, bandwidth_ratio=self.bandwidth_ratio, run_name=self.run_name
        )
        return d

    def replace(self, **kw) -> "Config":
        return dataclasses.replace(self, **kw)

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> "Config":
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
