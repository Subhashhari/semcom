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

    # --- arm selection (the 2x2)
    snr_adaptive: bool = False
    digital: bool = False
    modulation_order: int = 16

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
    batch_size: int = 128
    epochs: int = 1280
    patience: int = 50  # early-stopping patience, in epochs
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

    @property
    def k(self) -> int:
        return (self.image_size // 4) ** 2 * self.c_out // 2

    @property
    def bandwidth_ratio(self) -> float:
        return self.k / (self.image_size**2 * self.in_channels)

    @property
    def arm(self) -> str:
        from .models import arm_name

        return arm_name(self.snr_adaptive, self.digital)

    @property
    def run_name(self) -> str:
        """Filesystem- and wandb-safe identifier that encodes the arm and its settings."""
        parts = [self.arm.lower().replace("-", ""), f"r{self.bandwidth_ratio:.4f}"]
        if self.digital:
            parts.append(f"m{self.modulation_order}")
        if self.snr_train_fixed is not None:
            parts.append(f"snr{self.snr_train_fixed:g}")
        else:
            parts.append(f"snr{self.snr_train_min:g}-{self.snr_train_max:g}")
        if self.channel != "awgn":
            parts.append(self.channel)
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
