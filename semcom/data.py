"""CIFAR-10 loaders and image-quality metrics.

No normalisation beyond ToTensor: the decoder ends in a sigmoid, so the model operates on
[0, 1] images directly and PSNR is computed against the same range.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def cifar10_loaders(
    root: str = "data",
    batch_size: int = 128,
    num_workers: int = 4,
    val_fraction: float = 0.1,
    seed: int = 0,
    download: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train, val, test) loaders.

    The validation split is carved out of the 50k training set so the 10k test set stays
    untouched until final evaluation.
    """
    to_tensor = transforms.ToTensor()
    train_tf = transforms.Compose([transforms.RandomHorizontalFlip(), to_tensor])

    train_full = datasets.CIFAR10(root, train=True, download=download, transform=train_tf)
    val_full = datasets.CIFAR10(root, train=True, download=download, transform=to_tensor)
    test_set = datasets.CIFAR10(root, train=False, download=download, transform=to_tensor)

    n_val = int(len(train_full) * val_fraction)
    n_train = len(train_full) - n_val
    gen = torch.Generator().manual_seed(seed)
    train_idx, val_idx = random_split(range(len(train_full)), [n_train, n_val], generator=gen)

    train_set = torch.utils.data.Subset(train_full, list(train_idx))
    val_set = torch.utils.data.Subset(val_full, list(val_idx))

    common = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, **common),
        DataLoader(val_set, batch_size=batch_size, shuffle=False, **common),
        DataLoader(test_set, batch_size=batch_size, shuffle=False, **common),
    )


def psnr(x: torch.Tensor, x_hat: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Per-image PSNR in dB for images in [0, max_val].

    Averaged over images by the caller, not here: averaging PSNR across a batch is not the
    same as the PSNR of the batch's mean MSE, and the literature reports the former.
    """
    mse = (x - x_hat).pow(2).flatten(1).mean(dim=1).clamp_min(1e-12)
    return 10.0 * torch.log10(max_val**2 / mse)


def sample_snr(
    batch_size: int,
    cfg,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw the training SNR for a batch.

    Fixed-SNR specialists get a constant; adaptive arms get a fresh draw from U[min, max]
    *per example*, so the network never sees a fixed channel and cannot collapse onto a
    single operating point.
    """
    if cfg.snr_train_fixed is not None:
        return torch.full((batch_size,), float(cfg.snr_train_fixed), device=device)
    span = cfg.snr_train_max - cfg.snr_train_min
    u = torch.rand(batch_size, device=device, generator=generator)
    return u * span + cfg.snr_train_min
