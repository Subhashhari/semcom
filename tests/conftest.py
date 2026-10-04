"""Shared test setup."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def require_cifar(n: int, seed: int = 0):
    """Return `n` real CIFAR-10 test images, or skip if the dataset is not local.

    Deliberately passes `download=False`. A test suite should not pull 170 MB mid-run,
    and it should not silently depend on whatever happens to be on the machine either -
    the original version of these tests passed only because CIFAR-10 had been downloaded
    earlier in the same session, so a fresh clone failed with five errors.

    Skipping keeps the suite hermetic and makes the dependency visible in the output
    rather than implicit. The data appears as soon as training is run once.
    """
    from semcom.separation import cifar_test_images

    try:
        return cifar_test_images(n, seed=seed, download=False)
    except Exception:
        pytest.skip(
            "CIFAR-10 not found locally. Run any training command once (it downloads "
            "automatically), or: python -c \"from torchvision import datasets; "
            "datasets.CIFAR10('data', train=False, download=True)\""
        )


@pytest.fixture
def fake_cifar(monkeypatch):
    """Replace the CIFAR-10 loaders with a small in-memory dataset.

    Avoids a 170 MB download in CI and keeps these tests to a few seconds, while leaving
    every other part of the pipeline real.
    """
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(0)
    # Low-frequency images: compressible, so the model can actually learn something.
    yy, xx = torch.meshgrid(torch.linspace(0, 1, 32), torch.linspace(0, 1, 32), indexing="ij")
    phases = torch.arange(32).view(32, 1, 1, 1) * 0.3
    x = (torch.sin(torch.stack([xx, yy, xx * yy])[None] * 6.28 + phases) * 0.5 + 0.5).clamp(0, 1)
    ds = TensorDataset(x, torch.zeros(32, dtype=torch.long))

    def loaders(*a, **kw):
        return (
            DataLoader(ds, batch_size=8, shuffle=True, drop_last=True),
            DataLoader(ds, batch_size=8),
            DataLoader(ds, batch_size=8),
        )

    for module in ("semcom.train", "semcom.evaluate", "semcom.analyze_gates", "semcom.blind_eval"):
        monkeypatch.setattr(f"{module}.cifar10_loaders", loaders, raising=False)
    return ds
