"""Shared test setup."""

import sys
from pathlib import Path

import pytest

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
