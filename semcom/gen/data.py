"""Data for the generative-JSCC prototype: download, preprocess once, cache, load.

    python -m semcom.gen.data --config configs/gen/p128.yaml        # build the cache

The VAE and DINOv2 are frozen, so their outputs for every image are computed once and stored
next to the image as memory-mapped .npy arrays:

    images      uint8   (N, 3, S, S)          the 128x128 image itself (targets, CNN input)
    latents     float16 (N, 4, S/8, S/8)      SD-VAE latent, already scaled
    dino        float16 (N, D, S/16, S/16)    DINOv2 patch features pooled to the symbol grid
    *_flip      the same for the horizontally flipped image (training split only)

Training then runs only the JSCC encoders and the DiT. Flip augmentation picks the cached
flipped copy, so augmentation costs nothing either.

Datasets: "imagenette" (fast.ai's 10-class ImageNet subset, 320 px release, downloaded on
first use), "folder" (any data_root/<name>/{train,val}/<class>/*.jpg tree), and "synthetic"
(random smooth images, for tests and offline smoke runs).
"""

from __future__ import annotations

import argparse
import json
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import GenConfig

IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPEG"}
N_VAL = 500  # images carved from the official val split for validation; the rest are test
SYNTHETIC_SIZES = {"train": 64, "val": 32}


# ----------------------------------------------------------------------------- sources


def _download_imagenette(root: Path) -> Path:
    target = root / "imagenette2-320"
    if (target / "train").is_dir():
        return target
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "imagenette2-320.tgz"
    if not archive.exists():
        from .frozen import retry

        print(f"downloading {IMAGENETTE_URL}", flush=True)
        part = archive.with_suffix(".part")  # a failed transfer never leaves a truncated archive
        retry(lambda: urllib.request.urlretrieve(IMAGENETTE_URL, part), "Imagenette download")
        part.replace(archive)
    with tarfile.open(archive) as tar:
        try:
            tar.extractall(root, filter="data")  # refuses absolute paths and links out
        except TypeError:  # Python without the extraction-filter backport
            tar.extractall(root)
    return target


def list_images(cfg: GenConfig, split: str) -> list[Path]:
    """Image files of the official split ("train" or "val"), in a fixed sorted order."""
    root = Path(cfg.data_root)
    if cfg.dataset == "imagenette":
        base = _download_imagenette(root) / split
    elif cfg.dataset == "folder":
        base = root / cfg.name / split
    else:
        raise ValueError(f"no image files for dataset {cfg.dataset!r}")
    files = sorted(p for p in base.rglob("*") if p.suffix in IMAGE_EXTS)
    if not files:
        raise FileNotFoundError(f"no images under {base}")
    return files


def load_image(path: Path, size: int) -> np.ndarray:
    """Resize the short side to `size`, centre-crop, return uint8 (3, size, size)."""
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = size / min(w, h)
        im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
        w, h = im.size
        left, top = (w - size) // 2, (h - size) // 2
        im = im.crop((left, top, left + size, top + size))
        return np.asarray(im, dtype=np.uint8).transpose(2, 0, 1).copy()


def synthetic_images(n: int, size: int, seed: int) -> np.ndarray:
    """Random smooth colour images: low-frequency noise upsampled, uint8 (n, 3, size, size)."""
    gen = torch.Generator().manual_seed(seed)
    low = torch.rand(n, 3, size // 16, size // 16, generator=gen)
    x = torch.nn.functional.interpolate(low, size=size, mode="bicubic", align_corners=False)
    return (x.clamp(0, 1) * 255).round().to(torch.uint8).numpy()


# ----------------------------------------------------------------------------- the cache


def _split_paths(cfg: GenConfig) -> dict[str, object]:
    """Source of each cached split: a list of files, or a synthetic image array."""
    if cfg.dataset == "synthetic":
        return {
            "train": synthetic_images(SYNTHETIC_SIZES["train"], cfg.image_size, 0),
            "val": synthetic_images(SYNTHETIC_SIZES["val"], cfg.image_size, 1),
            "test": synthetic_images(SYNTHETIC_SIZES["val"], cfg.image_size, 2),
        }
    official_val = list_images(cfg, "val")
    order = np.random.default_rng(0).permutation(len(official_val))
    val = [official_val[i] for i in sorted(order[:N_VAL])]
    test = [official_val[i] for i in sorted(order[N_VAL:])]
    return {"train": list_images(cfg, "train"), "val": val, "test": test}


def is_prepared(cfg: GenConfig) -> bool:
    return (cfg.cache_dir / "meta.json").exists()


def cache_meta(cfg: GenConfig) -> dict:
    """The cache's meta.json: counts, and the DINOv2 feature width the encoder must accept."""
    return json.loads((cfg.cache_dir / "meta.json").read_text(encoding="utf-8"))


@torch.no_grad()
def prepare(cfg: GenConfig, device: torch.device | None = None, batch: int = 64) -> Path:
    """Build the cache for every split. Idempotent: returns at once if it is complete."""
    from .frozen import load_frozen

    out = cfg.cache_dir
    if is_prepared(cfg):
        return out
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae, dino = load_frozen(cfg, device)
    out.mkdir(parents=True, exist_ok=True)
    s, g, lat = cfg.image_size, cfg.grid, cfg.image_size // 8
    counts = {}
    for split, source in _split_paths(cfg).items():
        n = len(source)
        flips = (False, True) if split == "train" else (False,)
        arrays = {"images": np.lib.format.open_memmap(out / f"{split}_images.npy", "w+", np.uint8, (n, 3, s, s))}
        for flip in flips:
            suffix = "_flip" if flip else ""
            arrays[f"latents{suffix}"] = np.lib.format.open_memmap(
                out / f"{split}_latents{suffix}.npy", "w+", np.float16, (n, 4, lat, lat))
            arrays[f"dino{suffix}"] = np.lib.format.open_memmap(
                out / f"{split}_dino{suffix}.npy", "w+", np.float16, (n, dino.dim, g, g))
        for start in range(0, n, batch):
            stop = min(start + batch, n)
            if isinstance(source, np.ndarray):
                imgs = source[start:stop]
            else:
                imgs = np.stack([load_image(p, s) for p in source[start:stop]])
            arrays["images"][start:stop] = imgs
            x = torch.from_numpy(imgs).to(device).float() / 255
            for flip in flips:
                xf = x.flip(-1) if flip else x
                suffix = "_flip" if flip else ""
                arrays[f"latents{suffix}"][start:stop] = vae.encode(xf).half().cpu().numpy()
                arrays[f"dino{suffix}"][start:stop] = dino(xf, g)[0].half().cpu().numpy()
            print(f"  {split}: {stop}/{n}", end="\r", flush=True)
        for a in arrays.values():
            a.flush()
        counts[split] = n
        print(f"  {split}: {n} images cached")
    meta = {"dataset": cfg.dataset, "image_size": s, "dino_model": cfg.dino_model,
            "dino_dim": dino.dim, "vae_model": cfg.vae_model, "fake": cfg.fake_frozen, "counts": counts}
    # Written last: its presence marks a complete cache.
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


class CachedSplit(Dataset):
    """One cached split. Items: dict(image [0,1] float, latent, dino, index).

    With `augment`, each item is the flipped copy with probability 1/2 (training only).
    """

    def __init__(self, cache_dir: Path, split: str, augment: bool = False):
        self.dir, self.split = Path(cache_dir), split
        self.augment = augment and (self.dir / f"{split}_latents_flip.npy").exists()
        self._arrays = None  # opened lazily, once per worker process
        self.n = np.load(self.dir / f"{split}_images.npy", mmap_mode="r").shape[0]

    def _open(self):
        if self._arrays is None:
            names = ["images", "latents", "dino"] + (["latents_flip", "dino_flip"] if self.augment else [])
            self._arrays = {k: np.load(self.dir / f"{self.split}_{k}.npy", mmap_mode="r") for k in names}
        return self._arrays

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        a = self._open()
        flip = self.augment and bool(torch.randint(2, ()).item())
        img = torch.from_numpy(np.array(a["images"][i])).float() / 255
        if flip:
            img = img.flip(-1)
        suffix = "_flip" if flip else ""
        return {
            "image": img,
            "latent": torch.from_numpy(np.array(a[f"latents{suffix}"][i])).float(),
            "dino": torch.from_numpy(np.array(a[f"dino{suffix}"][i])).float(),
            "index": i,
        }


def loaders(cfg: GenConfig) -> tuple[DataLoader, DataLoader, CachedSplit]:
    """(train loader, val loader, test split). The train order is seeded by cfg.seed."""
    common = dict(num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
                  persistent_workers=cfg.num_workers > 0)
    train = CachedSplit(cfg.cache_dir, "train", augment=True)
    if cfg.train_fraction < 1:
        # Evenly spaced indices: the train files are sorted by class, so every class keeps its share.
        keep = np.linspace(0, train.n - 1, max(1, round(train.n * cfg.train_fraction))).round().astype(int)
        train = torch.utils.data.Subset(train, np.unique(keep).tolist())
    val = CachedSplit(cfg.cache_dir, "val")
    gen = torch.Generator().manual_seed(cfg.seed)
    return (
        DataLoader(train, batch_size=cfg.batch_size, shuffle=True, drop_last=True, generator=gen, **common),
        DataLoader(val, batch_size=cfg.batch_size, shuffle=False, **common),
        CachedSplit(cfg.cache_dir, "test"),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=Path("configs/gen/p128.yaml"))
    p.add_argument("--dataset", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--fake", action="store_true", help="stand-in frozen models (offline tests)")
    args = p.parse_args()
    cfg = GenConfig.from_yaml(args.config, dataset=args.dataset, data_root=args.data_root,
                              fake_frozen=args.fake or None)
    print(f"cache: {prepare(cfg)}")


if __name__ == "__main__":
    main()
