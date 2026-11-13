"""Does the reconstruction keep the object class? Class preservation of reconstructions per SNR.

The classifier is a k-NN (k = 10, cosine) on DINOv2 CLS embeddings of TRAIN_PER_CLASS training
images per class, with the same 128 px preprocessing and DINOv2 path as the evaluation metrics.
Test labels come from the Imagenette file order (semcom.gen.data._split_paths). Each run
reconstructs N class-spread test images (the --subset spread indices) on the CPU with the
evaluation's channel and sampling seeds. Saved per (run, SNR) in OUT_DIR: the reconstructions
(uint8) and the predicted classes, for montages; the summary goes to OUT_DIR/summary.json.

Separate designs (AVIF + finite-blocklength code, as separation.py): "sep0" is fixed at 0 dB (it
fails below 0 dB: the receiver shows the mean colour), "sep" is re-designed for every SNR.

usage: semantic.py OUT_DIR N SNR[,SNR...] SPEC [SPEC ...]
  SPEC: RUN_DIR, RUN_DIR@G (classifier-free guidance G at the receiver), sep, or sep0"""
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

R = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(R))
from semcom.gen.config import GenConfig  # noqa: E402
from semcom.gen.data import _split_paths, load_image  # noqa: E402
from semcom.gen.evaluate import load_run, seed_of  # noqa: E402
from semcom.gen.frozen import load_frozen  # noqa: E402
from semcom.gen.metrics import Metrics  # noqa: E402

TRAIN_PER_CLASS, K_NN, STEPS = 100, 10, 10
out_dir, n = Path(sys.argv[1]), int(sys.argv[2])
snrs = [float(v) for v in sys.argv[3].split(",")]
specs = sys.argv[4:]
out_dir.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(16)
dev = torch.device("cpu")
base = GenConfig.from_yaml(R / "results/gen/p128/dit-S-v_cbr48_blind-5to5_s0/config.yaml")
vae, dino = load_frozen(base, dev)
metrics = Metrics(dino, dev, fid=False, lpips_net=False)
summary_path = out_dir / "summary.json"
summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}


@torch.no_grad()
def embed(x: torch.Tensor) -> torch.Tensor:
    """Unit-norm DINOv2 CLS embeddings of images in [0, 1], as Metrics.per_image."""
    out = []
    for a in range(0, len(x), 64):
        _, cls = metrics.dino(x[a:a + 64], base.grid)
        out.append(F.normalize(cls.float(), dim=1))
    return torch.cat(out)


# ---- labels and the k-NN classifier ----
paths = _split_paths(base)
classes = sorted({p.parent.name for p in paths["train"]})
test_labels = np.array([classes.index(p.parent.name) for p in paths["test"]])
total = np.load(base.cache_dir / "test_images.npy", mmap_mode="r").shape[0]
assert total == len(test_labels), (total, len(test_labels))
idx = np.linspace(0, total - 1, n).round().astype(int)  # = semcom.gen.evaluate --subset spread
labels = test_labels[idx]
bank_f = out_dir / f"train_bank_{TRAIN_PER_CLASS}.pt"
if bank_f.exists():
    bank, bank_y = torch.load(bank_f)
else:
    by_class = {c: [p for p in paths["train"] if p.parent.name == c] for c in classes}
    chosen = [(c, ps[i]) for c, ps in by_class.items()
              for i in np.linspace(0, len(ps) - 1, TRAIN_PER_CLASS).round().astype(int)]
    xt = torch.from_numpy(np.stack([load_image(p, base.image_size) for _, p in chosen])).float() / 255
    bank, bank_y = embed(xt), torch.tensor([classes.index(c) for c, _ in chosen])
    torch.save((bank, bank_y), bank_f)


def classify(e: torch.Tensor) -> np.ndarray:
    sim = e @ bank.T
    top, ti = sim.topk(K_NN, dim=1)
    votes = torch.zeros(len(e), len(classes)).scatter_add_(1, bank_y[ti], top)
    return votes.argmax(1).numpy()


x8 = np.asarray(np.load(base.cache_dir / "test_images.npy", mmap_mode="r")[idx])
x = torch.from_numpy(x8).float() / 255
orig_pred = classify(embed(x))
summary["_original"] = {"accuracy": float((orig_pred == labels).mean()), "n": n, "classes": classes}
print("classifier on the original test images:", summary["_original"]["accuracy"], flush=True)


def record(name: str, snr: float, xh8: np.ndarray, seconds: float):
    xh = torch.from_numpy(xh8).float() / 255
    pred = classify(embed(xh))
    per_class = {classes[c]: float((pred[labels == c] == c).mean()) for c in range(len(classes))}
    summary.setdefault(name, {})[f"{snr:g}"] = {
        "accuracy": float((pred == labels).mean()),
        "agrees_with_original": float((pred == orig_pred).mean()),
        "per_class": per_class, "seconds": seconds}
    np.savez_compressed(out_dir / f"{name}__{snr:g}.npz", recon=xh8, pred=pred, label=labels, index=idx)
    summary_path.write_text(json.dumps(summary, indent=1))
    print(f"{name} {snr:g} dB: class accuracy {summary[name][f'{snr:g}']['accuracy']:.3f} ({seconds:.0f}s)", flush=True)


# ---- separate designs (AVIF + finite-blocklength code), as separation.py ----
SCALES, QUALS, OVERHEAD = [128, 96, 64, 48, 32, 24, 16, 12, 8], list(range(0, 101, 5)), 290


def fbl_bytes(snr_db: float) -> float:
    p = 10 ** (snr_db / 10)
    c = math.log2(1 + p)
    v = p * (p + 2) / (p + 1) ** 2 * math.log2(math.e) ** 2
    return max(base.k * c - math.sqrt(base.k * v) * 3.090232 + math.log2(base.k) / 2, 0) / 8


def avif_options(img: np.ndarray, cap: float):
    im, outs = Image.fromarray(img.transpose(1, 2, 0)), []
    for s in SCALES:
        small = im.resize((s, s), Image.BICUBIC) if s != 128 else im
        for q in QUALS:
            buf = io.BytesIO()
            small.save(buf, "AVIF", quality=q, speed=6)
            size = len(buf.getvalue()) - OVERHEAD
            if size > cap:
                break
            rec = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB").resize((128, 128), Image.BICUBIC))
            rec = rec.transpose(2, 0, 1)
            outs.append((size, float(-np.mean((img.astype(float) - rec.astype(float)) ** 2)), rec))
    return outs


if any(s in ("sep", "sep0") for s in specs):
    t0 = time.time()
    cap = max(fbl_bytes(s) for s in snrs + [0.0])
    opts = [avif_options(im, cap) for im in x8]
    mean = x8.reshape(n, 3, -1).mean((0, 2)).round().astype(np.uint8)
    grey = np.broadcast_to(mean[:, None, None], x8.shape[1:])

    def decode(budget):
        return np.stack([max(ok, key=lambda t: t[1])[2] if (ok := [t for t in o if t[0] <= budget]) else grey
                         for o in opts])

    for s in snrs:
        if "sep" in specs:
            record("sep", s, decode(fbl_bytes(s)), time.time() - t0)
        if "sep0" in specs:  # designed for 0 dB: decodes (at the 0 dB quality) only when SNR >= 0
            record("sep0", s, decode(fbl_bytes(0.0)) if s >= 0 else np.broadcast_to(grey, x8.shape).copy(),
                   time.time() - t0)

# ---- JSCC runs ----
for spec in [s for s in specs if s not in ("sep", "sep0")]:
    run, _, g = spec.partition("@")
    guidance = float(g) if g else 1.0
    model, cfg = load_run(R / run, dev)
    name = Path(run).name + (f"@g{guidance:g}" if g else "")
    for s in snrs:
        if f"{s:g}" in summary.get(name, {}):
            continue
        t0, recs = time.time(), []
        eb = cfg.eval_batch
        with torch.no_grad():
            for b in range((n + eb - 1) // eb):
                sl = slice(b * eb, min(n, (b + 1) * eb))
                batch = {"image": x[sl],
                         "latent": torch.from_numpy(np.asarray(np.load(cfg.cache_dir / "test_latents.npy", mmap_mode="r")[idx[sl]])).float(),
                         "dino": torch.from_numpy(np.asarray(np.load(cfg.cache_dir / "test_dino.npy", mmap_mode="r")[idx[sl]])).float()}
                chg = torch.Generator().manual_seed(seed_of("channel", s, b))
                sg = torch.Generator().manual_seed(seed_of("sample", s, STEPS, b))
                snr_t = torch.full((sl.stop - sl.start,), s)
                xh = model.reconstruct(batch, snr_t, vae, STEPS, chg, sg, cfg_scale=guidance)
                recs.append((xh.float().clamp(0, 1) * 255).round().byte().numpy())
        record(name, s, np.concatenate(recs), time.time() - t0)
