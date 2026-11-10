"""Ideal separation baseline: an image codec at the AWGN capacity of the same channel uses.

Budget per image = k * log2(1 + SNR) bits, k = 1024 complex symbols (CBR 1/48 at 128 px): an
error-free, capacity-achieving channel code with infinite block length, an upper bound on any
real separate design at this bandwidth. The sender picks, per image, the downscale and quality
with the highest PSNR whose payload fits (container overhead not counted: 290 bytes for AVIF,
40 for WebP, measured on a flat 1x1 image); the receiver upsamples bicubically to 128 px. If
nothing fits, the receiver shows the test set's mean colour.
usage: separation.py OUT.json N codec[,codec] [first|spread] [capacity|fbl] [SNR,SNR,...]
"spread" takes N evenly spaced test images (every class), as semcom.gen.evaluate --subset spread;
it needs the whole test split cached (cache_test.py all). Per-image metrics, including the
retrieval rank (semcom.gen.metrics.retrieval_rank), go to OUT_<codec>_<snr>.npz.
"""
import io, sys, json, math
import numpy as np, torch
from PIL import Image
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(ROOT))
from semcom.gen.config import GenConfig
from semcom.gen.frozen import load_frozen
from semcom.gen.metrics import Metrics, retrieval_rank
torch.set_num_threads(8)
out, n, codecs = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(",")
cfg = GenConfig.from_yaml(ROOT / "results/gen/p128/dit-S-v_cbr48_told-5to5_s0/config.yaml")
subset = sys.argv[4] if len(sys.argv) > 4 else "first"
all_imgs = np.load(cfg.cache_dir / "test_images.npy", mmap_mode="r")
idx = (np.arange(n) if subset == "first"
       else np.linspace(0, 3425 - 1, n).round().astype(int))  # 3425 = test split size
if idx.max() >= len(all_imgs):
    raise SystemExit(f"cache holds {len(all_imgs)} test images; run cache_test.py all for 'spread'")
imgs = np.asarray(all_imgs[idx])
OVERHEAD = {"webp": 40, "avif": 290}
SCALES = [128, 96, 64, 48, 32, 24, 16, 12, 8]
QUALS = list(range(0, 101, 5))
def rate_bits(snr_db: float, n: int, mode: str) -> float:
    """Bits per image over n complex AWGN uses. "capacity": n*C (infinite block length).
    "fbl": the normal approximation at block error 1e-3 (Polyanskiy-Poor-Verdu 2010),
    n*C - sqrt(n*V)*Qinv(1e-3) + log2(n)/2, with V the complex AWGN dispersion."""
    p = 10 ** (snr_db / 10)
    c = math.log2(1 + p)
    if mode == "capacity":
        return n * c
    v = p * (p + 2) / (p + 1) ** 2 * math.log2(math.e) ** 2
    return n * c - math.sqrt(n * v) * 3.090232 + math.log2(n) / 2


budget_mode = sys.argv[5] if len(sys.argv) > 5 else "capacity"
snr_list = [float(v) for v in sys.argv[6].split(",")] if len(sys.argv) > 6 else [-5, 0, 5]
budgets = {snr: max(rate_bits(snr, cfg.k, budget_mode), 0) / 8 for snr in snr_list}
print("budget bytes:", {s: round(b, 1) for s, b in budgets.items()}, budget_mode, flush=True)

def psnr(a, b):
    return 10 * np.log10(255 ** 2 / max(np.mean((a.astype(float) - b.astype(float)) ** 2), 1e-9))

def options(img, codec):
    im = Image.fromarray(img.transpose(1, 2, 0)); outs = []
    for s in SCALES:
        small = im.resize((s, s), Image.BICUBIC) if s != 128 else im
        for q in QUALS:
            buf = io.BytesIO()
            if codec == "webp": small.save(buf, "WEBP", quality=q, method=6)
            else: small.save(buf, "AVIF", quality=q, speed=6)
            size = len(buf.getvalue()) - OVERHEAD[codec]
            if size > max(budgets.values()):
                break  # larger quality only grows
            rec = Image.open(io.BytesIO(buf.getvalue())).convert("RGB").resize((128, 128), Image.BICUBIC)
            rec = np.asarray(rec).transpose(2, 0, 1)
            outs.append((size, psnr(img, rec), rec, s, q))
    return outs

mean = imgs.reshape(n, 3, -1).mean((0, 2)).round().astype(np.uint8)
grey = np.broadcast_to(mean[:, None, None], imgs.shape[1:]).copy()
vae, dino = load_frozen(cfg, torch.device("cpu"))
metrics = Metrics(dino, torch.device("cpu"), fid=False, lpips_net=True)
x = torch.from_numpy(imgs).float() / 255
res = {}
for codec in codecs:
    opts = [options(imgs[i], codec) for i in range(n)]
    for snr, budget in budgets.items():
        recs, fits, scales = [], 0, []
        for o in opts:
            ok = [t for t in o if t[0] <= budget]
            if ok:
                best = max(ok, key=lambda t: t[1]); fits += 1; recs.append(best[2]); scales.append(best[3])
            else:
                recs.append(grey)
        xh = torch.from_numpy(np.stack(recs)).float() / 255
        per, embs = {}, {"_cls": [], "_cls_hat": []}
        for a in range(0, n, 50):
            for k, v in metrics.per_image(x[a:a + 50], xh[a:a + 50], cfg.grid, embeddings=True).items():
                (embs[k] if k in embs else per.setdefault(k, [])).append(v)
        per["retrieval_rank"] = [retrieval_rank(torch.cat(embs["_cls"]), torch.cat(embs["_cls_hat"]))]
        r = {k: float(torch.cat(v).float().mean()) for k, v in per.items()}
        ranks = torch.cat(per["retrieval_rank"])
        r.update(retrieval_top1=float((ranks < 1).float().mean()), retrieval_top5=float((ranks < 5).float().mean()))
        r.update(budget_bytes=budget, fit_fraction=fits / n,
                 median_scale=float(np.median(scales)) if scales else None)
        res[f"{codec}_{snr:g}"] = r
        np.savez_compressed(out.replace(".json", f"_{codec}_{snr:g}.npz"),
                            **{k: torch.cat(v).numpy() for k, v in per.items()}, index=idx)
        print(codec, snr, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
json.dump(res, open(out, "w"), indent=2)
