"""Ideal separation baseline: an image codec at the AWGN capacity of the same channel uses.

Budget per image = k * log2(1 + SNR) bits, k = 1024 complex symbols (CBR 1/48 at 128 px): an
error-free, capacity-achieving channel code with infinite block length, an upper bound on any
real separate design at this bandwidth. The sender picks, per image, the downscale and quality
with the highest PSNR whose payload fits (container overhead not counted: 290 bytes for AVIF,
40 for WebP, measured on a flat 1x1 image); the receiver upsamples bicubically to 128 px. If
nothing fits, the receiver shows the test set's mean colour.
usage: separation.py OUT.json N codec[,codec]
"""
import io, sys, json, math
import numpy as np, torch
from PIL import Image
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(ROOT))
from semcom.gen.config import GenConfig
from semcom.gen.frozen import load_frozen
from semcom.gen.metrics import Metrics
torch.set_num_threads(8)
out, n, codecs = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(",")
cfg = GenConfig.from_yaml(ROOT / "results/gen/p128/dit-S-v_cbr48_told-5to5_s0/config.yaml")
imgs = np.asarray(np.load(cfg.cache_dir / "test_images.npy", mmap_mode="r")[:n])
OVERHEAD = {"webp": 40, "avif": 290}
SCALES = [128, 96, 64, 48, 32, 24, 16, 12, 8]
QUALS = list(range(0, 101, 5))
budgets = {snr: cfg.k * math.log2(1 + 10 ** (snr / 10)) / 8 for snr in (-5, 0, 5)}

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
        per = {}
        for a in range(0, n, 50):
            for k, v in metrics.per_image(x[a:a + 50], xh[a:a + 50], cfg.grid).items():
                per.setdefault(k, []).append(v)
        r = {k: float(torch.cat(v).mean()) for k, v in per.items()}
        r.update(budget_bytes=budget, fit_fraction=fits / n,
                 median_scale=float(np.median(scales)) if scales else None)
        res[f"{codec}_{snr}"] = r
        np.save(out.replace(".json", f"_{codec}_{snr}_psnr.npy"), torch.cat(per["psnr"]).numpy())
        print(codec, snr, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
json.dump(res, open(out, "w"), indent=2)
