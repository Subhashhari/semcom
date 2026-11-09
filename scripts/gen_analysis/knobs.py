"""Receiver-side knobs without retraining, on CPU: classifier-free guidance scale and sampling steps.
Scores PSNR, LPIPS, DINOv2 cosine and retrieval top-1/top-5 (own original among the N evaluated)
on N class-spread test images (needs the whole test split cached: cache_test.py all). Channel and
sampling seeds follow semcom.gen.evaluate (CPU generators: paired across these probes only).

usage: knobs.py OUT.json N RUN_DIR SNR[,SNR] STEPS[,STEPS] CFG[,CFG]"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(R))
from semcom.gen.evaluate import load_run, seed_of  # noqa: E402
from semcom.gen.frozen import load_frozen  # noqa: E402
from semcom.gen.metrics import Metrics, retrieval_rank  # noqa: E402

out, n, run = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
snrs = [float(v) for v in sys.argv[4].split(",")]
steps_l = [int(v) for v in sys.argv[5].split(",")]
cfgs = [float(v) for v in sys.argv[6].split(",")]
torch.set_num_threads(16)
dev = torch.device("cpu")
model, cfg = load_run(run, dev)
vae, dino = load_frozen(cfg, dev)
metrics = Metrics(dino, dev, fid=False, lpips_net=True)
c = cfg.cache_dir
total = np.load(c / "test_images.npy", mmap_mode="r").shape[0]
idx = np.linspace(0, total - 1, n).round().astype(int)
x = torch.from_numpy(np.asarray(np.load(c / "test_images.npy", mmap_mode="r")[idx])).float() / 255
lat = torch.from_numpy(np.asarray(np.load(c / "test_latents.npy", mmap_mode="r")[idx])).float()
dn = torch.from_numpy(np.asarray(np.load(c / "test_dino.npy", mmap_mode="r")[idx])).float()
res = json.load(open(out)) if Path(out).exists() else {}
eb = cfg.eval_batch
for snr, steps, g in itertools.product(snrs, steps_l, cfgs):
    key = f"{run.name}|{snr:g}|{steps}|{g:g}"
    if key in res:
        continue
    t0 = time.time()
    per, embs = {}, {"_cls": [], "_cls_hat": []}
    with torch.no_grad():
        for b in range((n + eb - 1) // eb):
            sl = slice(b * eb, min(n, (b + 1) * eb))
            batch = {"image": x[sl], "latent": lat[sl], "dino": dn[sl]}
            s = torch.full((x[sl].shape[0],), snr)
            ch = torch.Generator().manual_seed(seed_of("channel", snr, b))
            sg = torch.Generator().manual_seed(seed_of("sample", snr, steps, b))
            xh = model.reconstruct(batch, s, vae, steps=steps, channel_gen=ch, sample_gen=sg, cfg_scale=g).float().clamp(0, 1)
            for k, v in metrics.per_image(x[sl], xh, cfg.grid, embeddings=True).items():
                (embs[k] if k in embs else per.setdefault(k, [])).append(v)
    ranks = retrieval_rank(torch.cat(embs["_cls"]), torch.cat(embs["_cls_hat"]))
    r = {k: float(torch.cat(v).float().mean()) for k, v in per.items()}
    r.update(retrieval_top1=float((ranks < 1).float().mean()), retrieval_top5=float((ranks < 5).float().mean()))
    res[key] = r
    json.dump(res, open(out, "w"), indent=1)
    print(key, {k: round(v, 3) for k, v in r.items()}, f"({time.time() - t0:.0f}s)", flush=True)
