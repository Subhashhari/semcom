"""Metrics when a separate (source + channel code) design fails to decode: the receiver shows the
test set's mean colour, scored on the same N class-spread test images as the other evaluations.
This is the "cliff" of separation below its design SNR. usage: failure_floor.py OUT.json [N]"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(R))
from semcom.gen.config import GenConfig  # noqa: E402
from semcom.gen.frozen import load_frozen  # noqa: E402
from semcom.gen.metrics import Metrics, retrieval_rank  # noqa: E402

out, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1000
cfg = GenConfig.from_yaml(R / "results/gen/p128/dit-S-v_cbr48_blind-5to5_s0/config.yaml")
all_imgs = np.load(cfg.cache_dir / "test_images.npy", mmap_mode="r")
idx = np.linspace(0, len(all_imgs) - 1, n).round().astype(int)
x = torch.from_numpy(np.asarray(all_imgs[idx])).float() / 255
mean = (x.mean((0, 2, 3)) * 255).round() / 255  # mean colour, 8-bit
xh = mean[None, :, None, None].expand_as(x).contiguous()
_, dino = load_frozen(cfg, torch.device("cpu"))
m = Metrics(dino, torch.device("cpu"), fid=False, lpips_net=True)
per, embs = {}, {"_cls": [], "_cls_hat": []}
with torch.no_grad():
    for a in range(0, n, 50):
        for k, v in m.per_image(x[a:a + 50], xh[a:a + 50], cfg.grid, embeddings=True).items():
            (embs[k] if k in embs else per.setdefault(k, [])).append(v)
ranks = retrieval_rank(torch.cat(embs["_cls"]), torch.cat(embs["_cls_hat"]))
res = {k: float(torch.cat(v).float().mean()) for k, v in per.items()}
res.update(retrieval_top1=float((ranks < 1).float().mean()), retrieval_top5=float((ranks < 5).float().mean()),
           n_images=n, subset="spread", note="decoding failure: the test set's mean colour")
json.dump(res, open(out, "w"), indent=2)
print(res)
