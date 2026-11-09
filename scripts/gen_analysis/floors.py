"""Chance floors of the eval metrics: score each test image against an unrelated test image, and
against the test set's mean image. Tells what DINOv2 similarity / LPIPS / PSNR mean in absolute terms."""
import sys
from pathlib import Path

import numpy as np
import torch

R = Path(r"C:\Users\Subhash\Documents\VScodeProjects\SemCom")
sys.path.insert(0, str(R))
from semcom.gen.config import GenConfig  # noqa: E402
from semcom.gen.frozen import load_frozen  # noqa: E402
from semcom.gen.metrics import Metrics  # noqa: E402

cfg = GenConfig.from_yaml(R / "results/gen/p128/dit-S-v_cbr48_blind-5to5_s0/config.yaml")
dev = torch.device("cpu")
vae, dino = load_frozen(cfg, dev)
m = Metrics(dino, dev, fid=False, lpips_net=True)
n = 200
x = torch.from_numpy(np.load(R / "data/gen_cache/imagenette_128_vitb14/test_images.npy", mmap_mode="r")[:n].copy()).float() / 255
out = {}
with torch.no_grad():
    for name, xh in (("unrelated image", torch.roll(x, 1, 0)),
                     ("mean image", x.mean(0, keepdim=True).expand_as(x).contiguous()),
                     ("VAE round trip", torch.cat([vae.decode(vae.encode(x[i:i + 50])).float() for i in range(0, n, 50)])
                      if hasattr(vae, "encode") else None)):
        if xh is None:
            continue
        per = {}
        for i in range(0, n, 50):
            for k, v in m.per_image(x[i:i + 50], xh[i:i + 50].clamp(0, 1), cfg.grid).items():
                per.setdefault(k, []).append(v)
        print(name, {k: round(float(torch.cat(v).mean()), 3) for k, v in per.items()}, flush=True)
