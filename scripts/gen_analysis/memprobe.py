"""Peak GPU memory and speed of one DiT training step at 256 px, before committing a long run
(E-P7, plan of 26 Nov: does DiT-B with REPA fit an 8 GB card, and how fast is it?).

Random tensors of the real shapes, the training step of semcom.gen.train (fp16 autocast, grad
scaler, AdamW, clipping); prints peak allocated memory and steps per second.

usage: memprobe.py [--size B] [--batch 64] [--grad-checkpoint] [--repa 0.5] [--steps 30]
"""
import sys
import time
from pathlib import Path

import torch

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R))
from semcom.gen.config import GenConfig  # noqa: E402
from semcom.gen.model import build_model  # noqa: E402

a = sys.argv[1:]


def opt(flag, default):
    if flag in a:
        i = a.index(flag)
        v = a[i + 1]
        del a[i:i + 2]
        return v
    return default


size, batch = opt("--size", "B"), int(opt("--batch", "64"))
repa, steps = float(opt("--repa", "0.5")), int(opt("--steps", "30"))
ckpt = "--grad-checkpoint" in a
cfg = GenConfig.from_yaml(R / "configs/gen/p256.yaml", dit_size=size, digital=True, modulation_order=16,
                          snr_mode="blank", repa_weight=repa, repa_depth=4, grad_checkpoint=ckpt, prediction="v")
dev = torch.device("cuda")
model = build_model(cfg, 768).to(dev).train()
opt_ = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler("cuda")
b = {"image": torch.rand(batch, 3, 256, 256, device=dev), "latent": torch.randn(batch, 4, 32, 32, device=dev),
     "dino": torch.randn(batch, 768, 16, 16, device=dev)}
snr = torch.empty(batch, device=dev).uniform_(-5, 5)
torch.cuda.reset_peak_memory_stats()
for i in range(steps):
    if i == 5:
        torch.cuda.synchronize()
        t0 = time.time()
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model.loss(b, snr)["loss"]
    opt_.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt_)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt_)
    scaler.update()
torch.cuda.synchronize()
print(f"DiT-{size} batch {batch} REPA {repa} checkpointing {ckpt}: "
      f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f} M params, "
      f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB, {(steps - 5) / (time.time() - t0):.2f} it/s")
