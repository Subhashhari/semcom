"""What the two encoders of a DiT run actually send: a dump for plotting (scripts/gen_analysis).

For the 200 kept class-spread test images (every 5th of the 1,000 `--subset spread` images, the
same indexing as dump.py, so kept image i is the deck's tile i), saves to OUT:

    image       uint8 (200, 3, S, S)   the originals
    dino        float16 (200, D, g, g)  the cached DINOv2 patch features the semantic encoder reads
    sem, det    float16 (200, C, g, g)  each encoder's raw output (before power normalisation)
    zs, zd      float16 (200, k, 2)     power-normalised symbols, before the QAM snap
    qs, qd      uint8 (200, k)          the constellation point each symbol is snapped to
    points      float32 (M, 2)          the constellation
    y_<snr>     float16 (len(SHOW), 4096, 2)  received symbols for the SHOW images at each SNR
    usage_sem, usage_det  int64 (M,)    point counts over all 1,000 spread images

usage: encoder_view.py RUN_DIR OUT.npz [--show 31,60,69,132] [--snrs -5,0,5]
"""
import sys
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R))
from semcom.gen.config import GenConfig  # noqa: E402
from semcom.gen.evaluate import load_run, seed_of  # noqa: E402
from semcom.gen.model import channel, to_symbols  # noqa: E402
from semcom.channel import power_normalise  # noqa: E402

argv = sys.argv[1:]


def opt(flag, default):
    if flag in argv:
        i = argv.index(flag)
        v = argv[i + 1]
        del argv[i:i + 2]
        return v
    return default


show = [int(v) for v in opt("--show", "31,60,69,132").split(",")]
snrs = [float(v) for v in opt("--snrs", "-5,0,5").split(",")]
run, out = R / argv[0], Path(argv[1])
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = GenConfig.from_yaml(run / "config.yaml")
model, cfg = load_run(run, dev)
model.eval()
images = np.load(cfg.cache_dir / "test_images.npy", mmap_mode="r")
dino_all = np.load(cfg.cache_dir / "test_dino.npy", mmap_mode="r")
print("images", images.shape, "dino", dino_all.shape, "grid", cfg.grid, flush=True)
idx = np.linspace(0, images.shape[0] - 1, 1000).round().astype(int)  # = evaluate --subset spread
kept = idx[np.arange(0, 1000, 5)]
q = model.quantiser
pts = q.points.float()


def batch_of(ii):
    return {"image": torch.from_numpy(np.asarray(images[ii])).float().div(255).to(dev),
            "dino": torch.from_numpy(np.asarray(dino_all[ii])).float().to(dev)}


@torch.no_grad()
def encode(ii):
    b = batch_of(ii)
    sem, det = model.sem_enc(b["dino"]), model.det_enc(b["image"])
    zs = power_normalise(to_symbols(sem), cfg.avg_power).float()
    zd = power_normalise(to_symbols(det), cfg.avg_power).float()
    near = lambda z: torch.cdist(z.reshape(-1, 2), pts).argmin(1).reshape(z.shape[:2])  # noqa: E731
    return sem, det, zs, zd, near(zs), near(zd)


# the energy reference, as evaluate/dump: measured on the 1,000 spread images
with torch.no_grad():
    es = torch.cat([model.channel_input(batch_of(idx[a:a + 50])).float().cpu() for a in range(0, 1000, 50)])
if hasattr(model, "es_ref"):
    model.es_ref.fill_(es.pow(2).sum(-1).mean().item())
usage_s = torch.zeros(len(pts), dtype=torch.long)
usage_d = torch.zeros(len(pts), dtype=torch.long)
for a in range(0, 1000, 50):
    *_, qs, qd = encode(idx[a:a + 50])
    usage_s += torch.bincount(qs.flatten().cpu(), minlength=len(pts))
    usage_d += torch.bincount(qd.flatten().cpu(), minlength=len(pts))

res = {k: [] for k in ("sem", "det", "zs", "zd", "qs", "qd")}
for a in range(0, len(kept), 50):
    for k, v in zip(res, encode(kept[a:a + 50])):
        res[k].append(v.cpu())
res = {k: torch.cat(v) for k, v in res.items()}
# sanity: the quantiser's own snap equals the nearest-point indices
with torch.no_grad():
    zq = model.channel_input(batch_of(kept[:8])).float()
k_s = res["zs"].shape[1]
assert torch.allclose(zq[:, :k_s].cpu(), pts.cpu()[res["qs"][:8].long()], atol=1e-5)

ys = {}
with torch.no_grad():
    sel = kept[show]
    z = model.channel_input(batch_of(sel)).float()
    e = model.signal_energy(z)
    for s in snrs:
        g = torch.Generator(device=dev).manual_seed(seed_of("channel", s, 0))
        snr_t = torch.full((len(sel),), s, device=dev)
        ys[f"y_{s:g}"] = channel(z, snr_t, cfg.channel, e, g).half().cpu().numpy()

np.savez_compressed(
    out, image=np.asarray(images[kept]), dino=np.asarray(dino_all[kept]).astype(np.float16),
    sem=res["sem"].half().numpy(), det=res["det"].half().numpy(), zs=res["zs"].half().numpy(),
    zd=res["zd"].half().numpy(), qs=res["qs"].byte().numpy(), qd=res["qd"].byte().numpy(),
    points=pts.cpu().numpy(), show=np.array(show), usage_sem=usage_s.numpy(), usage_det=usage_d.numpy(),
    es=np.float32(es.pow(2).sum(-1).mean()), **ys)
print("wrote", out, "usage sem", usage_s.tolist(), "det", usage_d.tolist())
