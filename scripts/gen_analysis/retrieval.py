"""Chance-normalised semantic test: for each reconstruction, rank the N original test images by
DINOv2 CLS cosine to it; report top-1 / top-5 retrieval accuracy of its own original (chance 1/N,
5/N). Same channel/sampling seeds as semcom.gen.evaluate (CPU generators).

usage: retrieval.py N STEPS RUN_DIR [RUN_DIR...]"""
import sys
from pathlib import Path

import numpy as np
import torch

R = Path(r"C:\Users\Subhash\Documents\VScodeProjects\SemCom")
sys.path.insert(0, str(R))
from semcom.gen.evaluate import load_run, seed_of  # noqa: E402
from semcom.gen.frozen import load_frozen  # noqa: E402

n, steps, runs = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
dev = torch.device("cpu")
c = R / "data/gen_cache/imagenette_128_vitb14"
x = torch.from_numpy(np.load(c / "test_images.npy", mmap_mode="r")[:n].copy()).float() / 255
lat = torch.from_numpy(np.load(c / "test_latents.npy", mmap_mode="r")[:n].copy()).float()
dn = torch.from_numpy(np.load(c / "test_dino.npy", mmap_mode="r")[:n].copy()).float()
vae = dino = None


def cls(imgs, grid):
    out = []
    for i in range(0, len(imgs), 50):
        _, c_ = dino(imgs[i:i + 50], grid)
        out.append(torch.nn.functional.normalize(c_.float(), dim=1))
    return torch.cat(out)


def report(name, ref, rec):
    sim = rec @ ref.T  # (n recon, n originals)
    rank = (sim > sim.diag()[:, None]).sum(1)
    print(f"{name}: top1 {(rank < 1).float().mean():.3f} top5 {(rank < 5).float().mean():.3f} "
          f"own-sim {sim.diag().mean():.3f} other-sim {(sim.sum() - sim.diag().sum()) / (n * n - n):.3f} "
          f"(chance top1 {1 / n:.3f} top5 {5 / n:.3f})", flush=True)


for rd in runs:
    model, cfg = load_run(Path(rd), dev)
    if vae is None:
        vae, dino = load_frozen(cfg, dev)
        ref = cls(x, cfg.grid)
        report("VAE round trip", ref, cls(torch.cat([vae.decode(lat[i:i + 50]).float() for i in range(0, n, 50)]).clamp(0, 1), cfg.grid))
    eb = cfg.eval_batch
    for snr in (-5.0, 0.0, 5.0):
        recs = []
        with torch.no_grad():
            for b in range(0, (n + eb - 1) // eb):
                sl = slice(b * eb, min(n, (b + 1) * eb))
                batch = {"image": x[sl], "latent": lat[sl], "dino": dn[sl]}
                s = torch.full((x[sl].shape[0],), snr)
                ch = torch.Generator().manual_seed(seed_of("channel", snr, b))
                st = steps if cfg.model == "dit" else None
                sg = torch.Generator().manual_seed(seed_of("sample", snr, st, b))
                recs.append(model.reconstruct(batch, s, vae, steps=st, channel_gen=ch, sample_gen=sg,
                                              cfg_scale=cfg.cfg_scale).float().clamp(0, 1))
        report(f"{cfg.run_name} {snr:g} dB", ref, cls(torch.cat(recs), cfg.grid))
