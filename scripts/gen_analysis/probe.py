"""Eval-only probes on saved DiT-JSCC weights (CPU). Channel SNR and the SNR slot value are set
separately, so a told receiver can be given a wrong SNR. Seeds follow semcom.gen.evaluate, so all
local probes are paired with each other (not with the Kaggle GPU evals: CPU and CUDA generators
differ).

usage: probe.py OUT.npz N RUN_DIR [RUN_DIR...] --cfg ch:slot:steps [ch:slot:steps ...]
       slot "t" means the true channel SNR.
"""
import sys, time, argparse
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader, Subset
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(ROOT))
from semcom.gen.evaluate import load_run, seed_of
from semcom.gen.data import CachedSplit
from semcom.gen.frozen import load_frozen
from semcom.gen.metrics import Metrics
from semcom.gen.checks import symbol_stats

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("n", type=int); ap.add_argument("runs", nargs="+")
ap.add_argument("--cfg", nargs="+", required=True)
ap.add_argument("--batch", type=int, default=50)
a = ap.parse_args()
torch.set_num_threads(16)
dev = torch.device("cpu")
store = dict(np.load(a.out)) if Path(a.out).exists() else {}
vae = dino = metrics = None
for rd in a.runs:
    model, cfg = load_run(Path(rd), dev)
    if vae is None:
        vae, dino = load_frozen(cfg, dev)
        metrics = Metrics(dino, dev, fid=False, lpips_net=True)
    test = Subset(CachedSplit(cfg.cache_dir, "test"), range(a.n))
    loader = DataLoader(test, batch_size=a.batch, shuffle=False, num_workers=0)
    if getattr(model, "quantiser", None) is not None and hasattr(model, "es_ref"):
        full = DataLoader(Subset(CachedSplit(cfg.cache_dir, "test"), range(1000)), batch_size=50)
        p = np.mean([symbol_stats(model, b)["qam_power"] for b in full]); model.es_ref.fill_(float(p))
    name = Path(rd).name
    for c in a.cfg:
        ch, slot, steps = c.split(":"); ch = float(ch); slot = ch if slot == "t" else float(slot); steps = int(steps)
        key = f"{name}|{ch:g}|{slot:g}|{steps}"
        if key + "|psnr" in store:
            continue
        t0 = time.time(); per = {}
        with torch.no_grad():
            for b, batch in enumerate(loader):
                x = batch["image"]; B = x.shape[0]
                ch_g = torch.Generator().manual_seed(seed_of("channel", ch, b))
                sg = torch.Generator().manual_seed(seed_of("sample", ch, steps, b))
                y_s, y_d = model.received(batch, torch.full((B,), ch), ch_g)
                z = model.sample_latent(y_s, y_d, torch.full((B,), slot), steps, sg, 0.0, cfg.cfg_scale)
                xh = vae.decode(z.float()).float()
                for k, v in metrics.per_image(x, xh, cfg.grid).items():
                    per.setdefault(k, []).append(v.cpu())
        for k, v in per.items():
            store[f"{key}|{k}"] = torch.cat(v).numpy()
        print(f"{key}: " + " ".join(f"{k}={store[f'{key}|{k}'].mean():.4f}" for k in per) + f"  ({time.time()-t0:.0f}s)", flush=True)
        np.savez(a.out, **store)
