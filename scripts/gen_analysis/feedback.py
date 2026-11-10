"""SNR-aware transmitter with a wrong SNR: the transmitter is told SNR_TX (e.g. stale or quantised
feedback) while the channel is at SNR_CH; the receiver is whatever the run has (blind for the
txsnr runs). Scored on N class-spread test images, CPU, seeds as semcom.gen.evaluate (paired with
other CPU probes only). Also scores comparison runs (e.g. the plain blind run) at SNR_CH.

usage: feedback.py OUT.json N STEPS RUN_DIR CH:TX[,CH:TX...] [COMPARE_RUN_DIR ...]"""
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
from semcom.gen.model import channel, from_symbols  # noqa: E402

out, n, steps, run = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
pairs = [tuple(float(v) for v in p.split(":")) for p in sys.argv[5].split(",")]
compare = [Path(p) for p in sys.argv[6:]]
torch.set_num_threads(16)
dev = torch.device("cpu")
c = None
res = json.load(open(out)) if Path(out).exists() else {}
vae = dino = metrics = None


def reconstruct(model, cfg, batch, snr_ch, snr_tx, ch_gen, sg_gen):
    """GenJSCC.reconstruct with the transmitter and the channel at different SNRs."""
    b = batch["image"].shape[0]
    s_ch, s_tx = torch.full((b,), snr_ch), torch.full((b,), snr_tx)
    z_s, z_d = model.transmit(batch, s_tx)
    es = model.signal_energy(torch.cat([z_s, z_d], dim=1))
    y_s = from_symbols(channel(z_s, s_ch, cfg.channel, es, ch_gen), cfg.c_sem, cfg.grid)
    y_d = from_symbols(channel(z_d, s_ch, cfg.channel, es, ch_gen), cfg.c_det, cfg.grid)
    z = model.sample_latent(y_s, y_d, s_ch, steps, sg_gen, cfg_scale=cfg.cfg_scale)
    return vae.decode(z.float()).float().clamp(0, 1)


for rd, todo in [(run, pairs)] + [(p, [(ch, ch) for ch, _ in pairs]) for p in compare]:
    model, cfg = load_run(rd, dev)
    if vae is None:
        vae, dino = load_frozen(cfg, dev)
        metrics = Metrics(dino, dev, fid=False, lpips_net=True)
        cache = cfg.cache_dir
        total = np.load(cache / "test_images.npy", mmap_mode="r").shape[0]
        idx = np.linspace(0, total - 1, n).round().astype(int)
        x = torch.from_numpy(np.asarray(np.load(cache / "test_images.npy", mmap_mode="r")[idx])).float() / 255
        lat = torch.from_numpy(np.asarray(np.load(cache / "test_latents.npy", mmap_mode="r")[idx])).float()
        dn = torch.from_numpy(np.asarray(np.load(cache / "test_dino.npy", mmap_mode="r")[idx])).float()
    eb = cfg.eval_batch
    for ch, tx in dict.fromkeys(todo):
        key = f"{rd.name}|ch{ch:g}|tx{tx:g}|{steps}"
        if key in res:
            continue
        t0 = time.time()
        per, embs = {}, {"_cls": [], "_cls_hat": []}
        with torch.no_grad():
            for b in range((n + eb - 1) // eb):
                sl = slice(b * eb, min(n, (b + 1) * eb))
                batch = {"image": x[sl], "latent": lat[sl], "dino": dn[sl]}
                chg = torch.Generator().manual_seed(seed_of("channel", ch, b))
                sg = torch.Generator().manual_seed(seed_of("sample", ch, steps, b))
                xh = reconstruct(model, cfg, batch, ch, tx, chg, sg)
                for k, v in metrics.per_image(x[sl], xh, cfg.grid, embeddings=True).items():
                    (embs[k] if k in embs else per.setdefault(k, [])).append(v)
        ranks = retrieval_rank(torch.cat(embs["_cls"]), torch.cat(embs["_cls_hat"]))
        r = {k: float(torch.cat(v).float().mean()) for k, v in per.items()}
        r.update(retrieval_top1=float((ranks < 1).float().mean()), retrieval_top5=float((ranks < 5).float().mean()))
        res[key] = r
        json.dump(res, open(out, "w"), indent=1)
        print(key, {k: round(v, 3) for k, v in r.items()}, f"({time.time() - t0:.0f}s)", flush=True)
