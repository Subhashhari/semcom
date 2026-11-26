"""E-A2 (month plan, plan of 26 Nov): is the diffusion receiver a posterior sampler?

For an exact sampler of p(x | received symbols), K independent draws x_1..x_K satisfy
    E||x_1 - x||^2 = 2 MMSE,  E||mean_k - x||^2 = (1 + 1/k) MMSE,  E[spread_K] = MMSE,
where spread_K is the unbiased per-pixel variance across the K draws (Blau & Michaeli 2019,
Thm 2; Yan et al. 2021). So the PSNR of the mean of k draws rises by 10 log10(2k / (k + 1)) dB
over one draw, and spread / error-of-the-mean -> 1. A sampler that is too narrow (overconfident)
gains less and has spread / error < 1.

On the N class-spread test images (same indexing, channel seeds and draw-0 sampling seed as the
evaluation, so draw 0 is the scored reconstruction), per SNR and DDIM eta, K draws with the same
channel noise. Saves OUT.npz:
    mse_<eta>_<snr>      (K, N)  per-image MSE of the mean of the first k draws, k = 1..K
    spread_<eta>_<snr>   (N,)    per-image mean unbiased variance across the K draws
    draw0_<eta>_<snr>, mean_<eta>_<snr>  uint8 (len(kept), 3, H, W)  draw 0 and the K-draw mean
                                 of every KEEP-th image, for perceptual metrics elsewhere
and prints a summary (PSNR of one draw vs the k-draw means, the predicted gain, spread ratio).

usage: posterior.py RUN_DIR OUT.npz [--n 1000] [--k 8] [--snrs -5,0,5] [--etas 0,1] [--keep 5]
                    [--steps 10]
"""
import sys
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R))
from semcom.gen.evaluate import load_run, seed_of  # noqa: E402
from semcom.gen.frozen import load_frozen  # noqa: E402

argv = sys.argv[1:]


def opt(flag, default):
    if flag in argv:
        i = argv.index(flag)
        v = argv[i + 1]
        del argv[i:i + 2]
        return v
    return default


n, K = int(opt("--n", "1000")), int(opt("--k", "8"))
snrs = [float(v) for v in opt("--snrs", "-5,0,5").split(",")]
etas = [float(v) for v in opt("--etas", "0,1").split(",")]
keep, steps = int(opt("--keep", "5")), int(opt("--steps", "10"))
run, out = R / argv[0], Path(argv[1])
if K < 2:
    raise SystemExit("--k must be >= 2 (the spread needs two draws)")

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, cfg = load_run(run, dev)
vae, _ = load_frozen(cfg, dev)
images = np.load(cfg.cache_dir / "test_images.npy", mmap_mode="r")
lat_all = np.load(cfg.cache_dir / "test_latents.npy", mmap_mode="r")
dino_all = np.load(cfg.cache_dir / "test_dino.npy", mmap_mode="r")
idx = np.linspace(0, images.shape[0] - 1, n).round().astype(int)  # = evaluate --subset spread
kept = np.arange(0, n, keep)
eb = cfg.eval_batch


def batch_of(sl):
    ii = idx[sl]
    return {"image": torch.from_numpy(np.asarray(images[ii])).float().div(255).to(dev),
            "latent": torch.from_numpy(np.asarray(lat_all[ii])).float().to(dev),
            "dino": torch.from_numpy(np.asarray(dino_all[ii])).float().to(dev)}


if hasattr(model, "es_ref"):  # as evaluate and dump.py: the energy actually sent on these images
    with torch.no_grad():
        z = torch.cat([model.channel_input(batch_of(slice(a, a + 50))).float().cpu() for a in range(0, n, 50)])
    model.es_ref.fill_(z.pow(2).sum(-1).mean().item())


def to8(t):
    return (t.clamp(0, 1) * 255).round().byte().cpu().numpy()


res, lines = {}, []
for eta in etas:
    for s in snrs:
        mse = np.zeros((K, n), np.float64)
        spread = np.zeros(n, np.float64)
        d0, mn = [], []
        for b in range((n + eb - 1) // eb):
            sl = slice(b * eb, min(n, (b + 1) * eb))
            batch, m = batch_of(sl), sl.stop - sl.start
            snr_t = torch.full((m,), s, device=dev)
            x = batch["image"]
            total = torch.zeros_like(x)
            sq = torch.zeros_like(x)
            for j in range(K):
                chg = torch.Generator(device=dev).manual_seed(seed_of("channel", s, b))  # same channel noise
                key = ("sample", s, steps, b) if j == 0 else ("draw", s, b, j)
                sg = torch.Generator(device=dev).manual_seed(seed_of(*key))
                with torch.no_grad():
                    xj = model.reconstruct(batch, snr_t, vae, steps, chg, sg, eta=eta).float()
                if j == 0:
                    d0.append(to8(xj[[p - sl.start for p in kept if sl.start <= p < sl.stop]]))
                total += xj
                sq += xj * xj
                mse[j, sl] = ((total / (j + 1) - x) ** 2).mean(dim=(1, 2, 3)).double().cpu().numpy()
            mean = total / K
            spread[sl] = ((sq - K * mean * mean) / (K - 1)).mean(dim=(1, 2, 3)).double().cpu().numpy()
            mn.append(to8(mean[[p - sl.start for p in kept if sl.start <= p < sl.stop]]))
        tag = f"{eta:g}_{s:g}"
        res[f"mse_{tag}"], res[f"spread_{tag}"] = mse, spread
        res[f"draw0_{tag}"], res[f"mean_{tag}"] = np.concatenate(d0), np.concatenate(mn)
        psnr = lambda e: 10 * np.log10(1 / e).mean()  # noqa: E731 (mean per-image PSNR)
        ks = [k for k in (1, 2, 4, 8, 16, 32) if k <= K]
        gains = " ".join(f"k{k} {psnr(mse[k - 1]) - psnr(mse[0]):+.2f} (ideal {10 * np.log10(2 * k / (k + 1)):+.2f})"
                         for k in ks[1:])
        # error of the K-draw mean = (1 + 1/K) MMSE for an exact sampler; spread estimates MMSE
        ratio = spread.mean() / (mse[K - 1].mean() / (1 + 1 / K))
        lines.append(f"eta {eta:g}, {s:+g} dB: one draw {psnr(mse[0]):.2f} dB; mean gains {gains}; "
                     f"spread / MMSE-estimate {ratio:.2f} (1 for an exact posterior sampler)")
        print(lines[-1], flush=True)

np.savez_compressed(out, index=idx, kept=kept, K=K, steps=steps, **res)
print("wrote", out)
print("\n".join(lines))
