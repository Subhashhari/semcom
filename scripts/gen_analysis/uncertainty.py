"""Can the diffusion receiver tell when it is wrong? Several reconstructions from the SAME received
symbols (same channel noise, different sampling noise); their disagreement is an uncertainty the
receiver can compute without the original. GPU job. Exploratory analysis of 23 Nov (month plan).

usage: PYTHONPATH=~/evalpkgs python scripts/gen_analysis/uncertainty.py OUT.npz N SNR[,SNR...] DIT_RUN \
           [--draws 4] [--base configs/gen/p256.yaml] [--bank results/gen/dump/p256-conv] [--lpips-dump DIR CNN_RUN]

Per SNR: the evaluation's own draw (sample seed as semcom.gen.evaluate, so it is the image scored in
eval-spread*.json) plus DRAWS extra draws; for every draw the unit DINOv2 CLS and CLIP ViT-L/14
embeddings and the DINOv2 k-NN class (as dump.py). The originals' embeddings and labels too.
--lpips-dump: also LPIPS-VGG (the CNN's own fine-tuning loss) of the DiT and the CNN reconstructions
in a full dump (dump.py --keep 1), to check the DiT's LPIPS-Alex lead on the CNN's training metric.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R))
from semcom.gen.config import GenConfig  # noqa: E402
from semcom.gen.data import _split_paths  # noqa: E402
from semcom.gen.evaluate import load_run, seed_of  # noqa: E402
from semcom.gen.frozen import load_frozen  # noqa: E402

argv = sys.argv[1:]


def opt(name, default, nargs=1):
    if name in argv:
        i = argv.index(name)
        v = argv[i + 1:i + 1 + nargs]
        del argv[i:i + 1 + nargs]
        return v if nargs > 1 else v[0]
    return default


K = int(opt("--draws", "4"))
base_cfg = opt("--base", "configs/gen/p256.yaml")
bank_dir = Path(opt("--bank", "results/gen/dump/p256-conv"))
lp = opt("--lpips-dump", None, nargs=2)
out, n, snrs, run = Path(argv[0]), int(argv[1]), [float(s) for s in argv[2].split(",")], argv[3]
STEPS, K_NN = 10, 10
dev = torch.device("cuda")
base = GenConfig.from_yaml(R / base_cfg)
vae, dino = load_frozen(base, dev)
paths = _split_paths(base)
classes = sorted({p.parent.name for p in paths["train"]})
test_labels = np.array([classes.index(p.parent.name) for p in paths["test"]])
images = np.load(base.cache_dir / "test_images.npy", mmap_mode="r")
idx = np.linspace(0, images.shape[0] - 1, n).round().astype(int)  # = evaluate --subset spread
labels = test_labels[idx]
bank, bank_y = torch.load(next(bank_dir.glob("train_bank_*.pt")), map_location="cpu")

import open_clip  # noqa: E402

clip, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", device=dev)
clip.eval()
cm = torch.tensor(open_clip.OPENAI_DATASET_MEAN, device=dev).view(1, 3, 1, 1)
cs = torch.tensor(open_clip.OPENAI_DATASET_STD, device=dev).view(1, 3, 1, 1)


@torch.no_grad()
def embed(x):  # x on dev, [0, 1]
    _, cls = dino(x, base.grid)
    xc = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
    with torch.autocast("cuda", dtype=torch.float16):
        ce = clip.encode_image((xc - cm) / cs).float()
    return F.normalize(cls.float(), dim=1).cpu(), F.normalize(ce, dim=1).cpu()


def classify(e):
    top, ti = (e @ bank.T).topk(K_NN, dim=1)
    return torch.zeros(len(e), len(classes)).scatter_add_(1, bank_y[ti], top).argmax(1).numpy()


x = torch.from_numpy(np.asarray(images[idx])).float() / 255
res = {"label": labels, "index": idx, "snrs": np.array(snrs), "draws": K}
od, oc = zip(*[embed(x[a:a + 50].to(dev)) for a in range(0, n, 50)])
res["orig_dino"], res["orig_clip"] = torch.cat(od).half().numpy(), torch.cat(oc).half().numpy()

model, cfg = load_run(R / run, dev)
lat_all = np.load(base.cache_dir / "test_latents.npy", mmap_mode="r")
dino_all = np.load(base.cache_dir / "test_dino.npy", mmap_mode="r")
if hasattr(model, "es_ref"):  # as evaluate and dump.py: SNR relative to the energy sent on these images
    with torch.no_grad():
        z = torch.cat([model.channel_input({
            "image": x[a:a + 50].to(dev),
            "dino": torch.from_numpy(np.asarray(dino_all[idx[a:a + 50]])).float().to(dev)}).float().cpu()
            for a in range(0, n, 50)])
    model.es_ref.fill_(z.pow(2).sum(-1).mean().item())
eb = cfg.eval_batch
for s in snrs:
    D, C = [[] for _ in range(K + 1)], [[] for _ in range(K + 1)]
    with torch.no_grad():
        for b in range((n + eb - 1) // eb):
            sl = slice(b * eb, min(n, (b + 1) * eb))
            batch = {"image": x[sl].to(dev),
                     "latent": torch.from_numpy(np.asarray(lat_all[idx[sl]])).float().to(dev),
                     "dino": torch.from_numpy(np.asarray(dino_all[idx[sl]])).float().to(dev)}
            snr_t = torch.full((sl.stop - sl.start,), s, device=dev)
            for j in range(K + 1):
                chg = torch.Generator(device=dev).manual_seed(seed_of("channel", s, b))
                seed = seed_of("sample", s, STEPS, b) if j == 0 else seed_of("draw", s, b, j - 1)
                xh = model.reconstruct(batch, snr_t, vae, STEPS, chg, torch.Generator(device=dev).manual_seed(seed),
                                       cfg_scale=1.0).float().clamp(0, 1)
                d, c = embed(xh)
                D[j].append(d)
                C[j].append(c)
    dd = torch.stack([torch.cat(v) for v in D])  # (K+1, N, 768)
    cc = torch.stack([torch.cat(v) for v in C])
    res[f"dino_{s:g}"], res[f"clip_{s:g}"] = dd.half().numpy(), cc.half().numpy()
    res[f"pred_{s:g}"] = np.stack([classify(dd[j]) for j in range(K + 1)])
    print(f"{s:g} dB: class kept by the evaluation draw {(res[f'pred_{s:g}'][0] == labels).mean():.3f}", flush=True)

if lp:
    import lpips
    vgg = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
    ldir, cnn_run = Path(lp[0]), lp[1]
    orig = np.load(ldir / "_original.npz")["image"]
    for key, rn in (("dit", Path(run).name), ("cnn", cnn_run)):
        for s in snrs:
            rec = np.load(ldir / f"{rn}__{s:g}.npz")["recon"]
            v = []
            with torch.no_grad():
                for a in range(0, len(rec), 32):
                    xo = torch.from_numpy(orig[a:a + 32]).float().div(255).to(dev) * 2 - 1
                    xr = torch.from_numpy(rec[a:a + 32]).float().div(255).to(dev) * 2 - 1
                    v.append(vgg(xr, xo).flatten().cpu())
            res[f"lpips_vgg_{key}_{s:g}"] = torch.cat(v).numpy()
            print(f"LPIPS-VGG {key} {s:g} dB: {res[f'lpips_vgg_{key}_{s:g}'].mean():.4f}", flush=True)
np.savez_compressed(out, **res)
print("wrote", out)
