"""How much does the SD-VAE round trip cost at 128 px and at 256 px? (CPU; 18 Nov question.)

The 18 Nov refiner sweep (g-s3) found that passing the CNN's 128 px output through the SD-VAE
(encode, decode) already lowers DINOv2 similarity from 0.29 to 0.23 at 0 dB. Before any 256 px
run, this measures the same two things at both resolutions, on the same test images:

  clean   the original test image, round trip at 128 px and at 256 px (the VAE's ceiling)
  cnn     a CNN reconstruction (from a dump, 128 px), round trip at 128 px, and upsampled to
          256 px then round trip there (does a native-resolution VAE keep more of what the CNN
          transmitted?)

Metrics against the original at the same resolution: PSNR, LPIPS (VGG) and DINOv2 CLS cosine;
plus DINOv2 10-NN class agreement with the original's class where a dump's train bank exists.

usage: vae_resolution.py DUMP_DIR CNN_NAME SNR [N] [--out OUT.json]"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R))
from semcom.gen.config import GenConfig  # noqa: E402
from semcom.gen.data import _split_paths, load_image  # noqa: E402
from semcom.gen.frozen import load_frozen  # noqa: E402

args = sys.argv[1:]
out_f = None
if "--out" in args:
    i = args.index("--out")
    out_f = Path(args[i + 1])
    del args[i:i + 2]
dump, cnn_name, snr = Path(args[0]), args[1], float(args[2])
n_max = int(args[3]) if len(args) > 3 else 200
torch.set_num_threads(12)
dev = torch.device("cpu")
base = GenConfig.from_yaml(R / "results/gen/p128/dit-S-v_cbr48_blind-5to5_s0/config.yaml")
vae, dino = load_frozen(base, dev)
import lpips  # noqa: E402

lp = lpips.LPIPS(net="vgg", verbose=False).eval()
meta = json.loads((dump / "dump.json").read_text())
d = np.load(dump / f"{cnn_name}__{snr:g}.npz")
kept = np.array(meta["kept"])[:n_max]
test_idx = np.array(meta["index"])[kept]
paths = _split_paths(base)["test"]
x128 = torch.from_numpy(np.stack([load_image(paths[i], 128) for i in test_idx])).float() / 255
x256 = torch.from_numpy(np.stack([load_image(paths[i], 256) for i in test_idx])).float() / 255
cached = np.load(base.cache_dir / "test_images.npy", mmap_mode="r")
assert np.abs(np.asarray(cached[test_idx[:4]]).astype(int) - (x128[:4] * 255).round().numpy().astype(int)).max() <= 1, \
    "the 128 px reload must match the cache"
c128 = torch.from_numpy(d["recon"][:len(kept)]).float() / 255
c256 = F.interpolate(c128, size=256, mode="bicubic", align_corners=False).clamp(0, 1)


@torch.no_grad()
def batched(fn, x, b=16):
    return torch.cat([fn(x[a:a + b]) for a in range(0, len(x), b)])


@torch.no_grad()
def roundtrip(x):
    return batched(lambda v: vae.decode(vae.encode(v)), x)


@torch.no_grad()
def cls(x):
    return batched(lambda v: F.normalize(dino(v, base.grid)[1].float(), dim=1), x)


def psnr(a, b):
    return 10 * torch.log10(1 / ((a - b) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-10))


bank_f = dump / "train_bank_100.pt"
bank = torch.load(bank_f, map_location="cpu") if bank_f.exists() else None


def klass(e):
    if bank is None:
        return None
    b, y = bank
    top, ti = (e @ b.T).topk(10, dim=1)
    return torch.zeros(len(e), int(y.max()) + 1).scatter_add_(1, y[ti], top).argmax(1)


rows = {}
for res, x, cnn in ((128, x128, c128), (256, x256, c256)):
    e_x = cls(x)
    k_x = klass(cls(x128))  # class of the original, judged at the classifier's 128 px
    for name, y in (("clean round trip", roundtrip(x)), ("CNN output", cnn), ("CNN output round trip", roundtrip(cnn))):
        e_y = cls(y)
        with torch.no_grad():
            lpips_v = torch.cat([lp(x[a:a + 8] * 2 - 1, y[a:a + 8] * 2 - 1).flatten() for a in range(0, len(x), 8)])
        r = {"psnr": psnr(x, y).mean().item(), "lpips": lpips_v.mean().item(),
             "dino_sim": (e_x * e_y).sum(1).mean().item()}
        if k_x is not None:
            y128 = y if res == 128 else F.interpolate(y, size=128, mode="bicubic", align_corners=False).clamp(0, 1)
            r["class_agree"] = (klass(cls(y128)) == k_x).float().mean().item()
        rows[f"{res} px, {name}"] = r
        print(f"{res} px  {name:22s} " + "  ".join(f"{k} {v:.3f}" for k, v in r.items()), flush=True)

out = {"n": len(kept), "snr": snr, "cnn": cnn_name, "rows": rows}
if out_f:
    out_f.parent.mkdir(parents=True, exist_ok=True)
    out_f.write_text(json.dumps(out, indent=1), encoding="utf-8")
