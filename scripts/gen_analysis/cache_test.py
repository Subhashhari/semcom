"""Cache the first N test images (default 1000; "all" for the whole split) exactly as
semcom.gen.data.prepare would (same order, same frozen models), without building the train split.
No meta.json is written. usage: cache_test.py [N|all]"""
import sys, numpy as np, torch
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(ROOT))
from semcom.gen.config import GenConfig
from semcom.gen.data import _split_paths, load_image
from semcom.gen.frozen import load_frozen
torch.set_num_threads(16)
cfg = GenConfig.from_yaml(ROOT / "results/gen/p128/dit-S-v_cbr48_told-5to5_s0/config.yaml")
src = _split_paths(cfg)["test"]
src = src if len(sys.argv) > 1 and sys.argv[1] == "all" else src[:int(sys.argv[1]) if len(sys.argv) > 1 else 1000]
vae, dino = load_frozen(cfg, torch.device("cpu"))
out = cfg.cache_dir; out.mkdir(parents=True, exist_ok=True)
s, g, lat, n = cfg.image_size, cfg.grid, cfg.image_size // 8, len(src)
im = np.lib.format.open_memmap(out / "test_images.npy", "w+", np.uint8, (n, 3, s, s))
la = np.lib.format.open_memmap(out / "test_latents.npy", "w+", np.float16, (n, 4, lat, lat))
di = np.lib.format.open_memmap(out / "test_dino.npy", "w+", np.float16, (n, dino.dim, g, g))
with torch.no_grad():
    for a in range(0, n, 50):
        b = min(a + 50, n)
        imgs = np.stack([load_image(p, s) for p in src[a:b]]); im[a:b] = imgs
        x = torch.from_numpy(imgs).float() / 255
        la[a:b] = vae.encode(x).half().numpy(); di[a:b] = dino(x, g)[0].half().numpy()
        print(f"{b}/{n}", flush=True)
for arr in (im, la, di): arr.flush()
print("cache", out)
