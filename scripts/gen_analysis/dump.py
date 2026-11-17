"""Reconstructions for the analysis suite (scripts/gen_analysis/suite.py): what each system
actually outputs, not only its mean metrics. GPU job (Kaggle runner "dump", or a local GPU).

For each SPEC and SNR, on the N class-spread test images (the `--subset spread` indices) with the
evaluation's channel and sampling seeds (batches of the run's eval_batch, so the reconstructions
are the very ones scored in eval-spread*.json when N matches):

    pred, emb       DINOv2 k-NN class (as semantic.py) and the unit DINOv2 CLS embedding (float16)
                    of every reconstruction
    recon           uint8 images of every KEEP-th of them (default every 5th: 200 of 1000)
    cnn             refiner runs only: the frozen CNN's output for the same kept images, i.e.
                    what the refiner started from
    samples         DiT and refiner runs: SAMPLES more draws of the sampling noise for the first
                    DIVERSITY kept images, same channel noise (how much the receiver makes up)

Saved as OUT_DIR/<name>__<snr>.npz, plus OUT_DIR/dump.json (settings and the index lists).

usage: dump.py OUT_DIR N SNR[,SNR...] SPEC [SPEC ...] [--keep 5] [--samples 4] [--diversity 40] [--dry-run]
  SPEC: RUN_DIR (relative to the repo root) or RUN_DIR@G (classifier-free guidance G)
  ("--wandb-mode X", which the Kaggle runner appends, is ignored.)"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

R = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(R))
from semcom.gen.config import GenConfig  # noqa: E402
from semcom.gen.data import _split_paths, load_image  # noqa: E402
from semcom.gen.evaluate import load_run, seed_of  # noqa: E402
from semcom.gen.frozen import load_frozen  # noqa: E402

TRAIN_PER_CLASS, K_NN, STEPS = 100, 10, 10


def option(argv: list[str], flag: str, default: int) -> int:
    if flag in argv:
        i = argv.index(flag)
        v = int(argv[i + 1])
        del argv[i:i + 2]
        return v
    return default


argv = sys.argv[1:]
if "--wandb-mode" in argv:
    i = argv.index("--wandb-mode")
    del argv[i:i + 2]
keep = option(argv, "--keep", 5)
n_samples = option(argv, "--samples", 4)
n_div = option(argv, "--diversity", 40)
dry = "--dry-run" in argv
argv = [a for a in argv if a != "--dry-run"]
out_dir, n = Path(argv[0]), int(argv[1])
snrs = [float(v) for v in argv[2].split(",")]
specs = argv[3:]
for spec in specs:
    run = spec.partition("@")[0]
    if not dry and not (R / run / "config.yaml").exists():
        raise SystemExit(f"{run}: no config.yaml")
if dry:
    print(f"dump: {n} images at {snrs} dB, keep every {keep}, {n_samples} samples of {n_div}: {specs}")
    sys.exit(0)

out_dir.mkdir(parents=True, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(16)
base = GenConfig.from_yaml(R / "results/gen/p128/dit-S-v_cbr48_blind-5to5_s0/config.yaml")
if not (base.cache_dir / "test_images.npy").exists():
    from semcom.gen.data import prepare  # noqa: E402

    prepare(base, dev)
vae, dino = load_frozen(base, dev)


@torch.no_grad()
def embed(x: torch.Tensor) -> torch.Tensor:
    out = []
    for a in range(0, len(x), 64):
        _, cls = dino(x[a:a + 64].to(dev), base.grid)
        out.append(F.normalize(cls.float(), dim=1).cpu())
    return torch.cat(out)


# ---- labels and the k-NN classifier (as semantic.py) ----
paths = _split_paths(base)
classes = sorted({p.parent.name for p in paths["train"]})
test_labels = np.array([classes.index(p.parent.name) for p in paths["test"]])
images = np.load(base.cache_dir / "test_images.npy", mmap_mode="r")
assert images.shape[0] == len(test_labels)
idx = np.linspace(0, images.shape[0] - 1, n).round().astype(int)  # = evaluate --subset spread
labels = test_labels[idx]
kept = np.arange(0, n, keep)  # positions within idx whose images are saved
div = kept[:n_div]
bank_f = out_dir / f"train_bank_{TRAIN_PER_CLASS}.pt"
if bank_f.exists():
    bank, bank_y = torch.load(bank_f, map_location="cpu")
else:
    by_class = {c: [p for p in paths["train"] if p.parent.name == c] for c in classes}
    chosen = [(c, ps[i]) for c, ps in by_class.items()
              for i in np.linspace(0, len(ps) - 1, TRAIN_PER_CLASS).round().astype(int)]
    xt = torch.from_numpy(np.stack([load_image(p, base.image_size) for _, p in chosen])).float() / 255
    bank, bank_y = embed(xt), torch.tensor([classes.index(c) for c, _ in chosen])
    torch.save((bank, bank_y), bank_f)


def classify(e: torch.Tensor) -> np.ndarray:
    top, ti = (e @ bank.T).topk(K_NN, dim=1)
    return torch.zeros(len(e), len(classes)).scatter_add_(1, bank_y[ti], top).argmax(1).numpy()


x = torch.from_numpy(np.asarray(images[idx])).float() / 255
orig_emb = embed(x)
meta_f = out_dir / "dump.json"
meta = json.loads(meta_f.read_text()) if meta_f.exists() else {"runs": {}}
meta.update(n=n, index=idx.tolist(), kept=kept.tolist(), diversity=div.tolist(), classes=classes,
            labels=labels.tolist(), steps=STEPS, device=torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu")
np.savez_compressed(out_dir / "_original.npz", pred=classify(orig_emb), emb=orig_emb.half().numpy(),
                    label=labels, index=idx)
lat_all = np.load(base.cache_dir / "test_latents.npy", mmap_mode="r")
dino_all = np.load(base.cache_dir / "test_dino.npy", mmap_mode="r")


def to8(t: torch.Tensor) -> np.ndarray:
    return (t.float().clamp(0, 1) * 255).round().byte().cpu().numpy()


for spec in specs:
    run, _, g = spec.partition("@")
    guidance = float(g) if g else 1.0
    model, cfg = load_run(R / run, dev)
    name = Path(run).name + (f"@g{guidance:g}" if g else "")
    refiner = cfg.model == "refiner"
    generative = cfg.model != "cnn"
    if hasattr(model, "es_ref"):  # as evaluate: the SNR refers to the energy sent on these images
        with torch.no_grad():
            z = torch.cat([model.channel_input({"image": x[a:a + 50].to(dev)}).float().cpu()
                           for a in range(0, n, 50)])
        model.es_ref.fill_(z.pow(2).sum(-1).mean().item())
    for s in snrs:
        f = out_dir / f"{name}__{s:g}.npz"
        if f.exists():
            continue
        t0 = time.time()
        recs, cnns, sams = [], [], []
        eb = cfg.eval_batch
        with torch.no_grad():
            for b in range((n + eb - 1) // eb):
                sl = slice(b * eb, min(n, (b + 1) * eb))
                batch = {"image": x[sl].to(dev),
                         "latent": torch.from_numpy(np.asarray(lat_all[idx[sl]])).float().to(dev),
                         "dino": torch.from_numpy(np.asarray(dino_all[idx[sl]])).float().to(dev)}
                snr_t = torch.full((sl.stop - sl.start,), s, device=dev)
                chg = torch.Generator(device=dev).manual_seed(seed_of("channel", s, b))
                sg = torch.Generator(device=dev).manual_seed(seed_of("sample", s, STEPS if generative else None, b))
                recs.append(model.reconstruct(batch, snr_t, vae, STEPS if generative else None, chg, sg,
                                              cfg_scale=guidance).float().cpu())
                if refiner:
                    chg = torch.Generator(device=dev).manual_seed(seed_of("channel", s, b))
                    cnns.append(model.cnn_output(batch, snr_t, chg).float().cpu())
                # Extra sampling-noise draws for the diversity images in this batch, same channel noise.
                here = [int(p) for p in div if sl.start <= p < sl.stop]
                if generative and here and n_samples:
                    draws = []
                    for j in range(n_samples):
                        # The whole batch is redrawn so the channel generator replays this batch's
                        # noise exactly; only the diversity rows are kept.
                        chg = torch.Generator(device=dev).manual_seed(seed_of("channel", s, b))
                        full = model.reconstruct(batch, snr_t, vae, STEPS, chg,
                                                 torch.Generator(device=dev).manual_seed(seed_of("draw", s, b, j)),
                                                 cfg_scale=guidance).float().cpu()
                        draws.append(full[[p - sl.start for p in here]])
                    sams.append(torch.stack(draws, 1))  # (len(here), K, 3, H, W)
        xh = torch.cat(recs)
        emb = embed(xh)
        out = {"pred": classify(emb), "emb": emb.half().numpy(), "recon": to8(xh[kept]), "label": labels, "index": idx}
        if refiner:
            xc = torch.cat(cnns)
            ec = embed(xc)
            out.update(cnn=to8(xc[kept]), cnn_pred=classify(ec), cnn_emb=ec.half().numpy())
        if sams:
            out["samples"] = to8(torch.cat(sams))
        np.savez_compressed(f, **out)
        acc = float((out["pred"] == labels).mean())
        meta["runs"].setdefault(name, {"run": run, "model": cfg.model, "guidance": guidance})[f"{s:g}"] = {
            "class_kept": acc, "seconds": time.time() - t0}
        meta_f.write_text(json.dumps(meta, indent=1))
        print(f"DUMP {name} {s:g} dB: class kept {acc:.3f} ({time.time() - t0:.0f}s)", flush=True)
