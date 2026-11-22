"""Per-image semantic, perceptual and no-reference metrics on a full dump (dump.py --keep 1), for the
exploratory DiT-vs-CNN analysis of 22 Nov (protocol in docs/proposal/gen-jscc-month-plan.md, section 8).
GPU job; the metric packages live outside the training environment:

    PYTHONPATH=~/evalpkgs python scripts/gen_analysis/semantic_metrics.py DUMP_DIR OUT_DIR KEY=RUN_NAME ... \
        [--snrs -5,0,5] [--batch 32]

For the originals and every (system, SNR) reconstruction set, per image:
  dreamsim        DreamSim distance to the original (ensemble; lower = more similar to humans)
  dists           DISTS distance to the original (structure + texture statistics; lower is better)
  clip_sim        CLIP ViT-L/14 (OpenAI) image-image cosine to the original
  clip_rank       how many of the N originals are closer (CLIP cosine) than its own: 0 = retrieved first
  in1k_top1/5     ConvNeXt-L ImageNet-1k top-1 class and top-5 classes (the originals' too)
  in1k_prob       the full ImageNet-1k distribution (float16), for agreement and divergence later
  zs_top1         CLIP zero-shot ImageNet-1k top-1 class
  caption         BLIP-large caption (beam 3); clip_txt its CLIP text embedding (float16)
  capscore        CLIP cosine of the image with the ORIGINAL's caption
  det             DETR-R50 detections above 0.7: (label, box) lists, for set and box F1 later
  clipiqa         CLIP-IQA no-reference quality (higher is better); brisque (lower is better)
Saved as OUT_DIR/<KEY>__<snr>.npz and OUT_DIR/original.npz, captions and detections in JSON.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

argv = sys.argv[1:]


def opt(name, default):
    if name in argv:
        i = argv.index(name)
        v = argv[i + 1]
        del argv[i:i + 2]
        return v
    return default


snrs = [float(s) for s in opt("--snrs", "-5,0,5").split(",")]
B = int(opt("--batch", "32"))
dump, out = Path(argv[0]), Path(argv[1])
systems = dict(a.split("=", 1) for a in argv[2:])
out.mkdir(parents=True, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True

sets = {"original": np.load(dump / "_original.npz")["image"]}
for key, run in systems.items():
    for s in snrs:
        sets[f"{key}__{s:g}"] = np.load(dump / f"{run}__{s:g}.npz")["recon"]
n = len(sets["original"])
assert all(len(v) == n for v in sets.values()), {k: len(v) for k, v in sets.items()}
print(f"{n} images, sets: {list(sets)}", flush=True)
res = {k: {} for k in sets}
captions, dets, det_names = {}, {}, {}
partial = out / "partial.pkl"  # results after each finished stage, so a crash costs only its stage
done = set()
if partial.exists():
    import pickle
    res, captions, dets, det_names, done = pickle.loads(partial.read_bytes())
    print(f"resuming after stages {sorted(done)}", flush=True)


def finish(name):
    import pickle
    done.add(name)
    partial.write_bytes(pickle.dumps((res, captions, dets, det_names, done)))


def batches(arr, size=B):
    for i in range(0, len(arr), size):
        yield torch.from_numpy(np.asarray(arr[i:i + size])).float().div(255).to(dev)


def resize(x, s):
    return F.interpolate(x, size=(s, s), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)


def stage(name):
    print(f"[{time.strftime('%H:%M:%S')}] {name}", flush=True)


def run(name, fn):
    if name in done:
        print(f"skip {name} (done)", flush=True)
        return
    stage(name)
    fn()
    torch.cuda.empty_cache()
    finish(name)


def s_dreamsim():
    from dreamsim import dreamsim
    # torch.hub's fork check fails without a GitHub token (KeyError: Authorization); DreamSim loads
    # its DINO backbone through torch.hub, so skip the check.
    torch.hub._validate_not_a_forked_repo = lambda *a, **k: None
    ds, _ = dreamsim(pretrained=True, device=dev, cache_dir=str(Path.home() / ".cache/dreamsim"))
    with torch.no_grad():
        for k in sets:
            if k != "original":
                d = [ds(resize(xo, 224), resize(xr, 224)).float().cpu()
                     for xo, xr in zip(batches(sets["original"]), batches(sets[k]))]
                res[k]["dreamsim"] = torch.cat(d).numpy()


def s_dists():
    import piq
    dists = piq.DISTS(reduction="none").to(dev)
    with torch.no_grad():
        for k in sets:
            if k != "original":
                d = [dists(xr, xo).float().cpu() for xo, xr in zip(batches(sets["original"]), batches(sets[k]))]
                res[k]["dists"] = torch.cat(d).numpy()


def s_convnext():
    import torchvision
    cnx = torchvision.models.convnext_large(weights="IMAGENET1K_V1").to(dev).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
        for k in sets:
            p_ = torch.cat([F.softmax(cnx((resize(x, 224) - mean) / std).float(), 1).cpu() for x in batches(sets[k])])
            res[k]["in1k_prob"] = p_.half().numpy()
            res[k]["in1k_top1"] = p_.argmax(1).numpy()
            res[k]["in1k_top5"] = p_.topk(5, 1).indices.numpy()


def load_clip():
    import open_clip
    clip, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", device=dev)
    clip.eval()
    tok = open_clip.get_tokenizer("ViT-L-14")
    cm = torch.tensor(open_clip.OPENAI_DATASET_MEAN, device=dev).view(1, 3, 1, 1)
    cs = torch.tensor(open_clip.OPENAI_DATASET_STD, device=dev).view(1, 3, 1, 1)

    def img(arr):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            return torch.cat([F.normalize(clip.encode_image((resize(x, 224) - cm) / cs).float(), dim=1).cpu()
                              for x in batches(arr)])

    def txt(texts):
        o = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            for i in range(0, len(texts), 256):
                o.append(F.normalize(clip.encode_text(tok(texts[i:i + 256]).to(dev)).float(), dim=1).cpu())
        return torch.cat(o)
    return open_clip, img, txt


def s_clip():
    open_clip, img, txt = load_clip()
    templates = getattr(open_clip, "SIMPLE_IMAGENET_TEMPLATES", None) or open_clip.OPENAI_IMAGENET_TEMPLATES[:7]
    cls_txt = torch.stack([F.normalize(txt([t(c) for t in templates]).mean(0), dim=0)
                           for c in open_clip.IMAGENET_CLASSNAMES])
    emb = {k: img(v) for k, v in sets.items()}
    for k in sets:
        res[k]["zs_top1"] = (emb[k] @ cls_txt.T).argmax(1).numpy()
        res[k]["clip_img"] = emb[k].half().numpy()
        if k != "original":
            res[k]["clip_sim"] = (emb[k] * emb["original"]).sum(1).numpy()
            sim = emb[k] @ emb["original"].T
            res[k]["clip_rank"] = (sim > sim.diag()[:, None]).sum(1).numpy()


def s_blip():
    from transformers import BlipForConditionalGeneration, BlipProcessor
    bp = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    bm = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large",
                                                      torch_dtype=torch.float16).to(dev).eval()
    with torch.no_grad():
        for k, arr in sets.items():
            caps = []
            for i in range(0, n, B):
                imgs = [np.transpose(a, (1, 2, 0)) for a in arr[i:i + B]]
                inp = bp(images=imgs, return_tensors="pt").to(dev, torch.float16)
                ids = bm.generate(**inp, max_new_tokens=30, num_beams=3)
                caps += [c.strip() for c in bp.batch_decode(ids, skip_special_tokens=True)]
            captions[k] = caps
            print(f"  {k}: {caps[0]!r}", flush=True)
    (out / "captions.json").write_text(json.dumps(captions, indent=0))


def s_captext():
    _, _, txt = load_clip()
    cap = {k: txt(v) for k, v in captions.items()}
    for k in sets:
        im = torch.from_numpy(res[k]["clip_img"]).float()
        res[k]["clip_txt"] = cap[k].half().numpy()
        res[k]["capscore"] = (im * cap["original"]).sum(1).numpy()  # image vs the ORIGINAL's caption
        if k != "original":
            res[k]["caption_sim"] = (cap[k] * cap["original"]).sum(1).numpy()


def s_detr():
    from transformers import DetrForObjectDetection, DetrImageProcessor
    dp = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    dm = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm").to(dev).eval()
    det_names.update({int(i): v for i, v in dm.config.id2label.items()})
    with torch.no_grad():
        for k, arr in sets.items():
            found = []
            for i in range(0, n, B):
                imgs = [np.transpose(a, (1, 2, 0)) for a in arr[i:i + B]]
                o = dm(**dp(images=imgs, return_tensors="pt").to(dev))
                sizes = torch.tensor([[a.shape[1], a.shape[2]] for a in arr[i:i + B]], device=dev)
                for r in dp.post_process_object_detection(o, threshold=0.7, target_sizes=sizes):
                    found.append([[int(l), [round(float(v), 1) for v in b]] for l, b in zip(r["labels"], r["boxes"])])
            dets[k] = found
    (out / "detections.json").write_text(json.dumps({"labels": det_names, "sets": dets}))


def s_noref():
    import piq
    ciqa = piq.CLIPIQA(data_range=1.0).to(dev)
    with torch.no_grad():
        for k, arr in sets.items():
            res[k]["clipiqa"] = torch.cat([ciqa(x).float().view(-1).cpu() for x in batches(arr)]).numpy()
            res[k]["brisque"] = torch.cat([piq.brisque(x, data_range=1.0, reduction="none").float().cpu()
                                           for x in batches(arr)]).numpy()


for name, fn in (("dreamsim", s_dreamsim), ("dists", s_dists), ("convnext", s_convnext), ("clip", s_clip),
                 ("blip", s_blip), ("captext", s_captext), ("detr", s_detr), ("noref", s_noref)):
    run(name, fn)

for k, v in res.items():
    np.savez_compressed(out / f"{k}.npz", **v)
(out / "meta.json").write_text(json.dumps({"dump": str(dump), "systems": systems, "snrs": snrs, "n": n}, indent=1))
stage("done")
