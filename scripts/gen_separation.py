"""Separate source and channel coding baseline (AVIF + LDPC + QAM) on the JSCC's test images and
channel uses: k = 4096 complex symbols per 256 px image (bandwidth ratio 1/48), the same 1,000
spread test images, and the same metrics (PSNR, LPIPS-Alex, DINOv2 similarity, retrieval, FID).

usage:
  python scripts/gen_separation.py channel   # LDPC codes, BLER tables, QAM capacities
  python scripts/gen_separation.py source    # AVIF at every bit budget, per-image metrics
  python scripts/gen_separation.py combine   # AWGN and Rayleigh evaluations per scheme

Steps 1 and 2 are independent (run them side by side); step 3 needs both. Each is resumable.

Schemes (the transmitter picks the scheme for the average SNR; the receiver needs the SNR, and
under fading the fade, for its LLRs: separation is a "told" system at both ends):
  LDPC M-QAM rate R   M in 4/16/64, R in 1/2, 2/3, 3/4, 5/6: n = 2048 regular (3, dc) PEG codes,
                      normalised min-sum, 50 iterations. The image is lost (shown as a flat image
                      at the training-set mean colour) if any of its codewords fails.
  ideal 16-QAM rate r a capacity-achieving code on 16-QAM: the image arrives iff 4r bits per
                      symbol is below the 16-QAM mutual information at the (effective) SNR. An
                      upper bound on separation with the JSCC's constellation.
Envelopes pick, per SNR, the scheme with the best mean PSNR or the best mean LPIPS (a genie
choice of scheme, which favours the baseline).

Rayleigh: block fading, one fade h ~ CN(0, 1) per image, perfect CSI at the receiver (the
effective SNR is snr + 10 log10 |h|^2), none at the transmitter, the same fades for every scheme.

Inputs: data/separation/spread1000_256.npz (the 1,000 spread test images and the training-set
mean colour, extracted from the 256 px cache). Outputs: results/gen/separation/.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

K = 4096  # complex channel uses per image
N_CODE = 2048
MODS = (4, 16, 64)
RATES = {"1-2": 1 / 2, "2-3": 2 / 3, "3-4": 3 / 4, "5-6": 5 / 6}
# QPSK also gets the low rates a real system uses near and below 0 dB (5G NR goes down to about
# 0.12 with QPSK); without them every LDPC scheme fails below about 2 dB and the baseline is unfair.
LOW_RATES = {"1-8": 1 / 8, "1-5": 1 / 5, "1-4": 1 / 4, "1-3": 1 / 3, "2-5": 2 / 5}
IDEAL_RATES = [round(0.1 * i, 2) for i in range(1, 11)]  # plus the exact AWGN capacity budgets
SNRS = (-5, 0, 5, 10, 15, 20)
OUT = ROOT / "results/gen/separation"
WORK = ROOT / "data/separation"
DATA = WORK / "spread1000_256.npz"
SNR_GRID = np.round(np.arange(-12, 30.01, 0.25), 2)


def ldpc_schemes() -> dict:
    out = {}
    for M in MODS:
        m = int(math.log2(M))
        blocks = K * m // N_CODE
        for name, r in ({**LOW_RATES, **RATES} if M == 4 else RATES).items():
            k = int(round(N_CODE * r))
            out[f"ldpc_qam{M}_r{name}"] = {"M": M, "rate": k / N_CODE, "k": k, "blocks": blocks,
                                          "bytes": blocks * k // 8}
    return out


def ideal_schemes(cap16: dict | None = None) -> dict:
    out = {f"ideal_qam16_r{r:.2f}": {"M": 16, "rate": r, "bytes": int(K * 4 * r) // 8} for r in IDEAL_RATES}
    if cap16 is not None:  # the exact AWGN capacity budget at each evaluated SNR
        for s in SNRS:
            c = float(np.interp(s, SNR_GRID, cap16))
            out[f"ideal_qam16_cap{s:g}"] = {"M": 16, "rate": c / 4, "bytes": int(K * c) // 8}
    return out


# ---------------------------------------------------------------------------------------------
# step 1: channel


def step_channel() -> None:
    import torch

    from semcom.gen.separation import MinSum, bler_point, girth_hint, peg_ldpc, qam_capacity

    torch.set_num_threads(int(os.environ.get("SEP_THREADS", "6")))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "channel.json"
    state = json.loads(path.read_text()) if path.exists() else {}
    state.setdefault("method", {
        "code": f"regular (3, dc) LDPC by progressive edge growth, n = {N_CODE}, seed 0",
        "decoder": "normalised min-sum, alpha 0.8, flooding, up to 50 iterations",
        "bler": "all-zero codeword through an i.i.d. scrambler, exact LLRs, up to 4000 words or 200 errors per point",
        "snr": "Es/N0, Es = average symbol energy (unit), complex noise variance N0 = 10^(-snr/10)"})
    state["snr_grid"] = SNR_GRID.tolist()
    if "capacity" not in state:
        state["capacity"] = {str(M): [qam_capacity(M, float(s), n=100_000) for s in SNR_GRID] for M in MODS}
        path.write_text(json.dumps(state))
        print("capacities done", flush=True)
    state.setdefault("bler", {})
    codes = {}
    for name, sc in ldpc_schemes().items():
        if name in state["bler"] and state["bler"][name].get("done"):
            continue
        k = sc["k"]
        if k not in codes:
            t0 = time.time()
            codes[k] = peg_ldpc(N_CODE, k, seed=0)
            print(f"code k={k}: {time.time() - t0:.0f} s, 4-cycles {girth_hint(codes[k])}", flush=True)
        code = codes[k]
        dec = MinSum(code)
        M, m = sc["M"], int(math.log2(sc["M"]))
        cap = np.array(state["capacity"][str(M)])
        thr = float(np.interp(m * sc["rate"], cap, SNR_GRID))  # Shannon limit of this scheme
        entry = {"threshold_db": thr, "snr": [], "bler": [], "words": []}
        zeros = 0
        for s in SNR_GRID[SNR_GRID >= thr - 0.5]:
            t0 = time.time()
            b, e, w = bler_point(code, dec, M, float(s), seed=int((s + 20) * 100))
            entry["snr"].append(float(s))
            entry["bler"].append(b)
            entry["words"].append(w)
            print(f"{name} {s:+.2f} dB BLER {b:.4f} ({e}/{w}) {time.time() - t0:.0f} s", flush=True)
            zeros = zeros + 1 if e == 0 else 0
            if zeros >= 2:
                break
        b = np.array(entry["bler"])
        sn = np.array(entry["snr"])
        hit = np.flatnonzero(b <= 1e-2)
        entry["snr_at_bler_1e-2"] = float(sn[hit[0]]) if len(hit) else None
        entry["gap_to_limit_db"] = (entry["snr_at_bler_1e-2"] - thr) if len(hit) else None
        entry["done"] = True
        state["bler"][name] = entry
        path.write_text(json.dumps(state))
    print("channel done", flush=True)


def bler_at(entry: dict, snr: np.ndarray) -> np.ndarray:
    """BLER at arbitrary SNRs: 1 below the simulated range, 0 above it, log-linear in between."""
    sn = np.array(entry["snr"])
    b = np.clip(np.array(entry["bler"]), 1e-6, 1)
    out = np.exp(np.interp(snr, sn, np.log(b)))
    out[snr < sn[0]] = 1.0
    out[snr > sn[-1]] = 0.0
    last_zero = np.array(entry["bler"])[-1] == 0
    if last_zero:
        z = sn[np.flatnonzero(np.array(entry["bler"]) == 0)[0]]
        out[snr >= z] = 0.0
    return out


# ---------------------------------------------------------------------------------------------
# step 2: source


def _fit(args):
    from semcom.gen.separation import avif_fit

    img, budget = args
    r = avif_fit(img, budget)
    if r is None:
        return None, 0, -1, 0
    return r["image"], r["bytes"], r["quality"], r["scale"]


class Scorer:
    """Per-image metrics as semcom.gen.metrics / evaluate.py compute them (CPU)."""

    def __init__(self, originals: np.ndarray):
        import torch
        from torchmetrics.image.fid import FrechetInceptionDistance

        from semcom.gen.config import GenConfig
        from semcom.gen.frozen import FrozenDINO
        from semcom.gen.metrics import Metrics

        torch.set_num_threads(int(os.environ.get("SEP_THREADS", "6")))
        self.torch = torch
        cfg = GenConfig.from_yaml(ROOT / "configs/gen/p256.yaml")
        self.grid = cfg.grid
        self.dino = FrozenDINO(cfg.dino_model, cfg.dino_input).eval()
        self.metrics = Metrics(self.dino, torch.device("cpu"), fid=False, lpips_net=True)
        self.inception = FrechetInceptionDistance(feature=2048, normalize=True).inception.eval()
        self.x = originals  # (N, 3, H, W) uint8
        ref = WORK / "originals.npz"
        if ref.exists():
            d = np.load(ref)
            self.cls, self.cls_n, self.incep = d["cls"], d["cls_n"], d["incep"]
        else:
            cls, incep = self._embed(originals)
            self.cls, self.incep = cls, incep
            self.cls_n = cls / np.linalg.norm(cls, axis=1, keepdims=True)
            np.savez(ref, cls=self.cls, cls_n=self.cls_n, incep=self.incep)

    def _batches(self, a, bs=50):
        for i in range(0, len(a), bs):
            yield i, self.torch.from_numpy(np.ascontiguousarray(a[i:i + bs])).float() / 255

    def _embed(self, imgs):
        torch = self.torch
        cls, inc = [], []
        with torch.no_grad():
            for _, x in self._batches(imgs):
                cls.append(self.dino(x, self.grid)[1].float().numpy())
                inc.append(self.inception((x * 255).byte()).double().numpy())  # as torchmetrics, normalize=True
        return np.concatenate(cls), np.concatenate(inc)

    def score(self, recon: np.ndarray) -> dict:
        """recon (N, 3, H, W) uint8 -> per-image psnr, lpips, dino_sim, cls_hat_n, incep."""
        torch = self.torch
        from semcom.data import psnr

        ps, lp = [], []
        with torch.no_grad():
            for i, xh in self._batches(recon):
                x = torch.from_numpy(self.x[i:i + len(xh)]).float() / 255
                ps.append(psnr(x, xh).numpy())
                lp.append(self.metrics.lpips(x * 2 - 1, xh * 2 - 1).flatten().numpy())
        cls_hat, inc = self._embed(recon)
        a = self.cls / np.linalg.norm(self.cls, axis=1, keepdims=True)
        b = cls_hat / np.linalg.norm(cls_hat, axis=1, keepdims=True)
        return {"psnr": np.concatenate(ps), "lpips": np.concatenate(lp), "dino_sim": (a * b).sum(1),
                "cls_hat_n": b.astype(np.float32), "incep": inc}


def step_source() -> None:
    from multiprocessing import Pool

    d = np.load(DATA)
    x = d["images"]  # (N, 3, H, W) uint8
    hwc = x.transpose(0, 2, 3, 1)
    mean = d["train_mean"]
    budgets_dir = WORK / "budgets"
    budgets_dir.mkdir(parents=True, exist_ok=True)
    scorer = Scorer(x)
    lost = budgets_dir / "lost.npz"
    if not lost.exists():
        flat = np.broadcast_to((np.round(mean * 255)).astype(np.uint8)[None, :, None, None], x.shape).copy()
        np.savez(lost, **scorer.score(flat))
        print("lost (flat mean image) scored", flush=True)
    cap16 = None
    ch = OUT / "channel.json"
    if ch.exists() and "capacity" in json.loads(ch.read_text()):
        cap16 = json.loads(ch.read_text())["capacity"]["16"]
    budgets = sorted({sc["bytes"] for sc in {**ldpc_schemes(), **ideal_schemes(cap16)}.values()})
    workers = int(os.environ.get("SEP_WORKERS", "10"))
    print(f"{len(budgets)} budgets: {budgets}", flush=True)
    with Pool(workers) as pool:
        for bud in budgets:
            f = budgets_dir / f"b{bud:06d}.npz"
            if f.exists():
                continue
            t0 = time.time()
            res = pool.map(_fit, [(hwc[i], bud) for i in range(len(hwc))], chunksize=4)
            sent = np.array([r[0] is not None for r in res])
            recon = np.stack([(r[0] if r[0] is not None else np.zeros_like(hwc[0])) for r in res]).transpose(0, 3, 1, 2)
            t1 = time.time()
            sc = scorer.score(recon)
            np.savez(f, sent=sent, bytes=np.array([r[1] for r in res]), quality=np.array([r[2] for r in res]),
                     scale=np.array([r[3] for r in res]), **sc)
            if bud == budgets[len(budgets) // 2]:  # a sample grid for the record
                from PIL import Image
                Image.fromarray(np.concatenate(list(recon[:8].transpose(0, 2, 3, 1)), 1)).save(OUT / f"samples_b{bud}.png")
            print(f"budget {bud} B: sent {sent.mean():.3f}, PSNR {sc['psnr'][sent].mean() if sent.any() else float('nan'):.2f}, "
                  f"LPIPS {sc['lpips'][sent].mean() if sent.any() else float('nan'):.3f}, encode {t1 - t0:.0f} s, "
                  f"score {time.time() - t1:.0f} s", flush=True)
    print("source done", flush=True)


# ---------------------------------------------------------------------------------------------
# step 3: combine


def fid(real: np.ndarray, fake: np.ndarray) -> float:
    """FID as torchmetrics computes it (double precision, trace of sqrt via eigenvalues)."""
    import torch

    r, f = torch.from_numpy(real).double(), torch.from_numpy(fake).double()
    mu1, mu2 = r.mean(0), f.mean(0)
    s1, s2 = torch.cov(r.T), torch.cov(f.T)
    a = (mu1 - mu2).square().sum()
    b = s1.trace() + s2.trace()
    c = torch.linalg.eigvals(s1 @ s2).sqrt().real.sum()
    return float(a + b - 2 * c)


def step_combine() -> None:
    ch = json.loads((OUT / "channel.json").read_text())
    cap16 = np.array(ch["capacity"]["16"])
    budgets_dir = WORK / "budgets"
    ref = np.load(WORK / "originals.npz")
    lost = dict(np.load(budgets_dir / "lost.npz"))
    N = len(ref["cls"])
    cache = {}

    def budget(b):
        if b not in cache:
            cache[b] = dict(np.load(budgets_dir / f"b{b:06d}.npz"))
        return cache[b]

    schemes = {**ldpc_schemes(), **ideal_schemes(cap16)}
    rng = np.random.default_rng(2026)
    h = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) * math.sqrt(0.5)
    fade_db = 10 * np.log10(np.abs(h) ** 2)
    summary = {"schemes": schemes, "channels": {}}
    for chan in ("awgn", "rayleigh"):
        per_scheme = {}
        for s in SNRS:
            eff = np.full(N, float(s)) if chan == "awgn" else s + fade_db
            u = np.random.default_rng(1000 + s + (0 if chan == "awgn" else 500)).random((N, 12))
            for name, sc in schemes.items():
                if name.startswith("ideal_qam16_cap") and chan == "rayleigh":
                    continue
                b = budget(sc["bytes"])
                if name.startswith("ldpc"):
                    p = bler_at(ch["bler"][name], eff)
                    ok = (u[:, :sc["blocks"]] > p[:, None]).all(1)
                else:
                    ok = np.interp(eff, ch["snr_grid"], cap16) >= 4 * sc["rate"] - 1e-9
                ok &= b["sent"]
                pick = lambda key: np.where(ok if b[key].ndim == 1 else ok[:, None], b[key], lost[key])  # noqa: E731
                per = {k: pick(k) for k in ("psnr", "lpips", "dino_sim")}
                cls_hat, inc = pick("cls_hat_n"), pick("incep")
                sim = cls_hat @ ref["cls_n"].T
                per["retrieval_rank"] = (sim > np.diag(sim)[:, None]).sum(1)
                entry = {k: float(v.mean()) for k, v in per.items()}
                entry.update(retrieval_top1=float((per["retrieval_rank"] < 1).mean()),
                             retrieval_top5=float((per["retrieval_rank"] < 5).mean()),
                             fid=fid(ref["incep"], inc), delivered=float(ok.mean()), n_images=N)
                per_scheme.setdefault(name, {})[f"{s:g}"] = (entry, per)
        # envelopes: per SNR, the scheme with the best mean PSNR / LPIPS
        fams = {"ldpc_qam16": [n for n in per_scheme if n.startswith("ldpc_qam16_")],
                "ldpc_all": [n for n in per_scheme if n.startswith("ldpc_")],
                "ideal_qam16": [n for n in per_scheme if n.startswith("ideal_")]}
        env = {}
        for fam, names in fams.items():
            for crit, sign in (("psnr", 1), ("lpips", -1)):
                key = f"{fam}_best{crit}"
                env[key] = {}
                for s in SNRS:
                    best = max(names, key=lambda n: sign * per_scheme[n][f"{s:g}"][0][crit])
                    e, per = per_scheme[best][f"{s:g}"]
                    env[key][f"{s:g}"] = ({**e, "scheme": best}, per)
        # write: envelopes as run folders (paired.py reads eval_images-spread.npz), all schemes in summary
        for key, rows in {**env, **{f"mcs/{n}": v for n, v in per_scheme.items()}}.items():
            d = OUT / chan / key
            d.mkdir(parents=True, exist_ok=True)
            res = {s: e for s, (e, _) in rows.items()}
            (d / "eval-spread.json").write_text(json.dumps(
                {"run_name": f"separation_{chan}_{key.replace('/', '_')}", "subset": "spread", "n_images": N,
                 "model": "separation", "channel": chan, "snrs": list(SNRS), "results": res}, indent=2))
            np.savez_compressed(d / "eval_images-spread.npz",
                                **{f"{k}_{s}": v for s, (_, per) in rows.items() for k, v in per.items()})
        summary["channels"][chan] = {k: {s: e for s, (e, _) in v.items()} for k, v in env.items()}
        print(f"== {chan}")
        for key in env:
            print(f"  {key:22s} " + "  ".join(f"{s}:{env[key][s][0]['psnr']:.2f}/{env[key][s][0]['lpips']:.3f}"
                                             for s in env[key]))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("combine done", flush=True)


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    {"channel": step_channel, "source": step_source, "combine": step_combine}.get(
        step, lambda: sys.exit(__doc__))()
