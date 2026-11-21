"""Separate source and channel coding baseline for the generative JSCC: an image codec (AVIF), an
LDPC code and Gray-mapped QAM over the same channel uses as the JSCC (k complex symbols per image).

The pieces:
  qam(M)                 Gray-mapped square M-QAM at unit average energy, with its bit labels
  llr(y, N0, M)          exact per-bit log-likelihood ratios (log-sum-exp, positive = bit 0)
  peg_ldpc(n, k)         a (3, dc) LDPC code built by progressive edge growth (Hu et al., 2005)
  MinSum(code)           normalised min-sum decoder (flooding), batched in torch
  bler(code, M, snr_db)  block error rate over AWGN, by simulation (see below)
  qam_capacity(M, snr)   mutual information of uniform M-QAM over AWGN (coded-modulation bound)
  avif_fit(img, budget)  the best AVIF encoding of an image whose payload fits a byte budget

BLER simulation sends the all-zero codeword through an i.i.d. scrambler: the transmitted bits are a
uniform random sequence s, and the receiver flips the sign of the LLRs where s = 1. That makes the
QAM bit channels symmetric, so the all-zero word stands for any codeword (the usual "channel
adapter" argument), and no encoder is needed.

The SNR is Es/N0 with Es the average symbol energy, as in the JSCC (power_ref "actual"): complex
noise of total variance N0 = Es / 10^(snr/10).
"""

from __future__ import annotations

import io
import math
import struct

import numpy as np
import torch

# ---------------------------------------------------------------------------------------------
# QAM


def qam(M: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(points (M,) complex64 at unit average energy, labels (M, log2 M) of 0/1).

    Square Gray mapping: the first half of each label picks the in-phase level, the second half
    the quadrature level, each by a binary-reflected Gray code (adjacent points differ in one bit)."""
    m = int(round(math.log2(M)))
    side = int(round(math.sqrt(M)))
    if side * side != M or m % 2:
        raise ValueError(f"square QAM needs M = 4, 16, 64, ...; got {M}")
    h = m // 2
    levels = np.arange(side) * 2 - (side - 1)  # -3, -1, 1, 3 for 16-QAM
    gray = np.arange(side) ^ (np.arange(side) >> 1)  # Gray code of each level index
    bits_of = lambda v: [(v >> (h - 1 - i)) & 1 for i in range(h)]  # noqa: E731
    pts, labels = [], []
    for i in range(side):
        for q in range(side):
            pts.append(levels[i] + 1j * levels[q])
            labels.append(bits_of(gray[i]) + bits_of(gray[q]))
    pts = np.array(pts)
    pts = pts / np.sqrt((np.abs(pts) ** 2).mean())
    return torch.tensor(pts, dtype=torch.complex64), torch.tensor(labels, dtype=torch.uint8)


def modulate(bits: torch.Tensor, M: int) -> torch.Tensor:
    """(B, L) 0/1 bits, L a multiple of log2 M -> (B, L / log2 M) complex symbols."""
    pts, _ = qam(M)
    m = int(round(math.log2(M)))
    side = int(round(math.sqrt(M)))
    h = m // 2
    b = bits.view(bits.shape[0], -1, m).long()
    w = 2 ** torch.arange(h - 1, -1, -1)
    gi, gq = (b[..., :h] * w).sum(-1), (b[..., h:] * w).sum(-1)
    # invert the Gray code: level index from its Gray value
    inv = torch.empty(side, dtype=torch.long)
    inv[torch.arange(side) ^ (torch.arange(side) >> 1)] = torch.arange(side)
    return pts[inv[gi] * side + inv[gq]]


def llr(y: torch.Tensor, n0: torch.Tensor | float, M: int) -> torch.Tensor:
    """Exact bit LLRs log P(b=0|y) / P(b=1|y) for (B, S) received symbols and noise variance n0
    (scalar or (B,)); returns (B, S * log2 M) float32."""
    pts, labels = qam(M)
    n0 = torch.as_tensor(n0, dtype=torch.float32)
    n0 = n0.view(-1, 1, 1) if n0.dim() else n0
    metric = -(y.unsqueeze(-1) - pts).abs().pow(2) / n0  # (B, S, M)
    m = labels.shape[1]
    out = []
    for i in range(m):
        zero = labels[:, i] == 0
        out.append(torch.logsumexp(metric[..., zero], -1) - torch.logsumexp(metric[..., ~zero], -1))
    return torch.stack(out, -1).flatten(1).float()


def qam_capacity(M: int, snr_db: float, n: int = 200_000, seed: int = 0) -> float:
    """Mutual information (bits per symbol) of uniform M-QAM over complex AWGN, by Monte Carlo:
    I = log2 M - E[ log2 sum_j exp((|y - x|^2 - |y - x_j|^2) / N0) ]."""
    g = torch.Generator().manual_seed(seed)
    pts, _ = qam(M)
    n0 = 10 ** (-snr_db / 10)
    x = pts[torch.randint(M, (n,), generator=g)]
    z = (torch.randn(n, generator=g) + 1j * torch.randn(n, generator=g)) * math.sqrt(n0 / 2)
    y = x + z
    d = (-(y.unsqueeze(-1) - pts).abs().pow(2) + (y - x).abs().pow(2).unsqueeze(-1)) / n0
    return float(math.log2(M) - (torch.logsumexp(d.double(), -1) / math.log(2)).mean())


# ---------------------------------------------------------------------------------------------
# LDPC


def peg_ldpc(n: int, k: int, dv: int = 3, seed: int = 0) -> dict:
    """A (dv, ~dc) LDPC code of length n and dimension k (m = n - k checks) by progressive edge
    growth: each new edge of a variable node goes to a check node outside its current
    neighbourhood (or the deepest one if all are reached), lowest degree first, ties at random.
    This maximises the local girth. Returns {"n", "k", "vn", "cn"} with the edge lists."""
    rng = np.random.default_rng(seed)
    m = n - k
    vn_adj = [[] for _ in range(n)]
    cn_adj = [[] for _ in range(m)]
    cdeg = np.zeros(m, dtype=np.int64)
    for j in range(n):
        for e in range(dv):
            if e == 0:
                cand = np.flatnonzero(cdeg == cdeg.min())
            else:
                # breadth-first search from j over the current graph, by check-node layers
                seen_c = np.zeros(m, dtype=bool)
                seen_v = np.zeros(n, dtype=bool)
                seen_v[j] = True
                frontier_v = [j]
                last = None
                while True:
                    new_c = {c for v in frontier_v for c in vn_adj[v] if not seen_c[c]}
                    if not new_c:
                        break
                    last = seen_c.copy()
                    seen_c[list(new_c)] = True
                    if seen_c.all():
                        break
                    new_v = {v for c in new_c for v in cn_adj[c] if not seen_v[v]}
                    if not new_v:
                        break
                    seen_v[list(new_v)] = True
                    frontier_v = list(new_v)
                if not seen_c.all():
                    pool = np.flatnonzero(~seen_c)
                else:  # every check reached: take those reached last (the deepest layer)
                    pool = np.flatnonzero(~last) if last is not None else np.arange(m)
                pool = pool[~np.isin(pool, vn_adj[j])]
                cand = pool[cdeg[pool] == cdeg[pool].min()]
            c = int(rng.choice(cand))
            vn_adj[j].append(c)
            cn_adj[c].append(j)
            cdeg[c] += 1
    vn = np.array([j for j in range(n) for _ in vn_adj[j]], dtype=np.int64)
    cn = np.array([c for j in range(n) for c in vn_adj[j]], dtype=np.int64)
    return {"n": n, "k": k, "vn": vn, "cn": cn}


def girth_hint(code: dict) -> int:
    """Number of length-4 cycles (pairs of checks sharing two variable nodes); 0 for a good code."""
    from collections import Counter

    rows = {}
    for v, c in zip(code["vn"], code["cn"]):
        rows.setdefault(int(c), []).append(int(v))
    pairs = Counter()
    for vs in rows.values():
        vs = sorted(vs)
        for a in range(len(vs)):
            for b in range(a + 1, len(vs)):
                pairs[(vs[a], vs[b])] += 1
    return sum(1 for x in pairs.values() if x > 1)


class MinSum:
    """Normalised min-sum decoding (flooding schedule), batched over codewords.

    Check-node messages are alpha * prod(signs) * min(|other messages|); decoding stops early for
    the whole batch once every codeword satisfies its parity checks."""

    def __init__(self, code: dict, alpha: float = 0.8, iters: int = 50):
        self.n, self.alpha, self.iters = code["n"], alpha, iters
        vn = torch.as_tensor(code["vn"])
        cn = torch.as_tensor(code["cn"])
        self.vn, self.cn, self.E = vn, cn, len(vn)
        self.m = int(cn.max()) + 1
        # padded check -> edge table (m, dmax), -1 for padding
        order = torch.argsort(cn, stable=True)
        deg = torch.bincount(cn, minlength=self.m)
        dmax = int(deg.max())
        slot = torch.arange(self.E) - torch.repeat_interleave(torch.cumsum(deg, 0) - deg, deg)
        table = torch.full((self.m, dmax), -1, dtype=torch.long)
        table[cn[order], slot] = order
        self.table, self.pad = table, table < 0
        self.flat = table.clamp_min(0).flatten()
        # where each edge sits in the padded table (to scatter back)
        self.pos = torch.empty(self.E, dtype=torch.long)
        self.pos[order] = cn[order] * dmax + slot

    @torch.no_grad()
    def decode(self, L: torch.Tensor) -> torch.Tensor:
        """Channel LLRs (B, n), positive = 0 -> hard decisions (B, n) uint8."""
        B = L.shape[0]
        c2v = torch.zeros(B, self.E)
        big = torch.tensor(1e9)
        for _ in range(self.iters):
            total = L.index_add(1, self.vn, c2v)
            hard = total < 0
            syn = torch.zeros(B, self.m, dtype=torch.long).index_add_(1, self.cn, hard[:, self.vn].long()) % 2
            if not syn.any():
                return hard.to(torch.uint8)
            v2c = total[:, self.vn] - c2v
            t = v2c[:, self.flat].view(B, self.m, -1)
            mag = torch.where(self.pad, big, t.abs())
            sgn = torch.where(self.pad, 1.0, torch.sign(t) + (t == 0).float())
            prod = sgn.prod(-1, keepdim=True)
            two = torch.topk(mag, 2, dim=-1, largest=False)
            m1, m2 = two.values[..., :1], two.values[..., 1:2]
            is_min = torch.arange(mag.shape[-1]) == two.indices[..., :1]
            new = self.alpha * prod * sgn * torch.where(is_min, m2, m1)
            c2v = new.flatten(1)[:, self.pos]
        return (L.index_add(1, self.vn, c2v) < 0).to(torch.uint8)


def bler_point(code: dict, decoder: MinSum, M: int, snr_db: float, max_words: int = 4000,
               min_errors: int = 200, batch: int = 500, seed: int = 0) -> tuple[float, int, int]:
    """(BLER, errors, words) over AWGN at one SNR (see the module docstring for the method)."""
    m = int(round(math.log2(M)))
    n = code["n"]
    if n % m:
        raise ValueError(f"code length {n} is not a multiple of {m} bits per symbol")
    g = torch.Generator().manual_seed(seed)
    n0 = 10 ** (-snr_db / 10)
    err = words = 0
    while words < max_words and err < min_errors:
        s = torch.randint(2, (batch, n), generator=g, dtype=torch.uint8)
        x = modulate(s, M)
        z = (torch.randn(x.shape, generator=g) + 1j * torch.randn(x.shape, generator=g)) * math.sqrt(n0 / 2)
        L = llr(x + z, n0, M) * (1 - 2 * s.float())  # unscramble: LLRs of the all-zero codeword
        hard = decoder.decode(L)
        err += int(hard.any(1).sum())
        words += batch
    return err / words, err, words


# ---------------------------------------------------------------------------------------------
# AVIF


def avif_payload(data: bytes) -> int:
    """Bytes of coded image data: the size of the 'mdat' box's contents. The rest of the file
    (about 275 bytes here) is container metadata fixed by the codec configuration, which a fixed
    system would not transmit per image."""
    i = 0
    while i < len(data):
        size, typ = struct.unpack(">I4s", data[i:i + 8])
        hdr = 8
        if size == 1:
            size = struct.unpack(">Q", data[i + 8:i + 16])[0]
            hdr = 16
        if typ == b"mdat":
            return size - hdr
        if size == 0:
            break
        i += size
    raise ValueError("no mdat box in the AVIF file")


def _avif(img, q: int, speed: int) -> tuple[bytes, int]:
    bio = io.BytesIO()
    img.save(bio, format="AVIF", quality=q, speed=speed)
    b = bio.getvalue()
    return b, avif_payload(b)


def avif_fit(image: np.ndarray, budget_bytes: int, speed: int = 4) -> dict | None:
    """The highest-quality AVIF encoding of `image` (H, W, 3 uint8) whose payload fits the budget.

    Searches the integer quality 0..100 by bisection at full resolution; if even quality 0 does
    not fit, tries half resolution (decoded and upsampled bicubically, which helps at the lowest
    rates). Returns {"image": decoded (H, W, 3) uint8, "bytes", "quality", "scale"}, or None if
    nothing fits (the image cannot be sent at this budget)."""
    from PIL import Image

    h, w = image.shape[:2]
    for scale in (1, 2):
        img = Image.fromarray(image)
        if scale > 1:
            img = img.resize((w // scale, h // scale), Image.BICUBIC)
        b0, p0 = _avif(img, 0, speed)
        if p0 > budget_bytes:
            continue
        lo, hi, best = 0, 100, (b0, p0, 0)
        while lo < hi:  # largest q whose payload fits (payload is monotone in q, nearly always)
            mid = (lo + hi + 1) // 2
            b, p = _avif(img, mid, speed)
            if p <= budget_bytes:
                lo, best = mid, (b, p, mid)
            else:
                hi = mid - 1
        dec = Image.open(io.BytesIO(best[0])).convert("RGB")
        if scale > 1:
            dec = dec.resize((w, h), Image.BICUBIC)
        return {"image": np.asarray(dec, dtype=np.uint8).copy(), "bytes": best[1], "quality": best[2],
                "scale": scale}
    return None
