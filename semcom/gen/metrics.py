"""Image-quality metrics for the generative-JSCC prototype.

Per image (so arms can be compared pairwise): PSNR, LPIPS (AlexNet), and DINOv2 similarity
(cosine between the CLS embeddings of the original and the reconstruction; the paper's
"DINOv2" column measures semantic agreement the same way). Per set: FID.

LPIPS and FID need optional packages (`lpips`, and `torch-fidelity` behind torchmetrics). A
missing package disables that metric with a warning instead of failing the evaluation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..data import psnr

__all__ = ["psnr", "Metrics"]


class Metrics:
    def __init__(self, dino, device: torch.device, fid: bool = True, lpips_net: bool = True):
        self.dino, self.device = dino, device
        self.lpips = None
        if lpips_net:
            try:
                import lpips

                self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
            except Exception as exc:  # noqa: BLE001
                print(f"[metrics] LPIPS disabled ({exc})")
        self.fid_ok = fid
        self._fid = None

    def new_fid(self):
        """A fresh FID accumulator, or None if unavailable."""
        if not self.fid_ok:
            return None
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance

            return FrechetInceptionDistance(feature=2048, normalize=True).to(self.device)
        except Exception as exc:  # noqa: BLE001
            print(f"[metrics] FID disabled ({exc})")
            self.fid_ok = False
            return None

    @torch.no_grad()
    def per_image(self, x: torch.Tensor, x_hat: torch.Tensor, grid: int,
                  embeddings: bool = False) -> dict[str, torch.Tensor]:
        """Per-image PSNR, LPIPS, DINOv2 CLS cosine. `embeddings` also returns the unit-norm CLS
        embeddings of x and x_hat under "_cls" / "_cls_hat" (for retrieval, see retrieval_rank)."""
        out = {"psnr": psnr(x, x_hat)}
        if self.lpips is not None:
            out["lpips"] = self.lpips(x * 2 - 1, x_hat * 2 - 1).flatten()
        _, cls = self.dino(x, grid)
        _, cls_hat = self.dino(x_hat, grid)
        out["dino_sim"] = F.cosine_similarity(cls.float(), cls_hat.float(), dim=1)
        if embeddings:
            out["_cls"] = F.normalize(cls.float(), dim=1)
            out["_cls_hat"] = F.normalize(cls_hat.float(), dim=1)
        return out


def retrieval_rank(cls: torch.Tensor, cls_hat: torch.Tensor) -> torch.Tensor:
    """For each reconstruction, how many of the evaluated originals are closer to it (DINOv2 CLS
    cosine) than its own original: 0 means its original is retrieved first. Unlike the raw cosine,
    whose floor is high (two unrelated natural images score about 0.5), this is chance-normalised:
    a reconstruction that carries nothing image-specific has rank about N/2."""
    sim = cls_hat @ cls.T
    return (sim > sim.diag()[:, None]).sum(1)
