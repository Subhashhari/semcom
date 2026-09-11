"""Feature Learning (FL) and Attention Feature (AF) modules.

FL modules are the conventional backbone, inherited from DeepJSCC-F via BDJSCC.
AF modules are the ADJSCC contribution (Xu et al., arXiv:2012.00533): a
Squeeze-and-Excitation block with the channel SNR spliced into the squeeze vector.

Terminology trap, flagged in the ADJSCC paper's own footnote and worth repeating:
"channel-wise" here refers to the *feature channels* of the tensor, not the wireless
channel. The AF module attends over convolutional feature maps, using the wireless
channel's SNR only as a conditioning scalar.
"""

import torch
import torch.nn as nn

from .gdn import GDN

# SNR values are normalised by this before entering the AF MLP, so the conditioning
# scalar sits on a comparable scale to the pooled activations it is concatenated with.
SNR_NORM_DB = 20.0


class FLModule(nn.Module):
    """Conv -> GDN -> PReLU (encoder) or TransposedConv -> IGDN -> PReLU (decoder).

    The final module of each side drops the PReLU: the encoder's last module ends at GDN,
    and the decoder's last module substitutes a sigmoid to bound the output to [0, 1].
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        transposed: bool = False,
        activation: str = "prelu",
    ):
        super().__init__()
        padding = kernel_size // 2

        if transposed:
            self.conv = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                output_padding=stride - 1,
            )
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding
            )

        self.norm = GDN(out_channels, inverse=transposed)

        if activation == "prelu":
            self.act = nn.PReLU()
        elif activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif activation == "none":
            self.act = nn.Identity()
        else:
            raise ValueError(f"unknown activation: {activation!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class AFModule(nn.Module):
    """Attention Feature module: SNR-conditioned channel-wise gating.

    Three steps, following ADJSCC section III:
      (a) context extraction  - global average pool each feature map, concatenate SNR
      (b) factor prediction   - two-layer MLP -> sigmoid -> gates in (0, 1)
      (c) recalibration       - channel-wise multiply

    The sigmoid is load-bearing: it confines every factor to (0, 1), making this a gate
    that can only attenuate, never amplify. Keeping the MLP to two FC layers is what
    holds the parameter overhead to well under 1% of the backbone.
    """

    def __init__(self, num_channels: int, reduction: int = 16):
        super().__init__()
        self.num_channels = int(num_channels)
        hidden = max(4, num_channels // reduction)

        self.mlp = nn.Sequential(
            nn.Linear(num_channels + 1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        gates = self.compute_gates(x, snr_db)
        return x * gates.view(x.shape[0], self.num_channels, 1, 1)

    def compute_gates(self, x: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        """Return the per-channel scaling factors S, shape (batch, num_channels).

        Exposed separately from `forward` so `analyze_gates.py` can record the factors
        without re-running the recalibration.
        """
        pooled = x.mean(dim=(2, 3))
        mu = (snr_db.to(x.dtype).view(-1, 1) / SNR_NORM_DB).expand(x.shape[0], 1)
        context = torch.cat([mu, pooled], dim=1)
        return self.mlp(context)
