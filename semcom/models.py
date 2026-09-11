"""The 2x2: one model class, two booleans.

                    analog (digital=False)   digital (digital=True)
    fixed SNR       BDJSCC                   DeepJSCC-Q
    SNR-adaptive    ADJSCC                   ADJSCC-Q   <- the target

Every arm shares the same backbone, the same rate, and the same channel. The ablation is
a flag change, which is what makes the comparison controlled.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .channel import get_channel, power_normalise
from .constellation import SoftToHardQuantiser
from .modules import AFModule, FLModule

ARM_NAMES = {
    (False, False): "BDJSCC",
    (True, False): "ADJSCC",
    (False, True): "DeepJSCC-Q",
    (True, True): "ADJSCC-Q",
}


def arm_name(snr_adaptive: bool, digital: bool) -> str:
    return ARM_NAMES[(bool(snr_adaptive), bool(digital))]


class Encoder(nn.Module):
    """Five FL modules; AF modules after FL1-FL4 when SNR-adaptive.

    Strides 2, 2, 1, 1, 1 take 32x32 down to 8x8. The final module emits `c_out`
    channels, which is the knob that sets the bandwidth ratio.
    """

    def __init__(
        self,
        c_out: int,
        in_channels: int = 3,
        hidden: int = 256,
        snr_adaptive: bool = False,
    ):
        super().__init__()
        self.snr_adaptive = bool(snr_adaptive)

        self.fl = nn.ModuleList(
            [
                FLModule(in_channels, hidden, stride=2),
                FLModule(hidden, hidden, stride=2),
                FLModule(hidden, hidden, stride=1),
                FLModule(hidden, hidden, stride=1),
                # Final encoder module ends at GDN, no PReLU.
                FLModule(hidden, c_out, stride=1, activation="none"),
            ]
        )
        self.af = (
            nn.ModuleList([AFModule(hidden) for _ in range(4)]) if snr_adaptive else None
        )

    def forward(
        self, x: torch.Tensor, snr_db: torch.Tensor, collect_gates: bool = False
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        gates: list[torch.Tensor] = []
        for i, fl in enumerate(self.fl):
            x = fl(x)
            if self.af is not None and i < len(self.af):
                if collect_gates:
                    gates.append(self.af[i].compute_gates(x, snr_db).detach())
                x = self.af[i](x, snr_db)
        return x, gates


class Decoder(nn.Module):
    """Mirror of the encoder: transposed convs with IGDN, sigmoid on the final module."""

    def __init__(
        self,
        c_in: int,
        out_channels: int = 3,
        hidden: int = 256,
        snr_adaptive: bool = False,
    ):
        super().__init__()
        self.snr_adaptive = bool(snr_adaptive)

        self.fl = nn.ModuleList(
            [
                FLModule(c_in, hidden, stride=1, transposed=True),
                FLModule(hidden, hidden, stride=1, transposed=True),
                FLModule(hidden, hidden, stride=1, transposed=True),
                FLModule(hidden, hidden, stride=2, transposed=True),
                FLModule(
                    hidden, out_channels, stride=2, transposed=True, activation="sigmoid"
                ),
            ]
        )
        self.af = (
            nn.ModuleList([AFModule(hidden) for _ in range(4)]) if snr_adaptive else None
        )

    def forward(self, x: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        for i, fl in enumerate(self.fl):
            x = fl(x)
            if self.af is not None and i < len(self.af):
                x = self.af[i](x, snr_db)
        return x


class JSCC(nn.Module):
    """Joint source-channel coding model spanning all four ablation arms.

    Args:
        c_out: encoder output channel count; sets the bandwidth ratio.
        snr_adaptive: insert AF modules and condition on SNR (ADJSCC mechanism).
        digital: quantise the channel input to a finite M-QAM alphabet (DeepJSCC-Q).
        modulation_order: M, used only when `digital`.
        channel: "awgn" or "rayleigh".
        image_size: input spatial size; the backbone downsamples by 4.
    """

    def __init__(
        self,
        c_out: int,
        snr_adaptive: bool = False,
        digital: bool = False,
        modulation_order: int = 16,
        channel: str = "awgn",
        image_size: int = 32,
        in_channels: int = 3,
        hidden: int = 256,
        avg_power: float = 1.0,
        sigma_q_init: float = 5.0,
        anneal_period: int = 10_000,
    ):
        super().__init__()
        if c_out % 2 != 0:
            raise ValueError(f"c_out must be even to pair into (I, Q), got {c_out}")

        self.snr_adaptive = bool(snr_adaptive)
        self.digital = bool(digital)
        self.c_out = int(c_out)
        self.avg_power = float(avg_power)
        self.channel_name = channel
        self.channel_fn = get_channel(channel)

        self.latent_spatial = image_size // 4
        self.k = self.latent_spatial**2 * c_out // 2
        self.source_dim = image_size * image_size * in_channels
        self.bandwidth_ratio = self.k / self.source_dim

        self.encoder = Encoder(c_out, in_channels, hidden, snr_adaptive)
        self.decoder = Decoder(c_out, in_channels, hidden, snr_adaptive)

        self.quantiser = (
            SoftToHardQuantiser(
                modulation_order,
                avg_power=avg_power,
                sigma_q_init=sigma_q_init,
                anneal_period=anneal_period,
            )
            if digital
            else None
        )

    @property
    def name(self) -> str:
        return arm_name(self.snr_adaptive, self.digital)

    def _to_symbols(self, feat: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, k, 2), pairing channels into (I, Q)."""
        b = feat.shape[0]
        return feat.reshape(b, -1, 2)

    def _from_symbols(self, z: torch.Tensor) -> torch.Tensor:
        """(B, k, 2) -> (B, C, H, W)."""
        b = z.shape[0]
        s = self.latent_spatial
        return z.reshape(b, self.c_out, s, s)

    def transmit(self, x: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        """Run the transmit chain and return the channel input, shape (B, k, 2).

        Order: encode (with AF gating) -> power normalise -> quantise. When `digital`,
        the returned tensor contains only exact constellation points.
        """
        feat, _ = self.encoder(x, snr_db)
        z = self._to_symbols(feat)
        z = power_normalise(z, self.avg_power)
        if self.quantiser is not None:
            z = self.quantiser(z)
        return z

    def forward(
        self, x: torch.Tensor, snr_db: torch.Tensor, collect_gates: bool = False
    ) -> dict:
        """Full end-to-end pass.

        Returns a dict with the reconstruction, the channel input, and (when digital) the
        KL-to-uniform term for the codebook-collapse regulariser.
        """
        if snr_db.dim() == 0:
            snr_db = snr_db.expand(x.shape[0])

        feat, gates = self.encoder(x, snr_db, collect_gates=collect_gates)
        z = self._to_symbols(feat)
        z = power_normalise(z, self.avg_power)

        if self.quantiser is not None:
            z = self.quantiser(z)
            kl = self.quantiser.kl_to_uniform()
        else:
            kl = torch.zeros((), device=x.device, dtype=x.dtype)

        # The receiver feeds equalised symbols straight into the decoder network. No
        # demapping to LLRs, no channel decoding, no CRC.
        y = self.channel_fn(z, snr_db, self.avg_power)
        x_hat = self.decoder(self._from_symbols(y), snr_db)

        return {"x_hat": x_hat, "z": z, "kl": kl, "gates": gates}

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
