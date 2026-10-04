"""The 2x2: one model class, two booleans - plus the blind-JSCC arms.

                    analog (digital=False)   digital (digital=True)
    fixed SNR       BDJSCC                   DeepJSCC-Q
    SNR-adaptive    ADJSCC                   ADJSCC-Q

Every arm shares the same backbone, the same rate, and the same channel. The ablation is
a flag change, which is what makes the comparison controlled.

The blind-JSCC study (docs/proposal/blind-jscc-proposal.md) splits ADJSCC's SNR
conditioning into its two ends. `encoder_snr` puts AF modules with the SNR in the encoder;
`decoder_input` says what fills the decoder AF modules' conditioning slot:

    "none"         no decoder AF modules at all                      (C; BDJSCC/DeepJSCC-Q)
    "snr"          the true SNR (or an override, for sensitivity)    (B; A with encoder_snr)
    "blank"        a constant: same modules as B, no information     (C-att)
    "energy"       genie per-image transmitted energy                (C+E, digital only)
    "energy_shuf"  another image's transmitted energy                (C+E-shuf)
    "stats"        receiver-computed statistics of the whole block   (D)
    "stats_shuf"   another image's statistics                        (D-shuf)

`snr_adaptive=True` keeps its original meaning: SNR at both ends.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .channel import get_channel, power_normalise
from .constellation import SoftToHardQuantiser
from .modules import AFModule, FLModule, snr_condition
from .snr_estimation import receiver_statistics, statistics_dim

ARM_NAMES = {
    (False, False): "BDJSCC",
    (True, False): "ADJSCC",
    (False, True): "DeepJSCC-Q",
    (True, True): "ADJSCC-Q",
}

DECODER_INPUTS = ("none", "snr", "blank", "energy", "energy_shuf", "stats", "stats_shuf")

#: Labels of the blind-study arms, keyed by decoder input, for encoder-blind models.
BLIND_ARM_LABELS = {
    "snr": "B",
    "blank": "C-att",
    "energy": "C+E",
    "energy_shuf": "C+E-shuf",
    "stats": "D",
    "stats_shuf": "D-shuf",
}


def arm_name(snr_adaptive: bool, digital: bool) -> str:
    return ARM_NAMES[(bool(snr_adaptive), bool(digital))]


def resolve_conditioning(
    snr_adaptive: bool, encoder_snr: bool | None, decoder_input: str | None, digital: bool
) -> tuple[bool, str]:
    """Return (encoder_snr, decoder_input) with the legacy `snr_adaptive` flag folded in.

    `decoder_input="auto"` (or None) means "follow snr_adaptive". Raises on combinations
    that contradict each other or that carry no meaning.
    """
    dec = ("snr" if snr_adaptive else "none") if decoder_input in (None, "auto") else decoder_input
    enc = bool(snr_adaptive) if encoder_snr is None else bool(encoder_snr)
    if dec not in DECODER_INPUTS:
        raise ValueError(f"decoder_input must be one of {DECODER_INPUTS} or 'auto', got {dec!r}")
    if snr_adaptive and not (enc and dec == "snr"):
        raise ValueError(
            "snr_adaptive=True means SNR at both encoder and decoder; set encoder_snr / "
            "decoder_input instead of snr_adaptive to build a one-sided arm"
        )
    if dec.startswith("energy") and not digital:
        raise ValueError(
            "the energy arms need digital=True: analog transmit energy is exactly k by "
            "construction, so a genie-energy input would carry no information"
        )
    return enc, dec


def arm_label(encoder_snr: bool, decoder_input: str, digital: bool) -> str:
    """Human-readable arm name. Legacy 2x2 arms keep their published names."""
    if decoder_input == "snr" and encoder_snr:
        return arm_name(True, digital)
    if decoder_input == "none" and not encoder_snr:
        return arm_name(False, digital)
    if not encoder_snr:
        return BLIND_ARM_LABELS[decoder_input]
    return f"EncSNR-{decoder_input}"


def decoder_cond_dim(decoder_input: str, digital: bool) -> int | None:
    """Width of the decoder AF conditioning slot, or None for no decoder AF modules."""
    if decoder_input == "none":
        return None
    if decoder_input.startswith("stats"):
        return statistics_dim(digital)
    return 1


def shuffle_across_batch(cond: torch.Tensor) -> torch.Tensor:
    """Give every image another image's conditioning value.

    A cyclic shift by one: deterministic, consumes no random numbers (so it cannot disturb
    the paired noise streams across arms), and never returns an image its own row. With a
    batch of one there is no other image, so the input is blanked instead.
    """
    if cond.shape[0] < 2:
        return torch.zeros_like(cond)
    return torch.roll(cond, shifts=1, dims=0)


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
    """Mirror of the encoder: transposed convs with IGDN, sigmoid on the final module.

    AF modules follow FL1-FL4 when `cond_dim` is set (or `snr_adaptive`, which means a
    one-wide SNR slot). `forward` takes the conditioning input: an SNR in dB, shape
    (batch,), or a prepared (batch, cond_dim) tensor; it is ignored without AF modules.
    """

    def __init__(
        self,
        c_in: int,
        out_channels: int = 3,
        hidden: int = 256,
        snr_adaptive: bool = False,
        cond_dim: int | None = None,
    ):
        super().__init__()
        if cond_dim is None and snr_adaptive:
            cond_dim = 1
        self.cond_dim = cond_dim
        self.snr_adaptive = cond_dim is not None

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
            nn.ModuleList([AFModule(hidden, cond_dim=cond_dim) for _ in range(4)])
            if cond_dim is not None
            else None
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        for i, fl in enumerate(self.fl):
            x = fl(x)
            if self.af is not None and i < len(self.af):
                x = self.af[i](x, cond)
        return x


class JSCC(nn.Module):
    """Joint source-channel coding model spanning the 2x2 and the blind-JSCC arms.

    Args:
        c_out: encoder output channel count; sets the bandwidth ratio.
        snr_adaptive: SNR-conditioned AF modules at both ends (ADJSCC mechanism).
        digital: quantise the channel input to a finite M-QAM alphabet (DeepJSCC-Q).
        modulation_order: M, used only when `digital`.
        channel: "awgn" or "rayleigh".
        image_size: input spatial size; the backbone downsamples by 4.
        encoder_snr: SNR-conditioned AF modules in the encoder (None: follow snr_adaptive).
        decoder_input: what fills the decoder AF slot; see the module docstring.
        hidden_enc, hidden_dec: per-side widths (None: use `hidden`), for capacity sweeps.
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
        encoder_snr: bool | None = None,
        decoder_input: str = "auto",
        hidden_enc: int | None = None,
        hidden_dec: int | None = None,
    ):
        super().__init__()
        if c_out % 2 != 0:
            raise ValueError(f"c_out must be even to pair into (I, Q), got {c_out}")

        self.encoder_snr, self.decoder_input = resolve_conditioning(
            snr_adaptive, encoder_snr, decoder_input, digital
        )
        # Kept with its original meaning - "the encoder has SNR-conditioned AF modules" -
        # which is what analyze_gates.py reads it as.
        self.snr_adaptive = self.encoder_snr
        self.digital = bool(digital)
        self.c_out = int(c_out)
        self.avg_power = float(avg_power)
        self.channel_name = channel
        self.channel_fn = get_channel(channel)

        self.latent_spatial = image_size // 4
        self.k = self.latent_spatial**2 * c_out // 2
        self.source_dim = image_size * image_size * in_channels
        self.bandwidth_ratio = self.k / self.source_dim

        self.encoder = Encoder(c_out, in_channels, hidden_enc or hidden, self.encoder_snr)
        self.decoder = Decoder(
            c_out,
            in_channels,
            hidden_dec or hidden,
            cond_dim=decoder_cond_dim(self.decoder_input, self.digital),
        )

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
        return arm_label(self.encoder_snr, self.decoder_input, self.digital)

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

    def decoder_condition(
        self,
        snr_db: torch.Tensor,
        z: torch.Tensor,
        y: torch.Tensor,
        decoder_snr: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Build the decoder AF conditioning input for this arm, shape (batch, cond_dim).

        `decoder_snr` replaces the SNR the decoder is told (decoder_input="snr" only)
        without touching the channel, for the sensitivity curve and plug-in estimates.
        Genie energy and the receiver statistics are detached: they are fixed functions of
        the signal, not paths for the encoder to shape.
        """
        mode = self.decoder_input
        b = y.shape[0]
        if mode == "none":
            return None
        if mode == "snr":
            s = snr_db if decoder_snr is None else decoder_snr
            if s.dim() == 0:
                s = s.expand(b)
            return snr_condition(s, b, y.dtype)
        if decoder_snr is not None:
            raise ValueError(f"decoder_snr only applies to decoder_input='snr', not {mode!r}")
        if mode == "blank":
            return y.new_zeros(b, 1)
        if mode.startswith("energy"):
            # Standardised so per-image variation is O(1): for uniform 16-QAM the mean
            # power of a k-symbol block has standard deviation about sqrt(0.32 / k).
            p = z.detach().pow(2).sum(dim=-1).mean(dim=1)
            cond = ((p - self.avg_power) * math.sqrt(self.k)).unsqueeze(1).to(y.dtype)
        else:
            points = self.quantiser.points if self.quantiser is not None else None
            cond = receiver_statistics(y.detach().float(), points, self.avg_power).to(y.dtype)
        if mode.endswith("_shuf"):
            cond = shuffle_across_batch(cond)
        return cond

    def decode(
        self,
        y: torch.Tensor,
        snr_db: torch.Tensor,
        z: torch.Tensor,
        decoder_snr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decoder side only: received symbols (B, k, 2) -> reconstruction.

        `z` is the transmitted block; only the genie-energy arms read it.
        """
        if snr_db.dim() == 0:
            snr_db = snr_db.expand(y.shape[0])
        cond = self.decoder_condition(snr_db, z, y, decoder_snr)
        return self.decoder(self._from_symbols(y), cond)

    def forward(
        self,
        x: torch.Tensor,
        snr_db: torch.Tensor,
        collect_gates: bool = False,
        erase_mask: torch.Tensor | None = None,
        decoder_snr: torch.Tensor | None = None,
    ) -> dict:
        """Full end-to-end pass.

        Args:
            erase_mask: optional boolean (batch, k) or (k,) marking symbols the receiver
                never gets. Masked symbols are zeroed *after* the channel, which is what
                a lost packet looks like to the decoder: no observation at all, rather
                than a noisy one. Used by `semcom.importance` to measure how much each
                part of the latent actually contributes to the reconstruction. It changes
                nothing when left as None, so training and evaluation are unaffected.
            decoder_snr: SNR told to the decoder instead of the true one
                (decoder_input="snr" only). The channel still uses `snr_db`.

        Returns a dict with the reconstruction, the channel input and output, and (when
        digital) the KL-to-uniform term for the codebook-collapse regulariser.
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

        if erase_mask is not None:
            # Erasure is applied at the receiver, so the transmitted signal - and hence
            # the constellation invariant and the transmit power - are untouched.
            mask = erase_mask.to(device=y.device, dtype=torch.bool)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0).expand(y.shape[0], -1)
            y = y.masked_fill(mask.unsqueeze(-1), 0.0)

        x_hat = self.decode(y, snr_db, z, decoder_snr)

        return {"x_hat": x_hat, "z": z, "y": y, "kl": kl, "gates": gates}

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
