"""Tests for the separation baseline.

The baseline's job is to be a *fair and generous* reference point. If it is accidentally
crippled, the learned arms win by default and the whole comparison is worthless - so most
of these tests check that it is not being shortchanged, rather than that it is correct in
some narrow numerical sense.
"""

import math

import numpy as np
import pytest
from PIL import Image

from semcom.separation import (
    capacity_bits_per_symbol,
    compress_to_budget,
    evaluate_separation,
    shannon_threshold_db,
)


@pytest.fixture
def photo():
    """A smooth, compressible 32x32 image - something a codec can actually work with."""
    yy, xx = np.meshgrid(np.linspace(0, 1, 32), np.linspace(0, 1, 32), indexing="ij")
    arr = np.stack([xx, yy, (xx + yy) / 2], axis=-1)
    return (arr * 255).astype(np.uint8)


# ------------------------------------------------------------------- information theory


@pytest.mark.parametrize("snr_db,expected", [(0.0, 1.0), (10.0, math.log2(11.0))])
def test_capacity_matches_shannon(snr_db, expected):
    assert capacity_bits_per_symbol(snr_db) == pytest.approx(expected)


def test_capacity_increases_with_snr():
    values = [capacity_bits_per_symbol(s) for s in range(0, 21, 5)]
    assert values == sorted(values)


def test_threshold_inverts_capacity():
    """The cliff edge must be exactly where capacity equals spectral efficiency."""
    for se in (1.0, 2.0, 4.5):
        snr = shannon_threshold_db(se)
        assert capacity_bits_per_symbol(snr) == pytest.approx(se, abs=1e-9)


def test_threshold_rises_with_spectral_efficiency():
    assert shannon_threshold_db(1.0) < shannon_threshold_db(2.0) < shannon_threshold_db(6.0)


# ------------------------------------------------------------------------- compression


def test_compression_respects_the_byte_budget(photo):
    """Overshooting the budget would give the baseline free bandwidth."""
    img = Image.fromarray(photo)
    for budget in (120, 200, 400, 1000):
        out = compress_to_budget(img, budget)
        if out.feasible:
            assert out.bytes_used <= budget, f"exceeded budget {budget}"


def test_compression_uses_the_budget_it_is_given(photo):
    """The baseline must not leave most of its allowance unspent.

    A binary search that collapses to the floor would silently hand the learned arms a
    win, so require that a generous budget is actually consumed.
    """
    out = compress_to_budget(Image.fromarray(photo), 1200)
    assert out.feasible
    assert out.bytes_used > 1200 * 0.3


def test_more_budget_never_gives_worse_quality(photo):
    """Monotonicity: quality must not fall as the rate rises."""
    from semcom.separation import _psnr

    qualities = []
    for budget in (150, 300, 600, 1200):
        out = compress_to_budget(Image.fromarray(photo), budget)
        if out.feasible:
            qualities.append(_psnr(photo, out.decoded))
    assert len(qualities) >= 3
    # Allow a small non-monotonicity from codec switching at the boundaries.
    assert qualities[-1] >= qualities[0] - 0.5
    assert qualities[-1] > qualities[0]


def test_infeasible_budget_is_reported_not_faked(photo):
    """A budget below every codec's floor must report failure and name the floor.

    Returning a fabricated number here would be the single most misleading thing this
    module could do, because it is exactly the regime where CIFAR-sized classical coding
    breaks down.
    """
    out = compress_to_budget(Image.fromarray(photo), 4)
    assert out.feasible is False
    assert out.floor_bytes > 4
    assert out.decoded is None


def test_decoded_image_has_the_right_shape(photo):
    out = compress_to_budget(Image.fromarray(photo), 800)
    assert out.feasible
    assert out.decoded.shape == photo.shape
    assert out.decoded.dtype == np.uint8


# -------------------------------------------------------------------------- the sweeps


def test_ideal_mode_budget_tracks_capacity(photo):
    """In ideal mode the rate must rise with SNR - that is what 'oracle AMC' means."""
    res = evaluate_separation(photo[None], k=256, snrs=[0.0, 10.0, 20.0], mode="ideal")
    budgets = [res["by_snr"][s]["budget_bytes"] for s in (0.0, 10.0, 20.0)]
    assert budgets == sorted(budgets)
    assert budgets[0] == pytest.approx(256 * 1.0 / 8)


def test_ideal_mode_never_declares_a_decode_failure(photo):
    """A capacity-achieving code at a capacity-matched rate always decodes."""
    res = evaluate_separation(photo[None], k=256, snrs=[0.0, 5.0, 20.0], mode="ideal")
    assert all(v["decoded"] for v in res["by_snr"].values())


def test_fixed_mcs_produces_a_cliff(photo):
    """The defining behaviour: nothing below threshold, flat above it."""
    res = evaluate_separation(
        photo[None], k=256, snrs=list(range(0, 21)),
        mode="fixed_mcs", bits_per_symbol=4, code_rate=0.5,
    )
    threshold = res["threshold_db"]
    assert 0 < threshold < 20, f"threshold {threshold} outside the swept range"

    below = [v for s, v in res["by_snr"].items() if s < threshold]
    above = [v for s, v in res["by_snr"].items() if s >= threshold]

    assert below and above
    # Below: no image at all, not a degraded one.
    assert all(v["psnr"] is None and not v["decoded"] for v in below)
    # Above: delivered, and at a quality that does not vary with SNR.
    assert all(v["decoded"] for v in above)
    psnrs = [v["psnr"] for v in above if v["psnr"] is not None]
    if len(psnrs) > 1:
        assert max(psnrs) - min(psnrs) < 1e-6, "fixed MCS quality must be flat above threshold"


def test_fixed_mcs_budget_is_independent_of_snr(photo):
    """The source rate is locked in before transmission - that is why surplus SNR is wasted."""
    res = evaluate_separation(
        photo[None], k=256, snrs=[10.0, 15.0, 20.0], mode="fixed_mcs",
        bits_per_symbol=4, code_rate=0.5,
    )
    budgets = {v["budget_bytes"] for v in res["by_snr"].values()}
    assert len(budgets) == 1


def test_higher_order_mcs_moves_the_cliff_right(photo):
    """Higher spectral efficiency buys quality but needs a better channel."""
    lo = evaluate_separation(photo[None], k=256, snrs=[10.0], mode="fixed_mcs",
                             bits_per_symbol=2, code_rate=0.5)
    hi = evaluate_separation(photo[None], k=256, snrs=[10.0], mode="fixed_mcs",
                             bits_per_symbol=6, code_rate=0.75)
    assert hi["threshold_db"] > lo["threshold_db"]


def test_infeasibility_is_surfaced_in_the_sweep(photo):
    """At a tiny symbol budget the codec floor bites, and the sweep must say so."""
    res = evaluate_separation(photo[None], k=8, snrs=[0.0], mode="ideal")
    entry = res["by_snr"][0.0]
    assert entry["feasible_fraction"] == 0.0
    assert entry["psnr"] is None
    assert entry["codec_floor_bytes"] > entry["budget_bytes"]


def test_rejects_unknown_mode(photo):
    with pytest.raises(ValueError):
        evaluate_separation(photo[None], k=256, snrs=[10.0], mode="turbo")


def test_status_distinguishes_channel_outage_from_codec_infeasibility(photo):
    """Two very different failures must not look the same in the results.

    A PSNR column shows `None` for both "the link could not carry it" and "the codec
    could not compress that small". Only the first is a statement about wireless; the
    second is a 32x32 header-floor artefact. Conflating them would misattribute a source
    coding limitation to the channel.
    """
    # Channel outage: budget is comfortably above the codec floor, SNR is below threshold.
    outage = evaluate_separation(
        photo[None], k=2048, snrs=[0.0], mode="fixed_mcs",
        bits_per_symbol=6, code_rate=0.75,
    )["by_snr"][0.0]
    assert outage["status"] == "channel_outage"
    assert outage["decoded"] is False

    # Codec infeasible: channel is fine, but the byte budget is below any codec's floor.
    starved = evaluate_separation(photo[None], k=8, snrs=[20.0], mode="ideal")["by_snr"][20.0]
    assert starved["status"] == "codec_infeasible"
    assert starved["decoded"] is True, "the channel was never the problem here"


def test_viable_mcs_shows_a_clean_cliff(photo):
    """With a budget above the codec floor, the cliff is the *only* failure mode."""
    res = evaluate_separation(
        photo[None], k=512, snrs=list(range(0, 21)), mode="fixed_mcs",
        bits_per_symbol=4, code_rate=0.5,
    )
    statuses = {v["status"] for v in res["by_snr"].values()}
    assert statuses == {"channel_outage", "ok"}, statuses


# --------------------------------------------------------------- monotonicity regressions


def test_codec_is_chosen_by_quality_not_file_size():
    """Regression: selecting the largest file picked a worse codec.

    Within one codec, more bytes means better quality. Across codecs it does not: at
    32x32 JPEG2000 becomes feasible around 270 bytes and emits a larger file than WebP
    while reconstructing worse. The first version selected on `bytes_used`, which made
    the capacity-bound curve collapse at exactly the SNR where the budget crossed that
    floor.
    """
    from semcom.separation import _psnr, cifar_test_images

    for arr in cifar_test_images(6, seed=1):
        img = Image.fromarray(arr)
        for budget in (150, 280, 320, 500):
            chosen = compress_to_budget(img, budget)
            if not chosen.feasible:
                continue
            for single in ("webp", "jpeg2000", "jpeg"):
                alt = compress_to_budget(img, budget, codecs=(single,))
                if alt.feasible:
                    assert _psnr(arr, chosen.decoded) >= _psnr(arr, alt.decoded) - 1e-9, (
                        f"{single} beat the chosen codec at budget {budget}"
                    )


def test_capacity_bound_is_monotonic_in_snr():
    """The headline sanity check: more channel must never mean a worse image.

    Guards both monotonicity bugs at once — quality-based codec selection, and excluding
    SNRs where only the most compressible images fit.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.run_ablation import separation_curve
    from semcom.separation import cifar_test_images

    imgs = cifar_test_images(12, seed=2)
    sweep = evaluate_separation(imgs, k=512, snrs=[float(s) for s in range(0, 21)],
                                mode="ideal")
    curve = separation_curve(sweep)

    assert len(curve) >= 8, "too few usable points to judge monotonicity"
    values = [curve[s] for s in sorted(curve)]
    assert values == sorted(values), f"capacity bound is not monotonic: {values}"


def test_partial_feasibility_points_are_excluded_from_the_curve():
    """A mean over only the compressible images is biased and must not be plotted."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.run_ablation import separation_curve

    sweep = {"by_snr": {
        0.0: {"psnr": 30.0, "feasible_fraction": 0.4},   # biased subset - drop
        5.0: {"psnr": 22.0, "feasible_fraction": 1.0},   # keep
        9.0: {"psnr": None, "feasible_fraction": 0.0},   # nothing delivered - drop
    }}
    assert separation_curve(sweep) == {5.0: 22.0}
    # The unfiltered view is still available for reporting.
    assert set(separation_curve(sweep, require_all=False)) == {0.0, 5.0}
