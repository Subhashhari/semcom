"""Tests for the ablation driver's analysis logic.

The analysis is what turns raw curves into the claims that go in the writeup, so it is
worth testing on synthetic results where the right answer is known by construction. A bug
here would not crash anything - it would just quietly report the wrong conclusion.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_ablation import SPECIALIST_SNRS, analyse, arm_configs, curve_of, envelope  # noqa: E402
from semcom.config import Config  # noqa: E402

# Mirrors the real setup: the eval grid contains every specialist training SNR.
SNRS = [0.0, 1.0, 4.0, 7.0, 13.0, 19.0, 20.0]


def result(arm, curve, *, params=1000, snr_fixed=None):
    return {
        "arm": arm,
        "parameters": params,
        "snr_train_fixed": snr_fixed,
        # Written with string keys, as they come back from JSON.
        "psnr_by_snr": {str(s): v for s, v in zip(SNRS, curve)},
    }


def synthetic_results(adaptive_bonus_digital=2.0, adaptive_bonus_analog=1.0):
    """Four arms where the adaptive models beat their specialists by a known margin."""
    results = []
    for snr_fixed in SPECIALIST_SNRS[:3]:
        # A specialist peaks near its training SNR and falls off either side.
        curve = [20.0 - abs(s - snr_fixed) * 0.3 for s in SNRS]
        results.append(result("BDJSCC", curve, snr_fixed=snr_fixed))
        results.append(result("DeepJSCC-Q", curve, snr_fixed=snr_fixed))

    digital_env = [max(20.0 - abs(s - f) * 0.3 for f in SPECIALIST_SNRS[:3]) for s in SNRS]
    results.append(
        result("ADJSCC", [v + adaptive_bonus_analog for v in digital_env], params=1006)
    )
    results.append(
        result("ADJSCC-Q", [v + adaptive_bonus_digital for v in digital_env], params=1006)
    )
    return results


# ------------------------------------------------------------------------- run planning


def test_arm_configs_covers_the_full_2x2_with_no_duplicate_run_names():
    runs = arm_configs(Config())
    assert len(runs) == 2 + 2 * len(SPECIALIST_SNRS)
    assert len({r.run_name for r in runs}) == len(runs), "run name collision"
    assert {r.arm for r in runs} == {"ADJSCC", "ADJSCC-Q", "BDJSCC", "DeepJSCC-Q"}


def test_arm_configs_gives_adaptive_arms_a_range_and_specialists_a_point():
    for cfg in arm_configs(Config()):
        if cfg.snr_adaptive:
            assert cfg.snr_train_fixed is None
        else:
            assert cfg.snr_train_fixed in SPECIALIST_SNRS


def test_arm_configs_does_not_mutate_the_base():
    base = Config()
    arm_configs(base)
    assert base.snr_adaptive is False and base.snr_train_fixed is None


# ------------------------------------------------------------------------------ helpers


def test_envelope_takes_the_best_specialist_at_each_snr():
    peaked = [1, 9, 1, 1, 1, 1, 1]
    flat = [5] * 7
    assert envelope([result("X", peaked), result("X", flat)], SNRS) == dict(
        zip(SNRS, [5, 9, 5, 5, 5, 5, 5])
    )


def test_curve_and_envelope_accept_both_string_and_float_snr_keys():
    """Curves arrive as JSON (string keys) or in-memory (float keys); both must work."""
    as_float = {"arm": "X", "parameters": 1, "snr_train_fixed": None,
                "psnr_by_snr": {s: 3.0 for s in SNRS}}
    assert curve_of(as_float, SNRS) == {s: 3.0 for s in SNRS}
    assert envelope([as_float], SNRS) == {s: 3.0 for s in SNRS}


# ----------------------------------------------------------------------------- analysis


def test_analyse_requires_all_four_arms():
    with pytest.raises(RuntimeError, match="missing arms"):
        analyse([result("ADJSCC-Q", [1] * len(SNRS))], SNRS)


def test_q1_detects_a_clean_sweep_at_matched_points():
    a = analyse(synthetic_results(), SNRS)["q1_matched_point"]
    assert a["adaptive_wins_everywhere"] is True
    assert a["wins"] == a["of"] == 3
    assert all(m["margin_db"] > 0 for m in a["per_snr"])


def test_q1_detects_when_a_specialist_wins_its_home_turf():
    """The matched-point claim is the sharpest one, so a loss must be reported, not hidden."""
    results = synthetic_results(adaptive_bonus_digital=-1.0)
    a = analyse(results, SNRS)["q1_matched_point"]
    assert a["adaptive_wins_everywhere"] is False
    assert a["wins"] == 0


def test_q1_compares_each_specialist_at_its_own_training_snr():
    results = synthetic_results()
    a = analyse(results, SNRS)["q1_matched_point"]
    for m in a["per_snr"]:
        specialist = next(
            r for r in results
            if r["arm"] == "DeepJSCC-Q" and r["snr_train_fixed"] == m["snr_db"]
        )
        assert m["specialist_psnr"] == pytest.approx(
            specialist["psnr_by_snr"][str(m["snr_db"])]
        )


def test_q2_reports_the_hypothesis_holding_when_the_digital_gap_is_larger():
    q2 = analyse(synthetic_results(2.0, 1.0), SNRS)["q2_conditioning_gain_db"]
    assert q2["digital (ADJSCC-Q - DeepJSCC-Q)"] == pytest.approx(2.0)
    assert q2["analog (ADJSCC - BDJSCC)"] == pytest.approx(1.0)
    assert q2["quantisation_amplifies_conditioning"] is True


def test_q2_reports_the_hypothesis_failing_when_it_fails():
    """A null result is a legitimate outcome and must not be silently flipped."""
    q2 = analyse(synthetic_results(0.5, 1.5), SNRS)["q2_conditioning_gain_db"]
    assert q2["quantisation_amplifies_conditioning"] is False


def test_q3_storage_compares_one_model_against_the_ensemble():
    q3 = analyse(synthetic_results(), SNRS)["q3_storage"]
    assert q3["adaptive_models"] == 1
    assert q3["ensemble_models"] == 3
    # 1006 params vs 3 x 1000.
    assert q3["storage_fraction"] == pytest.approx(1006 / 3000)
    assert q3["adaptive_mb"] == pytest.approx(1006 * 4 / 1024**2)


def test_curves_contain_all_four_arms():
    curves = analyse(synthetic_results(), SNRS)["curves"]
    assert set(curves) == {
        "ADJSCC-Q",
        "ADJSCC",
        "DeepJSCC-Q (specialist envelope)",
        "BDJSCC (specialist envelope)",
    }
    for curve in curves.values():
        assert sorted(curve) == SNRS


def test_specialist_outside_the_eval_grid_raises_rather_than_reporting_a_vacuous_win():
    """Regression: a silent skip here produced the most misleading answer available.

    The matched-point comparison needs each specialist evaluated at its own training SNR.
    The first version skipped any specialist missing from the eval grid, so with a
    mismatched grid `matched` came back empty - and `all([])` is True, which reported
    "adaptive wins everywhere" having made zero comparisons.
    """
    results = synthetic_results()
    off_grid = [s for s in SNRS if s != 99.0]
    for r in results:
        if r["arm"] == "DeepJSCC-Q":
            r["snr_train_fixed"] = 99.0

    with pytest.raises(RuntimeError, match="matched-point test cannot run"):
        analyse(results, off_grid)


# ------------------------------------------------------ separation reference integration


def test_separation_curve_drops_undelivered_points():
    """Gaps must stay gaps. Imputing a PSNR where nothing arrived would erase the cliff."""
    from scripts.run_ablation import separation_curve

    sweep = {"by_snr": {0.0: {"psnr": None}, 5.0: {"psnr": 20.0}, 10.0: {"psnr": 22.0}}}
    assert separation_curve(sweep) == {5.0: 20.0, 10.0: 22.0}
    assert separation_curve(None) == {}


def test_separation_reference_reports_both_modes_and_the_codec_floor():
    from scripts.run_ablation import separation_reference

    snrs = [float(s) for s in range(0, 21, 4)]
    ref = separation_reference(k=512, snrs=snrs, n_images=4)

    assert ref["codec_floor_bytes"] > 0
    assert ref["ideal"]["mode"] == "ideal"
    # At k=512 an MCS with an in-range cliff and a usable budget does exist.
    assert ref["mcs"] is not None
    assert ref["fixed_mcs"]["mode"] == "fixed_mcs"
    assert min(snrs) < ref["mcs"]["threshold_db"] < max(snrs)


def test_separation_reference_survives_a_rate_with_no_viable_mcs():
    """At a tiny symbol budget no MCS works; that must be reported, not crash the run."""
    from scripts.run_ablation import separation_reference

    ref = separation_reference(k=8, snrs=[0.0, 10.0, 20.0], n_images=2)
    assert ref["mcs"] is None
    assert ref["fixed_mcs"] is None
    assert ref["ideal"] is not None


def test_plot_and_report_handle_the_separation_block(tmp_path, capsys):
    """The figure and the printed report must both survive a real separation block."""
    from scripts.run_ablation import analyse, plot, report, separation_reference

    analysis = analyse(synthetic_results(), SNRS)
    analysis["separation"] = separation_reference(k=512, snrs=SNRS, n_images=3)

    out = tmp_path / "ablation.png"
    plot(analysis, out)
    assert out.exists() and out.stat().st_size > 0

    report(analysis)
    text = capsys.readouterr().out
    assert "External reference" in text
    assert "codec floor" in text
