"""The pre-registered decision machinery, on inputs where the right answer is known."""

import numpy as np
import pytest
from scipy import stats

from semcom.blind_stats import (
    classify,
    clearly_nonzero,
    cue_fit,
    equivalent,
    gate,
    holm,
    paired_interval,
    superiority_p,
)


def test_interval_uses_student_t_not_normal_quantiles():
    d = [0.1, 0.2, 0.15]
    ci = paired_interval(d, 0.95)
    half = stats.t.ppf(0.975, 2) * np.std(d, ddof=1) / np.sqrt(3)
    assert ci.hi - ci.mean == pytest.approx(half)


def test_two_seeds_essentially_cannot_show_equivalence():
    # Max passable sigma at n = 2 is 0.034 dB (proposal 4.5).
    assert not equivalent([0.0, 0.1])
    assert equivalent([0.0, 0.01, -0.01, 0.005, 0.0])


def test_equivalence_uses_the_90_percent_interval():
    d = np.array([0.05, 0.12, 0.08, 0.1, 0.07])
    ci90 = paired_interval(d, 0.90)
    assert equivalent(d) == (ci90.lo > -0.15 and ci90.hi < 0.15)


def test_classify_verdicts():
    assert classify([0.5, 0.6, 0.55, 0.52, 0.58])["verdict"] == "nonzero"
    assert classify([0.0, 0.01, -0.01, 0.005, 0.0])["verdict"] == "equivalent"
    assert classify([0.4, -0.3, 0.1])["verdict"] == "inconclusive"


def test_superiority_and_holm():
    assert superiority_p([0.3, 0.25, 0.35, 0.3]) < 0.01
    assert superiority_p([-0.3, -0.25, -0.35]) > 0.99
    assert holm([0.01, 0.04, 0.5]) == [True, False, False]


def test_gate_rows():
    eq = [0.0, 0.01, -0.01, 0.005, 0.0]
    pos = [0.8, 0.9, 0.85, 0.82, 0.88]
    assert gate(eq, pos, {"18.0": pos, "10.0": eq})["outcome"] == "beats plug-in energy"
    assert gate(pos, pos, {"18.0": pos})["outcome"] == "G consistent with P"
    assert gate(eq, eq, {"18.0": eq, "10.0": eq})["outcome"] == "decoder insensitive"
    assert gate([0.4, -0.3, 0.1], pos, {"18.0": pos})["outcome"] == "inconclusive"


def synthetic_fit(truth="dd", noise=0.05, seeds=5, images=400, seed=0):
    rng = np.random.default_rng(seed)
    curves = {
        "energy": np.array([-2.0, -3.5, -4.5]),
        "dd": np.array([-0.4, -1.0, -1.8]),
        "m2m4": np.array([0.0, 0.0, 0.0]),
    }
    shape = (seeds, images, 3)
    preds = {c: v + rng.normal(0, noise, shape) for c, v in curves.items()}
    r = curves[truth] + rng.normal(0, noise, shape) if truth in curves else rng.normal(3.0, noise, shape)
    return r, preds


def test_cue_fit_identifies_the_true_cue():
    r, preds = synthetic_fit("dd")
    out = cue_fit(r, preds, n_boot=300)
    assert out["outcome"] == "dd" and out["discriminable"]


def test_cue_fit_reports_tracks_none_when_no_curve_fits():
    r, preds = synthetic_fit("unlisted")
    assert cue_fit(r, preds, n_boot=200)["outcome"] == "tracks none"


def test_cue_fit_reports_not_discriminable_when_curves_coincide():
    rng = np.random.default_rng(1)
    shape = (5, 200, 3)
    same = np.zeros(3)
    preds = {"energy": same + rng.normal(0, 0.01, shape), "dd": same + rng.normal(0, 0.01, shape)}
    r = rng.normal(0, 0.5, shape)
    assert cue_fit(r, preds, n_boot=100)["outcome"] == "not discriminable"


def test_clearly_nonzero():
    assert clearly_nonzero([0.5, 0.6, 0.55])
    assert not clearly_nonzero([0.5, -0.6, 0.05])
