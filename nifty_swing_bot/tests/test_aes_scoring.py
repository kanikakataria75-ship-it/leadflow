"""Unit tests for the AES section 4 weighted confidence score."""

from __future__ import annotations

import pytest

from nifty_swing_bot.aes.params import RelativeStrengthParams, ScoreParams
from nifty_swing_bot.aes.scoring import score_setup


def test_every_factor_maxed_scores_a_perfect_one():
    result = score_setup(gap_days=5, resistance_absorbed_count=10, rs_capture=0.0, prior_cycles=5)
    assert result.score == pytest.approx(1.0)
    assert result.bucket == "high_conviction"
    assert all(v == pytest.approx(1.0) for v in result.subscores.values())


def test_every_factor_at_worst_scores_zero():
    result = score_setup(gap_days=999, resistance_absorbed_count=0, rs_capture=5.0, prior_cycles=0)
    assert result.score == pytest.approx(0.0)
    assert result.bucket == "reject"


def test_fast_resolution_is_binary_at_its_own_boundary():
    params = ScoreParams()
    at_boundary = score_setup(
        gap_days=params.fast_resolution_days, resistance_absorbed_count=0, rs_capture=None, prior_cycles=0
    )
    one_day_over = score_setup(
        gap_days=params.fast_resolution_days + 1, resistance_absorbed_count=0, rs_capture=None, prior_cycles=0
    )
    assert at_boundary.subscores["fast_resolution"] == 1.0
    assert one_day_over.subscores["fast_resolution"] == 0.0
    assert at_boundary.score > one_day_over.score


def test_unknown_gap_scores_the_fast_resolution_factor_at_its_minimum():
    """A setup that has not yet broken out cannot be credited for a fast
    resolution it has not had -- the factor describes how the breakout
    happened, not something to assume in advance."""
    result = score_setup(gap_days=None, resistance_absorbed_count=0, rs_capture=None, prior_cycles=0)
    assert result.subscores["fast_resolution"] == 0.0


def test_missing_relative_strength_scores_the_midpoint_not_a_penalty():
    """Absence of enough data to measure RS must not be treated as fragility."""
    # 1.0 happens to be the exact midpoint of the default 0.70-1.30 band, so
    # a genuinely fragile reading is used here to make "worse than unknown"
    # an unambiguous comparison.
    fragile = score_setup(gap_days=None, resistance_absorbed_count=0, rs_capture=1.30, prior_cycles=0)
    without_data = score_setup(gap_days=None, resistance_absorbed_count=0, rs_capture=None, prior_cycles=0)
    assert without_data.subscores["rs_capture"] == pytest.approx(0.5)
    assert without_data.score > fragile.score   # 1.30 (fragile) is worse than "unknown"


def test_rs_capture_reuses_the_relative_strength_bands_exactly():
    """The resilient/fragile boundary must match RelativeStrengthParams, not a second copy of it."""
    rs_params = RelativeStrengthParams(capture_resilient_max=0.5, capture_fragile_min=1.5)
    at_resilient_edge = score_setup(
        gap_days=None, resistance_absorbed_count=0, rs_capture=0.5, prior_cycles=0, rs_params=rs_params
    )
    at_fragile_edge = score_setup(
        gap_days=None, resistance_absorbed_count=0, rs_capture=1.5, prior_cycles=0, rs_params=rs_params
    )
    midpoint = score_setup(
        gap_days=None, resistance_absorbed_count=0, rs_capture=1.0, prior_cycles=0, rs_params=rs_params
    )
    assert at_resilient_edge.subscores["rs_capture"] == pytest.approx(1.0)
    assert at_fragile_edge.subscores["rs_capture"] == pytest.approx(0.0)
    assert midpoint.subscores["rs_capture"] == pytest.approx(0.5)


def test_rs_capture_beyond_the_bands_clips_rather_than_extrapolates():
    result = score_setup(gap_days=None, resistance_absorbed_count=0, rs_capture=-3.0, prior_cycles=0)
    assert result.subscores["rs_capture"] == pytest.approx(1.0)
    result2 = score_setup(gap_days=None, resistance_absorbed_count=0, rs_capture=10.0, prior_cycles=0)
    assert result2.subscores["rs_capture"] == pytest.approx(0.0)


def test_absorbed_and_cycles_are_capped_not_unbounded():
    params = ScoreParams()
    at_cap = score_setup(
        gap_days=None, resistance_absorbed_count=params.absorbed_count_cap,
        rs_capture=None, prior_cycles=params.prior_cycles_cap,
    )
    way_over = score_setup(
        gap_days=None, resistance_absorbed_count=params.absorbed_count_cap * 10,
        rs_capture=None, prior_cycles=params.prior_cycles_cap * 10,
    )
    assert at_cap.subscores["resistance_absorbed"] == pytest.approx(1.0)
    assert way_over.subscores["resistance_absorbed"] == pytest.approx(1.0)
    assert at_cap.score == pytest.approx(way_over.score)


def test_bucket_boundaries_match_score_params():
    params = ScoreParams()
    just_below_reject = score_setup(
        gap_days=None, resistance_absorbed_count=0, rs_capture=None, prior_cycles=0
    )
    assert just_below_reject.score < params.reject_below
    assert just_below_reject.bucket == "reject"

    top = score_setup(gap_days=1, resistance_absorbed_count=100, rs_capture=-10, prior_cycles=100)
    assert top.score >= params.high_conviction_at_or_above
    assert top.bucket == "high_conviction"


def test_weights_sum_to_a_normalised_average_not_an_unbounded_total():
    """The score must stay in [0, 1] regardless of how the weights are set,
    since they are not required to sum to exactly 1."""
    params = ScoreParams(w_fast_resolution=2.0, w_absorbed=3.0, w_rs_capture=0.5, w_prior_cycles=1.0)
    result = score_setup(
        gap_days=5, resistance_absorbed_count=10, rs_capture=0.0, prior_cycles=5, params=params
    )
    assert result.score == pytest.approx(1.0)
