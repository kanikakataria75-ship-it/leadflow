"""The section 4 weighted confidence score.

"This is the single most important implementation note. The original
process was a **weighted confidence score**, not a set of binary filters."
Four inputs go into it here, not the spec's eleven, because Phase 3's
calibration (``research/aes_calibration.py``, written up in
``RESEARCH_AES.md``) measured all of them against 787 discovery-period
signals and found that most do not clear a defensible significance bar. The
four kept are the ones that did:

============================  ========  ===========================================
factor                        weight    evidence
============================  ========  ===========================================
fast resolution               0.40      the strongest signal in the whole project,
                                         reproduced under three independent
                                         constructions without being searched for
resistance absorption (§3.3)  0.25      p=0.037, n=787, cleanly monotonic
drawdown relative strength    0.20      directionally consistent across every run;
(§3.4)                                  significance itself is sample-size dependent
prior completed cycles (§0)   0.15      the spec's own central thesis; positive in
                                         both runs, significant in the larger one only
============================  ========  ===========================================

Everything else the spec's §4 table names -- box quality, higher lows, zone,
duration-vs-own-norm, timeframe confluence, resistance headroom as a
near-gate -- is still measured and reported on every signal (nothing is
deleted), but carries **zero weight** here, because none of it showed a
measurable relationship with net forward return at this sample size. That is
the literal, blunt answer the brief asked for: most of the spec's named
factors are noise on the data available so far. A future, larger sample may
revive some of them; this module does not pretend that has already happened.
"""

from __future__ import annotations

import warnings

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .params import RelativeStrengthParams, ScoreParams

if TYPE_CHECKING:
    from .signals import Signal


@dataclass(frozen=True, slots=True)
class ScoreResult:
    """One setup's conviction score, with the arithmetic behind it kept."""

    score: float                      # weighted average in [0, 1]
    bucket: str                       # "high_conviction" | "wait_and_watch" | "reject"
    subscores: dict[str, float]       # each factor's own [0, 1] contribution


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def score_setup(
    *,
    gap_days: float | None,
    resistance_absorbed_count: int,
    rs_capture: float | None,
    prior_cycles: int,
    params: ScoreParams | None = None,
    rs_params: RelativeStrengthParams | None = None,
) -> ScoreResult:
    """Score one setup from its already-measured Phase 1-3 factors.

    Args:
        gap_days: Calendar days from the screener admission that led to this
            setup to the breakout itself. ``None`` if unknown (e.g. scoring a
            setup before it has actually broken out -- in that case this
            factor contributes its minimum, since fast resolution is a
            property of *how* the breakout happened, not something to credit
            in advance).
        resistance_absorbed_count: From ``resistance.resistance_headroom``.
        rs_capture: Downside-capture ratio from
            ``relative_strength.drawdown_relative_strength``. ``None`` if
            there was insufficient data to measure it (contributes the
            midpoint -- absence of evidence is not evidence of fragility).
        prior_cycles: From ``boxes.NestedBox.prior_cycles``.
        params: Score weights and normalisation bounds.
        rs_params: Supplies the resilient/fragile capture bands, reused
            rather than re-specified.

    Returns:
        The score, its bucket, and the per-factor breakdown.
    """
    params = params or ScoreParams()
    rs_params = rs_params or RelativeStrengthParams()

    fast = 1.0 if (gap_days is not None and gap_days <= params.fast_resolution_days) else 0.0

    absorbed = _clip01(resistance_absorbed_count / params.absorbed_count_cap)

    if rs_capture is None:
        capture_score = 0.5
    else:
        span = rs_params.capture_fragile_min - rs_params.capture_resilient_max
        capture_score = _clip01((rs_params.capture_fragile_min - rs_capture) / span) if span else 0.5

    cycles = _clip01(prior_cycles / params.prior_cycles_cap)

    weights = {
        "fast_resolution": (params.w_fast_resolution, fast),
        "resistance_absorbed": (params.w_absorbed, absorbed),
        "rs_capture": (params.w_rs_capture, capture_score),
        "prior_cycles": (params.w_prior_cycles, cycles),
    }
    total_weight = sum(w for w, _ in weights.values())
    score = sum(w * v for w, v in weights.values()) / total_weight if total_weight else 0.0

    if score >= params.high_conviction_at_or_above:
        bucket = "high_conviction"
    elif score < params.reject_below:
        bucket = "reject"
    else:
        bucket = "wait_and_watch"

    return ScoreResult(score=score, bucket=bucket, subscores={k: v for k, (_, v) in weights.items()})


def _warn_if_default_weights(params: ScoreParams | None) -> None:
    """``ScoreParams()`` is the calibrated 0.40/0.25/0.20/0.15 scorer, REJECTED #4.

    Defaulting to it is the silent trap described in ``aes.locked``: it does not
    error, it just scores with weights this project rejected and moves `traded`
    from 551 to 405. Passing ``ScoreParams()`` explicitly is still allowed --
    only *omitting* it warns, because omission is what hides the choice.
    """
    if params is None:
        warnings.warn(
            "score params omitted: defaulting to the CALIBRATED 0.40/0.25/0.20/0.15 "
            "weights, which are REJECTED #4. The locked scorer is equal weight "
            "(aes.locked.LOCKED_SCORE_PARAMS). Pass it explicitly to silence this.",
            RuntimeWarning,
            stacklevel=3,
        )


def score_signal(
    signal: Signal, params: ScoreParams | None = None, rs_params: RelativeStrengthParams | None = None
) -> ScoreResult:
    """Score an already-built ``Signal`` (from ``signals`` or ``watchlist``).

    ``signal.breakout_date`` doubles as "entry trigger date" for every mode
    -- mode 3 fires before any breakout, but the field is still the bar the
    entry decision was made on, so the gap to admission is well-defined and
    causal for all three modes alike.
    """
    _warn_if_default_weights(params)
    gap_days = (signal.breakout_date - signal.admission_date).days
    return score_setup(
        gap_days=gap_days,
        resistance_absorbed_count=signal.resistance["resistance_absorbed_count"],
        rs_capture=signal.rel_strength.downside_capture,
        prior_cycles=signal.nested.prior_cycles,
        params=params,
        rs_params=rs_params,
    )


def attach_score(
    signal: Signal, params: ScoreParams | None = None, rs_params: RelativeStrengthParams | None = None
) -> Signal:
    """A copy of ``signal`` with its ``score``/``bucket`` fields populated.

    ``Signal`` is frozen, so this returns a new instance (``dataclasses.
    replace``) rather than mutating in place -- the same pattern used
    throughout ``aes/`` to keep every result immutable once built.
    """
    result = score_signal(signal, params, rs_params)
    return dataclasses.replace(signal, score=result.score, bucket=result.bucket)


__all__ = ["ScoreResult", "attach_score", "score_setup", "score_signal"]
