"""The locked configuration, as executable assertions rather than prose.

Two dataclass defaults in this package do not match the configuration every
walk-forward result in this project was produced on, and both fail *silently*
-- they produce a different, already-rejected configuration and a plausible
number rather than an error:

1. ``ScoreParams()`` defaults to the calibrated 0.40/0.25/0.20/0.15 weights.
   Those are **REJECTED #4**: three of the four factors were selected on the
   sample they score, none holds its pooled sign in a majority of folds. The
   locked configuration is equal weight, 0.25 x 4. Taking the default moves
   ``traded`` from 551 to 405 and every fold with it.

2. ``PortfolioParams()`` defaults to ``atr_stop_mult=None`` and
   ``max_hold_bars=None`` -- the spec SMA(10) stop, uncapped. The locked
   configuration is ATR(14) x 2.5 plus a hard 25-bar cap (**PROVEN #1**, the
   one component with fold-by-fold evidence behind it). Taking the default
   reproduces §9.2's *spec* column, not §9.1: 2015/2016/2018/2019 come out at
   1.16 / 3.19 / -2.67 / -5.36, which are the spec-stop numbers to the
   decimal.

Both traps were hit while reproducing the V0 baseline in Phase 15. The defaults
are deliberately **not** changed here -- they are the spec's own values, tests
depend on them, and silently redefining them would trade one invisible
substitution for another. Instead, any code that intends to run the locked
configuration calls ``assert_locked()``, and any code that intends to deviate
must *name* the fields it is deviating on. Unnamed deviation raises.

    assert_locked(score=sp, portfolio=pp)                    # must be locked
    assert_locked(portfolio=pp, deviations={"max_open_positions"},
                  reason="REJECTED #9 re-measurement")       # deliberate
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Iterable

from .params import PortfolioParams, ScoreParams

__all__ = [
    "LOCKED_BOOK_PARAMS",
    "LOCKED_PORTFOLIO_PARAMS",
    "LOCKED_SCORE_PARAMS",
    "LockedConfigError",
    "assert_live_config",
    "assert_locked",
    "locked_book_params",
    "locked_portfolio_params",
    "locked_score_params",
]


class LockedConfigError(AssertionError):
    """Raised when a run would silently use a non-locked configuration."""


#: Equal weight, 0.25 x 4 -- not because it is better, but because it is not
#: fitted. See STATE.md UNPROVEN #3.
LOCKED_SCORE_PARAMS = ScoreParams(
    w_fast_resolution=0.25,
    w_absorbed=0.25,
    w_rs_capture=0.25,
    w_prior_cycles=0.25,
)

#: ATR(14) x 2.5 chandelier trailing stop plus the hard 25-bar cap.
LOCKED_PORTFOLIO_PARAMS = PortfolioParams(atr_stop_mult=2.5, max_hold_bars=25)


#: 30 / 50 / 20, **annual** rebalance, sleeve basis, gold never drawn on.
#:
#: The third silent default, found in Phase 21 while auditing
#: ``book_results_final.json``: ``BookParams()`` defaults to
#: ``rebalance="quarterly"``, and the frozen book is **annual**. The file
#: generated on the freeze date carries 28 rebalances over 2015-2021 and 19
#: over 2022-2026 -- quarterly counts. Annual would be 7 and 5. Nothing
#: raised, because this module guarded ``ScoreParams`` and ``PortfolioParams``
#: and did not guard the book at all.
def _locked_book():
    from ..book.params import BookParams

    return BookParams(
        w_strategy=0.30, w_arbitrage=0.50, w_gold=0.20,
        arb_annual_rate=0.065, rebalance="annual",
        strategy_capital_basis="sleeve", gold_is_drawable=False,
    )


LOCKED_BOOK_PARAMS = _locked_book()


def locked_book_params(**overrides: Any):
    """The locked book config, optionally with named overrides applied."""
    from dataclasses import replace

    return replace(LOCKED_BOOK_PARAMS, **overrides) if overrides else LOCKED_BOOK_PARAMS


def locked_score_params() -> ScoreParams:
    return LOCKED_SCORE_PARAMS


def locked_portfolio_params(**overrides: Any) -> PortfolioParams:
    """The locked portfolio config, optionally with named overrides applied."""
    from dataclasses import replace

    return replace(LOCKED_PORTFOLIO_PARAMS, **overrides) if overrides else LOCKED_PORTFOLIO_PARAMS


def _diff(actual: Any, expected: Any) -> dict[str, tuple[Any, Any]]:
    out: dict[str, tuple[Any, Any]] = {}
    for f in fields(expected):
        a, e = getattr(actual, f.name), getattr(expected, f.name)
        if a != e:
            out[f.name] = (a, e)
    return out


def _check(
    label: str,
    actual: Any,
    expected: Any,
    deviations: Iterable[str],
    reason: str,
    note: str,
) -> None:
    diff = _diff(actual, expected)
    allowed = set(deviations)
    unnamed = {k: v for k, v in diff.items() if k not in allowed}
    if not unnamed:
        return
    lines = [
        f"{label} is not the locked configuration, and the deviation was not declared.",
        "",
        note,
        "",
        "Undeclared differences (actual -> locked):",
    ]
    for k, (a, e) in sorted(unnamed.items()):
        lines.append(f"    {k}: {a!r}  ->  should be {e!r}")
    lines += [
        "",
        "If this is deliberate, name the fields and say why:",
        f"    assert_locked({label.split()[0].lower()}=p, "
        f"deviations={{{', '.join(repr(k) for k in sorted(unnamed))}}}, reason='...')",
    ]
    if reason:
        lines += ["", f"(declared reason for other deviations: {reason})"]
    raise LockedConfigError("\n".join(lines))


_SCORE_NOTE = (
    "ScoreParams() defaults to the calibrated 0.40/0.25/0.20/0.15 weights, which are\n"
    "REJECTED #4. The locked scorer is equal weight, 0.25 x 4. Using the default\n"
    "silently changes `traded` from 551 to 405 and every fold with it."
)
_BOOK_NOTE = (
    "BookParams() defaults to rebalance='quarterly'. The frozen book\n"
    "rebalances ANNUALLY. Using the default changes every book-level metric\n"
    "and is not visible in any output except the rebalance count. See Phase 21."
)
_PORTFOLIO_NOTE = (
    "PortfolioParams() defaults to the spec SMA(10) stop (atr_stop_mult=None) with no\n"
    "hold cap. The locked exit is ATR(14) x 2.5 plus a hard 25-bar cap (PROVEN #1).\n"
    "Using the default reproduces RESEARCH_AES.md §9.2's *spec* column, not §9.1."
)


def assert_locked(
    *,
    score: ScoreParams | None = None,
    portfolio: PortfolioParams | None = None,
    book: Any | None = None,
    deviations: Iterable[str] = (),
    reason: str = "",
) -> None:
    """Raise unless the given params are the locked configuration.

    Args:
        score: ``ScoreParams`` actually being used, if any.
        portfolio: ``PortfolioParams`` actually being used, if any.
        book: ``BookParams`` actually being used, if any.
        deviations: Field names this run intends to differ on. Anything that
            differs and is *not* named here raises.
        reason: Why the named deviations are intended. Required whenever
            ``deviations`` is non-empty, so the record says what was being
            measured rather than only what was changed.

    Raises:
        LockedConfigError: on any undeclared deviation, or on declared
            deviations with no stated reason.
    """
    deviations = set(deviations)
    if deviations and not reason:
        raise LockedConfigError(
            "assert_locked(deviations=...) requires a reason. Name what is being "
            "measured, e.g. reason='REJECTED #9 re-measurement under edge_vs_base'."
        )
    if score is not None:
        _check("Score config", score, LOCKED_SCORE_PARAMS, deviations, reason, _SCORE_NOTE)
    if book is not None:
        _check("Book config", book, LOCKED_BOOK_PARAMS, deviations, reason, _BOOK_NOTE)
    if portfolio is not None:
        _check(
            "Portfolio config", portfolio, LOCKED_PORTFOLIO_PARAMS,
            deviations, reason, _PORTFOLIO_NOTE,
        )

# --------------------------------------------------------------------------- #
# The live freeze fingerprint
# --------------------------------------------------------------------------- #
#: Every field of ``aes_scanner.live_config`` that was frozen on 2026-09-17,
#: written out as literals here rather than read from that module.
#:
#: The distinction from ``LOCKED_*_PARAMS`` above matters and is easy to miss:
#: those pin the **Phase-15 V0 research baseline** (3 slots, 1% risk, the spec
#: 5/8/20 ladder) so a walk-forward reproduces §9.1. ``FROZEN_LIVE`` pins the
#: **Phase-20 live configuration** (6 slots, 3% risk, variant (c)), which
#: deliberately deviates from the research baseline. A server must serve the
#: second; a replication must reproduce the first. Checking one against the
#: other is a category error, so they are separate constants.
#:
#: These literals are duplicated on purpose. If ``live_config.py`` is edited,
#: the duplication is what turns a silent change into a startup failure -- and
#: the forward record genuinely does restart at that point.
FROZEN_LIVE: dict[str, dict[str, Any]] = {
    "SCORING": {
        "w_fast_resolution": 0.25, "w_absorbed": 0.25,
        "w_rs_capture": 0.25, "w_prior_cycles": 0.25,
    },
    "PORTFOLIO": {
        "max_open_positions": 6, "target_concurrent_positions": 5.0,
        "risk_per_trade_pct": 0.03, "atr_stop_mult": 2.5, "atr_stop_period": 14,
        "max_hold_bars": 25,
        "size_mult_wait_and_watch": 0.7, "size_mult_high_conviction": 1.2,
        "ladder1_trigger_pct": 0.12, "ladder1_fraction": 0.50,
        "ladder2_trigger_pct": 0.20, "ladder2_fraction": 0.20,
        "ladder3_fraction": 0.0,
    },
    "BOOK": {
        "w_strategy": 0.30, "w_arbitrage": 0.50, "w_gold": 0.20,
        "arb_annual_rate": 0.065, "rebalance": "annual",
        "strategy_capital_basis": "sleeve", "gold_is_drawable": False,
    },
}

#: The freeze date, duplicated for the same reason as the values above.
FROZEN_ON_EXPECTED = "2026-09-17"


def assert_live_config() -> None:
    """Raise unless ``live_config`` still holds the frozen values.

    Called at process start by anything that serves or trades. A deviation here
    is not a bad setting to be corrected at runtime -- it means the forward
    record before this process and after it are different experiments, so the
    right response is to refuse to start and make someone record the change.

    Raises:
        LockedConfigError: on any deviation, listing every field that moved.
    """
    from ..aes_scanner import live_config as lc

    problems: list[str] = []

    if str(lc.FROZEN_ON) != FROZEN_ON_EXPECTED:
        problems.append(
            f"    FROZEN_ON: {lc.FROZEN_ON}  ->  should be {FROZEN_ON_EXPECTED}"
        )

    for name, expected in FROZEN_LIVE.items():
        actual = getattr(lc, name, None)
        if actual is None:
            problems.append(f"    live_config.{name} is missing entirely")
            continue
        for field, want in expected.items():
            got = getattr(actual, field, "<absent>")
            if got != want:
                problems.append(f"    {name}.{field}: {got!r}  ->  should be {want!r}")

    if problems:
        raise LockedConfigError(
            "\n".join([
                "The running configuration is not the one frozen on "
                f"{FROZEN_ON_EXPECTED}. Refusing to start.",
                "",
                "The forward record is only meaningful while this stays fixed. If the",
                "change is deliberate, update FROZEN_LIVE in aes/locked.py, move",
                "FORWARD_CHECK_FROM, and restart the forward record -- the old record",
                "does not carry across a config change.",
                "",
                "Deviations (actual -> frozen):",
                *problems,
            ])
        )
