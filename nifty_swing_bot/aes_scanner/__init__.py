"""Production daily scanner for the Accumulation-Expansion Swing strategy.

Deliberately separate from ``research/``. This package reads only the settled
primitives in ``aes/`` -- the detector, the screener, the context features,
the scoring arithmetic and the render -- and never imports from ``research``,
so a research experiment cannot destabilise the tool that runs every evening.

What is settled and what is not, as of Phase 12 (see ``STATE.md``):

* **Settled and used here.** Box detection and the context features (Phases
  1-2); the persistent-watchlist state machine (Phase 3.6); the ATR 2.5x
  trailing stop with a 25-bar cap, which lowered drawdown and raised win rate
  in 7 of 7 folds; risk-per-share sizing; and the finding that edge is not
  concentrated in illiquid names, so the whole universe is scanned rather
  than a liquid subset.
* **Not settled, and therefore not relied on.** Whether the system makes
  money unsupervised. The scanner ranks and explains; it does not decide.
  Scoring is deliberately **equal-weight**, not the Phase 3 calibrated
  weights, which Phase 6 found overfit.
* **Breadth** is reported on every run because it is real at signal level and
  useful as context, but it is *not* used as a gate -- Phase 11/12 showed it
  is not separable from a single year and the one independent high-breadth
  year contradicted it.

Every run persists what it flagged, at what score, with which component
values, so a forward record accumulates without anyone having to remember to
write it down.
"""

from .store import AESScanStore
from .pipeline import ScanResult, run_scan, track_outcomes
from .sizing import TradePlan, plan_trade
from .breadth import BreadthReading, current_breadth

__all__ = [
    "AESScanStore",
    "BreadthReading",
    "ScanResult",
    "TradePlan",
    "current_breadth",
    "plan_trade",
    "run_scan",
    "track_outcomes",
]
