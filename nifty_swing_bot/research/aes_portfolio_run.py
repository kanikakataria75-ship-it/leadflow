"""Phase 5: run the AES portfolio backtest over AES_DISCOVERY.

Rebuilds the same persistent-watchlist signals Phase 3/4 calibrated on
(``discovery_signals_raw``), scores each with the calibrated section 4
weights (``scoring.attach_score``), and runs them through
``AESPortfolioBacktester`` -- section 6 exits, section 7 sizing, section 8
regime handling -- on a single shared account.

Usage::

    python -m nifty_swing_bot.research.aes_portfolio_run
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pandas as pd

from ..aes.params import PortfolioParams
from ..aes.portfolio import AESPortfolioBacktester
from ..aes.protocol import describe
from ..aes.scoring import attach_score
from ..config import get_config
from .aes_calibration import discovery_signals_raw, load_universe

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(describe())
    cfg = get_config()

    t0 = time.time()
    frames = load_universe(cfg.paths.ohlcv_dir)
    index_df = frames.pop("CRSLDX", None)
    if index_df is None:
        index_df = pd.read_parquet(os.path.join(str(cfg.paths.ohlcv_dir), "CRSLDX.parquet"))
    print(f"Loaded {len(frames)} symbols in {time.time() - t0:.0f}s")

    t0 = time.time()
    signals = discovery_signals_raw(frames, index_df)
    scored = [attach_score(s) for s in signals]
    print(f"Built and scored {len(scored)} signals in {time.time() - t0:.0f}s")
    from collections import Counter
    print("  bucket counts:", dict(Counter(s.bucket for s in scored)))

    bt = AESPortfolioBacktester(params=PortfolioParams())
    t0 = time.time()
    result = bt.run(frames, index_df, scored, min_bucket="wait_and_watch")
    print(f"Backtest complete in {time.time() - t0:.0f}s")

    print("\n=== Rejected entries ===")
    print(result.rejected)

    print("\n=== Stats ===")
    for k, v in result.stats.items():
        if k == "exit_reasons":
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")

    out_dir = Path(cfg.paths.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.trades_frame.to_csv(out_dir / "aes_portfolio_trades.csv", index=False)
    result.equity_curve.to_csv(out_dir / "aes_portfolio_equity.csv")
    print(f"\nSaved trades and equity curve to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
