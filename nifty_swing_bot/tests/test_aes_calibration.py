"""Unit tests for the AES Phase 3/4 calibration arithmetic.

These test the pure reporting functions against hand-built signal frames with
a known answer -- not the full screener-to-signal pipeline, which is already
covered end-to-end by ``test_aes_signals.py``. The point here is that the
sweep, threshold and edge-vs-baseline arithmetic is correct, independent of
where the signals came from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nifty_swing_bot.research.aes_calibration import (
    bar_level_edge_report,
    circuit_threshold_sweep,
    volume_multiplier_sweep,
)


def _signals(vol_ratios, net_20):
    n = len(vol_ratios)
    return pd.DataFrame({
        "vol_ratio": vol_ratios,
        "fwd_net_20": net_20,
        "circuit_hits_60d": np.zeros(n, dtype=int),
    })


def test_volume_sweep_buckets_by_realised_ratio():
    """Each signal's realised ratio, not a re-triggering threshold, decides its bucket."""
    d = _signals(
        vol_ratios=[0.5, 0.6, 0.7, 2.2, 2.3, 2.4, 5.0, 6.0],
        net_20=[0.01, 0.02, 0.00, 0.05, 0.04, 0.06, -0.02, -0.01],
    )
    out = volume_multiplier_sweep(d, horizon=20).set_index("bucket")
    assert out.loc["<1.0x", "n"] == 3
    assert out.loc["2.0-2.5x", "n"] == 3
    assert out.loc["3.0x+", "n"] == 2
    assert out.loc["2.0-2.5x", "mean_net_pct"] == pytest.approx(5.0, rel=1e-2)


def test_volume_sweep_small_bucket_reports_n_without_stats():
    """A bucket with under 3 signals should not report a fabricated t-stat."""
    d = _signals(vol_ratios=[0.5, 0.6, 4.0], net_20=[0.01, 0.02, 0.03])
    out = volume_multiplier_sweep(d, horizon=20).set_index("bucket")
    assert out.loc["3.0x+", "n"] == 1
    assert pd.isna(out.loc["3.0x+"].get("mean_net_pct", np.nan))


def test_circuit_sweep_counts_are_monotonic_and_partition_correctly():
    d = pd.DataFrame({
        "circuit_hits_60d": [0, 0, 1, 2, 3, 5],
        "fwd_net_20": [0.01, 0.02, -0.01, -0.02, -0.03, -0.05],
    })
    out = circuit_threshold_sweep(d, horizon=20).set_index("n_threshold")
    # N>0 excludes the 5 signals with 1+ hits; N>3 excludes only the single
    # 5-hit signal.
    assert out.loc[0, "excluded_n"] == 4
    assert out.loc[3, "excluded_n"] == 1
    assert out.loc[0, "kept_n"] + out.loc[0, "excluded_n"] == len(d)
    # Excluded-group mean must average exactly the excluded rows.
    assert out.loc[0, "excluded_mean_net_pct"] == pytest.approx(
        d.loc[d.circuit_hits_60d > 0, "fwd_net_20"].mean() * 100
    )


def test_circuit_sweep_handles_an_empty_excluded_group():
    d = pd.DataFrame({"circuit_hits_60d": [0, 0, 0], "fwd_net_20": [0.01, 0.02, 0.03]})
    out = circuit_threshold_sweep(d, horizon=20).set_index("n_threshold")
    assert out.loc[0, "excluded_n"] == 0
    assert pd.isna(out.loc[0, "excluded_mean_net_pct"])


def test_bar_level_edge_report_matches_hand_computed_excess():
    """Net excess = (signal's index-relative mean - base rate mean) - cost."""
    signals = pd.DataFrame({"excess_10": [0.02, 0.03, 0.01, 0.04]})   # mean = 0.025
    base_rate = {10: pd.Series([0.00, 0.005, -0.005, 0.01, 0.0])}     # mean = 0.002
    out = bar_level_edge_report(signals, base_rate, cost_pct=0.5).set_index("horizon")

    gross = 0.025 - 0.002
    assert out.loc[10, "gross_excess_pct"] == pytest.approx(gross * 100, rel=1e-6)
    assert out.loc[10, "net_excess_pct"] == pytest.approx(gross * 100 - 0.5, rel=1e-6)
    assert out.loc[10, "n_signals"] == 4
    assert out.loc[10, "n_base_bars"] == 5


def test_bar_level_edge_report_t_stat_uses_signal_dispersion():
    """The t-stat is computed on the signal group's own std, matching signal_lab.py."""
    signals = pd.DataFrame({"excess_5": [0.1, -0.1, 0.1, -0.1, 0.1, -0.1]})   # mean ~0, high spread
    base_rate = {5: pd.Series([0.0] * 100)}
    out = bar_level_edge_report(signals, base_rate, cost_pct=0.0).set_index("horizon")
    assert abs(out.loc[5, "t"]) < 1.0    # noisy, near-zero mean -> weak t despite n=6


def test_bar_level_edge_report_drops_nan_excess_rows():
    signals = pd.DataFrame({"excess_10": [0.02, np.nan, 0.04, np.nan]})
    base_rate = {10: pd.Series([0.0, 0.0])}
    out = bar_level_edge_report(signals, base_rate, cost_pct=0.0).set_index("horizon")
    assert out.loc[10, "n_signals"] == 2
    assert out.loc[10, "aes_idxrel_pct"] == pytest.approx(3.0, rel=1e-6)
