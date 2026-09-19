"""Unit tests for the production daily scanner.

The scanner is the piece that runs unattended every evening, so the tests here
are weighted towards the things that would fail silently: sizing drifting away
from the backtest's own rule, the store losing a column, breadth being
computed on admission days rather than sessions, and the forward record
measuring outcomes by a looser rule than the one that was validated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nifty_swing_bot.aes.params import PortfolioParams
from nifty_swing_bot.aes_scanner.breadth import breadth_series, current_breadth
from nifty_swing_bot.aes_scanner.pipeline import EQUAL_WEIGHTS, run_scan, track_outcomes
from nifty_swing_bot.aes_scanner.sizing import plan_trade
from nifty_swing_bot.aes_scanner.store import AESScanStore


# --------------------------------------------------------------------------- #
# Sizing -- must agree with the backtest engine's own arithmetic
# --------------------------------------------------------------------------- #
def test_sizing_matches_the_backtest_rule():
    """risk_budget / risk_per_share * price, capped at the notional ceiling."""
    p = PortfolioParams()
    equity, close, atr14 = 1_000_000.0, 100.0, 4.0
    plan = plan_trade(close=close, atr14=atr14, big_box_bottom=50.0,
                      bucket="high_conviction", equity=equity, params=p)

    mult = p.size_mult_high_conviction
    assert plan.stop == pytest.approx(close - (p.atr_stop_mult or 2.5) * atr14)
    assert plan.risk_per_share == pytest.approx(10.0)

    risk_budget = equity * p.risk_per_trade_pct * mult
    cap = equity * (1.0 / p.target_concurrent_positions) * mult
    notional = min(risk_budget / plan.risk_per_share * close, cap)
    assert plan.qty == int(notional // close)
    assert plan.risk_amount == pytest.approx(plan.qty * plan.risk_per_share)


def test_the_binding_stop_is_the_higher_of_atr_and_box_floor():
    near = plan_trade(close=100.0, atr14=10.0, big_box_bottom=90.0,
                      bucket="wait_and_watch", equity=1_000_000.0)
    assert near.stop_basis == "big_box_bottom"      # box floor sits above ATR level
    far = plan_trade(close=100.0, atr14=2.0, big_box_bottom=50.0,
                     bucket="wait_and_watch", equity=1_000_000.0)
    assert far.stop_basis == "atr"


def test_minimum_risk_floor_prevents_an_oversized_position():
    """A coincidentally razor-thin stop must not imply an enormous position."""
    plan = plan_trade(close=100.0, atr14=0.05, big_box_bottom=10.0,
                      bucket="wait_and_watch", equity=1_000_000.0)
    assert plan.stop_basis == "min_risk_floor"
    assert plan.risk_per_share == pytest.approx(2.0)     # 2% of price


def test_reject_bucket_is_never_sized():
    plan = plan_trade(close=100.0, atr14=4.0, big_box_bottom=50.0,
                      bucket="reject", equity=1_000_000.0)
    assert plan.qty == 0 and plan.risk_amount == 0


def test_weak_regime_reduces_size_but_does_not_zero_it():
    strong = plan_trade(close=100.0, atr14=4.0, big_box_bottom=50.0,
                        bucket="high_conviction", equity=1_000_000.0, regime_weak=False)
    weak = plan_trade(close=100.0, atr14=4.0, big_box_bottom=50.0,
                      bucket="high_conviction", equity=1_000_000.0, regime_weak=True)
    assert 0 < weak.qty < strong.qty


# --------------------------------------------------------------------------- #
# Breadth
# --------------------------------------------------------------------------- #
def _admissions(pairs):
    return pd.DataFrame([{"symbol": s, "date": pd.Timestamp(d)} for s, d in pairs])


def test_breadth_counts_distinct_names_not_admission_events():
    adm = _admissions([("A", "2024-01-01"), ("A", "2024-01-02"), ("A", "2024-01-03")])
    s = breadth_series(adm, window=63)
    assert s.max() == 1          # one name, three events


def test_breadth_decays_over_quiet_sessions():
    """A name admitted once must leave the window, not persist forever."""
    adm = _admissions([("A", "2024-01-01"), ("B", "2024-06-03")])
    s = breadth_series(adm, window=5)
    assert s.loc["2024-01-01"] == 1
    assert s.loc["2024-03-01"] == 0      # well past the 5-session window


def test_current_breadth_reports_context_without_a_recommendation():
    dates = pd.bdate_range("2020-01-01", periods=400)
    adm = _admissions([(f"S{i%30}", d) for i, d in enumerate(dates)])
    r = current_breadth(adm, window=63)
    assert r is not None
    assert 0 <= r.percentile <= 100
    assert isinstance(r.label, str) and r.label
    assert "breadth" in r.summary


def test_current_breadth_handles_no_admissions():
    assert current_breadth(pd.DataFrame(columns=["symbol", "date"])) is None


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #
def _store(tmp_path):
    return AESScanStore(path=tmp_path / "s.sqlite")


def test_store_roundtrips_every_declared_column(tmp_path):
    st = _store(tmp_path)
    row = {"symbol": "AAA", "bucket": "high_conviction", "score": 0.8, "actionable": 1,
           "box_bars": 20, "box_range_pct": 18.0, "sc_fast": 1.0, "sc_absorbed": 0.5,
           "sc_rs": 1.0, "sc_cycles": 0.5, "close": 100.0, "entry_ref": 100.0,
           "stop": 90.0, "stop_basis": "atr", "qty": 100, "risk_amount": 1000.0,
           "admission_gap_days": 12.0}
    st.record_scan("2026-01-05", universe_size=400, watchlist_size=40,
                   candidate_count=1, actionable_count=1, breadth=50)
    assert st.add_candidates([row], "2026-01-05") == 1
    got = st.candidates("2026-01-05")
    assert len(got) == 1
    assert got.loc[0, "symbol"] == "AAA"
    assert got.loc[0, "stop_basis"] == "atr"
    assert got.loc[0, "admission_gap_days"] == 12.0
    assert st.latest_scan_date() == "2026-01-05"


def test_rerunning_a_scan_replaces_that_day_rather_than_duplicating(tmp_path):
    st = _store(tmp_path)
    st.add_candidates([{"symbol": "AAA", "bucket": "reject", "score": 0.1}], "2026-01-05")
    st.add_candidates([{"symbol": "AAA", "bucket": "high_conviction", "score": 0.9}], "2026-01-05")
    got = st.candidates("2026-01-05")
    assert len(got) == 1 and got.loc[0, "score"] == 0.9


def test_only_actionable_rows_enter_the_forward_record(tmp_path):
    st = _store(tmp_path)
    st.add_candidates([
        {"symbol": "AAA", "bucket": "high_conviction", "score": 0.9, "actionable": 1},
        {"symbol": "BBB", "bucket": "wait_and_watch", "score": 0.5, "actionable": 0},
    ], "2026-01-05")
    rec = st.forward_record()
    assert list(rec["symbol"]) == ["AAA"]


# --------------------------------------------------------------------------- #
# End to end, on synthetic data
# --------------------------------------------------------------------------- #
def _series(n=420, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0006, 0.015, n)))
    return pd.DataFrame(
        {"open": close, "high": close * 1.012, "low": close * 0.988,
         "close": close, "volume": 5_000_000.0}, index=idx)


def test_run_scan_is_stable_on_data_with_no_setups(tmp_path):
    frames = {f"S{i}": _series(seed=i) for i in range(6)}
    index_df = _series(seed=99)
    res = run_scan(frames=frames, index_df=index_df, render=False,
                   store=_store(tmp_path), persist=True)
    assert res.universe_size == 6
    assert isinstance(res.candidates, pd.DataFrame)
    assert res.duration_s >= 0


def test_scoring_uses_equal_weights_not_the_calibrated_ones():
    """The calibrated weights are overfit (Phase 6); the tool must not use them."""
    w = EQUAL_WEIGHTS
    assert w.w_fast_resolution == w.w_absorbed == w.w_rs_capture == w.w_prior_cycles == 0.25


def test_track_outcomes_replays_the_adopted_exit(tmp_path):
    """A position that runs straight up must close on the 25-bar cap, not earlier."""
    st = _store(tmp_path)
    n = 60
    idx = pd.bdate_range(pd.Timestamp.today().normalize() - pd.Timedelta(days=90), periods=n)
    close = np.linspace(100, 160, n)
    df = pd.DataFrame({"open": close, "high": close * 1.005, "low": close * 0.995,
                       "close": close, "volume": 1e6}, index=idx)
    st.add_candidates([{"symbol": "UP", "bucket": "high_conviction", "score": 0.9,
                        "actionable": 1, "entry_ref": float(close[0]), "stop": 90.0,
                        "atr14": 1.0}], idx[0].strftime("%Y-%m-%d"))
    assert track_outcomes(st, {"UP": df}) == 1
    rec = st.forward_record()
    assert rec.loc[0, "exit_reason"] == "max_hold"
    assert rec.loc[0, "bars_elapsed"] == 25
    assert rec.loc[0, "realised_pct"] > 0


def test_track_outcomes_records_a_trailing_stop_exit(tmp_path):
    """Up then sharply down must stop out on the trail, below the peak."""
    st = _store(tmp_path)
    idx = pd.bdate_range(pd.Timestamp.today().normalize() - pd.Timedelta(days=90), periods=60)
    close = np.concatenate([np.linspace(100, 130, 12), np.linspace(129, 80, 48)])
    df = pd.DataFrame({"open": close, "high": close * 1.005, "low": close * 0.995,
                       "close": close, "volume": 1e6}, index=idx)
    st.add_candidates([{"symbol": "DN", "bucket": "high_conviction", "score": 0.9,
                        "actionable": 1, "entry_ref": float(close[0]), "stop": 90.0,
                        "atr14": 1.0}], idx[0].strftime("%Y-%m-%d"))
    track_outcomes(st, {"DN": df})
    rec = st.forward_record()
    assert rec.loc[0, "exit_reason"] == "atr_trail_stop"
    assert rec.loc[0, "stop_hit"] == 1
    assert rec.loc[0, "mfe_pct"] > 0        # it was up before it broke


# --------------------------------------------------------------------------- #
# Universe audit -- every symbol must get a verdict and a reason
# --------------------------------------------------------------------------- #
def test_audit_accounts_for_every_symbol():
    """No symbol may vanish: the audit is the check on the silent 95%."""
    from nifty_swing_bot.aes_scanner.audit import audit_universe

    frames = {f"S{i}": _series(seed=i) for i in range(8)}
    index_df = _series(seed=99)
    res = run_scan(frames=frames, index_df=index_df, render=False, persist=False, audit=False)
    a = audit_universe(frames, index_df, res.candidates)
    assert len(a) == len(frames)
    assert set(a["symbol"]) == set(frames)
    assert a["reason"].str.len().gt(0).all()
    assert a["verdict"].isin(["ACCEPTED", "WAIT & WATCH", "REJECTED"]).all()


def test_audit_diagnosis_agrees_with_the_detector():
    """``diagnose_big_box`` duplicates the anchor loop; it must not diverge."""
    from nifty_swing_bot.aes.boxes import detect_big_box
    from nifty_swing_bot.aes.params import BoxParams
    from nifty_swing_bot.aes_scanner.audit import diagnose_big_box

    bp = BoxParams()
    for seed in range(12):
        df = _series(n=300, seed=seed)
        i = len(df) - 1
        found = detect_big_box(df, i, bp) is not None
        diagnosed = diagnose_big_box(df, i, bp).reason == "a valid box exists"
        assert found == diagnosed, f"seed {seed}: detector={found} diagnosis={diagnosed}"


def test_audit_reason_names_the_binding_constraint():
    """A flat series fails the screener on the move test, and must say so."""
    from nifty_swing_bot.aes_scanner.audit import audit_universe

    flat = _series(seed=3)
    flat.loc[:, ["open", "high", "low", "close"]] = 100.0
    frames = {"FLAT": flat}
    a = audit_universe(frames, _series(seed=99), pd.DataFrame())
    assert len(a) == 1
    assert a.loc[0, "stage"] == "never_admitted"
    assert "10 sessions" in a.loc[0, "reason"]


def test_run_scan_persists_the_audit(tmp_path):
    st = _store(tmp_path)
    frames = {f"S{i}": _series(seed=i) for i in range(5)}
    run_scan(frames=frames, index_df=_series(seed=99), render=False,
             store=st, persist=True, audit=True)
    a = st.audit()
    assert len(a) == len(frames)
    assert a["verdict"].notna().all()


def test_cli_and_report_modules_import():
    """The CLI is the production entry point and nothing else imports it.

    Without this, a syntax error in ``cli.py`` passes the whole suite and only
    surfaces when the nightly job fails to start.
    """
    import importlib

    for mod in ("nifty_swing_bot.aes_scanner.cli",
                "nifty_swing_bot.aes_scanner.report",
                "nifty_swing_bot.aes_scanner.audit",
                "nifty_swing_bot.api.aes_routes"):
        assert importlib.import_module(mod) is not None


def test_cli_parses_its_flags():
    from nifty_swing_bot.aes_scanner.cli import main

    with pytest.raises(SystemExit):
        main(["--help"])
