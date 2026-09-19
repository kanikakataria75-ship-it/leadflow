"""The LeadFlow terminal API. The only backend the UI talks to.

This replaces the DAB/MR router. That router served a different engine: its
``/api/signals`` read ``cache/terminal.db`` (the DAB ledger), its
``/api/stock/{symbol}`` computed ``strategy.dab_strategy`` delivery features,
and its ``/api/backtest/*`` read DAB's ``stats_*.json``. None of that is
LeadFlow, and none of it is reachable from here -- the DAB store is not opened
by any endpoint in this module.

Honesty requirements are enforced at the API boundary rather than left to the
frontend, because a page can forget and a payload cannot: every window carries
``provenance``, every headline return ships with its ``ex_best_fold`` twin, and
position R is served under a name that says what it is with the row-mean figure
attached as a labelled warning.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from ..aes.locked import assert_live_config
from ..aes_scanner import live_config as LC
from ..config import get_config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

_cache: dict[str, Any] = {}


def _clean(o: Any) -> Any:
    """NaN and inf are not JSON; pandas produces both freely."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, date)):
        return str(o)
    return o


def ok(payload: Any) -> JSONResponse:
    return JSONResponse(content=_clean(payload))


def _results_dir() -> str:
    return str(get_config().paths.results_dir)


def _sector_map() -> dict[str, dict[str, str]]:
    """symbol -> {company, industry}, from the cached index constituents.

    The scan store does not populate ``company`` or ``industry`` -- both come
    back None -- so without this join the scanner shows a column of dashes and,
    worse, the sector map's colour channel silently reads zero signal density
    everywhere. Cached for the process lifetime; the constituent files change
    at index-review cadence, not intraday.
    """
    if "_sectors" in _cache:
        return _cache["_sectors"]
    import glob

    out: dict[str, dict[str, str]] = {}
    for p in glob.glob(os.path.join(str(get_config().paths.cache_dir), "universe", "*.parquet")):
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        for r in df.itertuples():
            sym = str(getattr(r, "symbol", "") or "")
            if sym and sym not in out:
                out[sym] = {
                    "company": str(getattr(r, "company", "") or ""),
                    "industry": str(getattr(r, "industry", "") or ""),
                }
    _cache["_sectors"] = out
    return out


def _enrich(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fill company/industry from the constituent map where the scan left them null."""
    sectors = _sector_map()
    for r in rows:
        meta = sectors.get(str(r.get("symbol") or ""))
        if not meta:
            continue
        if not r.get("company"):
            r["company"] = meta["company"] or None
        if not r.get("industry"):
            r["industry"] = meta["industry"] or None
    return rows


def _load(name: str) -> dict[str, Any]:
    if name not in _cache:
        p = os.path.join(_results_dir(), name)
        if not os.path.exists(p):
            raise HTTPException(status_code=503, detail=f"{name} not built yet")
        with open(p, encoding="utf-8") as fh:
            _cache[name] = json.load(fh)
    return _cache[name]


# --------------------------------------------------------------------------- #
# Config / provenance
# --------------------------------------------------------------------------- #
@router.get("/config")
def config() -> JSONResponse:
    """The frozen configuration, read-only, plus the forward-record clock."""
    from ..aes_scanner.store import AESScanStore

    try:
        store = AESScanStore()
        rec = store.forward_record()
        n_rec = int(len(rec))
        since = str(rec["scan_date"].min()) if n_rec and "scan_date" in rec else None
    except Exception:
        n_rec, since = 0, None

    frozen_on = str(LC.FROZEN_ON)
    days = (date.today() - LC.FROZEN_ON).days
    return ok({
        "product": "LeadFlow",
        "frozen_on": frozen_on,
        "forward_record_days": days,
        "forward_record_entries": n_rec,
        "forward_record_since": since,
        "describe": LC.describe(),
        "warning": (
            "Changing any frozen value restarts the forward record. The record "
            "before a change and after it are not the same experiment."
        ),
        "sizing": {
            "slots": LC.PORTFOLIO.max_open_positions,
            "risk_per_trade_pct": LC.PORTFOLIO.risk_per_trade_pct * 100,
            "conviction_multipliers": [LC.PORTFOLIO.size_mult_wait_and_watch,
                                       LC.PORTFOLIO.size_mult_high_conviction],
        },
        "exits": {
            "atr_period": LC.PORTFOLIO.atr_stop_period,
            "atr_mult": LC.PORTFOLIO.atr_stop_mult,
            "max_hold_bars": LC.PORTFOLIO.max_hold_bars,
            "ladder": [{"trigger_pct": t * 100, "fraction_pct": f * 100}
                       for t, f in LC.TRANCHES],
            "remainder_trails_pct": (1 - sum(f for _, f in LC.TRANCHES)) * 100,
            "ladder_name": "variant (c)",
            "ladder_status": "FORWARD-TEST CANDIDATE",
            "ladder_tradeoff": (
                "Variant (c) wins mean return in both windows but concentrates: "
                "best-fold share 48.5%->66.6% and 49.7%->69.0%, and ex-best-fold "
                "return falls 5.86->4.63 and 9.73->7.66. Best paired t = 1.98."
            ),
        },
        "book": {
            "w_strategy": LC.BOOK.w_strategy, "w_arbitrage": LC.BOOK.w_arbitrage,
            "w_gold": LC.BOOK.w_gold, "rebalance": LC.BOOK.rebalance,
            "arb_annual_rate": LC.BOOK.arb_annual_rate,
            "strategy_capital_basis": LC.BOOK.strategy_capital_basis,
            "gold_is_drawable": LC.BOOK.gold_is_drawable,
        },
        "scoring": {"weights": {"fast_resolution": LC.SCORING.w_fast_resolution,
                                "absorbed": LC.SCORING.w_absorbed,
                                "rs_capture": LC.SCORING.w_rs_capture,
                                "prior_cycles": LC.SCORING.w_prior_cycles},
                    "note": "equal weight 0.25 x 4; the calibrated weights are REJECTED #4"},
    })


# --------------------------------------------------------------------------- #
# Backtest / book
# --------------------------------------------------------------------------- #
def _headline(side: dict[str, Any], conc: dict[str, Any]) -> dict[str, Any]:
    """A headline return never travels without its ex-best-fold twin."""
    return {
        "total_return_pct": side.get("total_return_pct"),
        "cagr_pct": side.get("cagr_pct"),
        "ex_best_fold_pct": conc.get("ex_best_period"),
        "best_fold_share_pct": conc.get("best_period_share_pct"),
        "mean_fold_pct": conc.get("mean"),
        "median_fold_pct": conc.get("median"),
        "positive_folds": conc.get("positive"),
    }


@router.get("/results")
def results() -> JSONResponse:
    """Both windows, both ladders, full metric set, honestly labelled."""
    d = _load("leadflow_frozen_results.json")
    out: dict[str, Any] = {"generated": d["generated"], "frozen_on": d["frozen_on"],
                           "config": d["config"], "windows": {}}
    for wname, w in d["windows"].items():
        lad = {}
        for lname, lv in w["ladders"].items():
            strat, book = lv["strategy"], lv["book"]
            lad[lname] = {
                "params": lv["params"],
                "n_positions": lv["n_positions"],
                "n_traded_rows": lv["n_traded_rows"],
                "strategy": strat,
                "book": book,
                "strategy_headline": _headline(strat, lv["strategy_concentration"]),
                "book_headline": _headline(book, lv["book_concentration"]),
                "fold_years": lv["fold_years"],
                "strategy_folds_pct": lv["strategy_folds_pct"],
                "book_folds_pct": lv["book_folds_pct"],
                "position_r": {
                    "value": strat.get("avg_r_multiple"),
                    "basis": "POSITION level (ladder rows collapsed)",
                    "row_mean_do_not_use": strat.get("avg_r_multiple_row_mean_DO_NOT_USE"),
                    "why": ("A profit ladder splits one position into 2-4 rows and every "
                            "ladder row is profitable by construction, so the row mean is "
                            "inflated. Quote the position figure."),
                },
                "diagnostics": lv["diagnostics"],
                "gold_correlation_by_year": lv["gold_correlation_by_year"],
            }
        out["windows"][wname] = {
            "provenance": w["provenance"].upper(),
            "provenance_note": ("Nothing in this project is out-of-sample validated. "
                                "2015-2021 was the discovery window; 2022-2026 was spent "
                                "as discovery by Phase 16."),
            "start": w["start"], "end": w["end"], "signals": w["signals"],
            "ladders": lad,
        }
    return ok(out)


@router.get("/series/{window}")
def series(window: str) -> JSONResponse:
    """Daily series for one window: equity, benchmark, drawdown, sleeves, trades."""
    d = _load("leadflow_series.json")
    if window not in d["windows"]:
        raise HTTPException(404, f"unknown window {window!r}; have {list(d['windows'])}")
    w = dict(d["windows"][window])
    w["reconciliation"] = d["reconciliation"].get(window, {}).get("all_match")
    w["ladder"] = d["ladder"]
    return ok(w)


@router.get("/windows")
def windows() -> JSONResponse:
    d = _load("leadflow_frozen_results.json")
    return ok([{"name": k, "provenance": v["provenance"].upper(),
                "start": v["start"], "end": v["end"], "signals": v["signals"]}
               for k, v in d["windows"].items()])


# --------------------------------------------------------------------------- #
# Live scanner
# --------------------------------------------------------------------------- #
@router.get("/scan/latest")
def scan_latest(scan_date: str | None = Query(default=None),
                limit: int = Query(default=400, ge=1, le=2000)) -> JSONResponse:
    from ..aes_scanner.store import AESScanStore

    store = AESScanStore()
    d = scan_date or store.latest_scan_date()
    if d is None:
        return ok({"scan_date": None, "candidates": [], "rejected": []})
    cand = store.candidates(scan_date=d, limit=limit)
    audit = store.audit(scan_date=d)
    return ok({
        "scan_date": d,
        "candidates": _enrich(cand.to_dict("records")) if not cand.empty else [],
        "rejected": _enrich(audit.to_dict("records")) if not audit.empty else [],
    })


@router.get("/scan/history")
def scan_history(limit: int = Query(default=120, ge=1, le=500)) -> JSONResponse:
    from ..aes_scanner.store import AESScanStore

    s = AESScanStore().scans(limit=limit)
    return ok(s.to_dict("records") if not s.empty else [])


@router.get("/record")
def forward_record(since: str | None = Query(default=None)) -> JSONResponse:
    """The live forward record. Starts at FROZEN_ON; nothing before it counts."""
    from ..aes_scanner.store import AESScanStore

    rec = AESScanStore().forward_record(since=since)
    rows = rec.to_dict("records") if not rec.empty else []
    closed = [r for r in rows if r.get("status") == "closed"] if rows else []
    wins = [r for r in closed if (r.get("r_multiple") or 0) > 0]
    return ok({
        "frozen_on": str(LC.FROZEN_ON),
        "note": ("Results from any earlier engine are a different strategy and are "
                 "not carried forward into this record."),
        "entries": rows,
        "n_total": len(rows),
        "n_open": len(rows) - len(closed),
        "n_closed": len(closed),
        "win_rate_pct": (len(wins) / len(closed) * 100) if closed else None,
        "mean_r": (float(np.mean([r.get("r_multiple") or 0 for r in closed]))
                   if closed else None),
    })


@router.get("/breadth")
def breadth() -> JSONResponse:
    """Current regime and breadth context."""
    from ..aes_scanner.store import AESScanStore

    try:
        store = AESScanStore()
        d = store.latest_scan_date()
        scans = store.scans(limit=200)
        if scans.empty:
            return ok({"available": False})
        row = scans.iloc[0].to_dict()
        return ok({"available": True, "scan_date": d, "latest": row,
                   "history": scans.to_dict("records")})
    except Exception as exc:                                    # pragma: no cover
        logger.warning("breadth unavailable: %s", exc)
        return ok({"available": False})


@router.get("/stock/{symbol}")
def stock(symbol: str, bars: int = Query(default=260, ge=60, le=1200),
          equity: float = Query(default=1_000_000.0, gt=0)) -> JSONResponse:
    """Price, the detected box, the score breakdown and the trade plan.

    The trade plan is priced through ``aes_scanner.sizing.plan_trade`` under the
    frozen ``PortfolioParams``, so the levels shown here are the levels the
    backtest would have used -- not a second, prettier calculation.
    """
    from dataclasses import asdict

    from ..aes_scanner.sizing import plan_trade
    from ..aes_scanner.store import AESScanStore
    from ..strategy.indicators import atr

    cfg = get_config()
    path = os.path.join(str(cfg.paths.ohlcv_dir), f"{symbol}.parquet")
    if not os.path.exists(path):
        raise HTTPException(404, f"no price data for {symbol!r}")
    df = pd.read_parquet(path).tail(bars)
    if df.empty:
        raise HTTPException(404, f"empty series for {symbol!r}")

    a = atr(df["high"], df["low"], df["close"], LC.PORTFOLIO.atr_stop_period)
    atr14 = float(a.iloc[-1]) if np.isfinite(a.iloc[-1]) else float("nan")
    close = float(df["close"].iloc[-1])

    # the most recent scan row for this name, if it was surfaced
    cand: dict[str, Any] | None = None
    try:
        store = AESScanStore()
        d = store.latest_scan_date()
        if d:
            c = store.candidates(scan_date=d, limit=2000)
            if not c.empty:
                hit = c[c["symbol"] == symbol]
                if not hit.empty:
                    cand = _enrich([hit.iloc[0].to_dict()])[0]
    except Exception as exc:                                    # pragma: no cover
        logger.warning("candidate lookup failed for %s: %s", symbol, exc)

    box = None
    plan = None
    if cand is not None:
        box = {
            "top": cand.get("box_top"), "bottom": cand.get("box_bottom"),
            "bars": cand.get("box_bars"), "range_pct": cand.get("box_range_pct"),
            "date": cand.get("box_date"),
        }
        floor = cand.get("box_bottom")
        if floor is not None and np.isfinite(atr14):
            tp = plan_trade(
                close=close, atr14=atr14, big_box_bottom=float(floor),
                bucket=str(cand.get("bucket") or "wait_and_watch"),
                equity=equity, params=LC.PORTFOLIO,
            )
            plan = asdict(tp)
            plan["tranches"] = [
                {"price": t[0], "qty": t[1], "trigger_pct": t[2] * 100} for t in tp.tranches
            ]

    return ok({
        "symbol": symbol,
        "company": (cand or {}).get("company"),
        "industry": (cand or {}).get("industry"),
        "dates": [str(d.date()) for d in df.index],
        "close": df["close"].astype(float).tolist(),
        "high": df["high"].astype(float).tolist(),
        "low": df["low"].astype(float).tolist(),
        "volume": df["volume"].astype(float).tolist(),
        "atr14": atr14,
        "last_close": close,
        "in_latest_scan": cand is not None,
        "candidate": cand,
        "box": box,
        "plan": plan,
        "equity_basis": equity,
    })


@router.get("/universe")
def universe() -> JSONResponse:
    """Sector composition of the traded universe, for the market visual.

    Area encodes how much of the universe a sector is; the caller colours it by
    live signal density from the latest scan. Both are real quantities -- the
    visual is not decoration.
    """
    import glob

    cfg = get_config()
    frames: list[pd.DataFrame] = []
    for p in glob.glob(os.path.join(str(cfg.paths.cache_dir), "universe", "*.parquet")):
        try:
            frames.append(pd.read_parquet(p)[["symbol", "company", "industry"]])
        except Exception:
            continue
    if not frames:
        return ok({"available": False, "sectors": [], "n": 0})
    u = pd.concat(frames, ignore_index=True).drop_duplicates("symbol")

    # live signal density per sector, from the latest scan
    density: dict[str, dict[str, Any]] = {}
    try:
        from ..aes_scanner.store import AESScanStore

        store = AESScanStore()
        d = store.latest_scan_date()
        if d:
            c = store.candidates(scan_date=d, limit=2000)
            if not c.empty:
                c = pd.DataFrame(_enrich(c.to_dict("records")))
            if not c.empty and "industry" in c:
                g = c.dropna(subset=["industry"]).groupby("industry")
                density = {
                    str(k): {"candidates": int(len(v)),
                             "mean_score": float(v["score"].mean()),
                             "max_score": float(v["score"].max())}
                    for k, v in g
                }
    except Exception as exc:                                    # pragma: no cover
        logger.warning("scan density unavailable: %s", exc)

    sectors = []
    for name, grp in u.groupby("industry"):
        den = density.get(str(name), {})
        sectors.append({
            "industry": str(name),
            "symbols": int(len(grp)),
            "weight_pct": round(len(grp) / len(u) * 100, 3),
            "candidates": den.get("candidates", 0),
            "mean_score": den.get("mean_score"),
            "max_score": den.get("max_score"),
            "signal_density_pct": round(den.get("candidates", 0) / len(grp) * 100, 2),
        })
    sectors.sort(key=lambda s: -s["symbols"])
    return ok({"available": True, "n": int(len(u)), "sectors": sectors})


@router.get("/health")
def health() -> JSONResponse:
    """Includes the frozen-config assertion, so a drifted deploy is visible."""
    try:
        assert_live_config()
        locked = True
        detail = None
    except Exception as exc:
        locked = False
        detail = str(exc)
    from .auth import password_configured

    return ok({
        "product": "LeadFlow",
        "config_locked": locked,
        "config_error": detail,
        # Surfaced so a deploy that forgot APP_PASSWORD is visible from the
        # outside. The gate fails CLOSED when it is unset -- every login is
        # refused -- so false here means "locked out", never "wide open".
        "auth_configured": password_configured(),
        "database": "postgres" if os.getenv("DATABASE_URL") else "sqlite (ephemeral on most hosts)",
        "frozen_on": str(LC.FROZEN_ON),
        "artefacts": {
            n: os.path.exists(os.path.join(_results_dir(), n))
            for n in ("leadflow_frozen_results.json", "leadflow_series.json")
        },
    })


__all__ = ["router"]
