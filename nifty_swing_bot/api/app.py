"""FastAPI backend for the Nifty Swing Terminal.

Serves the signal ledger, backtest artefacts, walk-forward study, per-stock
chart data and LLM insights to the web front end, and serves the built front end
itself in production.

Run it with::

    uvicorn nifty_swing_bot.api.app:app --reload --port 8000

Heavy imports (yfinance, the backtest engine) are deliberately deferred into the
handlers that need them so the server starts instantly and a missing optional
dependency cannot take the whole API down.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import AppConfig, get_config, get_secrets, reload_config
from ..data.store import SignalStore

logger = logging.getLogger(__name__)

api = APIRouter(prefix="/api")

#: Set by ``POST /api/scan`` so the UI can show progress without a job queue.
_scan_state: dict[str, Any] = {"running": False, "started_at": None, "finished_at": None,
                               "error": None, "signals_found": None}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _clean(obj: Any) -> Any:
    """Recursively make a structure JSON-safe.

    NaN and infinity are not valid JSON. pandas produces both freely, and
    ``JSONResponse`` will happily emit bare ``NaN`` tokens that then break
    ``JSON.parse`` in the browser, so every payload is normalised here.
    """
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()[:19]
    if obj is pd.NaT or obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        return _clean(obj.to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return _clean(obj.to_dict())
    return obj


def ok(payload: Any) -> JSONResponse:
    """Return a cleaned JSON response."""
    return JSONResponse(content=_clean(payload))


def _results_dir(cfg: AppConfig | None = None) -> Path:
    return (cfg or get_config()).paths.results_dir


def _read_results_json(name: str, cfg: AppConfig | None = None) -> dict[str, Any] | None:
    path = _results_dir(cfg) / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Could not parse %s", path)
        return None


def _pick_results_file(stem: str, cfg: AppConfig | None = None) -> Path | None:
    """Find a results file, preferring the untagged one then the newest tagged.

    Backtests can be saved with a ``--tag``; the UI should show whichever run is
    most recent rather than silently showing nothing when only tagged runs exist.
    """
    directory = _results_dir(cfg)
    exact = directory / f"{stem}.json"
    if exact.exists():
        return exact
    candidates = sorted(
        directory.glob(f"{stem}_*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return candidates[0] if candidates else None


def _load_csv(stem: str, cfg: AppConfig | None = None) -> pd.DataFrame:
    """Load a results CSV, preferring untagged then newest tagged."""
    directory = _results_dir(cfg)
    exact = directory / f"{stem}.csv"
    path = exact if exact.exists() else None
    if path is None:
        candidates = sorted(
            directory.glob(f"{stem}_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        path = candidates[0] if candidates else None
    if path is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return pd.DataFrame()


# --------------------------------------------------------------------------- #
# Meta
# --------------------------------------------------------------------------- #
@api.get("/health")
def health() -> JSONResponse:
    """Liveness plus a summary of what data is actually available."""
    cfg = get_config()
    store = SignalStore(cfg=cfg)
    from ..data.fetch_delivery import cached_delivery_days

    delivery_days = cached_delivery_days(cfg)
    ohlcv = len(list(cfg.paths.ohlcv_dir.glob("*.parquet")))
    signals = store.get_signals(limit=1)

    return ok(
        {
            "status": "ok",
            "time": datetime.now().isoformat(timespec="seconds"),
            "data": {
                "ohlcv_symbols_cached": ohlcv,
                "delivery_trading_days": len(delivery_days),
                "delivery_from": delivery_days[0].isoformat() if delivery_days else None,
                "delivery_to": delivery_days[-1].isoformat() if delivery_days else None,
                "signals_recorded": int(len(store.get_signals(with_outcomes=False))),
                "latest_signal_date": (
                    signals["signal_date"].iloc[0] if not signals.empty else None
                ),
            },
            "artefacts": {
                "backtest": _pick_results_file("stats") is not None,
                "walk_forward": (_results_dir(cfg) / "walk_forward.json").exists(),
            },
            "llm_configured": bool(get_secrets().anthropic_api_key),
        }
    )


# --------------------------------------------------------------------------- #
# Config / Settings page
# --------------------------------------------------------------------------- #
class ConfigPatch(BaseModel):
    """Partial config update. Only the sections supplied are touched."""

    dab: dict[str, Any] | None = Field(default=None)
    risk: dict[str, Any] | None = Field(default=None)
    universe: dict[str, Any] | None = Field(default=None)
    execution: dict[str, Any] | None = Field(default=None)
    backtest: dict[str, Any] | None = Field(default=None)
    llm: dict[str, Any] | None = Field(default=None)


@api.get("/config")
def read_config() -> JSONResponse:
    """Current strategy configuration, with field descriptions for the UI."""
    cfg = get_config()
    schema = cfg.model_json_schema()

    def describe(section: str) -> dict[str, Any]:
        """Pull per-field descriptions out of the pydantic schema."""
        ref = schema.get("properties", {}).get(section, {}).get("$ref", "")
        name = ref.rsplit("/", 1)[-1]
        defs = schema.get("$defs", {}).get(name, {})
        return {
            key: {
                "description": spec.get("description"),
                "type": spec.get("type"),
                "enum": spec.get("enum"),
            }
            for key, spec in defs.get("properties", {}).items()
        }

    payload = cfg.model_dump(mode="json")
    payload.pop("paths", None)
    return ok(
        {
            "config": payload,
            "meta": {s: describe(s) for s in
                     ("dab", "risk", "universe", "execution", "backtest", "llm")},
        }
    )


@api.put("/config")
def update_config(patch: ConfigPatch) -> JSONResponse:
    """Validate and persist a config change, then reload the singleton.

    The whole config is re-validated through pydantic before anything is
    written, so an invalid value is rejected with a 422 rather than persisted
    and blowing up on the next backtest.
    """
    cfg = get_config()
    data = cfg.model_dump(mode="json")
    for section, values in patch.model_dump(exclude_none=True).items():
        data.setdefault(section, {}).update(values)

    try:
        candidate = AppConfig.model_validate(data)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid configuration: {exc}") from exc

    candidate.save()
    fresh = reload_config()
    payload = fresh.model_dump(mode="json")
    payload.pop("paths", None)
    return ok({"config": payload, "saved": True})


# --------------------------------------------------------------------------- #
# Signals / forward test
# --------------------------------------------------------------------------- #
@api.get("/signals")
def list_signals(
    signal_date: str | None = Query(default=None, alias="date"),
    since: str | None = Query(default=None),
    symbol: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=2000),
) -> JSONResponse:
    """Recorded signals joined to their realised outcomes."""
    store = SignalStore()
    frame = store.get_signals(
        signal_date=signal_date, since=since, symbol=symbol, limit=limit
    )
    return ok({"count": int(len(frame)), "signals": frame})


@api.get("/signals/latest")
def latest_signals() -> JSONResponse:
    """The most recent scan's signals, plus any cached LLM insights."""
    store = SignalStore()
    frame = store.get_signals(limit=500)
    if frame.empty:
        return ok({"date": None, "count": 0, "signals": []})

    latest = frame["signal_date"].iloc[0]
    today = frame[frame["signal_date"] == latest].copy()
    insights = {
        row["symbol"]: store.get_insight(row["symbol"], row["signal_date"])
        for _, row in today.iterrows()
    }
    return ok(
        {
            "date": latest,
            "count": int(len(today)),
            "signals": today,
            "insights": {k: v for k, v in insights.items() if v},
        }
    )


@api.get("/forward")
def forward_performance(since: str | None = Query(default=None)) -> JSONResponse:
    """Live forward-test statistics, alongside the backtest for comparison."""
    store = SignalStore()
    stats = store.forward_stats(since=since)
    backtest_file = _pick_results_file("stats")
    backtest: dict[str, Any] = {}
    if backtest_file is not None:
        try:
            backtest = json.loads(backtest_file.read_text(encoding="utf-8")).get("stats", {})
        except json.JSONDecodeError:
            backtest = {}

    return ok(
        {
            "forward": stats,
            "backtest_comparison": {
                k: backtest.get(k)
                for k in ("win_rate_pct", "avg_r_multiple", "profit_factor",
                          "avg_bars_held", "trade_count")
            },
            "scans": store.scan_history(limit=60),
        }
    )


@api.post("/scan")
def trigger_scan(background: BackgroundTasks, limit: int | None = None,
                 with_llm: bool = False) -> JSONResponse:
    """Kick off a scan in the background and return immediately."""
    if _scan_state["running"]:
        raise HTTPException(status_code=409, detail="A scan is already running.")

    def _run() -> None:
        _scan_state.update(running=True, started_at=datetime.now().isoformat(timespec="seconds"),
                           error=None, signals_found=None, finished_at=None)
        try:
            from ..scanner.daily_scan import scan, track_open_signals

            cfg = get_config()
            store = SignalStore(cfg=cfg)
            frame = scan(limit=limit, cfg=cfg)
            if not frame.empty:
                store.add_signals(frame.to_dict(orient="records"), scan_date=date.today())
                if with_llm:
                    from ..llm.insight_generator import generate_for_signals

                    generate_for_signals(frame.to_dict(orient="records"), store=store, cfg=cfg)
            track_open_signals(store, cfg=cfg)
            _scan_state["signals_found"] = int(len(frame))
        except Exception as exc:
            logger.exception("Background scan failed.")
            _scan_state["error"] = str(exc)
        finally:
            _scan_state.update(
                running=False, finished_at=datetime.now().isoformat(timespec="seconds")
            )

    background.add_task(_run)
    return ok({"started": True})


@api.get("/scan/status")
def scan_status() -> JSONResponse:
    """Progress of the most recent background scan."""
    return ok(_scan_state)


# --------------------------------------------------------------------------- #
# Backtest artefacts
# --------------------------------------------------------------------------- #
@api.get("/backtest")
def backtest_summary() -> JSONResponse:
    """Headline backtest statistics, parameters and the R-multiple histogram."""
    path = _pick_results_file("stats")
    if path is None:
        raise HTTPException(
            status_code=404,
            detail="No backtest results found. Run: python -m nifty_swing_bot.backtest.run_backtest",
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Corrupt results file: {exc}") from exc
    payload["source_file"] = path.name
    return ok(payload)


@api.get("/backtest/equity")
def backtest_equity() -> JSONResponse:
    """Daily equity curve for the results chart."""
    frame = _load_csv("equity_curve")
    if frame.empty:
        raise HTTPException(status_code=404, detail="No equity curve available.")
    date_col = frame.columns[0]
    frame = frame.rename(columns={date_col: "date"})
    return ok({"points": frame.to_dict(orient="records")})


@api.get("/backtest/trades")
def backtest_trades(
    limit: int = Query(default=1000, ge=1, le=10000),
    symbol: str | None = None,
    exit_reason: str | None = None,
) -> JSONResponse:
    """The backtest trade log, optionally filtered."""
    frame = _load_csv("trades")
    if frame.empty:
        raise HTTPException(status_code=404, detail="No trade log available.")
    if symbol:
        frame = frame[frame["symbol"] == symbol]
    if exit_reason:
        frame = frame[frame["exit_reason"] == exit_reason]
    return ok({"count": int(len(frame)), "trades": frame.head(limit)})


@api.get("/walkforward")
def walkforward() -> JSONResponse:
    """The walk-forward study, with its chained out-of-sample equity curve."""
    payload = _read_results_json("walk_forward.json")
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail="No walk-forward results. Run: python -m nifty_swing_bot.backtest.walk_forward",
        )
    equity_path = _results_dir() / "walk_forward_equity.csv"
    if equity_path.exists():
        try:
            frame = pd.read_csv(equity_path)
            frame = frame.rename(columns={frame.columns[0]: "date"})
            payload["equity"] = frame.to_dict(orient="records")
        except (OSError, pd.errors.ParserError):
            payload["equity"] = []
    return ok(payload)


# --------------------------------------------------------------------------- #
# Universe & per-stock detail
# --------------------------------------------------------------------------- #
@api.get("/universe")
def universe() -> JSONResponse:
    """The tradeable universe with sector breakdown, for the sector view."""
    from ..data.universe import build_universe

    frame = build_universe()
    if frame.empty:
        raise HTTPException(status_code=503, detail="Universe unavailable (NSE unreachable).")
    sectors = (
        frame.groupby("industry").size().sort_values(ascending=False)
        if "industry" in frame else pd.Series(dtype=int)
    )
    return ok(
        {
            "count": int(len(frame)),
            "symbols": frame[["symbol", "company", "industry"]],
            "sectors": [{"name": k, "count": int(v)} for k, v in sectors.items()],
        }
    )


@api.get("/stock/{symbol}")
def stock_detail(
    symbol: str,
    days: int = Query(default=180, ge=30, le=1000),
) -> JSONResponse:
    """Price, delivery and DAB feature series for one stock's detail page."""
    from ..data.fetch_delivery import fetch_delivery_history, to_symbol_panel
    from ..data.fetch_ohlcv import fetch_benchmark, fetch_ohlcv
    from ..strategy.dab_strategy import compute_features

    cfg = get_config()
    symbol = symbol.upper()
    start = pd.Timestamp.today().normalize() - pd.Timedelta(days=days + 200)

    prices = fetch_ohlcv([symbol], start=start, refresh=False, cfg=cfg, show_progress=False)
    ohlcv = prices.get(symbol)
    if ohlcv is None or ohlcv.empty:
        raise HTTPException(status_code=404, detail=f"No price data for {symbol}.")

    delivery = to_symbol_panel(
        fetch_delivery_history(start.date(), date.today(), symbols=[symbol],
                               cfg=cfg, progress_every=0)
    ).get(symbol)
    benchmark = fetch_benchmark(start=start, cfg=cfg)

    features = compute_features(ohlcv, delivery, benchmark, cfg=cfg)
    tail = features.tail(days).copy()
    tail = tail.reset_index().rename(columns={tail.index.name or "index": "date"})

    columns = [
        "date", "open", "high", "low", "close", "volume",
        "deliv_pct", "deliv_baseline", "deliv_ratio", "deliv_spike_ratio",
        "vol_ratio", "range_position", "rs_excess", "high_n", "dist_from_high",
        "atr_fast", "atr_slow", "atr_ratio", "atr_stop",
        "cond_delivery", "cond_volume", "cond_close", "cond_rs", "cond_setup",
        "is_deliv_spike", "is_breakout", "signal",
    ]
    present = [c for c in columns if c in tail.columns]

    store = SignalStore(cfg=cfg)
    signals = store.get_signals(symbol=symbol, limit=50)
    latest_insight = None
    if not signals.empty:
        latest_insight = store.get_insight(symbol, signals["signal_date"].iloc[0])

    return ok(
        {
            "symbol": symbol,
            "series": tail[present],
            "signal_dates": [
                d.strftime("%Y-%m-%d")
                for d in features.index[features["signal"].fillna(False).astype(bool)]
            ],
            "recorded_signals": signals,
            "insight": latest_insight,
            "insight_dates": store.insight_dates(symbol),
        }
    )


@api.get("/stock/{symbol}/insight")
def stock_insight(
    symbol: str,
    insight_date: str | None = Query(default=None, alias="date"),
    generate: bool = Query(default=False),
) -> JSONResponse:
    """Fetch a cached LLM insight, optionally generating it on demand."""
    cfg = get_config()
    store = SignalStore(cfg=cfg)
    symbol = symbol.upper()

    signals = store.get_signals(symbol=symbol, signal_date=insight_date, limit=1)
    target_date = insight_date or (
        signals["signal_date"].iloc[0] if not signals.empty else date.today().isoformat()
    )

    cached = store.get_insight(symbol, target_date)
    if cached is not None:
        return ok({"insight": cached, "cached": True})
    if not generate:
        return ok({"insight": None, "cached": False})

    if signals.empty:
        raise HTTPException(
            status_code=404, detail=f"No recorded signal for {symbol} on {target_date}."
        )

    from ..llm.insight_generator import InsightGenerator

    generator = InsightGenerator(cfg=cfg, store=store)
    if not generator.available:
        raise HTTPException(
            status_code=503,
            detail="LLM layer unavailable. Set ANTHROPIC_API_KEY in your .env file.",
        )
    payload = generator.generate(signals.iloc[0].to_dict())
    if payload is None:
        raise HTTPException(status_code=502, detail="Insight generation failed.")
    return ok({"insight": payload, "cached": False})


# --------------------------------------------------------------------------- #
# App assembly
# --------------------------------------------------------------------------- #
def create_app() -> FastAPI:
    """Build the FastAPI application."""
    # Phase 25: refuse to serve on a drifted configuration. A terminal that
    # renders numbers produced by a config other than the frozen one is worse
    # than one that does not start, because the forward record it displays
    # would no longer be the experiment it claims to be.
    from ..aes.locked import assert_live_config

    assert_live_config()

    app = FastAPI(
        title="LeadFlow",
        description=(
            "LeadFlow research terminal: accumulation-breakout signals across the "
            "NSE mid- and small-cap tier. Research tooling, not investment advice."
        ),
        version="2.0.0",
    )
    # The Vite dev server runs on another port, so CORS is needed in development.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173", "http://127.0.0.1:5173",
            "http://localhost:4173", "http://127.0.0.1:4173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # The DAB/MR router (`api` in this module) is NOT mounted. It served a
    # different engine -- the DAB signal ledger in cache/terminal.db, DAB
    # feature series, and DAB backtest artefacts -- none of which is LeadFlow.
    # Leaving it unmounted is what makes "the new UI cannot read the DAB store"
    # true by construction rather than by convention.
    from . import ratelimit
    from .auth import is_authenticated, is_public
    from .auth import router as auth_router
    from .leadflow import router as leadflow_router

    # Order matters: rate limiting runs before auth so a flood of bad logins is
    # rejected before it costs a signature check, and auth runs before any
    # route so there is no page or payload reachable without a session.
    @app.middleware("http")
    async def _gate(request, call_next):
        limited = ratelimit.check(request)
        if limited is not None:
            return limited

        path = request.url.path
        if not is_public(path) and not is_authenticated(request):
            if path.startswith("/api/"):
                return JSONResponse(status_code=401, content={"detail": "Authentication required."})
            # Everything else is the SPA shell. Serving it is safe -- it holds
            # no data and immediately shows the login screen -- and it is what
            # makes a bookmarked deep link work after signing in.
            return await call_next(request)
        return await call_next(request)

    app.include_router(auth_router)
    app.include_router(leadflow_router)

    from .aes_routes import aes as aes_router

    app.include_router(aes_router)

    # Serve the built front end when it exists, so a single uvicorn process is
    # enough in production. In development Vite serves it instead.
    dist = Path(__file__).resolve().parent.parent / "ui" / "dist"
    if dist.exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        #: Asset filenames are content-hashed by Vite, so they are safe to cache
        #: forever -- but ``index.html`` is the thing that *names* them. If a
        #: browser caches it, a rebuild leaves that browser pinned to the old
        #: bundle until someone thinks to hard-refresh, which is exactly the
        #: sort of stale-UI bug nobody reports and everybody works around.
        _NO_STORE = {"Cache-Control": "no-store, must-revalidate"}

        @app.get("/{full_path:path}")
        def spa(full_path: str) -> FileResponse:
            """Serve index.html for any non-API route so client routing works."""
            candidate = dist / full_path
            if full_path and candidate.is_file():
                headers = _NO_STORE if candidate.name == "index.html" else None
                return FileResponse(candidate, headers=headers)
            return FileResponse(dist / "index.html", headers=_NO_STORE)
    else:
        @app.get("/")
        def placeholder() -> JSONResponse:
            return ok(
                {
                    "message": "API is running. The front end has not been built yet.",
                    "build": "cd nifty_swing_bot/ui && npm install && npm run build",
                    "dev": "cd nifty_swing_bot/ui && npm run dev",
                    "docs": "/docs",
                }
            )

    return app


app = create_app()


__all__ = ["api", "app", "create_app"]
