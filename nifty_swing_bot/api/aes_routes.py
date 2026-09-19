"""API surface for the production AES scanner.

Kept in its own module and mounted under ``/api/aes`` so the scanner's
endpoints can change without touching the DAB/MR terminal's, which is the
same separation the packages themselves have.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from ..aes_scanner import live_config as LC
from ..aes_scanner.store import AESScanStore
from ..config import get_config

logger = logging.getLogger(__name__)

aes = APIRouter(prefix="/api/aes", tags=["aes"])

#: Mirrors the DAB scanner's pattern: a tiny bit of in-process state so the UI
#: can show progress without introducing a job queue.
_state: dict[str, Any] = {"running": False, "started_at": None, "finished_at": None,
                          "error": None, "candidates": None, "actionable": None}


def _ok(payload: Any) -> JSONResponse:
    from .app import _clean

    return JSONResponse(content=_clean(payload))


def _store() -> AESScanStore:
    return AESScanStore(cfg=get_config())


@aes.get("/scan")
def latest_scan(
    scan_date: str | None = Query(default=None, description="YYYY-MM-DD; defaults to latest"),
    bucket: str | None = Query(default=None),
    actionable_only: bool = Query(default=False),
) -> JSONResponse:
    """One evening's ranked candidates, with every score component."""
    st = _store()
    d = scan_date or st.latest_scan_date()
    if d is None:
        return _ok({"scan_date": None, "candidates": [], "scan": None,
                    "message": "No scan has been run yet."})
    scans = st.scans(limit=400)
    meta = scans[scans["scan_date"] == d]
    cands = st.candidates(d, bucket=bucket, actionable_only=actionable_only)
    return _ok({
        "scan_date": d,
        "scan": meta.to_dict(orient="records")[0] if len(meta) else None,
        "counts": {
            "total": int(len(cands)),
            "actionable": int(cands["actionable"].sum()) if len(cands) else 0,
            "high_conviction": int((cands["bucket"] == "high_conviction").sum()) if len(cands) else 0,
            "wait_and_watch": int((cands["bucket"] == "wait_and_watch").sum()) if len(cands) else 0,
            "reject": int((cands["bucket"] == "reject").sum()) if len(cands) else 0,
        },
        "candidates": cands.to_dict(orient="records"),
    })


@aes.get("/audit")
def audit(
    scan_date: str | None = Query(default=None),
    verdict: str | None = Query(default=None, description="ACCEPTED | WAIT & WATCH | REJECTED"),
    stage: str | None = Query(default=None),
) -> JSONResponse:
    """Every universe symbol with its stage, verdict and reason.

    The scan table shows only names that produced a box. This shows the whole
    universe, including the ~95% that dropped out and why -- which is what
    makes the ranking checkable by hand.
    """
    st = _store()
    d = scan_date or st.latest_scan_date()
    if d is None:
        return _ok({"scan_date": None, "rows": [], "funnel": []})
    df = st.audit(d, verdict=verdict, stage=stage)
    full = st.audit(d)
    funnel = (
        full.groupby(["stage_rank", "stage", "verdict"], dropna=False)
        .size().reset_index(name="count")
        .sort_values("stage_rank", ascending=False)
        .to_dict(orient="records")
        if len(full) else []
    )
    return _ok({"scan_date": d, "total": int(len(full)), "rows": df.to_dict(orient="records"),
                "funnel": funnel})


@aes.get("/audit.csv")
def audit_csv(scan_date: str | None = Query(default=None)) -> PlainTextResponse:
    """The same audit as a CSV, for checking against a broker terminal."""
    st = _store()
    d = scan_date or st.latest_scan_date()
    df = st.audit(d) if d else None
    body = "" if df is None or df.empty else df.to_csv(index=False)
    return PlainTextResponse(
        body, media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="aes_audit_{d or "empty"}.csv"'},
    )


@aes.get("/history")
def scan_history(limit: int = Query(default=120, ge=1, le=500)) -> JSONResponse:
    """Per-day scan metadata, including the breadth series."""
    return _ok({"scans": _store().scans(limit=limit).to_dict(orient="records")})


@aes.get("/record")
def forward_record(since: str | None = Query(default=None)) -> JSONResponse:
    """The accumulating live record: what was flagged, at what score, what it did."""
    df = _store().forward_record(since=since)
    summary: dict[str, Any] = {"flagged": int(len(df))}
    if len(df) and "realised_pct" in df:
        closed = df[df["realised_pct"].notna()]
        summary.update(
            closed=int(len(closed)),
            open=int(len(df) - len(closed)),
            win_rate_pct=round(float((closed["realised_pct"] > 0).mean() * 100), 1) if len(closed) else None,
            mean_pct=round(float(closed["realised_pct"].mean()), 2) if len(closed) else None,
            median_pct=round(float(closed["realised_pct"].median()), 2) if len(closed) else None,
        )
    return _ok({"summary": summary, "rows": df.to_dict(orient="records")})


@aes.get("/chart/{scan_date}/{symbol}")
def chart(scan_date: str, symbol: str) -> FileResponse:
    """The rendered box overlay for one candidate."""
    cfg = get_config()
    base = Path(cfg.paths.results_dir) / "aes_scan_charts" / scan_date
    path = base / f"{symbol.upper()}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No chart for {symbol} on {scan_date}.")
    return FileResponse(path, media_type="image/png")


@aes.post("/scan")
def trigger(background: BackgroundTasks, equity: float | None = None,
            render: bool = True) -> JSONResponse:
    """Run a scan in the background and return immediately."""
    if _state["running"]:
        raise HTTPException(status_code=409, detail="A scan is already running.")

    def _run() -> None:
        _state.update(running=True, started_at=datetime.now().isoformat(timespec="seconds"),
                      error=None, candidates=None, actionable=None, finished_at=None)
        try:
            from ..aes_scanner.pipeline import run_scan, track_outcomes

            cfg = get_config()
            st = AESScanStore(cfg=cfg)
            # The FROZEN configuration is passed explicitly, exactly as
            # aes_scanner/cli.py does for the scheduled scan. Taking the
            # dataclass defaults here would run this scan at 1% risk over 3
            # slots instead of the frozen 3% over 6 -- and persist=True means
            # those rows would enter the same forward record as the scheduled
            # ones, which is two different experiments in one table.
            result = run_scan(
                cfg, equity=equity, render=render, store=st, persist=True,
                screener_params=LC.SCREENER, box_params=LC.BOX,
                watch_params=LC.WATCHLIST, portfolio_params=LC.PORTFOLIO,
                score_params=LC.SCORING,
            )
            _state["candidates"] = int(len(result.candidates))
            _state["actionable"] = (
                int(result.candidates["actionable"].sum()) if len(result.candidates) else 0
            )
            try:
                from ..data.fetch_ohlcv import fetch_benchmark, fetch_ohlcv
                from ..data.universe import build_universe

                uni = build_universe(cfg=cfg)
                frames = fetch_ohlcv(uni["symbol"].tolist(), cfg=cfg, show_progress=False)
                track_outcomes(st, frames, fetch_benchmark(cfg=cfg))
            except Exception:
                logger.exception("Outcome tracking failed; the scan itself is recorded.")
        except Exception as exc:
            logger.exception("AES scan failed.")
            _state["error"] = str(exc)
        finally:
            _state.update(running=False,
                          finished_at=datetime.now().isoformat(timespec="seconds"))

    background.add_task(_run)
    return _ok({"started": True})


@aes.get("/scan/status")
def status() -> JSONResponse:
    """Progress of the most recent background scan."""
    return _ok(_state)


__all__ = ["aes"]
