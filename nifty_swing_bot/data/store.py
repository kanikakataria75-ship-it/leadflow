"""SQLite store for live signals, their realised outcomes, and LLM insights.

This is the forward-test ledger. Every daily scan appends its signals here, and
a follow-up pass fills in what actually happened to each one using the same exit
rules the backtester applies. That makes the live record directly comparable to
the backtest statistics, which is the only honest way to tell whether a
backtested edge is real.

SQLite rather than parquet because these tables are append-and-update, small,
and queried by the API on every page load.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import AppConfig, get_config

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    scan_date       TEXT PRIMARY KEY,
    universe_size   INTEGER NOT NULL,
    signal_count    INTEGER NOT NULL,
    duration_s      REAL,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS signals (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_date         TEXT NOT NULL,
    symbol              TEXT NOT NULL,
    scan_date           TEXT NOT NULL,
    close               REAL,
    entry_ref           REAL,
    stop                REAL,
    risk_per_share      REAL,
    stop_pct            REAL,
    atr14               REAL,
    deliv_pct           REAL,
    deliv_ratio         REAL,
    deliv_spike_ratio   REAL,
    deliv_spike_age     INTEGER,
    vol_ratio           REAL,
    range_position      REAL,
    rs_excess           REAL,
    dist_from_high      REAL,
    atr_ratio           REAL,
    is_breakout         INTEGER,
    suggested_qty       INTEGER,
    risk_amount         REAL,
    company             TEXT,
    industry            TEXT,
    created_at          TEXT NOT NULL,
    UNIQUE(signal_date, symbol)
);

CREATE INDEX IF NOT EXISTS idx_signals_date   ON signals(signal_date);
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);

CREATE TABLE IF NOT EXISTS outcomes (
    signal_id       INTEGER PRIMARY KEY REFERENCES signals(id) ON DELETE CASCADE,
    status          TEXT NOT NULL,
    entry_date      TEXT,
    entry_price     REAL,
    exit_date       TEXT,
    exit_price      REAL,
    exit_reason     TEXT,
    bars_held       INTEGER,
    return_pct      REAL,
    r_multiple      REAL,
    mae_r           REAL,
    mfe_r           REAL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_insights (
    symbol          TEXT NOT NULL,
    insight_date    TEXT NOT NULL,
    payload         TEXT NOT NULL,
    model           TEXT,
    created_at      TEXT NOT NULL,
    PRIMARY KEY (symbol, insight_date)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SignalStore:
    """Thin, typed wrapper over the SQLite forward-test database."""

    def __init__(self, path: Path | None = None, cfg: AppConfig | None = None) -> None:
        cfg = cfg or get_config()
        self.path = Path(path or cfg.paths.db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def connect(self):
        """Yield a row-dict connection with foreign keys enforced."""
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    # ------------------------------------------------------------- signals --
    def record_scan(
        self, scan_date: date, universe_size: int, signal_count: int, duration_s: float
    ) -> None:
        """Log that a scan ran, so gaps in the forward record are visible."""
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO scans (scan_date, universe_size, signal_count, duration_s, created_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(scan_date) DO UPDATE SET "
                "universe_size=excluded.universe_size, signal_count=excluded.signal_count, "
                "duration_s=excluded.duration_s, created_at=excluded.created_at",
                (scan_date.isoformat(), universe_size, signal_count, duration_s, _now()),
            )

    def add_signals(self, signals: Sequence[Mapping[str, Any]], scan_date: date) -> int:
        """Insert signals, ignoring any already recorded for that date and symbol.

        Returns:
            The number of genuinely new rows, so a re-run of the same scan
            reports 0 rather than silently double-counting.
        """
        if not signals:
            return 0
        columns = (
            "signal_date", "symbol", "scan_date", "close", "entry_ref", "stop",
            "risk_per_share", "stop_pct", "atr14", "deliv_pct", "deliv_ratio",
            "deliv_spike_ratio", "deliv_spike_age", "vol_ratio", "range_position",
            "rs_excess", "dist_from_high", "atr_ratio", "is_breakout",
            "suggested_qty", "risk_amount", "company", "industry", "created_at",
        )
        placeholders = ", ".join("?" for _ in columns)
        rows = []
        for s in signals:
            rows.append(
                tuple(
                    [
                        s.get("date"),
                        s.get("symbol"),
                        scan_date.isoformat(),
                        s.get("close"), s.get("entry_ref"), s.get("stop"),
                        s.get("risk_per_share"), s.get("stop_pct"), s.get("atr14"),
                        s.get("deliv_pct"), s.get("deliv_ratio"),
                        s.get("deliv_spike_ratio"), s.get("deliv_spike_age"),
                        s.get("vol_ratio"), s.get("range_position"),
                        s.get("rs_excess"), s.get("dist_from_high"), s.get("atr_ratio"),
                        int(bool(s.get("is_breakout"))),
                        s.get("suggested_qty"), s.get("risk_amount"),
                        s.get("company"), s.get("industry"), _now(),
                    ]
                )
            )
        with self.connect() as conn:
            before = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
            conn.executemany(
                f"INSERT OR IGNORE INTO signals ({', '.join(columns)}) VALUES ({placeholders})",
                rows,
            )
            after = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        return after - before

    def get_signals(
        self,
        *,
        signal_date: date | str | None = None,
        symbol: str | None = None,
        since: date | str | None = None,
        limit: int | None = None,
        with_outcomes: bool = True,
    ) -> pd.DataFrame:
        """Query recorded signals, optionally joined to their outcomes."""
        select = "SELECT s.*"
        join = ""
        if with_outcomes:
            select += (
                ", o.status, o.entry_date, o.entry_price, o.exit_date, o.exit_price, "
                "o.exit_reason, o.bars_held, o.return_pct, o.r_multiple, o.mae_r, o.mfe_r"
            )
            join = " LEFT JOIN outcomes o ON o.signal_id = s.id"

        where: list[str] = []
        params: list[Any] = []
        if signal_date is not None:
            where.append("s.signal_date = ?")
            params.append(str(signal_date))
        if symbol is not None:
            where.append("s.symbol = ?")
            params.append(symbol)
        if since is not None:
            where.append("s.signal_date >= ?")
            params.append(str(since))

        sql = f"{select} FROM signals s{join}"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY s.signal_date DESC, s.deliv_spike_ratio DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"

        with self.connect() as conn:
            return pd.read_sql_query(sql, conn, params=params)

    def open_signals(self) -> pd.DataFrame:
        """Signals with no recorded outcome, or an outcome still marked open."""
        with self.connect() as conn:
            return pd.read_sql_query(
                "SELECT s.* FROM signals s "
                "LEFT JOIN outcomes o ON o.signal_id = s.id "
                "WHERE o.signal_id IS NULL OR o.status = 'open' "
                "ORDER BY s.signal_date",
                conn,
            )

    # ------------------------------------------------------------ outcomes --
    def upsert_outcome(self, signal_id: int, outcome: Mapping[str, Any]) -> None:
        """Insert or update the realised result for one signal."""
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO outcomes (signal_id, status, entry_date, entry_price, exit_date, "
                "exit_price, exit_reason, bars_held, return_pct, r_multiple, mae_r, mfe_r, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(signal_id) DO UPDATE SET "
                "status=excluded.status, entry_date=excluded.entry_date, "
                "entry_price=excluded.entry_price, exit_date=excluded.exit_date, "
                "exit_price=excluded.exit_price, exit_reason=excluded.exit_reason, "
                "bars_held=excluded.bars_held, return_pct=excluded.return_pct, "
                "r_multiple=excluded.r_multiple, mae_r=excluded.mae_r, mfe_r=excluded.mfe_r, "
                "updated_at=excluded.updated_at",
                (
                    signal_id,
                    outcome.get("status", "open"),
                    outcome.get("entry_date"), outcome.get("entry_price"),
                    outcome.get("exit_date"), outcome.get("exit_price"),
                    outcome.get("exit_reason"), outcome.get("bars_held"),
                    outcome.get("return_pct"), outcome.get("r_multiple"),
                    outcome.get("mae_r"), outcome.get("mfe_r"), _now(),
                ),
            )

    def forward_stats(self, since: date | str | None = None) -> dict[str, Any]:
        """Aggregate realised forward performance, comparable to backtest stats."""
        frame = self.get_signals(since=since, with_outcomes=True)
        closed = frame[frame.get("status") == "closed"] if "status" in frame else pd.DataFrame()
        out: dict[str, Any] = {
            "signals_total": int(len(frame)),
            "signals_closed": int(len(closed)),
            "signals_open": int(len(frame) - len(closed)),
        }
        if closed.empty:
            return out

        r = pd.to_numeric(closed["r_multiple"], errors="coerce").dropna()
        wins = r[r > 0]
        gross_win = float(r[r > 0].sum())
        gross_loss = float(-r[r < 0].sum())
        out.update(
            {
                "win_rate_pct": round(len(wins) / len(r) * 100, 2) if len(r) else 0.0,
                "avg_r_multiple": round(float(r.mean()), 3) if len(r) else 0.0,
                "median_r_multiple": round(float(r.median()), 3) if len(r) else 0.0,
                "profit_factor": round(gross_win / gross_loss, 3) if gross_loss > 0 else None,
                "best_r": round(float(r.max()), 3) if len(r) else 0.0,
                "worst_r": round(float(r.min()), 3) if len(r) else 0.0,
                "avg_bars_held": round(
                    float(pd.to_numeric(closed["bars_held"], errors="coerce").mean()), 2
                ),
                "total_r": round(float(r.sum()), 3),
            }
        )
        reasons = closed["exit_reason"].value_counts().to_dict() if "exit_reason" in closed else {}
        out["exit_reasons"] = {str(k): int(v) for k, v in reasons.items()}
        return out

    def scan_history(self, limit: int = 90) -> pd.DataFrame:
        """Recent scan log, newest first."""
        with self.connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM scans ORDER BY scan_date DESC LIMIT ?", conn, params=[limit]
            )

    # ----------------------------------------------------------- insights --
    def get_insight(self, symbol: str, insight_date: date | str) -> dict[str, Any] | None:
        """Fetch a cached LLM insight, or None."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT payload FROM llm_insights WHERE symbol = ? AND insight_date = ?",
                (symbol, str(insight_date)),
            ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row["payload"])
        except json.JSONDecodeError:
            logger.warning("Corrupt cached insight for %s on %s.", symbol, insight_date)
            return None

    def put_insight(
        self, symbol: str, insight_date: date | str, payload: Mapping[str, Any], model: str
    ) -> None:
        """Cache an LLM insight for a symbol on a date."""
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO llm_insights (symbol, insight_date, payload, model, created_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(symbol, insight_date) DO UPDATE SET "
                "payload=excluded.payload, model=excluded.model, created_at=excluded.created_at",
                (symbol, str(insight_date), json.dumps(payload, default=str), model, _now()),
            )

    def insight_dates(self, symbol: str) -> list[str]:
        """All dates for which we hold an insight on a symbol."""
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT insight_date FROM llm_insights WHERE symbol = ? ORDER BY insight_date DESC",
                (symbol,),
            ).fetchall()
        return [r["insight_date"] for r in rows]


__all__ = ["SCHEMA", "SignalStore"]
