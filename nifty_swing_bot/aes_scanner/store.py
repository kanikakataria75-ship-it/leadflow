"""SQLite persistence for the daily AES scan.

Separate database file from the DAB/MR ``SignalStore``: that schema is shaped
around delivery-spike columns this strategy does not use, and sharing a file
would couple two tools that should be able to change independently.

The point of this store is the **forward record**. Every candidate is written
with the component values that put it where it ranked, so that months later
it is possible to ask "what did the things I scored 0.7 actually do?" without
re-running anything. ``aes_outcomes`` is refreshed in place as price history
extends.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import date, datetime
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import AppConfig, get_config

#: When set, the store talks to Postgres instead of a local SQLite file.
#:
#: This exists because the deploy target's filesystem is ephemeral. The forward
#: record is the only genuinely out-of-sample evidence this project has, and a
#: SQLite file on a free-tier web host is deleted on every redeploy and every
#: idle spin-down -- the record would silently restart from zero and still look
#: healthy. Postgres is the difference between a forward record and a rolling
#: 15-minute window.
#:
#: Local development sets nothing and keeps the SQLite file, so the research
#: workflow is unchanged.
def _database_url() -> str | None:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        return None
    # Heroku/Render style URLs use the legacy scheme psycopg3 does not accept.
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    return url


def _is_pg(url: str | None) -> bool:
    return bool(url and url.startswith("postgresql://"))


#: SQLite DDL -> Postgres. Deliberately a small, explicit list rather than a
#: general translator: these are the only four constructs the schema uses that
#: the two dialects spell differently, and an explicit list fails loudly if a
#: fifth is ever added.
def _pg_schema(sql: str) -> str:
    sql = sql.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY")
    sql = sql.replace("REAL", "DOUBLE PRECISION")
    return sql


def _pg_sql(sql: str) -> str:
    """Rewrite a SQLite statement for Postgres.

    Only two things differ in the queries themselves: the parameter marker, and
    SQLite's ``julianday`` date arithmetic. ``ON CONFLICT ... DO UPDATE SET ...
    EXCLUDED.x`` is spelled identically in both, so every upsert passes through
    untouched.
    """
    sql = re.sub(
        r"julianday\('now'\)\s*-\s*julianday\(([^)]+)\)",
        r"(CURRENT_DATE - (\1)::date)",
        sql,
    )
    return sql.replace("?", "%s")

SCHEMA = """
CREATE TABLE IF NOT EXISTS aes_scans (
    scan_date        TEXT PRIMARY KEY,
    universe_size    INTEGER NOT NULL,
    watchlist_size   INTEGER NOT NULL,
    candidate_count  INTEGER NOT NULL,
    actionable_count INTEGER NOT NULL,
    breadth          INTEGER,
    breadth_pctile   REAL,
    breadth_label    TEXT,
    duration_s       REAL,
    created_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS aes_candidates (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_date           TEXT NOT NULL,
    symbol              TEXT NOT NULL,
    company             TEXT,
    industry            TEXT,
    bucket              TEXT NOT NULL,
    score               REAL,
    actionable          INTEGER NOT NULL DEFAULT 0,
    entry_mode          INTEGER,
    admission_gap_days  REAL,
    -- box geometry (section 2)
    box_date            TEXT,
    box_bars            INTEGER,
    box_range_pct       REAL,
    box_top             REAL,
    box_bottom          REAL,
    box_position        REAL,
    higher_lows         INTEGER,
    pct_closes_above_mid REAL,
    box_quality         REAL,
    small_zone          TEXT,
    small_top           REAL,
    small_bottom        REAL,
    small_bars          INTEGER,
    prior_cycles        INTEGER,
    vol_dryup_ratio     REAL,
    -- context (section 3)
    resistance_level    REAL,
    resistance_dist_pct REAL,
    resistance_age      INTEGER,
    resistance_absorbed INTEGER,
    resistance_rejected INTEGER,
    tf_daily            INTEGER,
    tf_weekly           INTEGER,
    tf_monthly          INTEGER,
    rs_capture          REAL,
    rs_class            TEXT,
    -- score components (equal weight)
    sc_fast             REAL,
    sc_absorbed         REAL,
    sc_rs               REAL,
    sc_cycles           REAL,
    -- trade plan
    close               REAL,
    atr14               REAL,
    entry_ref           REAL,
    stop                REAL,
    stop_basis          TEXT,
    risk_per_share      REAL,
    stop_pct            REAL,
    qty                 INTEGER,
    notional            REAL,
    risk_amount         REAL,
    chart_path          TEXT,
    created_at          TEXT NOT NULL,
    UNIQUE(scan_date, symbol)
);

CREATE INDEX IF NOT EXISTS idx_aes_cand_date   ON aes_candidates(scan_date);
CREATE INDEX IF NOT EXISTS idx_aes_cand_symbol ON aes_candidates(symbol);
CREATE INDEX IF NOT EXISTS idx_aes_cand_bucket ON aes_candidates(bucket);

CREATE TABLE IF NOT EXISTS aes_audit (
    scan_date    TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    company      TEXT,
    industry     TEXT,
    stage        TEXT NOT NULL,
    stage_rank   INTEGER,
    verdict      TEXT NOT NULL,
    reason       TEXT NOT NULL,
    score        REAL,
    sc_fast      REAL,
    sc_absorbed  REAL,
    sc_rs        REAL,
    sc_cycles    REAL,
    box_bars     INTEGER,
    box_range_pct REAL,
    box_position REAL,
    actionable   INTEGER,
    qty          INTEGER,
    price        REAL,
    move_10d_pct REAL,
    below_52wh_pct REAL,
    turnover_cr  REAL,
    last_admission TEXT,
    admission_age_bars INTEGER,
    created_at   TEXT NOT NULL,
    PRIMARY KEY (scan_date, symbol)
);

CREATE INDEX IF NOT EXISTS idx_aes_audit_stage ON aes_audit(scan_date, stage_rank);

CREATE TABLE IF NOT EXISTS aes_outcomes (
    candidate_id  INTEGER PRIMARY KEY REFERENCES aes_candidates(id) ON DELETE CASCADE,
    as_of         TEXT,
    bars_elapsed  INTEGER,
    ret_5         REAL,
    ret_10        REAL,
    ret_15        REAL,
    ret_20        REAL,
    ret_25        REAL,
    excess_20     REAL,
    mfe_pct       REAL,
    mae_pct       REAL,
    stop_hit      INTEGER,
    stop_hit_bar  INTEGER,
    exit_reason   TEXT,
    realised_pct  REAL,
    updated_at    TEXT NOT NULL
);
"""

_CAND_COLUMNS = [
    "scan_date", "symbol", "company", "industry", "bucket", "score", "actionable",
    "entry_mode", "admission_gap_days", "box_date", "box_bars", "box_range_pct", "box_top", "box_bottom",
    "box_position", "higher_lows", "pct_closes_above_mid", "box_quality", "small_zone",
    "small_top", "small_bottom", "small_bars", "prior_cycles", "vol_dryup_ratio",
    "resistance_level", "resistance_dist_pct", "resistance_age", "resistance_absorbed",
    "resistance_rejected", "tf_daily", "tf_weekly", "tf_monthly", "rs_capture", "rs_class",
    "sc_fast", "sc_absorbed", "sc_rs", "sc_cycles", "close", "atr14", "entry_ref", "stop", "stop_basis",
    "risk_per_share", "stop_pct", "qty", "notional", "risk_amount", "chart_path",
]

_AUDIT_COLUMNS = [
    "scan_date", "symbol", "company", "industry", "stage", "stage_rank", "verdict",
    "reason", "score", "sc_fast", "sc_absorbed", "sc_rs", "sc_cycles", "box_bars",
    "box_range_pct", "box_position", "actionable", "qty", "price", "move_10d_pct",
    "below_52wh_pct", "turnover_cr", "last_admission", "admission_age_bars",
]

_OUTCOME_COLUMNS = [
    "as_of", "bars_elapsed", "ret_5", "ret_10", "ret_15", "ret_20", "ret_25",
    "excess_20", "mfe_pct", "mae_pct", "stop_hit", "stop_hit_bar", "exit_reason",
    "realised_pct",
]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _d(value: date | str) -> str:
    return value if isinstance(value, str) else value.strftime("%Y-%m-%d")


class AESScanStore:
    """Daily scan output and the forward record built from it."""

    def __init__(
        self,
        path: Path | None = None,
        cfg: AppConfig | None = None,
        url: str | None = None,
    ) -> None:
        cfg = cfg or get_config()
        self.url = url if url is not None else _database_url()
        self.is_pg = _is_pg(self.url)
        if self.is_pg:
            self.path = None
        else:
            self.path = (
                Path(path) if path is not None
                else Path(cfg.paths.cache_dir) / "aes_scans.sqlite"
            )
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def q(self, sql: str) -> str:
        """Dialect-correct SQL. Call sites keep writing SQLite."""
        return _pg_sql(sql) if self.is_pg else sql

    def ex(self, conn, sql: str, params: Any = ()):
        """One statement, either driver.

        psycopg3 exposes ``execute`` on the connection but NOT ``executemany``
        -- that lives on the cursor -- so both go through here rather than
        being called on the connection directly.
        """
        sql = self.q(sql)
        if self.is_pg:
            cur = conn.cursor()
            cur.execute(sql, tuple(params) if params else None)
            return cur
        return conn.execute(sql, params)

    def exmany(self, conn, sql: str, seq) -> None:
        sql = self.q(sql)
        if self.is_pg:
            with conn.cursor() as cur:
                cur.executemany(sql, list(seq))
            return
        conn.executemany(sql, seq)

    def frame(self, conn, sql: str, params: Any = None) -> pd.DataFrame:
        """read_sql_query with the SQL translated for the live dialect."""
        return pd.read_sql_query(self.q(sql), conn, params=params)

    @contextmanager
    def connect(self):
        if self.is_pg:
            import psycopg
            from psycopg.rows import dict_row

            conn = psycopg.connect(self.url, row_factory=dict_row)
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()
            return

        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self.connect() as conn:
            if self.is_pg:
                # psycopg has no executescript; the schema is a handful of
                # idempotent CREATE IF NOT EXISTS statements.
                with conn.cursor() as cur:
                    for stmt in _pg_schema(SCHEMA).split(";"):
                        if stmt.strip():
                            cur.execute(stmt)
            else:
                conn.executescript(SCHEMA)

    # ------------------------------------------------------------- writing --
    def record_scan(
        self,
        scan_date: date | str,
        *,
        universe_size: int,
        watchlist_size: int,
        candidate_count: int,
        actionable_count: int,
        breadth: int | None = None,
        breadth_pctile: float | None = None,
        breadth_label: str | None = None,
        duration_s: float | None = None,
    ) -> None:
        with self.connect() as conn:
            self.ex(
                conn,
                """INSERT INTO aes_scans (scan_date, universe_size, watchlist_size,
                       candidate_count, actionable_count, breadth, breadth_pctile,
                       breadth_label, duration_s, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(scan_date) DO UPDATE SET
                       universe_size=excluded.universe_size,
                       watchlist_size=excluded.watchlist_size,
                       candidate_count=excluded.candidate_count,
                       actionable_count=excluded.actionable_count,
                       breadth=excluded.breadth,
                       breadth_pctile=excluded.breadth_pctile,
                       breadth_label=excluded.breadth_label,
                       duration_s=excluded.duration_s""",
                (_d(scan_date), universe_size, watchlist_size, candidate_count,
                 actionable_count, breadth, breadth_pctile, breadth_label, duration_s, _now()),
            )

    def add_candidates(self, rows: Sequence[Mapping[str, Any]], scan_date: date | str) -> int:
        """Upsert one day's candidates. Re-running a scan replaces that day."""
        if not rows:
            return 0
        sd = _d(scan_date)
        cols = _CAND_COLUMNS
        placeholders = ",".join("?" * (len(cols) + 1))
        updates = ",".join(f"{c}=excluded.{c}" for c in cols if c not in ("scan_date", "symbol"))
        # ``actionable`` and ``bucket`` are NOT NULL: default them rather than
        # letting a partially-populated row abort the whole evening's write.
        defaults = {"actionable": 0, "bucket": "reject"}
        payload = []
        for r in rows:
            vals = []
            for c in cols:
                if c == "scan_date":
                    vals.append(sd)
                    continue
                v = r.get(c)
                if v is None and c in defaults:
                    v = defaults[c]
                vals.append(v)
            payload.append(tuple(vals) + (_now(),))
        with self.connect() as conn:
            self.exmany(
                conn,
                f"INSERT INTO aes_candidates ({','.join(cols)},created_at) "
                f"VALUES ({placeholders}) "
                f"ON CONFLICT(scan_date,symbol) DO UPDATE SET {updates}",
                payload,
            )
        return len(payload)

    def add_audit(self, rows: Sequence[Mapping[str, Any]], scan_date: date | str) -> int:
        """Upsert the full-universe audit for one day. Re-running replaces it."""
        if rows is None or not len(rows):
            return 0
        sd = _d(scan_date)
        cols = _AUDIT_COLUMNS
        placeholders = ",".join("?" * (len(cols) + 1))
        updates = ",".join(f"{c}=excluded.{c}" for c in cols if c not in ("scan_date", "symbol"))
        payload = []
        for r in rows:
            vals = []
            for c in cols:
                if c == "scan_date":
                    vals.append(sd)
                    continue
                v = r.get(c)
                if v is not None and isinstance(v, float) and pd.isna(v):
                    v = None
                vals.append(v)
            payload.append(tuple(vals) + (_now(),))
        with self.connect() as conn:
            self.exmany(
                conn,
                f"INSERT INTO aes_audit ({','.join(cols)},created_at) "
                f"VALUES ({placeholders}) "
                f"ON CONFLICT(scan_date,symbol) DO UPDATE SET {updates}",
                payload,
            )
        return len(payload)

    def audit(
        self,
        scan_date: date | str | None = None,
        *,
        verdict: str | None = None,
        stage: str | None = None,
    ) -> pd.DataFrame:
        """The full-universe audit: every symbol, its stage and its reason."""
        q = "SELECT * FROM aes_audit"
        where, params = [], []
        if scan_date is not None:
            where.append("scan_date = ?")
            params.append(_d(scan_date))
        if verdict is not None:
            where.append("verdict = ?")
            params.append(verdict)
        if stage is not None:
            where.append("stage = ?")
            params.append(stage)
        if where:
            q += " WHERE " + " AND ".join(where)
        q += " ORDER BY stage_rank DESC, score DESC, symbol"
        with self.connect() as conn:
            return self.frame(conn, q, params)

    def upsert_outcome(self, candidate_id: int, outcome: Mapping[str, Any]) -> None:
        cols = _OUTCOME_COLUMNS
        placeholders = ",".join("?" * (len(cols) + 2))
        updates = ",".join(f"{c}=excluded.{c}" for c in cols)
        vals = (candidate_id, *[outcome.get(c) for c in cols], _now())
        with self.connect() as conn:
            self.ex(
                conn,
                f"INSERT INTO aes_outcomes (candidate_id,{','.join(cols)},updated_at) "
                f"VALUES ({placeholders}) "
                f"ON CONFLICT(candidate_id) DO UPDATE SET {updates}, updated_at=excluded.updated_at",
                vals,
            )

    # ------------------------------------------------------------- reading --
    def candidates(
        self,
        scan_date: date | str | None = None,
        *,
        bucket: str | None = None,
        actionable_only: bool = False,
        limit: int | None = None,
    ) -> pd.DataFrame:
        q = "SELECT * FROM aes_candidates"
        where, params = [], []
        if scan_date is not None:
            where.append("scan_date = ?")
            params.append(_d(scan_date))
        if bucket is not None:
            where.append("bucket = ?")
            params.append(bucket)
        if actionable_only:
            where.append("actionable = 1")
        if where:
            q += " WHERE " + " AND ".join(where)
        q += " ORDER BY scan_date DESC, score DESC"
        if limit:
            q += f" LIMIT {int(limit)}"
        with self.connect() as conn:
            return self.frame(conn, q, params)

    def latest_scan_date(self) -> str | None:
        with self.connect() as conn:
            row = self.ex(conn, "SELECT MAX(scan_date) AS d FROM aes_scans").fetchone()
        return row["d"] if row and row["d"] else None

    def scans(self, limit: int = 180) -> pd.DataFrame:
        with self.connect() as conn:
            return self.frame(
                conn, "SELECT * FROM aes_scans ORDER BY scan_date DESC LIMIT ?", [limit]
            )

    def pending_outcomes(self, max_age_days: int = 120) -> pd.DataFrame:
        """Candidates whose forward record is missing or not yet 25 bars old."""
        with self.connect() as conn:
            return self.frame(
                conn,
                """SELECT c.id, c.scan_date, c.symbol, c.entry_ref, c.stop, c.atr14
                   FROM aes_candidates c
                   LEFT JOIN aes_outcomes o ON o.candidate_id = c.id
                   WHERE c.actionable = 1
                     AND julianday('now') - julianday(c.scan_date) <= ?
                     AND (o.candidate_id IS NULL OR COALESCE(o.bars_elapsed, 0) < 25)
                   ORDER BY c.scan_date""",
                [max_age_days],
            )

    def forward_record(self, since: date | str | None = None) -> pd.DataFrame:
        """The accumulating live record: what was flagged, at what score, what it did."""
        q = """SELECT c.scan_date, c.symbol, c.bucket, c.score, c.entry_mode,
                      c.sc_fast, c.sc_absorbed, c.sc_rs, c.sc_cycles,
                      c.entry_ref, c.stop, c.qty, c.risk_amount,
                      o.bars_elapsed, o.ret_5, o.ret_10, o.ret_20, o.excess_20,
                      o.mfe_pct, o.mae_pct, o.stop_hit, o.realised_pct, o.exit_reason
               FROM aes_candidates c
               LEFT JOIN aes_outcomes o ON o.candidate_id = c.id
               WHERE c.actionable = 1"""
        params: list[Any] = []
        if since is not None:
            q += " AND c.scan_date >= ?"
            params.append(_d(since))
        q += " ORDER BY c.scan_date DESC"
        with self.connect() as conn:
            return self.frame(conn, q, params)


__all__ = ["AESScanStore", "SCHEMA"]
