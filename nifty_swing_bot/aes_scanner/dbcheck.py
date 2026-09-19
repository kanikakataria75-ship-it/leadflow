"""Verify the scan store against whatever database DATABASE_URL points at.

Run this once after provisioning Postgres, BEFORE trusting the deploy with the
forward record:

    DATABASE_URL="postgresql://..." python -m nifty_swing_bot.aes_scanner.dbcheck

It creates the schema, round-trips a row through every table the scanner
writes, reads it back through the same query paths the API uses, and then
removes what it inserted. It exits non-zero on any failure, so it can also be
used as a deploy gate in CI.

Why this exists as a script rather than a test: the Postgres path cannot be
exercised without a real Postgres, and the machine this was written on did not
have one. Rather than claim the migration works, the check is shipped so the
first person with a database can prove it in one command.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta

from .store import AESScanStore, _database_url, _is_pg

PROBE_DATE = "1990-01-02"   # far outside any real scan range
PROBE_SYMBOL = "__DBCHECK__"


def main() -> int:
    url = _database_url()
    kind = "postgres" if _is_pg(url) else "sqlite"
    print(f"target      : {kind}")
    if _is_pg(url):
        # never print credentials
        host = url.split("@")[-1].split("/")[0] if "@" in url else "?"
        print(f"host        : {host}")
    else:
        print("DATABASE_URL not set -- checking the local SQLite file instead.")

    ok = True
    store = AESScanStore()
    print("schema      : created / already present")

    try:
        store.record_scan(
            scan_date=PROBE_DATE, universe_size=1, watchlist_size=1,
            candidate_count=1, actionable_count=0, breadth=50.0,
            breadth_pctile=50.0, breadth_label="dbcheck", duration_s=0.0,
        )
        print("write scan  : ok")

        n = store.add_candidates(
            [{
                "symbol": PROBE_SYMBOL, "company": "dbcheck", "industry": "dbcheck",
                "bucket": "wait_and_watch", "score": 0.5, "actionable": 0,
            }],
            scan_date=PROBE_DATE,
        )
        print(f"write cand  : ok ({n} row)")

        scans = store.scans(limit=500)
        got_scan = (scans["scan_date"].astype(str) == PROBE_DATE).any() if not scans.empty else False
        cands = store.candidates(scan_date=PROBE_DATE)
        got_cand = (cands["symbol"] == PROBE_SYMBOL).any() if not cands.empty else False
        print(f"read back   : scan={'ok' if got_scan else 'MISSING'} candidate={'ok' if got_cand else 'MISSING'}")
        ok = ok and got_scan and got_cand

        # exercises the one query with date arithmetic, which is the only
        # statement whose SQL differs between the two dialects
        pend = store.pending_outcomes(max_age_days=10)
        print(f"date query  : ok ({len(pend)} pending rows)")

        rec = store.forward_record()
        print(f"record query: ok ({len(rec)} entries)")

    except Exception as exc:                                    # pragma: no cover
        ok = False
        print(f"FAILED      : {type(exc).__name__}: {exc}")

    # ---- clean up the probe rows -------------------------------------------
    try:
        with store.connect() as conn:
            store.ex(conn, "DELETE FROM aes_candidates WHERE scan_date = ?", (PROBE_DATE,))
            store.ex(conn, "DELETE FROM aes_audit WHERE scan_date = ?", (PROBE_DATE,))
            store.ex(conn, "DELETE FROM aes_scans WHERE scan_date = ?", (PROBE_DATE,))
        print("cleanup     : probe rows removed")
    except Exception as exc:                                    # pragma: no cover
        ok = False
        print(f"cleanup     : FAILED {type(exc).__name__}: {exc}")

    print()
    print("RESULT      :", "PASS - the store works against this database" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
