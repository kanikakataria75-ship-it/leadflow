"""Rate limiting.

Two buckets, because two kinds of request cost wildly different things:

* **Read endpoints** serve JSON that is already in memory or on disk. They are
  cheap; the limit exists only so a loop cannot pin the free-tier CPU.
* **The scan trigger** starts a job that fetches ~400 symbols from Yahoo
  Finance. Hammering it would get the host's IP throttled or blocked by Yahoo,
  which breaks the forward record for everyone including the owner. It gets a
  hard, deliberately small allowance.

In-memory, per-process, fixed-window. That is the right size for a single-user
tool on one instance: it needs no Redis, no extra service and no dependency. It
also means the counter resets if the host restarts the process, which for a
free tier that spins down on idle is an accepted limitation rather than a
surprise -- it is written here so nobody discovers it later.
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict

from fastapi import Request
from fastapi.responses import JSONResponse

#: (requests, window seconds) for ordinary API reads.
READ_LIMIT = (int(os.getenv("RATE_LIMIT_READS", "120")), 60)
#: The scan trigger. Two per hour is generous for a once-daily job.
SCAN_LIMIT = (int(os.getenv("RATE_LIMIT_SCANS", "2")), 3600)
#: Login attempts, to make brute-forcing the single password impractical.
LOGIN_LIMIT = (int(os.getenv("RATE_LIMIT_LOGINS", "8")), 900)

#: Paths that start real outbound work rather than reading cached state.
EXPENSIVE = ("/api/aes/scan",)

_lock = threading.Lock()
_hits: dict[tuple[str, str], list[float]] = defaultdict(list)


def _client(request: Request) -> str:
    # Render and Vercel both sit behind a proxy, so the socket address is the
    # proxy's. The first X-Forwarded-For entry is the real client.
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _bucket_for(path: str, method: str) -> tuple[str, tuple[int, int]]:
    if path.startswith("/api/auth/login"):
        return "login", LOGIN_LIMIT
    if method == "POST" and path.startswith(EXPENSIVE):
        return "scan", SCAN_LIMIT
    return "read", READ_LIMIT


def check(request: Request) -> JSONResponse | None:
    """None when allowed, or the 429 to return."""
    path = request.url.path
    if not path.startswith("/api/"):
        return None

    name, (limit, window) = _bucket_for(path, request.method)
    key = (_client(request), name)
    now = time.time()

    with _lock:
        hits = _hits[key]
        cutoff = now - window
        # drop expired timestamps in place
        hits[:] = [t for t in hits if t > cutoff]
        if len(hits) >= limit:
            retry = int(window - (now - hits[0])) + 1
            return JSONResponse(
                status_code=429,
                content={
                    "detail": (
                        f"Rate limit reached for {name} requests "
                        f"({limit} per {window}s). Try again in {retry}s."
                    )
                },
                headers={"Retry-After": str(retry)},
            )
        hits.append(now)
    return None


__all__ = ["LOGIN_LIMIT", "READ_LIMIT", "SCAN_LIMIT", "check"]
