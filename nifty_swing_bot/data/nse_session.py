"""Warmed-up HTTP session for NSE India.

NSE actively blocks naive scrapers: a bare ``requests.get`` against an archive
URL returns 401/403 because the edge expects a browser-like client that has
already picked up cookies from the main site. The working pattern is:

1. Issue a GET to ``https://www.nseindia.com`` with full browser headers so the
   ``nsit``/``nseappid`` cookies land in the jar.
2. Reuse that same session (cookies + headers + ``Referer``) for archive CSVs.
3. Re-warm transparently whenever a request comes back with an auth-ish status.

This module owns that dance so the fetchers can stay simple.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from typing import Final

import requests

from ..config import get_config

logger = logging.getLogger(__name__)

_BROWSER_HEADERS: Final[dict[str, str]] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}

# Statuses that indicate the session lost its blessing rather than the resource
# genuinely being absent.
_REWARM_STATUSES: Final[frozenset[int]] = frozenset({401, 403, 405, 429})


class NSESession:
    """A thread-safe, self-re-warming :class:`requests.Session` wrapper."""

    def __init__(self, warm_ttl_s: float = 600.0) -> None:
        cfg = get_config().data
        self._base_url = cfg.nse_base_url
        self._timeout = cfg.nse_timeout_s
        self._max_retries = cfg.nse_max_retries
        self._delay = cfg.nse_request_delay_s
        self._warm_ttl_s = warm_ttl_s

        self._session = requests.Session()
        self._session.headers.update(_BROWSER_HEADERS)
        self._warmed_at: float = 0.0
        self._lock = threading.Lock()
        self._last_request_at: float = 0.0

    # ------------------------------------------------------------- warm-up --
    def warm(self, force: bool = False) -> bool:
        """Prime the cookie jar from the NSE homepage.

        Returns:
            True if the session holds cookies afterwards.
        """
        with self._lock:
            fresh = (time.monotonic() - self._warmed_at) < self._warm_ttl_s
            if fresh and not force and len(self._session.cookies) > 0:
                return True
            try:
                # The homepage sets the base cookies; the market-data page tends
                # to add the ones the archive edge actually checks.
                for url in (self._base_url, f"{self._base_url}/market-data/live-equity-market"):
                    resp = self._session.get(url, timeout=self._timeout)
                    logger.debug("Warm-up GET %s -> %s", url, resp.status_code)
                    time.sleep(self._delay)
                self._warmed_at = time.monotonic()
            except requests.RequestException as exc:
                logger.warning("NSE warm-up failed: %s", exc)
                return False
            return len(self._session.cookies) > 0

    # ------------------------------------------------------------- requests --
    def _throttle(self) -> None:
        """Keep a polite minimum gap between requests, with a little jitter."""
        elapsed = time.monotonic() - self._last_request_at
        wait = self._delay - elapsed
        if wait > 0:
            time.sleep(wait + random.uniform(0, 0.25))
        self._last_request_at = time.monotonic()

    def get(
        self,
        url: str,
        *,
        referer: str | None = None,
        accept: str | None = None,
        allow_404: bool = True,
    ) -> requests.Response | None:
        """GET a URL through the warmed session, retrying and re-warming.

        Args:
            url: Absolute URL to fetch.
            referer: Referer header; defaults to the NSE homepage.
            accept: Override the Accept header (CSVs prefer ``text/csv``).
            allow_404: Return None on 404 instead of raising. NSE returns 404
                for market holidays, which is an expected, non-exceptional case.

        Returns:
            The response on success, or None if the resource is genuinely absent
            or every retry failed.
        """
        self.warm()
        headers = {"Referer": referer or f"{self._base_url}/"}
        if accept:
            headers["Accept"] = accept
        headers["Sec-Fetch-Dest"] = "empty"
        headers["Sec-Fetch-Mode"] = "cors"
        headers["Sec-Fetch-Site"] = "same-site"

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            self._throttle()
            try:
                resp = self._session.get(url, headers=headers, timeout=self._timeout)
            except requests.RequestException as exc:
                last_exc = exc
                logger.debug("GET %s attempt %d errored: %s", url, attempt + 1, exc)
                time.sleep(min(2**attempt, 8))
                continue

            if resp.status_code == 200:
                return resp
            if resp.status_code == 404 and allow_404:
                logger.debug("GET %s -> 404 (treated as absent)", url)
                return None
            if resp.status_code in _REWARM_STATUSES:
                logger.debug(
                    "GET %s -> %s; re-warming session (attempt %d)",
                    url,
                    resp.status_code,
                    attempt + 1,
                )
                self.warm(force=True)
                time.sleep(min(2**attempt, 8))
                continue

            logger.debug("GET %s -> unexpected %s", url, resp.status_code)
            time.sleep(min(2**attempt, 8))

        if last_exc is not None:
            logger.warning("GET %s exhausted retries: %s", url, last_exc)
        else:
            logger.warning("GET %s exhausted retries.", url)
        return None

    def close(self) -> None:
        """Release the underlying connection pool."""
        self._session.close()


_SHARED: NSESession | None = None
_SHARED_LOCK = threading.Lock()


def get_nse_session() -> NSESession:
    """Return the process-wide shared NSE session, creating it on first use."""
    global _SHARED
    with _SHARED_LOCK:
        if _SHARED is None:
            _SHARED = NSESession()
        return _SHARED


__all__ = ["NSESession", "get_nse_session"]
