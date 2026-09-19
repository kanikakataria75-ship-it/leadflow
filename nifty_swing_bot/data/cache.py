"""Local caching layer.

Every network fetch in this project goes through :class:`ParquetCache` so that
re-runs are cheap and offline-repeatable. The cache is a plain directory of
parquet files plus a JSON sidecar holding fetch metadata, which keeps it easy
to inspect and trivial to blow away.

Two knobs control freshness:

``refresh=True``
    Ignore whatever is on disk and re-fetch.
``ttl_hours``
    Treat an entry older than this as stale. ``None`` means never expire, which
    is the right setting for historical bhavcopy files that can never change.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_SAFE_KEY = re.compile(r"[^A-Za-z0-9._-]+")


def _slugify(key: str) -> str:
    """Turn an arbitrary cache key into a filesystem-safe stem.

    Long or exotic keys are hashed so that we never exceed Windows' path
    limits, while ordinary keys such as ``RELIANCE`` stay human-readable.
    """
    safe = _SAFE_KEY.sub("_", key).strip("_")
    if not safe or len(safe) > 80:
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        prefix = safe[:40].rstrip("_")
        return f"{prefix}_{digest}" if prefix else digest
    return safe


class ParquetCache:
    """A namespaced, TTL-aware parquet cache for pandas DataFrames."""

    def __init__(self, root: Path, namespace: str = "") -> None:
        self.root = Path(root) / namespace if namespace else Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._meta_path = self.root / "_meta.json"

    # ---------------------------------------------------------------- paths --
    def path_for(self, key: str) -> Path:
        """Return the parquet path a given key maps to."""
        return self.root / f"{_slugify(key)}.parquet"

    # ------------------------------------------------------------- metadata --
    def _read_meta(self) -> dict[str, Any]:
        if not self._meta_path.exists():
            return {}
        try:
            return json.loads(self._meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt cache metadata at %s; resetting.", self._meta_path)
            return {}

    def _write_meta(self, meta: dict[str, Any]) -> None:
        tmp = self._meta_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        tmp.replace(self._meta_path)

    def _touch(self, key: str, rows: int) -> None:
        meta = self._read_meta()
        meta[key] = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "rows": rows,
        }
        self._write_meta(meta)

    def fetched_at(self, key: str) -> datetime | None:
        """When this key was last written, or None if never."""
        entry = self._read_meta().get(key)
        if not entry:
            return None
        try:
            return datetime.fromisoformat(entry["fetched_at"])
        except (KeyError, ValueError):
            return None

    # ------------------------------------------------------------ accessors --
    def is_fresh(self, key: str, ttl_hours: float | None) -> bool:
        """True if the key exists on disk and is within its TTL."""
        if not self.path_for(key).exists():
            return False
        if ttl_hours is None:
            return True
        stamp = self.fetched_at(key)
        if stamp is None:
            # File exists but metadata was lost; treat as stale so we refetch.
            return False
        return datetime.now(timezone.utc) - stamp < timedelta(hours=ttl_hours)

    def get(self, key: str, ttl_hours: float | None = None) -> pd.DataFrame | None:
        """Read a cached frame, or None when missing/stale/unreadable."""
        if not self.is_fresh(key, ttl_hours):
            return None
        try:
            return pd.read_parquet(self.path_for(key))
        except (OSError, ValueError) as exc:  # corrupt/partial parquet
            logger.warning("Discarding unreadable cache entry %s: %s", key, exc)
            self.drop(key)
            return None

    def put(self, key: str, frame: pd.DataFrame) -> Path:
        """Write a frame to the cache atomically and stamp its metadata."""
        path = self.path_for(key)
        tmp = path.with_suffix(".parquet.tmp")
        # A non-default index (e.g. a DatetimeIndex) must survive the round trip.
        frame.to_parquet(tmp, index=True)
        tmp.replace(path)
        self._touch(key, len(frame))
        return path

    def drop(self, key: str) -> None:
        """Remove a single entry and its metadata."""
        self.path_for(key).unlink(missing_ok=True)
        meta = self._read_meta()
        if meta.pop(key, None) is not None:
            self._write_meta(meta)

    def clear(self) -> int:
        """Delete every parquet file in this namespace. Returns the count."""
        count = 0
        for path in self.root.glob("*.parquet"):
            path.unlink(missing_ok=True)
            count += 1
        self._write_meta({})
        return count

    def keys(self) -> list[str]:
        """All keys currently recorded in metadata."""
        return sorted(self._read_meta())

    # ------------------------------------------------------------- combined --
    def get_or_fetch(
        self,
        key: str,
        loader: Callable[[], pd.DataFrame | None],
        *,
        refresh: bool = False,
        ttl_hours: float | None = None,
        store_empty: bool = True,
    ) -> pd.DataFrame | None:
        """Return a cached frame, otherwise call ``loader`` and cache the result.

        Args:
            key: Cache key.
            loader: Zero-arg callable performing the actual fetch. It may return
                None or an empty frame to signal "nothing available".
            refresh: Bypass the cache and refetch.
            ttl_hours: Freshness window; None means never expire.
            store_empty: Cache empty results too. This is important for NSE
                bhavcopy on market holidays -- without it every run would
                re-request the same non-existent files.

        Returns:
            The cached or freshly fetched frame, or None if the loader produced
            nothing and there was no usable cache entry.
        """
        if not refresh:
            cached = self.get(key, ttl_hours)
            if cached is not None:
                return cached

        frame = loader()
        if frame is None:
            return None
        if frame.empty and not store_empty:
            return frame
        self.put(key, frame)
        return frame


def timed(label: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Small decorator that logs how long a fetch took."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            logger.info("%s took %.2fs", label, time.perf_counter() - start)
            return result

        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        return wrapper

    return decorator


__all__ = ["ParquetCache", "timed"]
