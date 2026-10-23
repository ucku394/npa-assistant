"""Access-control helpers shared by the web API and the Telegram bot."""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable

_MAX_TRACKED_KEYS = 10_000


class SlidingWindowRateLimiter:
    """In-memory per-key limiter: at most ``limit`` hits per ``window_seconds``.

    State is per process, which matches the single-instance deployment.
    A ``limit`` of 0 or less disables limiting.
    """

    def __init__(
        self,
        limit: int,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.limit = limit
        self.window_seconds = window_seconds
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.limit > 0

    def hit(self, key: str) -> float:
        """Register a request; return 0 if allowed, else seconds until retry."""
        if not self.enabled:
            return 0.0
        now = self._clock()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= self.window_seconds:
                hits.popleft()
            if len(hits) >= self.limit:
                return max(self.window_seconds - (now - hits[0]), 0.001)
            hits.append(now)
            if len(self._hits) > _MAX_TRACKED_KEYS:
                self._prune(now)
            return 0.0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()

    def _prune(self, now: float) -> None:
        stale = [
            key for key, hits in self._hits.items()
            if not hits or now - hits[-1] >= self.window_seconds
        ]
        for key in stale:
            del self._hits[key]
