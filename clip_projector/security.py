from __future__ import annotations

import hashlib
import hmac
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status

from clip_projector.config import Settings


class SlidingWindowRateLimiter:
    def __init__(self, limit: int, window_seconds: float = 60.0) -> None:
        self.limit, self.window = limit, window_seconds
        self.requests: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, identifier: str, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        history = self.requests[identifier]
        while history and history[0] <= current - self.window:
            history.popleft()
        if len(history) >= self.limit:
            return False
        history.append(current)
        return True


def client_identifier(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def verify_api_key(request: Request, settings: Settings) -> None:
    if not settings.require_api_key:
        return
    supplied = request.headers.get("X-API-Key", "")
    valid = any(hmac.compare_digest(supplied, candidate) for candidate in settings.key_set)
    if not valid:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "valid X-API-Key required")


def redact_key(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:12] if value else "anonymous"
