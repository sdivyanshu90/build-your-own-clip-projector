from fastapi import HTTPException
from starlette.requests import Request

from clip_projector.config import Settings
from clip_projector.security import SlidingWindowRateLimiter, verify_api_key


def request(key: str) -> Request:
    return Request({"type": "http", "headers": [(b"x-api-key", key.encode())]})


def test_rate_limiter_expires_old_requests() -> None:
    limiter = SlidingWindowRateLimiter(1, 10)
    assert limiter.allow("a", 0)
    assert not limiter.allow("a", 1)
    assert limiter.allow("a", 10)


def test_api_key_enforcement() -> None:
    settings = Settings(environment="test", api_keys="correct-key", require_api_key=True)
    verify_api_key(request("correct-key"), settings)
    try:
        verify_api_key(request("wrong"), settings)
    except HTTPException as exc:
        assert exc.status_code == 401
    else:
        raise AssertionError("invalid key was accepted")
