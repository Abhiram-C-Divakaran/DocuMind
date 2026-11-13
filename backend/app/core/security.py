import hashlib
import secrets
import threading
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, Optional, Tuple

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError

from backend.app.config import settings
from backend.app.logger import logger

_password_hasher = PasswordHasher(
    time_cost=2,
    memory_cost=65536,
    parallelism=2,
    hash_len=32,
    salt_len=16,
)


def hash_password(password: str) -> str:
    return _password_hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    if not password or not password_hash:
        return False
    try:
        return _password_hasher.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, Exception):
        return False


def hash_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def _get_signing_secret() -> str:
    return settings.validate_runtime_security()


def create_access_token(
    *,
    subject: str,
    email: str,
    role: str,
    expires_delta: Optional[timedelta] = None,
) -> Tuple[str, datetime]:
    now = datetime.now(timezone.utc)
    exp = now + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    payload: Dict[str, Any] = {
        "sub": subject,
        "email": email,
        "role": role,
        "type": "access",
        "jti": str(uuid.uuid4()),
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
    }
    token = jwt.encode(payload, _get_signing_secret(), algorithm=settings.JWT_ALGORITHM)
    return token, exp


def create_refresh_token(
    *,
    subject: str,
    expires_delta: Optional[timedelta] = None,
) -> Tuple[str, str, datetime]:
    now = datetime.now(timezone.utc)
    exp = now + (expires_delta or timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS))
    jti = str(uuid.uuid4())
    payload: Dict[str, Any] = {
        "sub": subject,
        "type": "refresh",
        "jti": jti,
        "nonce": secrets.token_hex(8),
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
    }
    token = jwt.encode(payload, _get_signing_secret(), algorithm=settings.JWT_ALGORITHM)
    return token, jti, exp


def decode_token(token: str, expected_type: Optional[str] = None) -> Dict[str, Any]:
    payload = jwt.decode(
        token,
        _get_signing_secret(),
        algorithms=[settings.JWT_ALGORITHM],
    )
    if expected_type and payload.get("type") != expected_type:
        raise jwt.InvalidTokenError("Unexpected token type.")
    return payload


class SlidingWindowRateLimiter:
    """
    Sliding-window rate limiter using Redis sorted sets when available, with an
    in-memory thread-safe fallback for single-process development and automated tests.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buckets: Dict[str, Deque[float]] = defaultdict(deque)
        self._redis_client: Any = None
        self._redis_checked: bool = False

    def _get_redis(self) -> Any:
        if self._redis_checked:
            return self._redis_client
        self._redis_checked = True
        try:
            import redis

            client = redis.Redis.from_url(settings.REDIS_URL, socket_connect_timeout=0.5, socket_timeout=0.5)
            client.ping()
            self._redis_client = client
        except Exception:
            self._redis_client = None
        return self._redis_client

    def reset_memory(self) -> None:
        with self._lock:
            self._buckets.clear()

    def check_and_consume(self, key: str, max_requests: int, window_seconds: int = 60) -> Tuple[bool, int]:
        if max_requests <= 0:
            return True, max_requests

        now = time.time()
        redis_client = self._get_redis()
        if redis_client is not None:
            try:
                redis_key = f"documind:rl:{key}"
                cutoff = now - window_seconds
                pipe = redis_client.pipeline()
                pipe.zremrangebyscore(redis_key, 0, cutoff)
                pipe.zcard(redis_key)
                pipe.zadd(redis_key, {f"{now}:{uuid.uuid4().hex[:8]}": now})
                pipe.expire(redis_key, window_seconds + 5)
                _, current_count, _, _ = pipe.execute()
                if int(current_count) >= max_requests:
                    return False, 0
                return True, max(0, max_requests - int(current_count) - 1)
            except Exception as exc:
                logger.warning("Redis rate limiter unavailable, using in-memory fallback", extra={"error_category": type(exc).__name__})

        cutoff = now - window_seconds
        with self._lock:
            dq = self._buckets[key]
            while dq and dq[0] <= cutoff:
                dq.popleft()
            if len(dq) >= max_requests:
                return False, 0
            dq.append(now)
            return True, max(0, max_requests - len(dq))


rate_limiter = SlidingWindowRateLimiter()

