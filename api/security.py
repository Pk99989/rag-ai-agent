"""JWT issuance/verification for the FastAPI backend.

Deliberately thin: this module only handles token mechanics. The actual
username/password check still goes through the existing auth.authenticate()
(the project's mock user store) -- this module never re-implements or
bypasses that. The role embedded in a token is copied verbatim from the
User object authenticate() returned for that login, at issuance time only;
nothing in this module lets a caller change the role of an existing token
without re-authenticating.
"""
import datetime as dt
import sys
from pathlib import Path

import jwt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root
from config import JWT_SECRET_KEY, JWT_ALGORITHM, JWT_ACCESS_TOKEN_EXPIRE_MINUTES  # noqa: E402


class TokenError(Exception):
    """Raised for any invalid/expired/malformed token -- callers should treat
    this uniformly as 401 Unauthorized rather than branching on the cause,
    to avoid leaking which specific check failed to a potential attacker."""


def create_access_token(username: str, role: str, name: str) -> tuple:
    """Returns (token, expires_in_minutes)."""
    now = dt.datetime.now(dt.timezone.utc)
    expire = now + dt.timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": username,
        "role": role,
        "name": name,
        "iat": now,
        "exp": expire,
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token, JWT_ACCESS_TOKEN_EXPIRE_MINUTES


def decode_access_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        raise TokenError("token expired") from exc
    except jwt.InvalidTokenError as exc:
        raise TokenError("invalid token") from exc

    for required in ("sub", "role", "name"):
        if required not in payload:
            raise TokenError(f"token missing required claim: {required}")
    return payload
