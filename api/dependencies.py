"""FastAPI dependencies: current-user extraction and role gating.

The critical property here, worth stating explicitly since it's a security
boundary: the role used everywhere downstream (RBAC document filtering,
SQL table access, monitoring logs) comes from `get_current_user`, which
reads it out of a cryptographically verified JWT claim -- never from a
request body/query-param the caller could set to anything they want. A
QueryRequest has no role field for exactly this reason.
"""
import sys
from pathlib import Path

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.security import decode_access_token, TokenError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root
from auth import User  # noqa: E402

_bearer_scheme = HTTPBearer(auto_error=True)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> User:
    try:
        payload = decode_access_token(credentials.credentials)
    except TokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return User(username=payload["sub"], name=payload["name"], role=payload["role"])


def require_roles(*allowed_roles: str):
    """Dependency factory: raises 403 unless the authenticated user's role
    is one of allowed_roles. Used for endpoints that aren't gated by the
    RAG/SQL RBAC layer itself (e.g. the monitoring summary), so access
    control isn't silently left to "whatever the frontend happens to show"."""

    def _dependency(user: User = Depends(get_current_user)) -> User:
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user.role}' is not permitted to access this endpoint.",
            )
        return user

    return _dependency
