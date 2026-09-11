"""Minimal mock authentication for demo purposes only.

NOT for production use. Before going live, replace this with real SSO
(Azure AD / Entra ID is the natural choice since the app deploys to Azure).
"""
import hashlib
from dataclasses import dataclass
from typing import Optional


@dataclass
class User:
    username: str
    name: str
    role: str


# username -> (sha256(password), display name, role)
_DEMO_USERS = {
    "alice.finance": (hashlib.sha256(b"finance123").hexdigest(), "Alice Chen", "finance"),
    "bob.hr": (hashlib.sha256(b"hr123").hexdigest(), "Bob Martinez", "hr"),
    "carol.ceo": (hashlib.sha256(b"exec123").hexdigest(), "Carol Whitfield", "executive"),
    "dave.eng": (hashlib.sha256(b"employee123").hexdigest(), "Dave Kim", "employee"),
}

_PLAINTEXT_FOR_DEMO_DISPLAY = {
    "alice.finance": "finance123",
    "bob.hr": "hr123",
    "carol.ceo": "exec123",
    "dave.eng": "employee123",
}


def authenticate(username: str, password: str) -> Optional[User]:
    record = _DEMO_USERS.get(username.strip().lower())
    if not record:
        return None
    hashed, name, role = record
    if hashlib.sha256(password.encode()).hexdigest() != hashed:
        return None
    return User(username=username, name=name, role=role)


def demo_credentials() -> list:
    """Return the demo login table for display in the UI/README."""
    return [
        {"username": u, "password": _PLAINTEXT_FOR_DEMO_DISPLAY[u], "role": rec[2], "name": rec[1]}
        for u, rec in _DEMO_USERS.items()
    ]
