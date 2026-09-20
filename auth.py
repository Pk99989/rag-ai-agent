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
# Phase 7: added one demo user per new role (sales, operations, manager,
# admin) so all roles from the RBAC upgrade are actually testable through
# the UI, not just present in config.ROLE_ACCESS.
# Olist-only pivot: removed bob.hr along with the "hr" role in config.py --
# there is nothing left for an HR account to demonstrate once the AtliQ
# employee-handbook/payroll-policy documents it existed to guard are gone.
# The remaining 7 users map 1:1 onto the 7 roles config.py now defines.
_DEMO_USERS = {
    "alice.finance": (hashlib.sha256(b"finance123").hexdigest(), "Alice Chen", "finance"),
    "carol.ceo": (hashlib.sha256(b"exec123").hexdigest(), "Carol Whitfield", "executive"),
    "dave.eng": (hashlib.sha256(b"employee123").hexdigest(), "Dave Kim", "employee"),
    "erin.sales": (hashlib.sha256(b"sales123").hexdigest(), "Erin Alves", "sales"),
    "frank.ops": (hashlib.sha256(b"ops123").hexdigest(), "Frank Silva", "operations"),
    "grace.mgr": (hashlib.sha256(b"manager123").hexdigest(), "Grace Nakamura", "manager"),
    "henry.admin": (hashlib.sha256(b"admin123").hexdigest(), "Henry Osei", "admin"),
}

_PLAINTEXT_FOR_DEMO_DISPLAY = {
    "alice.finance": "finance123",
    "carol.ceo": "exec123",
    "dave.eng": "employee123",
    "erin.sales": "sales123",
    "frank.ops": "ops123",
    "grace.mgr": "manager123",
    "henry.admin": "admin123",
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
