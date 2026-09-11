"""Role-based access control for document retrieval."""
from config import ROLE_ACCESS


def allowed_departments(role: str) -> set:
    return ROLE_ACCESS.get(role, set())


def build_chroma_filter(role: str):
    """Build a Chroma 'where' filter restricting results to departments the role can see."""
    depts = sorted(allowed_departments(role))
    if not depts:
        return {"department": "___none___"}  # matches nothing -> zero access
    if len(depts) == 1:
        return {"department": depts[0]}
    return {"department": {"$in": depts}}


def is_authorized(role: str, department: str) -> bool:
    return department in allowed_departments(role)
