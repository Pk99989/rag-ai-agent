"""New modular components for the Enterprise AI Knowledge & Analytics Assistant upgrade.

The original flat-file modules (config.py, rbac.py, guardrails.py, ingest.py,
rag_chain.py, monitoring.py, auth.py, app.py) remain in the repo root and
untouched, per the upgrade's "preserve working functionality, do not
blindly overwrite" constraint. New subsystems (SQL, router, and later
retrieval/generation/api upgrades) are added here, under the project
structure the spec asked for. A deliberate migration of the old modules
into this package is a separate, explicit later step -- not a side effect
of adding new features.
"""
