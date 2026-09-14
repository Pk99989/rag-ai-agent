"""Phase 11: structured, per-request monitoring. Coexists with the
existing root-level monitoring.py (CSV usage log + cost alerting, used by
app.py's sidebar and unchanged by this module) rather than replacing it --
this adds a richer JSONL record per request with the specific field list
Phase 11 asked for (route, per-stage latency, retrieved documents,
guardrail result, confidence), which the CSV schema was never designed to
hold and which several other tools (dashboard, future FastAPI /metrics
endpoint) need in a structured form.
"""
