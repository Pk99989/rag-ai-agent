"""Usage logging, cost tracking, and threshold alerting."""
import csv
import datetime as dt
from config import USAGE_LOG_PATH, MODEL_PRICING_PER_1M, GROQ_MODEL, DAILY_COST_ALERT_USD

_HEADERS = ["timestamp", "username", "role", "query", "blocked", "reason",
            "latency_ms", "tokens_in", "tokens_out", "cost_usd"]


def estimate_cost(tokens_in: int, tokens_out: int, model: str = GROQ_MODEL) -> float:
    pricing = MODEL_PRICING_PER_1M.get(model, {"input": 0.0, "output": 0.0})
    cost = (tokens_in / 1_000_000) * pricing["input"] + (tokens_out / 1_000_000) * pricing["output"]
    return round(cost, 8)


def _ensure_log_file():
    if not USAGE_LOG_PATH.exists():
        with open(USAGE_LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow(_HEADERS)


def log_interaction(user, role, query, result, latency_ms, tokens_in, tokens_out):
    _ensure_log_file()
    cost = estimate_cost(tokens_in, tokens_out)
    username = getattr(user, "username", str(user))
    with open(USAGE_LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow([
            dt.datetime.utcnow().isoformat(), username, role, query,
            result.get("blocked", False), result.get("reason"),
            round(latency_ms, 1), tokens_in, tokens_out, cost,
        ])
    check_daily_cost_alert()
    return cost


def today_usage_summary() -> dict:
    if not USAGE_LOG_PATH.exists():
        return {"queries": 0, "total_cost_usd": 0.0, "avg_latency_ms": 0.0, "blocked": 0}
    today = dt.date.today().isoformat()
    rows = []
    with open(USAGE_LOG_PATH, newline="") as f:
        for row in csv.DictReader(f):
            if row["timestamp"].startswith(today):
                rows.append(row)
    if not rows:
        return {"queries": 0, "total_cost_usd": 0.0, "avg_latency_ms": 0.0, "blocked": 0}
    total_cost = sum(float(r["cost_usd"]) for r in rows)
    avg_latency = sum(float(r["latency_ms"]) for r in rows) / len(rows)
    blocked = sum(1 for r in rows if r["blocked"] == "True")
    return {
        "queries": len(rows),
        "total_cost_usd": round(total_cost, 6),
        "avg_latency_ms": round(avg_latency, 1),
        "blocked": blocked,
    }


def check_daily_cost_alert():
    summary = today_usage_summary()
    if summary["total_cost_usd"] >= DAILY_COST_ALERT_USD:
        print(f"[COST ALERT] Today's spend ${summary['total_cost_usd']} has reached/exceeded "
              f"the ${DAILY_COST_ALERT_USD} threshold.")
    return summary
