"""Phase 11 monitoring dashboard -- reads the structured JSONL log written
by src/rag_agent/router/dispatch.py (one real record per request handled
through handle_query(), i.e. every request app.py has served since Phase
11). Does not call any LLM and costs nothing to run; it only reads
logs/structured_log.jsonl.

Run:
    streamlit run dashboard.py

If no requests have been logged yet, this shows an empty-state message
rather than fabricated placeholder charts.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from rag_agent.monitoring.structured_logger import read_structured_log

st.set_page_config(page_title="RAG Agent Monitoring", page_icon="📊", layout="wide")
st.title("📊 RAG Agent Monitoring Dashboard")
st.caption("Real per-request data from logs/structured_log.jsonl -- nothing here is simulated.")

records = read_structured_log()

if not records:
    st.info(
        "No requests logged yet. This dashboard reads logs/structured_log.jsonl, written by "
        "every call to handle_query() (see src/rag_agent/router/dispatch.py). Use the chat app "
        "(streamlit run app.py) to generate some real traffic, then reload this page."
    )
    st.stop()

df = pd.DataFrame(records)
df["timestamp"] = pd.to_datetime(df["timestamp"])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total requests", len(df))
col2.metric("Blocked", int(df["guardrail_result"].apply(lambda g: g.get("blocked", False)).sum()))
col3.metric("Total est. cost (USD)", f"${df['estimated_cost_usd'].sum():.6f}")
col4.metric("Avg total latency (ms)", f"{df['total_latency_ms'].mean():.0f}")

st.divider()

left, right = st.columns(2)
with left:
    st.subheader("Requests by route")
    st.bar_chart(df["query_route"].value_counts())
with right:
    st.subheader("Requests by confidence")
    conf_counts = df["confidence"].fillna("n/a").value_counts()
    st.bar_chart(conf_counts)

st.divider()

st.subheader("Latency breakdown by route (ms, mean)")
latency_cols = ["retrieval_latency_ms", "reranking_latency_ms", "generation_latency_ms",
                "sql_execution_time_ms", "vision_latency_ms", "total_latency_ms"]
latency_by_route = df.groupby("query_route")[latency_cols].mean(numeric_only=True).round(1)
st.dataframe(latency_by_route, use_container_width=True)

st.divider()

st.subheader("Requests over time")
by_minute = df.set_index("timestamp").resample("1min").size()
if len(by_minute) > 1:
    st.line_chart(by_minute)
else:
    st.caption("Not enough time spread yet to plot a trend -- all requests landed in the same minute.")

st.divider()

st.subheader("Recent requests")
display_cols = ["timestamp", "user_role", "query_route", "route_confidence", "confidence",
                 "total_latency_ms", "estimated_cost_usd", "question", "answer_preview"]
st.dataframe(df[display_cols].sort_values("timestamp", ascending=False).head(50),
             use_container_width=True)

st.divider()

st.subheader("Guardrail activity")
blocked_df = df[df["guardrail_result"].apply(lambda g: g.get("blocked", False))]
if blocked_df.empty:
    st.caption("No blocked requests logged yet.")
else:
    reasons = blocked_df["guardrail_result"].apply(lambda g: g.get("reason", "unknown"))
    st.bar_chart(reasons.value_counts())
