"""Generate meaningful business-report RAG documents from data/olist.db.

Usage:
    python scripts/generate_olist_rag_docs.py

Deliberate scope decision (confirmed with the project owner, not assumed):
this does NOT generate one document per raw order (~99k) or per raw product
(~33k). Those are exact-lookup questions Text-to-SQL (Phase 3) already
answers faster and more accurately than embedding six-figure counts of
near-identical tiny documents would. What RAG is actually good at --
summaries, comparisons, "how is X performing overall" -- is what gets
generated here:
    - one performance report per seller (record-level, ~3k documents)
    - product category performance (aggregated, ~70 documents)
    - monthly sales reports (aggregated, ~2 years of months)
    - regional (per-state) sales summaries (~27 Brazilian states)
    - payment method summaries (~5 payment types)
    - one overall delivery-performance report
    - one overall order-status report

Each document is written as a markdown file with a small frontmatter block
(source/document_type/department/access_level/record_id) that
ingest.py parses and attaches as ChromaDB metadata -- the same metadata
shape used for RBAC filtering as the existing docs_<department>_*.md files,
so both document sources retrieve through one RBAC-filtered index.
"""
import sqlite3
import sys
from pathlib import Path
from textwrap import dedent

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import OLIST_DB_PATH, RAG_DOCS_DIR


def _connect():
    if not OLIST_DB_PATH.exists():
        raise FileNotFoundError(
            f"{OLIST_DB_PATH} not found. Run scripts/prepare_olist.py first."
        )
    conn = sqlite3.connect(f"file:{OLIST_DB_PATH.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _slug(value: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(value)).strip("_").lower()


def _write_doc(document_type: str, department: str, record_id: str, body: str) -> None:
    RAG_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    frontmatter = dedent(f"""\
        ---
        source: olist
        document_type: {document_type}
        department: {department}
        access_level: internal
        record_id: {record_id}
        ---
        """)
    filename = f"olist_{document_type}_{_slug(record_id)}.md"
    (RAG_DOCS_DIR / filename).write_text(frontmatter + body.strip() + "\n", encoding="utf-8")


def generate_seller_performance(conn) -> int:
    rows = conn.execute("""
        SELECT s.seller_id, s.seller_state,
               COUNT(DISTINCT oi.order_id) AS num_orders,
               AVG(oi.price) AS avg_item_price,
               AVG(o.delivery_days) AS avg_delivery_days,
               AVG(r.review_score) AS avg_review_score
        FROM sellers s
        JOIN order_items oi ON s.seller_id = oi.seller_id
        JOIN orders o ON oi.order_id = o.order_id
        LEFT JOIN reviews r ON o.order_id = r.order_id
        GROUP BY s.seller_id, s.seller_state
    """).fetchall()
    for row in rows:
        # avg_delivery_days is NULL when every one of this seller's orders
        # is undelivered (canceled/still in transit) -- a real, legitimate
        # data condition (found by running this, not anticipated), not a
        # bug to paper over with a fabricated number.
        delivery_line = (
            f"Average Delivery Time: {row['avg_delivery_days']:.1f} days"
            if row["avg_delivery_days"] is not None
            else "Average Delivery Time: no delivered orders recorded"
        )
        review_line = (
            f"Average Review Score: {row['avg_review_score']:.2f} / 5"
            if row["avg_review_score"] is not None
            else "Average Review Score: no reviews recorded"
        )
        body = dedent(f"""\
            # Seller Performance Report

            Seller ID: {row['seller_id']}
            Seller State: {row['seller_state']}
            Number of Orders: {row['num_orders']}
            Average Item Price: R$ {row['avg_item_price']:.2f}
            {delivery_line}
            {review_line}
        """)
        _write_doc("seller_performance", "sellers", row["seller_id"], body)
    return len(rows)


def generate_category_performance(conn) -> int:
    rows = conn.execute("""
        SELECT p.product_category_name_english AS category,
               COUNT(DISTINCT oi.order_id) AS num_orders,
               SUM(oi.price + oi.freight_value) AS total_revenue,
               AVG(oi.price) AS avg_price,
               AVG(r.review_score) AS avg_review_score
        FROM order_items oi
        JOIN products p ON oi.product_id = p.product_id
        JOIN orders o ON oi.order_id = o.order_id
        LEFT JOIN reviews r ON o.order_id = r.order_id
        GROUP BY p.product_category_name_english
    """).fetchall()
    for row in rows:
        body = dedent(f"""\
            # Product Category Performance Report

            Category: {row['category']}
            Number of Orders: {row['num_orders']}
            Total Revenue: R$ {row['total_revenue']:.2f}
            Average Item Price: R$ {row['avg_price']:.2f}
            {'Average Review Score: %.2f / 5' % row['avg_review_score'] if row['avg_review_score'] is not None else 'Average Review Score: no reviews recorded'}
        """)
        _write_doc("category_performance", "products", row["category"], body)
    return len(rows)


def generate_monthly_sales(conn) -> int:
    rows = conn.execute("""
        SELECT strftime('%Y-%m', o.order_purchase_timestamp) AS month,
               COUNT(DISTINCT o.order_id) AS num_orders,
               SUM(oi.price + oi.freight_value) AS total_revenue,
               AVG(oi.price + oi.freight_value) AS avg_order_value
        FROM orders o
        JOIN order_items oi ON o.order_id = oi.order_id
        GROUP BY month ORDER BY month
    """).fetchall()
    for row in rows:
        body = dedent(f"""\
            # Monthly Sales Report -- {row['month']}

            Month: {row['month']}
            Number of Orders: {row['num_orders']}
            Total Revenue: R$ {row['total_revenue']:.2f}
            Average Order Value: R$ {row['avg_order_value']:.2f}
        """)
        _write_doc("monthly_sales_report", "revenue", row["month"], body)
    return len(rows)


def generate_regional_summary(conn) -> int:
    rows = conn.execute("""
        SELECT c.customer_state AS state,
               COUNT(DISTINCT o.order_id) AS num_orders,
               SUM(oi.price + oi.freight_value) AS total_revenue,
               AVG(r.review_score) AS avg_review_score
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        JOIN order_items oi ON o.order_id = oi.order_id
        LEFT JOIN reviews r ON o.order_id = r.order_id
        GROUP BY c.customer_state
    """).fetchall()
    for row in rows:
        body = dedent(f"""\
            # Regional Sales Summary -- {row['state']}

            Customer State: {row['state']}
            Number of Orders: {row['num_orders']}
            Total Revenue: R$ {row['total_revenue']:.2f}
            {'Average Review Score: %.2f / 5' % row['avg_review_score'] if row['avg_review_score'] is not None else 'Average Review Score: no reviews recorded'}
        """)
        _write_doc("regional_summary", "sales", row["state"], body)
    return len(rows)


def generate_payment_summary(conn) -> int:
    rows = conn.execute("""
        SELECT payment_type,
               COUNT(*) AS num_transactions,
               SUM(payment_value) AS total_value,
               AVG(payment_installments) AS avg_installments
        FROM payments
        GROUP BY payment_type
    """).fetchall()
    for row in rows:
        body = dedent(f"""\
            # Payment Method Summary -- {row['payment_type']}

            Payment Type: {row['payment_type']}
            Number of Transactions: {row['num_transactions']}
            Total Value: R$ {row['total_value']:.2f}
            Average Installments: {row['avg_installments']:.1f}
        """)
        _write_doc("payment_summary", "payments", row["payment_type"], body)
    return len(rows)


def generate_delivery_performance(conn) -> int:
    rows = conn.execute("""
        SELECT delivery_status, COUNT(*) AS num_orders, AVG(delivery_days) AS avg_days
        FROM orders GROUP BY delivery_status
    """).fetchall()
    lines = [f"- {r['delivery_status']}: {r['num_orders']} orders"
             + (f", average {r['avg_days']:.1f} days" if r["avg_days"] is not None else "")
             for r in rows]
    body = "# Delivery Performance Report (Overall)\n\n" + "\n".join(lines) + "\n"
    _write_doc("delivery_performance", "delivery", "overall", body)
    return 1


def generate_order_status_summary(conn) -> int:
    rows = conn.execute("""
        SELECT order_status, COUNT(*) AS num_orders FROM orders GROUP BY order_status
        ORDER BY num_orders DESC
    """).fetchall()
    lines = [f"- {r['order_status']}: {r['num_orders']} orders" for r in rows]
    body = "# Order Status Summary (Overall)\n\n" + "\n".join(lines) + "\n"
    _write_doc("order_status_summary", "orders", "overall", body)
    return 1


def main() -> None:
    conn = _connect()
    try:
        counts = {
            "seller_performance": generate_seller_performance(conn),
            "category_performance": generate_category_performance(conn),
            "monthly_sales_report": generate_monthly_sales(conn),
            "regional_summary": generate_regional_summary(conn),
            "payment_summary": generate_payment_summary(conn),
            "delivery_performance": generate_delivery_performance(conn),
            "order_status_summary": generate_order_status_summary(conn),
        }
    finally:
        conn.close()

    total = sum(counts.values())
    print(f"Generated {total} RAG documents in {RAG_DOCS_DIR}:")
    for doc_type, n in counts.items():
        print(f"  {doc_type}: {n}")
    print("\nNext: run ingest.py to embed these (along with the existing "
          "docs_*.md company documents) into ChromaDB.")


if __name__ == "__main__":
    main()
