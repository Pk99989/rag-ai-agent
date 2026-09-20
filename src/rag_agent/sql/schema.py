"""Static schema description for the Olist analytics database.

The database has only 8 tables + 1 view, so this uses a single static
schema description rather than a dynamic "schema selection" retrieval
step -- at this scale, retrieval would add latency and failure modes
without buying anything a fixed ~40-line string doesn't already give the
LLM. Revisit if/when the schema grows large enough that it stops fitting
comfortably in a prompt.

This module is also the single source of truth for which tables/columns
Text-to-SQL is allowed to reference at all (validator.py imports
ALLOWED_TABLES/ALLOWED_COLUMNS from here) -- so a table added to the
database without being added here is invisible to the LLM AND rejected
by the validator, rather than silently queryable.
"""

# table/view name -> {column_name: short type+meaning description}
SCHEMA: dict[str, dict[str, str]] = {
    "customers": {
        "customer_id": "TEXT, per-order customer identifier (join key for orders)",
        "customer_unique_id": "TEXT, the actual person -- repeats across their orders",
        "customer_zip_code_prefix": "TEXT, first digits of zip code",
        "customer_city": "TEXT",
        "customer_state": "TEXT, 2-letter Brazilian state code (e.g. SP, RJ)",
    },
    "orders": {
        "order_id": "TEXT primary key",
        "customer_id": "TEXT, FK -> customers.customer_id",
        "order_status": "TEXT, e.g. delivered, shipped, canceled",
        "order_purchase_timestamp": "TIMESTAMP",
        "order_approved_at": "TIMESTAMP, nullable",
        "order_delivered_carrier_date": "TIMESTAMP, nullable",
        "order_delivered_customer_date": "TIMESTAMP, nullable",
        "order_estimated_delivery_date": "TIMESTAMP",
        "delivery_status": "TEXT, derived: 'on_time' | 'late' | 'not_delivered'",
        "delivery_days": "REAL, derived: days between purchase and delivery, nullable",
    },
    "order_items": {
        "order_id": "TEXT, FK -> orders.order_id",
        "order_item_id": "INTEGER, line-item number within the order",
        "product_id": "TEXT, FK -> products.product_id",
        "seller_id": "TEXT, FK -> sellers.seller_id",
        "shipping_limit_date": "TIMESTAMP",
        "price": "REAL, item price in BRL (excludes freight)",
        "freight_value": "REAL, shipping cost in BRL",
    },
    "payments": {
        "order_id": "TEXT, FK -> orders.order_id",
        "payment_sequential": "INTEGER, installment/payment record number",
        "payment_type": "TEXT, e.g. credit_card, boleto, voucher",
        "payment_installments": "INTEGER",
        "payment_value": "REAL, amount in BRL for this payment record",
    },
    "reviews": {
        "review_id": "TEXT primary key",
        "order_id": "TEXT, FK -> orders.order_id",
        "review_score": "INTEGER, 1-5",
        "review_comment_title": "TEXT, often empty",
        "review_comment_message": "TEXT, often empty",
        "review_creation_date": "TIMESTAMP",
        "review_answer_timestamp": "TIMESTAMP",
    },
    "products": {
        "product_id": "TEXT primary key",
        "product_category_name": "TEXT, original Portuguese category",
        "product_category_name_english": "TEXT, translated category -- prefer this for readable output",
        "product_weight_g": "REAL",
        "product_length_cm": "REAL",
        "product_height_cm": "REAL",
        "product_width_cm": "REAL",
    },
    "sellers": {
        "seller_id": "TEXT primary key",
        "seller_zip_code_prefix": "TEXT",
        "seller_city": "TEXT",
        "seller_state": "TEXT, 2-letter Brazilian state code",
    },
    "categories": {
        "product_category_name": "TEXT primary key, original Portuguese",
        "product_category_name_english": "TEXT",
    },
    # Convenience view built by scripts/prepare_olist.py -- one row per order
    # line item, pre-joined so simple questions don't need a 5-table JOIN.
    "order_facts": {
        "order_id": "TEXT", "order_item_id": "INTEGER", "order_status": "TEXT",
        "order_purchase_timestamp": "TIMESTAMP",
        "customer_id": "TEXT", "customer_state": "TEXT", "customer_city": "TEXT",
        "product_id": "TEXT", "product_category_name": "TEXT", "product_category_name_english": "TEXT",
        "seller_id": "TEXT", "seller_state": "TEXT", "seller_city": "TEXT",
        "price": "REAL", "freight_value": "REAL",
        "delivery_status": "TEXT", "delivery_days": "REAL",
    },
}

ALLOWED_TABLES: frozenset[str] = frozenset(SCHEMA.keys())
ALLOWED_COLUMNS: frozenset[str] = frozenset(
    col for table_cols in SCHEMA.values() for col in table_cols
)


def render_schema_for_prompt() -> str:
    """Compact schema text handed to the LLM as part of the Text-to-SQL prompt."""
    lines = []
    for table, cols in SCHEMA.items():
        lines.append(f"TABLE {table}:")
        for col, desc in cols.items():
            lines.append(f"  - {col}: {desc}")
    return "\n".join(lines)
