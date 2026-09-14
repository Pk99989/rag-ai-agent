"""Prepare the Olist Brazilian E-Commerce dataset for SQL analytics and RAG ingestion.

Usage:
    python scripts/prepare_olist.py

Reads raw Kaggle CSVs from data/raw/ (not committed to git -- see README for
download instructions: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce),
validates and cleans them, writes cleaned per-table CSVs to data/processed/,
and builds a normalized SQLite database at data/olist.db with indexes on the
columns the acceptance-criteria business questions actually filter/join on.

Design choices, stated up front rather than buried in comments:
  - Malformed records are COUNTED and REPORTED, not silently dropped or
    silently kept. See data/processed/data_quality_report.txt after running.
  - Only rows with a missing primary key are dropped outright (they cannot be
    joined or referenced safely). Everything else that looks suspicious
    (e.g. negative price) is kept but flagged in the quality report, since
    dropping business data is a decision a human should confirm, not a script.
  - This script has NOT been executed by the assistant that wrote it (sandbox
    outage). Run it yourself and report the printed output back verbatim --
    including tracebacks -- so we debug against real output, not guesses.
"""
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, OLIST_DB_PATH, OLIST_REQUIRED_FILES

STATE_RE = re.compile(r"^[A-Z]{2}$")


# ---------------------------------------------------------------------------
# Step 1: validate raw files exist
# ---------------------------------------------------------------------------
def validate_raw_files() -> None:
    missing = [f for f in OLIST_REQUIRED_FILES if not (RAW_DATA_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Olist CSV file(s) in data/raw/: " + ", ".join(missing) + "\n"
            "Download the dataset from "
            "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce "
            "and place the CSVs in data/raw/ (see README)."
        )
    print(f"[1/8] All {len(OLIST_REQUIRED_FILES)} required CSV files found in {RAW_DATA_DIR}")


# ---------------------------------------------------------------------------
# Step 2: load + normalize column names
# ---------------------------------------------------------------------------
def load_raw() -> dict:
    def _load(name: str) -> pd.DataFrame:
        df = pd.read_csv(RAW_DATA_DIR / name)
        df.columns = [c.strip().lower() for c in df.columns]
        return df

    tables = {
        "customers": _load("olist_customers_dataset.csv"),
        "orders": _load("olist_orders_dataset.csv"),
        "order_items": _load("olist_order_items_dataset.csv"),
        "payments": _load("olist_order_payments_dataset.csv"),
        "reviews": _load("olist_order_reviews_dataset.csv"),
        "products": _load("olist_products_dataset.csv"),
        "sellers": _load("olist_sellers_dataset.csv"),
        "categories": _load("product_category_name_translation.csv"),
    }
    print("[2/8] Loaded raw tables: " + ", ".join(f"{k}={len(v)} rows" for k, v in tables.items()))
    return tables


# ---------------------------------------------------------------------------
# Step 3: validate expected columns are present (fail fast on schema drift)
# ---------------------------------------------------------------------------
# NOTE: "lenght" is not a typo in this script -- it is the actual (misspelled)
# column name Kaggle/Olist ships in olist_products_dataset.csv. Renaming it
# here would break the merge below.
EXPECTED_COLUMNS = {
    "customers": {"customer_id", "customer_unique_id", "customer_zip_code_prefix",
                  "customer_city", "customer_state"},
    "orders": {"order_id", "customer_id", "order_status", "order_purchase_timestamp",
               "order_approved_at", "order_delivered_carrier_date",
               "order_delivered_customer_date", "order_estimated_delivery_date"},
    "order_items": {"order_id", "order_item_id", "product_id", "seller_id",
                     "shipping_limit_date", "price", "freight_value"},
    "payments": {"order_id", "payment_sequential", "payment_type",
                 "payment_installments", "payment_value"},
    "reviews": {"review_id", "order_id", "review_score", "review_comment_title",
                "review_comment_message", "review_creation_date", "review_answer_timestamp"},
    "products": {"product_id", "product_category_name", "product_name_lenght",
                 "product_description_lenght", "product_photos_qty", "product_weight_g",
                 "product_length_cm", "product_height_cm", "product_width_cm"},
    "sellers": {"seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"},
    "categories": {"product_category_name", "product_category_name_english"},
}


def validate_columns(tables: dict) -> None:
    problems = []
    for name, df in tables.items():
        missing = EXPECTED_COLUMNS[name] - set(df.columns)
        if missing:
            problems.append(f"{name}: missing columns {sorted(missing)}")
    if problems:
        raise ValueError("Schema validation failed -- Kaggle may have changed the "
                          "dataset format:\n" + "\n".join(problems))
    print("[3/8] Column schema validated against the expected Olist schema for all 8 tables.")


# ---------------------------------------------------------------------------
# Step 4: parse dates, translate categories, handle missing values
# ---------------------------------------------------------------------------
DATE_COLUMNS = {
    "orders": ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
               "order_delivered_customer_date", "order_estimated_delivery_date"],
    "order_items": ["shipping_limit_date"],
    "reviews": ["review_creation_date", "review_answer_timestamp"],
}


def clean_tables(tables: dict) -> dict:
    for table_name, cols in DATE_COLUMNS.items():
        for col in cols:
            tables[table_name][col] = pd.to_datetime(tables[table_name][col], errors="coerce")

    # Reviews: free-text comment fields are legitimately optional -- most
    # reviews have no written comment. Missing != malformed here.
    tables["reviews"]["review_comment_title"] = tables["reviews"]["review_comment_title"].fillna("")
    tables["reviews"]["review_comment_message"] = tables["reviews"]["review_comment_message"].fillna("")

    # Products: a small number of rows in the real dataset have a null
    # category and null dimension fields. Fill category with an explicit
    # sentinel rather than dropping the product (it may still be referenced
    # by order_items and must remain joinable).
    tables["products"]["product_category_name"] = (
        tables["products"]["product_category_name"].fillna("unknown_category")
    )
    for dim_col in ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]:
        tables["products"][dim_col] = tables["products"][dim_col].fillna(0)

    # Translate categories: left-join so a category missing from the
    # translation table doesn't drop the product -- fall back to the
    # original Portuguese name instead of silently losing rows.
    tables["products"] = tables["products"].merge(
        tables["categories"], on="product_category_name", how="left"
    )
    tables["products"]["product_category_name_english"] = (
        tables["products"]["product_category_name_english"]
        .fillna(tables["products"]["product_category_name"])
    )

    print("[4/8] Parsed date columns, filled legitimate-optional nulls, translated categories "
          "(fallback to Portuguese name where no translation exists).")
    return tables


# ---------------------------------------------------------------------------
# Step 5: detect malformed records (report, do not silently drop)
# ---------------------------------------------------------------------------
# Primary-key column(s) per table. order_items and payments use composite
# keys (a single order can have multiple line items / payment installments).
PK_COLUMNS = {
    "customers": ["customer_id"],
    "orders": ["order_id"],
    "order_items": ["order_id", "order_item_id"],
    "payments": ["order_id", "payment_sequential"],
    "reviews": ["review_id"],
    "products": ["product_id"],
    "sellers": ["seller_id"],
}


def detect_malformed(tables: dict) -> list:
    """Returns a list of (table, issue_description, row_count) tuples.

    Also drops rows with a null primary key (unjoinable, unsafe to keep) and
    de-duplicates rows on the primary key, keeping the first occurrence.
    The real Olist reviews.csv ships with a small number of repeated
    review_id values (the same review record attached more than once) --
    this was discovered by actually running this script against the real
    data, not anticipated in advance, which is exactly why this step reports
    counts instead of assuming the raw data is clean.
    """
    issues = []

    for key, pk_cols in PK_COLUMNS.items():
        n_null_pk = tables[key][pk_cols].isna().any(axis=1).sum()
        if n_null_pk:
            before = len(tables[key])
            tables[key] = tables[key].dropna(subset=pk_cols)
            issues.append((key, f"dropped {n_null_pk} row(s) with null {'/'.join(pk_cols)}", n_null_pk))
            assert len(tables[key]) == before - n_null_pk

        n_dupes = tables[key].duplicated(subset=pk_cols).sum()
        if n_dupes:
            tables[key] = tables[key].drop_duplicates(subset=pk_cols, keep="first")
            issues.append((key, f"dropped {n_dupes} duplicate row(s) on primary key "
                                 f"{'/'.join(pk_cols)} (kept first occurrence)", n_dupes))

    bad_price = tables["order_items"][
        (tables["order_items"]["price"] <= 0) | (tables["order_items"]["freight_value"] < 0)
    ]
    if len(bad_price):
        issues.append(("order_items", "price <= 0 or freight_value < 0 (kept, flagged)", len(bad_price)))

    bad_score = tables["reviews"][~tables["reviews"]["review_score"].between(1, 5)]
    if len(bad_score):
        issues.append(("reviews", "review_score outside 1-5 (kept, flagged)", len(bad_score)))

    bad_dates = tables["orders"][
        tables["orders"]["order_delivered_customer_date"].notna()
        & tables["orders"]["order_purchase_timestamp"].notna()
        & (tables["orders"]["order_delivered_customer_date"] < tables["orders"]["order_purchase_timestamp"])
    ]
    if len(bad_dates):
        issues.append(("orders", "delivered before purchased (kept, flagged)", len(bad_dates)))

    for key in ("customers", "sellers"):
        state_col = f"{key[:-1]}_state" if key != "customers" else "customer_state"
        bad_state = tables[key][~tables[key][state_col].astype(str).str.match(STATE_RE)]
        if len(bad_state):
            issues.append((key, f"{state_col} not a valid 2-letter code (kept, flagged)", len(bad_state)))

    print(f"[5/8] Data-quality scan complete: {len(issues)} distinct issue type(s) found "
          f"(see data/processed/data_quality_report.txt).")
    return issues


# ---------------------------------------------------------------------------
# Step 6: derive delivery status + build the denormalized "order_facts" table
# ---------------------------------------------------------------------------
def add_derived_columns(tables: dict) -> dict:
    orders = tables["orders"]
    orders["delivery_status"] = "not_delivered"
    delivered_mask = orders["order_delivered_customer_date"].notna()
    on_time_mask = delivered_mask & (
        orders["order_delivered_customer_date"] <= orders["order_estimated_delivery_date"]
    )
    orders.loc[delivered_mask, "delivery_status"] = "late"
    orders.loc[on_time_mask, "delivery_status"] = "on_time"
    orders["delivery_days"] = (
        orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]
    ).dt.days
    tables["orders"] = orders
    print("[6/8] Derived orders.delivery_status (on_time/late/not_delivered) and delivery_days.")
    return tables


def build_order_facts(tables: dict) -> pd.DataFrame:
    """One row per order line item, enriched with everything a business
    question or a RAG document (Phase 5) would need -- avoids re-joining
    eight tables every time something downstream needs order context."""
    payments_agg = (
        tables["payments"].groupby("order_id")
        .agg(total_payment_value=("payment_value", "sum"))
        .reset_index()
    )
    primary_payment = (
        tables["payments"].sort_values("payment_value", ascending=False)
        .drop_duplicates(subset="order_id", keep="first")[["order_id", "payment_type"]]
        .rename(columns={"payment_type": "primary_payment_type"})
    )
    payments_agg = payments_agg.merge(primary_payment, on="order_id", how="left")

    reviews_agg = (
        tables["reviews"].sort_values("review_answer_timestamp", ascending=False)
        .drop_duplicates(subset="order_id", keep="first")[["order_id", "review_score"]]
    )

    facts = (
        tables["order_items"]
        .merge(tables["orders"], on="order_id", how="left")
        .merge(tables["customers"], on="customer_id", how="left")
        .merge(
            tables["products"][["product_id", "product_category_name", "product_category_name_english"]],
            on="product_id", how="left",
        )
        .merge(tables["sellers"], on="seller_id", how="left")
        .merge(payments_agg, on="order_id", how="left")
        .merge(reviews_agg, on="order_id", how="left")
    )

    keep = [
        "order_id", "order_item_id", "order_status", "order_purchase_timestamp",
        "customer_id", "customer_state", "customer_city",
        "product_id", "product_category_name", "product_category_name_english",
        "seller_id", "seller_state", "seller_city",
        "price", "freight_value", "total_payment_value", "primary_payment_type",
        "review_score", "delivery_status", "delivery_days",
    ]
    facts = facts[keep]
    print(f"[7/8] Built order_facts: {len(facts)} rows (1 per order line item), "
          f"{facts['order_id'].nunique()} distinct orders.")
    return facts


# ---------------------------------------------------------------------------
# Step 7: write processed CSVs + data quality report
# ---------------------------------------------------------------------------
def write_processed(tables: dict, order_facts: pd.DataFrame, issues: list) -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(PROCESSED_DATA_DIR / f"{name}.csv", index=False)
    order_facts.to_csv(PROCESSED_DATA_DIR / "order_facts.csv", index=False)

    report_path = PROCESSED_DATA_DIR / "data_quality_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Olist data preparation -- data quality report\n")
        f.write("=" * 50 + "\n")
        if not issues:
            f.write("No data quality issues detected.\n")
        for table, desc, count in issues:
            f.write(f"[{table}] {desc}: {count} row(s)\n")
    print(f"    Wrote processed CSVs + data_quality_report.txt to {PROCESSED_DATA_DIR}")


# ---------------------------------------------------------------------------
# Step 8: build the normalized SQLite database with indexes
# ---------------------------------------------------------------------------
SCHEMA_SQL = """
CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY,
    customer_unique_id TEXT,
    customer_zip_code_prefix TEXT,
    customer_city TEXT,
    customer_state TEXT
);
CREATE TABLE sellers (
    seller_id TEXT PRIMARY KEY,
    seller_zip_code_prefix TEXT,
    seller_city TEXT,
    seller_state TEXT
);
CREATE TABLE categories (
    product_category_name TEXT PRIMARY KEY,
    product_category_name_english TEXT
);
CREATE TABLE products (
    product_id TEXT PRIMARY KEY,
    product_category_name TEXT,
    product_category_name_english TEXT,
    product_name_lenght REAL,
    product_description_lenght REAL,
    product_photos_qty REAL,
    product_weight_g REAL,
    product_length_cm REAL,
    product_height_cm REAL,
    product_width_cm REAL
);
CREATE TABLE orders (
    order_id TEXT PRIMARY KEY,
    customer_id TEXT REFERENCES customers(customer_id),
    order_status TEXT,
    order_purchase_timestamp TEXT,
    order_approved_at TEXT,
    order_delivered_carrier_date TEXT,
    order_delivered_customer_date TEXT,
    order_estimated_delivery_date TEXT,
    delivery_status TEXT,
    delivery_days REAL
);
CREATE TABLE order_items (
    order_id TEXT REFERENCES orders(order_id),
    order_item_id INTEGER,
    product_id TEXT REFERENCES products(product_id),
    seller_id TEXT REFERENCES sellers(seller_id),
    shipping_limit_date TEXT,
    price REAL,
    freight_value REAL,
    PRIMARY KEY (order_id, order_item_id)
);
CREATE TABLE payments (
    order_id TEXT REFERENCES orders(order_id),
    payment_sequential INTEGER,
    payment_type TEXT,
    payment_installments INTEGER,
    payment_value REAL,
    PRIMARY KEY (order_id, payment_sequential)
);
CREATE TABLE reviews (
    review_id TEXT PRIMARY KEY,
    order_id TEXT REFERENCES orders(order_id),
    review_score INTEGER,
    review_comment_title TEXT,
    review_comment_message TEXT,
    review_creation_date TEXT,
    review_answer_timestamp TEXT
);

CREATE INDEX idx_orders_customer_id ON orders(customer_id);
CREATE INDEX idx_orders_purchase_ts ON orders(order_purchase_timestamp);
CREATE INDEX idx_order_items_order_id ON order_items(order_id);
CREATE INDEX idx_order_items_product_id ON order_items(product_id);
CREATE INDEX idx_order_items_seller_id ON order_items(seller_id);
CREATE INDEX idx_payments_order_id ON payments(order_id);
CREATE INDEX idx_reviews_order_id ON reviews(order_id);
CREATE INDEX idx_products_category ON products(product_category_name);
CREATE INDEX idx_customers_state ON customers(customer_state);
CREATE INDEX idx_sellers_state ON sellers(seller_state);

CREATE VIEW order_facts AS
SELECT
    oi.order_id, oi.order_item_id, o.order_status, o.order_purchase_timestamp,
    c.customer_id, c.customer_state, c.customer_city,
    p.product_id, p.product_category_name, p.product_category_name_english,
    s.seller_id, s.seller_state, s.seller_city,
    oi.price, oi.freight_value, o.delivery_status, o.delivery_days
FROM order_items oi
JOIN orders o ON oi.order_id = o.order_id
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON oi.product_id = p.product_id
JOIN sellers s ON oi.seller_id = s.seller_id;
"""

TABLE_LOAD_ORDER = ["customers", "sellers", "categories", "products", "orders", "order_items", "payments", "reviews"]


def build_sqlite(tables: dict) -> None:
    OLIST_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OLIST_DB_PATH.exists():
        OLIST_DB_PATH.unlink()  # rebuild from scratch every run -- no stale schema/data

    conn = sqlite3.connect(OLIST_DB_PATH)
    try:
        conn.executescript(SCHEMA_SQL)
        for name in TABLE_LOAD_ORDER:
            df = tables[name].copy()
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = df[col].astype(str).replace("NaT", None)
            df.to_sql(name, conn, if_exists="append", index=False)
        conn.commit()

        counts = {}
        for name in TABLE_LOAD_ORDER:
            counts[name] = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        print(f"[8/8] Built SQLite database at {OLIST_DB_PATH}")
        for name, n in counts.items():
            print(f"       {name}: {n} rows")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
def main() -> None:
    validate_raw_files()
    tables = load_raw()
    validate_columns(tables)
    tables = clean_tables(tables)
    issues = detect_malformed(tables)
    tables = add_derived_columns(tables)
    order_facts = build_order_facts(tables)
    write_processed(tables, order_facts, issues)
    build_sqlite(tables)
    print("\nDone. Next: inspect data/processed/data_quality_report.txt, then spot-check "
          "data/olist.db with e.g. `sqlite3 data/olist.db \"SELECT * FROM order_facts LIMIT 5;\"`")


if __name__ == "__main__":
    main()
