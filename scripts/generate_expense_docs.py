"""Generates the synthetic expense-policy / vendor-profile corpus backing
Phase 16 (multimodal document Q&A -- "why was this charge deducted?").

Honesty note, stated explicitly per the project's no-fabrication rule: these
are NOT real company documents, unlike the Olist RAG corpus (which is
generated from real Kaggle transaction data). There is no real "company"
behind this RBAC demo, so there is no real expense policy to source from.
Every document below is clearly synthetic demo content -- fake vendor
names where needed, fake policy numbers, explicitly labeled as such in both
the frontmatter (source: synthetic_demo) and the body. This mirrors the
same document shape ingest.py already knows how to load (frontmatter +
department tag), so no ingest.py changes were needed -- these just land in
the same RAG_DOCS_DIR as the real Olist documents, tagged with the new
"expense_docs" department (see config.py), which is gated to
finance/manager/executive/admin only.

Usage:
    python scripts/generate_expense_docs.py
"""
import sys
from pathlib import Path
from textwrap import dedent

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAG_DOCS_DIR


def _write_doc(record_id: str, body: str) -> None:
    RAG_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    frontmatter = dedent(f"""\
        ---
        source: synthetic_demo
        document_type: expense_policy
        department: expense_docs
        access_level: internal
        record_id: {record_id}
        ---
        """)
    filename = f"expense_{record_id}.md"
    (RAG_DOCS_DIR / filename).write_text(frontmatter + body.strip() + "\n", encoding="utf-8")


DOCUMENTS = {
    "policy_cloud_services": dedent("""\
        # Expense Policy: Cloud Infrastructure Services (SYNTHETIC DEMO DATA)

        This is a synthetic demo policy document, not a real company policy.

        Approved cloud infrastructure vendors: Amazon Web Services (AWS), Microsoft
        Azure, Google Cloud Platform (GCP).

        Monthly cloud infrastructure spend under $500 per team does not require
        pre-approval and is auto-classified as "Cloud Infrastructure - Operating
        Expense" on statements. Charges from AWS, Azure, or GCP in this range
        (e.g. EC2, Azure VMs, GCE compute usage) are routine recurring operating
        costs, not one-off or unusual charges.

        Spend above $500/month per team requires a cost-center approval on file;
        Finance can look up the approval record by team name and month.
    """),
    "policy_software_subscriptions": dedent("""\
        # Expense Policy: Software Subscriptions (SYNTHETIC DEMO DATA)

        This is a synthetic demo policy document, not a real company policy.

        Recurring SaaS subscriptions (e.g. Netflix, Spotify, Adobe Creative Cloud,
        Slack, Zoom, GitHub) charged to a corporate card fall into two buckets:

        - Business-justified subscriptions (GitHub, Slack, Zoom, Adobe Creative
          Cloud, and similar tools with a documented work use) are reimbursable
          operating expenses and appear on statements as "Software Subscription -
          Approved."
        - Personal-entertainment subscriptions charged in error (Netflix, Spotify,
          and similar) are NOT reimbursable; the cardholder is expected to
          reimburse the company directly and should flag the charge to Finance
          rather than assume it will be written off.
    """),
    "policy_travel_and_meals": dedent("""\
        # Expense Policy: Travel and Meals (SYNTHETIC DEMO DATA)

        This is a synthetic demo policy document, not a real company policy.

        Airfare, hotel, and ground transportation for approved business travel are
        reimbursable in full with a receipt. Meals during business travel are
        reimbursable up to $75/day per traveler without itemized receipts, and in
        full with itemized receipts. Alcohol is not reimbursable under any
        circumstance and will be flagged for cardholder reimbursement if found on
        a statement.
    """),
    "vendor_profile_amazon_web_services": dedent("""\
        # Vendor Profile: Amazon Web Services (SYNTHETIC DEMO DATA)

        This is a synthetic demo vendor profile, not real vendor data.

        Amazon Web Services (AWS) is a cloud computing infrastructure provider.
        Common line items from AWS on a corporate statement include "EC2 Usage"
        (virtual server compute time), "S3" (object storage), and "RDS" (managed
        databases). These are usage-based charges that vary month to month
        depending on how much compute/storage was consumed -- an unfamiliar
        exact dollar amount from AWS is normal and expected, not itself a sign of
        an error, unless it falls outside the approved monthly threshold in the
        cloud infrastructure expense policy.
    """),
    "vendor_profile_netflix": dedent("""\
        # Vendor Profile: Netflix (SYNTHETIC DEMO DATA)

        This is a synthetic demo vendor profile, not real vendor data.

        Netflix is a consumer video streaming subscription service. It has no
        legitimate business use case for a corporate expense account under the
        software subscriptions policy above -- a Netflix charge on a corporate
        statement should be treated as a personal charge requiring cardholder
        reimbursement, not a business expense to approve.
    """),
    "vendor_profile_grocery_and_retail": dedent("""\
        # Vendor Profile: Grocery / General Retail (SYNTHETIC DEMO DATA)

        This is a synthetic demo vendor profile, not real vendor data.

        Grocery store and general retail purchases (supermarkets, convenience
        stores, department stores) are personal by default and are not covered
        under any business expense policy in this demo corpus, with the sole
        exception of pre-approved office-supply purchases, which must be tagged
        as such at time of purchase to be reimbursable.
    """),
}


def main() -> None:
    for record_id, body in DOCUMENTS.items():
        _write_doc(record_id, body)
    print(f"Generated {len(DOCUMENTS)} synthetic expense/vendor documents in {RAG_DOCS_DIR}")
    print("These are clearly-labeled SYNTHETIC DEMO DATA, not real company documents "
          "(see this script's module docstring for why).")
    print("Run `python ingest.py` afterward to index them (department: expense_docs, "
          "gated to finance/manager/executive/admin -- see config.py).")


if __name__ == "__main__":
    main()
