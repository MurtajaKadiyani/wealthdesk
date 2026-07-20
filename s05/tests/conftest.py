"""
s05/tests/conftest.py
---------------------
Pytest configuration for Session 5 tests.

Sets dummy API keys and clears any stale wealthdesk modules at collection time.
The seeded_db fixture creates a temporary SQLite database with real schema and
sample data so tool tests can run actual SQL without touching the production db.
"""
import os
import sqlite3
import sys

import pytest

for _key in list(sys.modules):
    if _key == "wealthdesk" or _key.startswith("wealthdesk."):
        sys.modules.pop(_key)

os.environ.setdefault("GROQ_API_KEY", "test-key-not-real")
os.environ.setdefault("LANGSMITH_API_KEY", "test-langsmith-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")


@pytest.fixture
def seeded_db(tmp_path):
    """Temporary SQLite database with BNB schema and sample rows.

    Used by tool tests via monkeypatch.setattr("wealthdesk.tools.DB_PATH", seeded_db).
    The schema matches data/seed.py exactly -- tests remain valid as long as
    the production schema does not diverge from this fixture.
    """
    db_path = tmp_path / "test_bnb.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE loan_products (
            product_id          TEXT PRIMARY KEY,
            name                TEXT NOT NULL,
            interest_rate       REAL NOT NULL,
            tenure_min_years    INTEGER NOT NULL,
            tenure_max_years    INTEGER NOT NULL,
            eligibility_formula TEXT NOT NULL,
            processing_fee_info TEXT,
            effective_date      TEXT NOT NULL
        );
        CREATE TABLE fd_products (
            product_id      TEXT PRIMARY KEY,
            tenure_label    TEXT NOT NULL,
            tenure_months   INTEGER NOT NULL,
            interest_rate   REAL NOT NULL,
            senior_rate     REAL NOT NULL,
            min_deposit     INTEGER NOT NULL,
            effective_date  TEXT NOT NULL
        );
        CREATE TABLE branches (
            branch_id   TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            city        TEXT NOT NULL,
            address     TEXT NOT NULL,
            ifsc        TEXT NOT NULL,
            phone       TEXT NOT NULL
        );
        INSERT INTO loan_products VALUES
            ('home_loan',     'Home Loan',     8.5,  5, 30, 'income x 60', '0.5%', '2024-04-01'),
            ('personal_loan', 'Personal Loan', 12.0, 1,  5, 'income x 24', '2.0%', '2024-04-01'),
            ('car_loan',      'Car Loan',       9.5, 1,  7, 'income x 36', '1.0%', '2024-04-01');
        INSERT INTO fd_products VALUES
            ('fd_1year', '1 year',  12, 6.8, 0.5, 10000, '2024-04-01'),
            ('fd_2year', '2 years', 24, 7.1, 0.5, 10000, '2024-04-01'),
            ('fd_5year', '5 years', 60, 7.3, 0.5, 10000, '2024-04-01');
        INSERT INTO branches VALUES
            ('BNB001', 'BNB Koramangala', 'Bengaluru', '12th Main, Koramangala', 'BNBI0001001', '080-41234567'),
            ('BNB002', 'BNB Indiranagar', 'Bengaluru', '100 Feet Road, Indiranagar', 'BNBI0001002', '080-41234568'),
            ('BNB003', 'BNB Andheri West', 'Mumbai', 'SV Road, Andheri West', 'BNBI0002001', '022-41234567');
    """)
    conn.commit()
    conn.close()
    return db_path
