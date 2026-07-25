"""
s07/tests/conftest.py
---------------------
Pytest configuration and fixtures for Session 7 tests.

Creates a minimal in-memory SQLite database that mirrors the schema of
bnb_data.db so tests run without requiring data/seed.py to have been run.
"""

import sqlite3
import sys
from pathlib import Path

import pytest

# Make s07/solution importable
SOLUTION_DIR = Path(__file__).parent.parent / "solution"
sys.path.insert(0, str(SOLUTION_DIR))


@pytest.fixture()
def test_db(tmp_path):
    """Create a minimal bnb_data.db in a temp directory with known seed data."""
    db_path = tmp_path / "bnb_data.db"
    conn = sqlite3.connect(str(db_path))

    conn.executescript("""
        CREATE TABLE loan_products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            interest_rate REAL NOT NULL,
            tenure_min_years INTEGER NOT NULL,
            tenure_max_years INTEGER NOT NULL
        );

        CREATE TABLE fd_products (
            id INTEGER PRIMARY KEY,
            tenure_label TEXT NOT NULL,
            tenure_months INTEGER NOT NULL,
            interest_rate REAL NOT NULL,
            senior_rate REAL NOT NULL DEFAULT 0.5
        );

        CREATE TABLE branches (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            city TEXT NOT NULL,
            address TEXT NOT NULL,
            ifsc TEXT NOT NULL,
            phone TEXT NOT NULL
        );

        INSERT INTO loan_products VALUES
            (1, 'Home Loan',     8.5, 5,  30),
            (2, 'Personal Loan', 12.0, 1, 5),
            (3, 'Car Loan',      9.5, 1,  7);

        INSERT INTO fd_products VALUES
            (1, '1 Year',  12, 6.8, 0.5),
            (2, '2 Years', 24, 7.1, 0.5),
            (3, '3 Years', 36, 7.3, 0.5);

        INSERT INTO branches VALUES
            (1, 'BNB Indiranagar',   'Bengaluru', '100 CMH Road, Indiranagar',    'BNBI0001001', '080-25201234'),
            (2, 'BNB Koramangala',   'Bengaluru', '5th Block, Koramangala',        'BNBI0001002', '080-25202345'),
            (3, 'BNB Bandra',        'Mumbai',    'Hill Road, Bandra West',        'BNBI0002001', '022-26401234'),
            (4, 'BNB Anna Nagar',    'Chennai',   '2nd Avenue, Anna Nagar',        'BNBI0003001', '044-26201234');
    """)
    conn.commit()
    conn.close()
    return db_path
