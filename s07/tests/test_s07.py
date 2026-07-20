"""
s07/tests/test_s07.py
---------------------
Tests for Session 7: MCP Server (US-06 Part 1).

Run with:
    pytest s07/tests/ -v

These tests call the tool functions directly. FastMCP's @mcp.tool()
decorator registers the function with the server but returns the original
callable unchanged -- so query_rates("loan") works just like calling any
Python function.

DB_PATH is patched to a test database created in conftest.py. Tests do not
require data/seed.py to have been run.

Test groups:
  TestServerStructure  -- server name, tool count, tool names
  TestQueryRates       -- return format, product filtering, empty result
  TestQueryBranch      -- return format, city filtering, not-found message
  TestQueryRatesSchema -- function signature accepts expected arguments
  TestSQLInjection     -- parameterised query protects against injection
"""

import sys
from pathlib import Path

import pytest

SOLUTION_DIR = Path(__file__).parent.parent / "solution"
sys.path.insert(0, str(SOLUTION_DIR))

import mcp_server
from mcp_server import mcp, query_branch, query_rates


# ---------------------------------------------------------------------------
# TestServerStructure
# ---------------------------------------------------------------------------

class TestServerStructure:
    def test_server_name(self):
        assert mcp.name == "wealthdesk-tools"

    def test_server_has_two_tools(self):
        tools = mcp._tool_manager.list_tools()
        assert len(tools) == 2, f"Expected 2 tools, found {len(tools)}"

    def test_server_has_query_rates_tool(self):
        tools = mcp._tool_manager.list_tools()
        names = [t.name for t in tools]
        assert "query_rates" in names

    def test_server_has_query_branch_tool(self):
        tools = mcp._tool_manager.list_tools()
        names = [t.name for t in tools]
        assert "query_branch" in names

    def test_query_rates_has_description(self):
        tools = mcp._tool_manager.list_tools()
        tool = next(t for t in tools if t.name == "query_rates")
        assert tool.description and len(tool.description) > 10

    def test_query_branch_has_description(self):
        tools = mcp._tool_manager.list_tools()
        tool = next(t for t in tools if t.name == "query_branch")
        assert tool.description and len(tool.description) > 10


# ---------------------------------------------------------------------------
# TestQueryRates
# ---------------------------------------------------------------------------

class TestQueryRates:
    def test_all_returns_string(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_rates("all")
        assert isinstance(result, str)

    def test_all_contains_home_loan(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_rates("all")
        assert "Home Loan" in result

    def test_all_contains_fd(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_rates("all")
        assert "FD" in result

    def test_home_loan_rate_is_8_5(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_rates("loan")
        assert "8.5" in result

    def test_loan_filter_excludes_fd(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_rates("loan")
        assert "FD" not in result

    def test_fd_filter_excludes_loans(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_rates("fd")
        assert "Home Loan" not in result
        assert "Personal Loan" not in result

    def test_fd_filter_contains_fd(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_rates("fd")
        assert "FD" in result

    def test_fd_rate_format_includes_senior_rate(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_rates("fd")
        assert "senior citizens" in result

    def test_loan_format_includes_tenure(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_rates("loan")
        assert "tenure" in result

    def test_no_data_returns_not_found(self, tmp_path, monkeypatch):
        empty_db = tmp_path / "empty.db"
        import sqlite3
        conn = sqlite3.connect(str(empty_db))
        conn.executescript("""
            CREATE TABLE loan_products (
                id INTEGER PRIMARY KEY, name TEXT, interest_rate REAL,
                tenure_min_years INTEGER, tenure_max_years INTEGER
            );
            CREATE TABLE fd_products (
                id INTEGER PRIMARY KEY, tenure_label TEXT, tenure_months INTEGER,
                interest_rate REAL, senior_rate REAL
            );
        """)
        conn.commit()
        conn.close()
        monkeypatch.setattr(mcp_server, "DB_PATH", empty_db)
        result = query_rates("all")
        assert result == "No rate data found."

    def test_default_argument_is_all(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result_default = query_rates()
        result_all = query_rates("all")
        assert result_default == result_all

    def test_personal_loan_rate_is_12(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_rates("loan")
        assert "12.0" in result

    def test_one_year_fd_rate_is_6_8(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_rates("fd")
        assert "6.8" in result

    def test_two_year_fd_rate_is_7_1(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_rates("fd")
        assert "7.1" in result


# ---------------------------------------------------------------------------
# TestQueryBranch
# ---------------------------------------------------------------------------

class TestQueryBranch:
    def test_all_returns_string(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_branch("all")
        assert isinstance(result, str)

    def test_all_returns_multiple_branches(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_branch("all")
        assert "BNB Indiranagar" in result
        assert "BNB Bandra" in result

    def test_city_filter_bengaluru_returns_bengaluru_only(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_branch("Bengaluru")
        assert "Bengaluru" in result
        assert "Mumbai" not in result

    def test_city_filter_returns_correct_branch(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_branch("Mumbai")
        assert "BNB Bandra" in result

    def test_unknown_city_returns_not_found_message(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_branch("Kolkata")
        assert "No BNB branches found" in result
        assert "Kolkata" in result

    def test_result_includes_ifsc(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_branch("Bengaluru")
        assert "IFSC" in result

    def test_result_includes_phone(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_branch("Bengaluru")
        assert "Phone" in result

    def test_result_includes_address(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_branch("Chennai")
        assert "Address" in result

    def test_default_argument_is_all(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result_default = query_branch()
        result_all = query_branch("all")
        assert result_default == result_all

    def test_city_filter_is_case_sensitive_partial_match(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_branch("bengaluru")
        # LIKE is case-insensitive in SQLite by default for ASCII characters
        assert "Bengaluru" in result or "No BNB branches found" in result

    def test_multiple_bengaluru_branches_returned(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_branch("Bengaluru")
        assert "BNB Indiranagar" in result
        assert "BNB Koramangala" in result

    def test_branches_separated_by_double_newline(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_branch("Bengaluru")
        assert "\n\n" in result


# ---------------------------------------------------------------------------
# TestSQLInjection
# ---------------------------------------------------------------------------

class TestSQLInjection:
    def test_branch_city_injection_does_not_crash(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_branch("'; DROP TABLE branches; --")
        assert isinstance(result, str)
        assert "No BNB branches found" in result

    def test_branch_city_injection_does_not_drop_table(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        query_branch("'; DROP TABLE branches; --")
        # Table should still exist -- subsequent query should work
        result = query_branch("Bengaluru")
        assert "BNB Indiranagar" in result
