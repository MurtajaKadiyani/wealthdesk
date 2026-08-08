"""
WealthDesk -- Session 7: MCP Server (US-06 Part 1)
===================================================
STARTER FILE -- your task is to implement the two TODO sections below.

Goal
  Build a standalone MCP server that exposes WealthDesk's two database
  tools -- query_rates and query_branch -- over the MCP protocol.
  When finished, MCP Inspector should be able to discover both tools and
  call them without touching any agent code.

What is already done for you
  - FastMCP server created: mcp = FastMCP("wealthdesk-tools")
  - Both @mcp.tool() decorators and function signatures are in place
  - DB_PATH points to the same bnb_data.db used in Session 5
  - mcp.run() at the bottom starts the STDIO server

Your task
  Implement the SQL queries inside TODO 1 (query_rates) and TODO 2
  (query_branch). The logic is identical to s05/solution/wealthdesk/tools.py --
  open that file, find the two @tool functions, and adapt them here.
  The only change: replace @tool with @mcp.tool() (already done).

Run when done
  python s07/starter/mcp_server.py

Inspect with MCP Inspector
  npx @modelcontextprotocol/inspector python s07/starter/mcp_server.py
  Open http://localhost:5173 -- both tools should appear.
"""

import sqlite3
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Server instantiation -- already done for you
# ---------------------------------------------------------------------------

mcp = FastMCP("wealthdesk-tools")

# ---------------------------------------------------------------------------
# Configuration -- already done for you
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent.parent / "data"
DB_PATH  = DATA_DIR / "bnb_data.db"

# ---------------------------------------------------------------------------
# TODO 1: Implement query_rates
# Hint: copy the query_rates() function from s05/solution/wealthdesk/tools.py.
#       The SQL queries and return format are identical.
#       The only difference: @tool becomes @mcp.tool() (already in place).
# ---------------------------------------------------------------------------

@mcp.tool()
def query_rates(product_type: str = "all") -> str:
    """Fetch current BNB interest rates from the database.

    Args:
        product_type: Which rates to return. Options:
            "loan" -- all loan products (home, personal, car, education, gold)
            "fd"   -- all fixed deposit products
            "all"  -- both loans and FDs (default)

    Returns formatted rate information as a plain-text string.
    """
    conn  = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    lines = []
 
    if product_type in ("loan", "all"):
        rows = conn.execute(
            "SELECT name, interest_rate, tenure_min_years, tenure_max_years "
            "FROM loan_products ORDER BY interest_rate"
        ).fetchall()
        for name, rate, min_y, max_y in rows:
            lines.append(f"{name}: {rate:.2f}% p.a., tenure {min_y}-{max_y} years")
 
    if product_type in ("fd", "all"):
        rows = conn.execute(
            "SELECT tenure_label, interest_rate, senior_rate "
            "FROM fd_products ORDER BY tenure_months"
        ).fetchall()
        for label, rate, senior in rows:
            lines.append(
                f"FD {label}: {rate:.2f}% p.a. "
                f"(senior citizens: {rate + senior:.2f}%, extra +{senior:.2f}%)"
            )
 
    conn.close()
    return "\n".join(lines) if lines else "No rate data found."


# ---------------------------------------------------------------------------
# TODO 2: Implement query_branch
# Hint: copy the query_branch() function from s05/solution/wealthdesk/tools.py.
#       Same SQL, same return format, same @mcp.tool() decorator.
# ---------------------------------------------------------------------------

@mcp.tool()
def query_branch(city: str = "all") -> str:
    """Fetch BNB branch locations from the database.

    Args:
        city: Filter branches by city name. Examples: "Bengaluru", "Mumbai",
              "Chennai", "Hyderabad", "Delhi". Use "all" for every branch.

    Returns branch names, addresses, IFSC codes, and phone numbers.
    """
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)  # see query_rates for why
 
    if city.lower() == "all":
        rows = conn.execute(
            "SELECT name, city, address, ifsc, phone FROM branches ORDER BY city, name"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT name, city, address, ifsc, phone "
            "FROM branches WHERE city LIKE ? ORDER BY name",
            (f"%{city}%",),
        ).fetchall()
        if not rows:
            # Neighbourhood names (e.g. "Andheri West", "Koramangala") are stored
            # in the branch name, not the city column — retry by name.
            rows = conn.execute(
                "SELECT name, city, address, ifsc, phone "
                "FROM branches WHERE name LIKE ? ORDER BY name",
                (f"%{city}%",),
            ).fetchall()
 
    conn.close()
 
    if not rows:
        return f"No BNB branches found for city: '{city}'."
 
    parts = []
    for name, city_, address, ifsc, phone in rows:
        parts.append(
            f"{name} ({city_})\n"
            f"  Address: {address}\n"
            f"  IFSC: {ifsc}  |  Phone: {phone}"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Entry point -- already done for you
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()  # STDIO transport by default
