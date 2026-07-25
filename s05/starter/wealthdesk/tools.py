"""
wealthdesk/tools.py
-------------------
LLM clients and database tool functions for WealthDesk.

Session 5: adds query_rates() and query_branch() so the LLM can
look up live data instead of relying on hardcoded rates.
"""
import os
import sqlite3

from langchain_core.tools import tool
from langchain_groq import ChatGroq

from .config import DB_PATH, MODEL_NAME, CLASSIFIER_MODEL, TEMPERATURE, MAX_TOKENS,CLASSIFIER_MAX_TOKENS, CLASSIFIER_TEMPERATURE

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found.\n"
        "Did you copy .env.example to .env and fill in your key?\n"
        "  Windows:  copy .env.example .env\n"
        "  Mac/Linux: cp .env.example .env"
    )

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
)

classifier_llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=CLASSIFIER_MODEL,
    temperature=CLASSIFIER_TEMPERATURE,
    max_tokens=CLASSIFIER_MAX_TOKENS,
)


# ---------------------------------------------------------------------------
# TODO 1 of 4 -- Implement query_rates()
# ---------------------------------------------------------------------------
# Steps:
#   1. Open: conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
#   2. Build lines = []
#   3. If product_type in ("loan", "all"), query loan_products:
#        rows = conn.execute(
#            "SELECT name, interest_rate, tenure_min_years, tenure_max_years "
#            "FROM loan_products ORDER BY interest_rate"
#        ).fetchall()
#      For each row append:
#        f"{name}: {rate:.1f}% p.a., tenure {min_y}-{max_y} years"
#   4. If product_type in ("fd", "all"), query fd_products:
#        rows = conn.execute(
#            "SELECT tenure_label, interest_rate, senior_rate "
#            "FROM fd_products ORDER BY tenure_months"
#        ).fetchall()
#      For each row append:
#        f"FD {label}: {rate:.1f}% p.a. (senior citizens: {rate + senior:.1f}%, extra +{senior:.1f}%)"
#   5. conn.close()
#   6. Return "\n".join(lines) if lines else "No rate data found."
# ---------------------------------------------------------------------------
@tool
def query_rates(product_type: str = "all") -> str:
    """Fetch current BNB interest rates from the database.

    Args:
        product_type: Which rates to return. Options:
            "loan" -- all loan products (home, personal, car, education, gold)
            "fd"   -- all fixed deposit products
            "all"  -- both loans and FDs (default)

    Returns formatted rate information as a plain-text string.
    """
    # TODO: implement this tool
    pass


# ---------------------------------------------------------------------------
# TODO 2 of 4 -- Implement query_branch()
# ---------------------------------------------------------------------------
# Steps:
#   1. Open: conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
#   2. If city.lower() == "all":
#        rows = conn.execute(
#            "SELECT name, city, address, ifsc, phone FROM branches ORDER BY city, name"
#        ).fetchall()
#      Otherwise use a parameterised query (CRITICAL -- prevents SQL injection):
#        rows = conn.execute(
#            "SELECT name, city, address, ifsc, phone "
#            "FROM branches WHERE city LIKE ? ORDER BY name",
#            (f"%{city}%",),
#        ).fetchall()
#      IMPORTANT: if rows is empty after the city search, also try searching by
#      branch name (neighbourhood names like "Koramangala" or "Andheri West" appear
#      in the branch name column, not the city column):
#        if not rows:
#            rows = conn.execute(
#                "SELECT name, city, address, ifsc, phone "
#                "FROM branches WHERE name LIKE ? ORDER BY name",
#                (f"%{city}%",),
#            ).fetchall()
#   3. conn.close()
#   4. If not rows: return f"No BNB branches found for city: '{city}'."
#   5. Build parts = [] and for each row append:
#        f"{name} ({city_})\n  Address: {address}\n  IFSC: {ifsc}  |  Phone: {phone}"
#      Return "\n\n".join(parts)
# ---------------------------------------------------------------------------
@tool
def query_branch(city: str = "all") -> str:
    """Fetch BNB branch locations from the database.

    Args:
        city: Filter branches by city name. Examples: "Bengaluru", "Mumbai".
              Use "all" for every branch.

    Returns branch names, addresses, IFSC codes, and phone numbers.
    """
    # TODO: implement this tool
    pass


# ---------------------------------------------------------------------------
# TODO 3 of 4 -- Bind tools to the LLM
# ---------------------------------------------------------------------------
# Create llm_with_tools by binding both tools to llm:
#   llm_with_tools = llm.bind_tools([query_rates, query_branch])
#
# bind_tools() reads each function's type hints and docstring, converts them to a
# JSON schema, and includes that schema in every request. The LLM was fine-tuned
# on examples with this schema format, so it knows to output a structured tool call
# instead of guessing the answer from memory.
# llm_with_tools is used for the FIRST call in respond(). The second call
# (after tools have run) uses plain llm.
# ---------------------------------------------------------------------------
# TODO: add llm_with_tools = llm.bind_tools([query_rates, query_branch])


def _run_tool(tool_name: str, tool_args: dict) -> str:
    """Dispatch a tool call by name. Provided -- no changes needed."""
    _registry = {
        "query_rates":  query_rates,
        "query_branch": query_branch,
    }
    if tool_name not in _registry:
        return f"Unknown tool: {tool_name}"
    try:
        return _registry[tool_name].invoke(tool_args)
    except Exception as e:
        return f"Tool error ({tool_name}): {e}"
