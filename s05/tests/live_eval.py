"""
s05/tests/live_eval.py — Live evaluation for WealthDesk S05 solution
----------------------------------------------------------------------
Runs 15 test queries directly against graph.invoke() using the REAL Groq
LLM, the REAL ChromaDB vectorstore, and the REAL SQLite tools database.
No mocks.

Why this exists alongside pytest
─────────────────────────────────
`pytest test_s05.py` mocks the LLM, tools, and vectorstore — it runs in
~2 seconds and catches structural bugs (wrong node wired, tool not bound,
state field missing, etc.).

This script catches a different class of defects that only appear with a
real LLM:
  • Classifier brittleness — e.g. rate query classified COMPLEX
  • Tool not called    — LLM answers a rate question from memory instead
                         of calling query_rates() (violates Rule 3)
  • Wrong tool args    — query_rates("home") instead of query_rates("loan")
  • Stale history      — previous tool output leaks into follow-up answer
  • Out-of-scope drift — "Recommend stocks" sneaks through as SIMPLE

Run this script from the wealthdesk/ directory:
    python s05/tests/live_eval.py

Expected output: 15/15 passed
If any test fails, the response snippet is printed so you can diagnose.

Cost: ~15 Groq API calls (~8–12 seconds total, well within free tier).
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────
TESTS_DIR    = Path(__file__).parent
S05_DIR      = TESTS_DIR.parent
SOLUTION_DIR = S05_DIR / "solution"
WEALTHDESK_ROOT = S05_DIR.parent   # cohort-1/wealthdesk/

load_dotenv(WEALTHDESK_ROOT / ".env")
sys.path.insert(0, str(SOLUTION_DIR))

from langgraph.checkpoint.memory import MemorySaver
from wealthdesk.agent import build_graph
from wealthdesk.config import DECLINE_RESPONSE, ESCALATE_RESPONSE
from wealthdesk.tools import query_rates   # used to get actual DB rates for validation

graph = build_graph(checkpointer=MemorySaver())

# ── Pull live rates from the DB for validation ────────────────────────────────
# This proves tool calls return real data, not hallucinated values.
try:
    _all_rates = query_rates.invoke({"product_type": "all"})
    # Extract percentage numbers that appear in the DB (e.g. "8.5", "9.5", "6.8")
    import re
    _rate_values = set(re.findall(r'\d+\.\d+', _all_rates))
except Exception:
    _rate_values = set()


# ── Test cases ────────────────────────────────────────────────────────────────
# (label, query, expected_route, expected_behaviour)
#
# expected_route     : "SIMPLE" | "COMPLEX" | "OUT_OF_SCOPE"
# expected_behaviour :
#     "answer"           — not escalate, not decline (any substantive reply)
#     "answer_with_rate" — not escalate/decline AND response contains a
#                          rate value that matches the live DB (tool called)
#     "escalate"         — response == ESCALATE_RESPONSE
#     "decline"          — response == DECLINE_RESPONSE

TEST_CASES = [
    # ── Tool: interest rates ─────────────────────────────────────────────────
    ("Home loan rate",   "What is the home loan interest rate at BNB?",
     "SIMPLE", "answer_with_rate"),

    ("FD rate",          "What are BNB's current fixed deposit interest rates?",
     "SIMPLE", "answer_with_rate"),

    ("Car loan rate",    "What is the car loan rate?",
     "SIMPLE", "answer_with_rate"),

    ("Personal loan",    "Tell me the personal loan interest rate.",
     "SIMPLE", "answer_with_rate"),

    # ── Tool: branch locations ───────────────────────────────────────────────
    ("Mumbai branch",    "Where is the BNB branch in Mumbai?",
     "SIMPLE", "answer"),

    ("Bengaluru branch", "Do you have a branch in Bengaluru?",
     "SIMPLE", "answer"),

    # ── Multi-tool: rates AND branches in one query ──────────────────────────
    # Regression case: before the if→while fix, the second tool call (query_branch)
    # caused a Groq 400 error — "Tool choice is none, but model called a tool".
    ("Rates and branches", "Get me all the rates and branches",
     "SIMPLE", "answer_with_rate"),

    # ── RAG policy questions (no tool expected) ──────────────────────────────
    ("Loan docs",        "What documents do I need to apply for a home loan?",
     "SIMPLE", "answer"),

    ("Prepayment",       "What is BNB's prepayment policy for loans?",
     "SIMPLE", "answer"),

    # ── Complex → escalate ───────────────────────────────────────────────────
    ("Best loan",        "Which loan product is best for me?",
     "COMPLEX", "escalate"),

    ("FD vs loan",       "Should I invest in a fixed deposit or take a personal loan?",
     "COMPLEX", "escalate"),

    # ── Out of scope → decline ───────────────────────────────────────────────
    ("Weather",          "What is the weather in Mumbai today?",
     "OUT_OF_SCOPE", "decline"),

    ("Stocks",           "Recommend me a good stock to buy right now.",
     "OUT_OF_SCOPE", "decline"),

    ("Cricket",          "Who won the IPL this year?",
     "OUT_OF_SCOPE", "decline"),

    # ── Follow-up memory (same thread, second turn uses context) ─────────────
    ("Follow-up 1",      "What is the home loan interest rate at BNB?",
     "SIMPLE", "answer_with_rate"),

    ("Follow-up 2",      "And what about the education loan rate?",
     "SIMPLE", "answer_with_rate"),
]

FOLLOW_UP_START = 13   # index of "Follow-up 1" — share one thread from here
SHARED_THREAD   = "live-eval-memory-thread"


# ── Run ───────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  WealthDesk S05 — Live Evaluation  (real Groq + real SQLite tools)")
print(f"  DB rate values found : {sorted(_rate_values) or 'none — DB may be missing'}")
print("=" * 80)

results = []

for i, (label, query, exp_route, exp_behaviour) in enumerate(TEST_CASES):
    thread = SHARED_THREAD if i >= FOLLOW_UP_START else f"eval-{i}"
    cfg    = {"configurable": {"thread_id": thread}}

    result   = graph.invoke({"customer_message": query, "response": ""}, config=cfg)
    route    = result.get("query_type", "?")
    response = result["response"]

    if response == ESCALATE_RESPONSE:
        actual = "escalate"
    elif response == DECLINE_RESPONSE:
        actual = "decline"
    else:
        actual = "answer"

    # For answer_with_rate: verify the response contains at least one rate
    # value that exists in the live DB (proves tool was called, not hallucinated)
    if exp_behaviour == "answer_with_rate" and actual == "answer":
        resp_rates = set(re.findall(r'\d+\.\d+', response))
        if _rate_values and not resp_rates.isdisjoint(_rate_values):
            actual = "answer_with_rate"          # rates match DB ✓
        elif _rate_values and resp_rates.isdisjoint(_rate_values):
            actual = "answer_hallucinated"       # rates don't match DB ✗
        elif not _rate_values and "%" in response:
            actual = "answer_with_rate"          # DB unavailable, has % at least
        # else: actual stays "answer" — no rate found at all

    route_ok = route == exp_route
    act_ok   = actual == exp_behaviour
    passed = route_ok and act_ok
    results.append(passed)

    mark = "✓ PASS" if passed else "✗ FAIL"
    print(f"\n{mark}  [{label}]")
    print(f"     Q      : {query[:72]}")
    print(f"     Route  : {route} (expected {exp_route}) {'✓' if route_ok else '✗'}")
    print(f"     Action : {actual} (expected {exp_behaviour}) {'✓' if act_ok else '✗'}")
    if not passed:
        snippet = response[:200].replace("\n", " ")
        print(f"     Resp   : {snippet}...")

total  = len(results)
passed = sum(results)
print("\n" + "=" * 80)
print(f"  Result : {passed}/{total} passed")
if passed < total:
    print(f"  {'─' * 40}")
    print(f"  {total - passed} failure(s) above need fixing before S05 is release-ready.")
    print()
    print("  Common causes:")
    print("    answer_hallucinated → LLM answered rate question without calling query_rates()")
    print("    wrong route         → CLASSIFY_SYSTEM_PROMPT prompt needs tightening")
    print("    escalate got answer → COMPLEX case fell through to SIMPLE path")
print("=" * 80 + "\n")
