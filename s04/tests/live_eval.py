"""
s04/tests/live_eval.py — Live evaluation for WealthDesk S04 solution
----------------------------------------------------------------------
Runs 16 test queries directly against graph.invoke() using the REAL Groq
LLM and the REAL ChromaDB vectorstore. No mocks.

Why this exists alongside pytest
─────────────────────────────────
`pytest test_s04.py` mocks the LLM and vectorstore — it runs in ~2 seconds
and catches structural bugs (wrong node wired, state field missing, etc.).

This script catches a different class of defects that only appear with a
real LLM:
  • Classifier prompt brittleness  — e.g. FAQ query classified OUT_OF_SCOPE
  • Stale state leaking across turns — SqliteSaver persisting retrieved_docs
  • Score threshold miscalibration  — noise chunks sneaking into answers
  • LLM non-compliance with SYSTEM_PROMPT rules (rule 6 escalation)

Run this script:
  • Before every session (instructor pre-flight check)
  • After any change to CLASSIFY_SYSTEM, SYSTEM_PROMPT, or RETRIEVAL_K
  • After rebuilding the vectorstore (python data/ingest.py)
  • To self-check your starter implementation before asking for help

Usage (run from the wealthdesk/ directory):
  python s04/tests/live_eval.py

Expected output: 16/16 passed
If any test fails, the response snippet is printed so you can diagnose.

Cost: ~16 Groq API calls (~5–10 seconds total, well within free tier).
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────
TESTS_DIR    = Path(__file__).parent
S04_DIR      = TESTS_DIR.parent
SOLUTION_DIR = S04_DIR / "solution"
WEALTHDESK_ROOT = S04_DIR.parent  # cohort-1/wealthdesk/

load_dotenv(WEALTHDESK_ROOT / ".env")
sys.path.insert(0, str(SOLUTION_DIR))

from langgraph.checkpoint.memory import MemorySaver
from wealthdesk.agent import build_graph
from wealthdesk.config import DECLINE_RESPONSE, ESCALATE_RESPONSE, RETRIEVAL_SCORE_THRESHOLD

graph = build_graph(checkpointer=MemorySaver())

# ── Test cases ────────────────────────────────────────────────────────────────
# Each entry: (label, query, expected_route, expected_behaviour)
# expected_route    : "IN_SCOPE" | "OUT_OF_SCOPE"
# expected_behaviour: "answer" | "escalate" | "decline"
TEST_CASES = [
    # Factual RAG — retrieved docs should drive the answer
    ("FAQs",          "What different types of products does BNB provide?",        "IN_SCOPE",     "answer"),
    ("Home loan doc", "What documents do I need for a home loan?",                  "IN_SCOPE",     "answer"),
    ("FD tax",        "Is the 5-year FD tax deductible?",                          "IN_SCOPE",     "answer"),
    ("FD early exit", "What happens to my FD if I withdraw it early?",             "IN_SCOPE",     "answer"),
    ("Prepayment",    "What is BNB's policy on loan prepayment charges?",          "IN_SCOPE",     "answer"),

    # Personal advice — IN_SCOPE (about BNB) but respond() escalates via rule 6
    ("Advice 1",      "Should I invest in FDs or take a home loan?",               "IN_SCOPE",     "escalate"),
    ("Advice 2",      "Which loan is best for me?",                                "IN_SCOPE",     "escalate"),

    # Eligibility with salary — LLM gives general formula + RM referral (answer, not pure escalation)
    ("Advice 3",      "Can I afford a home loan on a salary of Rs. 50,000?",       "IN_SCOPE",     "answer"),

    # Fragment — cosine score 0.11–0.12, filtered by threshold → no docs → escalate
    ("Fragment",      "Which",                                                      "IN_SCOPE",     "escalate"),

    # Gibberish — classifier catches it as OUT_OF_SCOPE before retrieval
    ("Gibberish",     "asdfjkl; qwerty",                                           "OUT_OF_SCOPE", "decline"),

    # Out of scope
    ("Weather",       "What is the weather in Mumbai today?",                       "OUT_OF_SCOPE", "decline"),
    ("Poem",          "Write a poem about savings",                                 "OUT_OF_SCOPE", "decline"),
    ("Stocks",        "Can you help me invest in stocks?",                          "OUT_OF_SCOPE", "decline"),
    ("Cricket",       "Who won the cricket match yesterday?",                       "OUT_OF_SCOPE", "decline"),

    # Follow-up memory — same thread, second query should use prior context
    ("Follow-up 1",   "What documents do I need for a home loan?",                 "IN_SCOPE",     "answer"),
    ("Follow-up 2",   "And what about for a personal loan?",                       "IN_SCOPE",     "answer"),
]

FOLLOW_UP_START = 14   # index of "Follow-up 1" — share one thread from here
SHARED_THREAD   = "live-eval-memory-thread"

# ── Run ───────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  WealthDesk S04 — Live Evaluation  (real Groq, no mocks)")
print(f"  Retrieval score threshold : {RETRIEVAL_SCORE_THRESHOLD}")
print("=" * 80)

results = []
for i, (label, query, exp_route, exp_behaviour) in enumerate(TEST_CASES):
    thread = SHARED_THREAD if i >= FOLLOW_UP_START else f"eval-{i}"
    cfg    = {"configurable": {"thread_id": thread}}

    result   = graph.invoke({"customer_message": query, "response": ""}, config=cfg)
    route    = result.get("query_type", "?")
    docs     = result.get("retrieved_docs", [])
    response = result["response"]

    if response == ESCALATE_RESPONSE:
        actual = "escalate"
    elif response == DECLINE_RESPONSE:
        actual = "decline"
    else:
        actual = "answer"

    route_ok = route == exp_route
    act_ok   = actual == exp_behaviour
    passed   = route_ok and act_ok
    results.append(passed)

    sources = ""
    if docs and response != ESCALATE_RESPONSE:
        src_set = {d.split("]\n")[0].lstrip("[") for d in docs if "]\n" in d}
        sources = f"  [{len(docs)} chunk(s): {', '.join(sorted(src_set))}]"

    mark = "✓ PASS" if passed else "✗ FAIL"
    print(f"\n{mark}  [{label}]")
    print(f"     Q      : {query[:70]}")
    print(f"     Route  : {route} (expected {exp_route}) {'✓' if route_ok else '✗'}")
    print(f"     Action : {actual} (expected {exp_behaviour}) {'✓' if act_ok else '✗'}{sources}")
    if not passed:
        snippet = response[:150].replace("\n", " ")
        print(f"     Resp   : {snippet}...")

total  = len(results)
passed = sum(results)
print("\n" + "=" * 80)
print(f"  Result : {passed}/{total} passed")
if passed < total:
    print(f"  {'─' * 40}")
    print(f"  {total - passed} failure(s) above need fixing before this session is release-ready.")
print("=" * 80 + "\n")
