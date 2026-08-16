"""
live_eval_s12.py
----------------
Live evaluation for WealthDesk Session 12 — Compliance Agent.

Runs 6 test cases against the real Groq LLM (no mocks).
Tests specialist routing AND compliance behaviour.

Run from inside s12/solution/:
    python ../../scratchpad/live_eval_s12.py

Or from the cohort-1 root:
    python wealthdesk/s12/scratchpad/live_eval_s12.py
"""
import sys
import os
from pathlib import Path

# Add s12/solution to sys.path
SOLUTION_DIR = Path(__file__).parent.parent / "solution"
sys.path.insert(0, str(SOLUTION_DIR))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")  # cohort-1/.env

from langgraph.checkpoint.memory import MemorySaver

from wealthdesk.agent import build_graph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_query(graph, message: str, thread_id: str) -> dict:
    return graph.invoke(
        {
            "customer_message":  message,
            "response":          "",
            "specialist":        "",
            "retrieved_docs":    [],
            "compliance_status": "",
        },
        config={"configurable": {"thread_id": thread_id}},
    )


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    return condition


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  WealthDesk S12 — Live Eval (real Groq, no mocks)")
    print("  Architecture: Supervisor → Specialist → Compliance Agent")
    print("=" * 65)

    graph   = build_graph(checkpointer=MemorySaver())
    passed  = 0
    total   = 0

    # -----------------------------------------------------------------------
    # TC-1: RATES query → rates_agent → compliance_status set
    # -----------------------------------------------------------------------
    print("\nTC-1: Home loan rate (RATES → rates_agent, compliance required)")
    result = run_query(graph, "What is the home loan rate?", "eval-tc1")
    sp  = result.get("specialist", "")
    cs  = result.get("compliance_status", "")
    rsp = result.get("response", "")
    total += 3
    passed += check("specialist == rates_agent",                  sp == "rates_agent",       sp)
    passed += check("compliance_status in {PASS, REVISED}",       cs in {"PASS", "REVISED"}, cs)
    passed += check("response is non-empty",                       bool(rsp))
    print(f"  Response (first 120 chars): {rsp[:120]}")

    # -----------------------------------------------------------------------
    # TC-2: POLICY query → documents_agent → compliance_status set
    # -----------------------------------------------------------------------
    print("\nTC-2: Documents query (POLICY → documents_agent, compliance required)")
    result = run_query(graph, "What documents do I need for a home loan?", "eval-tc2")
    sp  = result.get("specialist", "")
    cs  = result.get("compliance_status", "")
    rsp = result.get("response", "")
    total += 3
    passed += check("specialist == documents_agent",              sp == "documents_agent",   sp)
    passed += check("compliance_status in {PASS, REVISED}",       cs in {"PASS", "REVISED"}, cs)
    passed += check("response is non-empty",                       bool(rsp))
    print(f"  Response (first 120 chars): {rsp[:120]}")

    # -----------------------------------------------------------------------
    # TC-3: COMPLEX query → escalated (compliance skipped)
    # -----------------------------------------------------------------------
    print("\nTC-3: Personal advice (COMPLEX → escalated, no compliance)")
    result = run_query(graph, "Should I invest in a fixed deposit or equities right now?", "eval-tc3")
    sp  = result.get("specialist", "")
    rsp = result.get("response", "")
    total += 2
    passed += check("specialist == escalated",                    sp == "escalated",         sp)
    passed += check("response mentions branch or phone",
                    "branch" in rsp.lower() or "1800" in rsp,    rsp[:80])
    print(f"  Response (first 120 chars): {rsp[:120]}")

    # -----------------------------------------------------------------------
    # TC-4: OUT_OF_SCOPE → declined (compliance skipped)
    # -----------------------------------------------------------------------
    print("\nTC-4: Out of scope (OUT_OF_SCOPE → declined)")
    result = run_query(graph, "Tell me a joke", "eval-tc4")
    sp  = result.get("specialist", "")
    rsp = result.get("response", "")
    total += 2
    passed += check("specialist == declined",                     sp == "declined",          sp)
    passed += check("response mentions BNB banking",
                    "bnb" in rsp.lower() or "banking" in rsp.lower(), rsp[:80])
    print(f"  Response (first 120 chars): {rsp[:120]}")

    # -----------------------------------------------------------------------
    # TC-5: compliance_status field is always present in state
    # -----------------------------------------------------------------------
    print("\nTC-5: compliance_status field present in all specialist responses")
    for msg, tid in [
        ("What is the FD interest rate for 1 year?", "eval-tc5a"),
        ("What are the eligibility criteria for a car loan?", "eval-tc5b"),
    ]:
        r = run_query(graph, msg, tid)
        sp_val = r.get("specialist", "")
        cs_val = r.get("compliance_status", None)
        total += 1
        passed += check(
            f"compliance_status field exists ({msg[:40]}…)",
            cs_val is not None,
            f"got: {cs_val!r}",
        )
    print("  (compliance field must be present even if empty for escalated/declined)")

    # -----------------------------------------------------------------------
    # TC-6: FD rate for senior citizens (RATES → rates_agent, non-empty response)
    # -----------------------------------------------------------------------
    print("\nTC-6: FD senior citizen rate (RATES → rates_agent)")
    result = run_query(graph, "What is the FD rate for senior citizens?", "eval-tc6")
    sp  = result.get("specialist", "")
    cs  = result.get("compliance_status", "")
    rsp = result.get("response", "")
    total += 3
    passed += check("specialist == rates_agent",                  sp == "rates_agent",       sp)
    passed += check("compliance_status in {PASS, REVISED}",       cs in {"PASS", "REVISED"}, cs)
    passed += check("response mentions senior or FD",
                    "senior" in rsp.lower() or "fd" in rsp.lower() or "fixed deposit" in rsp.lower(),
                    rsp[:80])
    print(f"  Response (first 120 chars): {rsp[:120]}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 65)
    print(f"  Result: {passed}/{total} checks passed")
    if passed == total:
        print("  STATUS: ALL PASS — S12 is live-eval ready")
    else:
        print(f"  STATUS: {total - passed} FAILURES — fix before class")
    print("=" * 65)
    return passed == total


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
