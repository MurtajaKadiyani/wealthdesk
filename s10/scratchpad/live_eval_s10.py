"""
s10/scratchpad/live_eval_s10.py
-------------------------------
Live evaluation for WealthDesk Session 10 (Multi-Agent Architecture).
Tests real Groq calls — no mocks.

Run from cohort-1/wealthdesk/:
    python s10/scratchpad/live_eval_s10.py
"""
import sys
from pathlib import Path
from uuid import uuid4

SOLUTION_DIR = Path(__file__).parent.parent / "solution"
sys.path.insert(0, str(SOLUTION_DIR))

from dotenv import load_dotenv
load_dotenv()

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from wealthdesk.agent import build_graph              # noqa: E402

TEST_CASES = [
    {
        "query":               "What is the home loan rate at BNB?",
        "expected_specialist": "rates_agent",
        "description":         "RATES → Rates Agent (MCP tool call for live rate)",
    },
    {
        "query":               "What documents do I need to apply for a home loan?",
        "expected_specialist": "documents_agent",
        "description":         "POLICY → Documents Agent (RAG retrieval, no tool call)",
    },
    {
        "query":               "Should I take a home loan now or wait for rates to fall?",
        "expected_specialist": "escalated",
        "description":         "COMPLEX → escalate (Relationship Manager referral)",
    },
    {
        "query":               "Who won the IPL this year?",
        "expected_specialist": "declined",
        "description":         "OUT_OF_SCOPE → decline (off-topic)",
    },
    {
        "query":               "What are BNB's car loan interest rates?",
        "expected_specialist": "rates_agent",
        "description":         "RATES (car loan) → Rates Agent",
    },
    {
        "query":               "What is the minimum deposit amount for a fixed deposit?",
        "expected_specialist": "documents_agent",
        "description":         "POLICY (FD terms) → Documents Agent",
    },
]


def run_eval() -> bool:
    graph  = build_graph(checkpointer=MemorySaver())
    passed = 0
    total  = len(TEST_CASES)

    print("=" * 65)
    print("  WealthDesk S10 — Live Evaluation (Multi-Agent Routing)")
    print("  Tests: real Groq calls, no mocks")
    print("=" * 65)

    for i, tc in enumerate(TEST_CASES, 1):
        thread_id = str(uuid4())
        config    = {"configurable": {"thread_id": thread_id}}
        try:
            result     = graph.invoke(
                {
                    "customer_message": tc["query"],
                    "response":         "",
                    "specialist":       "",
                    "retrieved_docs":   [],
                },
                config=config,
            )
            specialist = result.get("specialist", "?")
            ok         = specialist == tc["expected_specialist"]
            if ok:
                passed += 1
            status = "PASS" if ok else "FAIL"
            print(f"\n[{i}/{total}] {status} — {tc['description']}")
            print(f"  Query:     {tc['query']}")
            print(f"  Expected:  {tc['expected_specialist']}")
            print(f"  Got:       {specialist}")
            preview = result.get("response", "")[:100].replace("\n", " ")
            print(f"  Response:  {preview}...")
        except Exception as exc:
            print(f"\n[{i}/{total}] ERROR — {tc['description']}")
            print(f"  {exc}")

    print(f"\n{'='*65}")
    print(f"  Result: {passed}/{total} passed")
    if passed == total:
        print("  All routing cases pass. S10 is release-ready.")
    else:
        print(f"  {total - passed} case(s) failed. Fix routing before releasing.")
    print(f"{'='*65}")
    return passed == total


if __name__ == "__main__":
    ok = run_eval()
    sys.exit(0 if ok else 1)
