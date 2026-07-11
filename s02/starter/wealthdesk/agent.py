"""
wealthdesk/agent.py
-------------------
Graph construction and the terminal loop.

Run the agent from the repo root:
    cd cohort-1/wealthdesk/s01/starter
    python -m wealthdesk.agent

Session 1 graph:
    START --> respond --> END
"""
from langgraph.graph import END, StateGraph

from .nodes import respond
from .state import WealthDeskState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from uuid import uuid4
from .config import CHECKPOINT_DB

def build_graph(checkpointer = None):
    # Start -> Respond -> End
    builder = StateGraph(WealthDeskState)
    builder.add_node("respond", respond)
    builder.set_entry_point("respond") #START
    builder.add_edge("respond", END) # END
    if checkpointer is None:
        checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# Module-level graph instance required by langgraph.json for LangGraph Studio.
# run() uses this directly rather than building a second copy.
graph = build_graph()


# ---------------------------------------------------------------------------
# Terminal loop (provided -- no changes needed)
# ---------------------------------------------------------------------------

def run() -> None:
    conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
    _graph    = build_graph(checkpointer=SqliteSaver(conn))  # terminal app opts into disk persistence explicit
    thread_id = str(uuid4())
    config    = {"configurable": {"thread_id": thread_id}}
    print("=" * 55)
    print("  WealthDesk | Bharat National Bank")
    print("  Type 'quit' to exit")
    print("=" * 55)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nWealthDesk: Session ended. Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "bye"}:
            print("\nWealthDesk: Thank you for choosing Bharat National Bank. Goodbye!")
            break

        # "response": "" is a placeholder to satisfy the TypedDict contract.
        # respond() overwrites it; graph.invoke() returns the full merged state.
        result = _graph.invoke({"customer_message": user_input, "response": ""},config=config) # type: ignore
        print(f"\nWealthDesk: {result['response']}")


if __name__ == "__main__":
    run()
