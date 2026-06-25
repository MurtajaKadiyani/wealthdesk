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


# ---------------------------------------------------------------------------
# TODO 5 of 5 -- build_graph
# ---------------------------------------------------------------------------
# Implement build_graph() so it:
#
#   1. Creates a StateGraph:
#        builder = StateGraph(WealthDeskState)
#
#   2. Registers the respond node:
#        builder.add_node("respond", respond)
#
#   3. Sets the entry point (first node to run):
#        builder.set_entry_point("respond")
#
#   4. Connects respond → END (the graph exits after one response):
#        builder.add_edge("respond", END)
#
#   5. Compiles and returns the graph:
#        return builder.compile()
#
# ---------------------------------------------------------------------------

def build_graph():
    """Build and compile the WealthDesk LangGraph graph."""
    # TODO: implement this function
    pass


# Module-level graph instance (used by LangSmith Studio for deployment).
graph = build_graph()


# ---------------------------------------------------------------------------
# Terminal loop (provided -- no changes needed)
# ---------------------------------------------------------------------------

def run() -> None:
    _graph = build_graph()

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

        result = _graph.invoke({"customer_message": user_input, "response": ""})
        print(f"\nWealthDesk: {result['response']}")


if __name__ == "__main__":
    run()
