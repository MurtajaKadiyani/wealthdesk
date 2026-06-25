"""
wealthdesk/state.py
-------------------
The shared state that flows through the LangGraph graph.

Every node reads from this state and writes back a partial update.
Only define the shape here -- no logic.
"""
from typing import TypedDict


# ---------------------------------------------------------------------------
# TODO 3 of 5 -- State definition
# ---------------------------------------------------------------------------
# Define WealthDeskState as a TypedDict with exactly two fields:
#
#   customer_message : str   -- the question the customer typed
#   response         : str   -- the answer WealthDesk will return
#
# Pattern:
#   class WealthDeskState(TypedDict):
#       field_name: type
#
# ---------------------------------------------------------------------------

class WealthDeskState(TypedDict):
    pass  # TODO: replace 'pass' with the two field definitions
