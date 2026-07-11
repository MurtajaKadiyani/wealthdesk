"""
wealthdesk/nodes.py
-------------------
Node functions for the WealthDesk graph.

Each node is a plain Python function:
  - Input : the full WealthDeskState (read-only)
  - Output: a dict containing ONLY the keys this node changed
             (LangGraph merges it into the state automatically)
"""
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from .config import SYSTEM_PROMPT
from .state import WealthDeskState
from .tools import llm

def respond(state: WealthDeskState) -> dict:
    """Call the LLM and return the agent's reply."""
    messages = [
        SystemMessage(content=SYSTEM_PROMPT)
    ]
    history = state.get("history",[])
    for turn in history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"])) # type: ignore
        else:
            messages.append(AIMessage(content=turn["content"])) # type: ignore
    
    messages.append(HumanMessage(content=state["customer_message"])) # type: ignore
    
    try:
      result = llm.invoke(messages)
      response_text = result.content
      new_history = history + [{"role": "user", "content": state["customer_message"]}, # type: ignore
                             {"role": "assistant", "content": response_text}]
      return {"response": response_text, "history": new_history}
    except Exception as e:
      print(f"[WealthDesk] LLM error: {e}")
      return {"response": "I am temporarily unavailable. Please try again in a moment" }
     
    # State has two fields, only one field got updated as parr of this function execution,
    # node execution. So method will return only the modified values
    # evntually state will contain multiple fields , maybe history? Query type , information about retrieved docs ,
    # Compliance passed or not?  