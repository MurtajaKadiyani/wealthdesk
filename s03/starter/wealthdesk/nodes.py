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

from .config import CLASSIFY_SYSTEM_PROMPT, DECLINE_RESPONSE, ESCALATE_RESPONSE, SYSTEM_PROMPT
from .state import WealthDeskState
from .tools import llm, classifier_llm

BLOCKLIST = [
    "ignore all previous",
    "forget everything",
    "you are now",
    "disregard your system",
    "act as",
    "jailbreak",
]

def classify(state: WealthDeskState) -> dict:
    """Call the LLM and return the agent's reply."""
    msg = state["customer_message"].strip()

    if any(phrase in msg.lower() for phrase in BLOCKLIST):
        return {"query_type": "OUT_OF_SCOPE"}
    
    if not msg or len(msg) < 10 or len(msg) > 500:
        return {"query_type": "OUT_OF_SCOPE"}
    
    messages = [
        SystemMessage(content=CLASSIFY_SYSTEM_PROMPT),
        HumanMessage(content=state["customer_message"]),
    ]
 
    try:
       result = classifier_llm.invoke(messages)
       query_type = result.content.strip().upper() # type: ignore
       if query_type not in {"SIMPLE","COMPLEX","OUT_OF_SCOPE"}:
          query_type = "SIMPLE"
    except Exception as e:
        print(f"[WealthDesk] Classification error: {e}")
        query_type = "SIMPLE"
 
    return {"query_type": query_type}

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

def escalate(state: WealthDeskState) -> dict:
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": ESCALATE_RESPONSE},
    ]
    return {"response": ESCALATE_RESPONSE, "history": new_history}
 
def decline(state: WealthDeskState) -> dict:
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": DECLINE_RESPONSE},
    ]
    return {"response": DECLINE_RESPONSE, "history": new_history}
 
def route_query(state: WealthDeskState)->str:
   query_type = state.get("query_type","SIMPLE")
   if query_type == "COMPLEX":
      return "escalate"
   if query_type == "OUT_OF_SCOPE":
      return "decline"
   return "respond"