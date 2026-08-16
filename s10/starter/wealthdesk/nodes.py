"""
wealthdesk/nodes.py
-------------------
Graph nodes for WealthDesk supervisor and specialist agents.
 
Session 10 replaces the single agent with a supervisor pattern:
  classify() routes to one of two specialist agents or escalate/decline.
  The Documents Agent retrieves policy docs then responds with no tools.
  The Rates Agent uses MCP tools to fetch live rates and branch data.
"""
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, StateGraph
 
from .config import (
    CLASSIFY_SYSTEM, DECLINE_RESPONSE, DOCS_SYSTEM_PROMPT, EMBED_MODEL,
    ESCALATE_RESPONSE, RETRIEVAL_K, SYSTEM_PROMPT, VECTORSTORE_DIR,
)
from .state import WealthDeskState
from .tools import _run_tool, classifier_llm, llm, llm_with_tools
 
vectorstore = None
 
 
def _init_vectorstore() -> None:
    global vectorstore
    if vectorstore is not None:
        return
    try:
        embeddings  = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vectorstore = Chroma(
            persist_directory=str(VECTORSTORE_DIR),
            embedding_function=embeddings,
        )
    except Exception as e:
        print(f"[WealthDesk] Could not load vectorstore: {e}")
        print("  Run 'python data/ingest.py' to create it.")
 
 
# ---------------------------------------------------------------------------
# Specialist node functions (provided -- no changes needed)
# ---------------------------------------------------------------------------
 
def _doc_retrieve(state: WealthDeskState) -> dict:
    """Retrieve policy documents for the Documents Agent."""
    _init_vectorstore()
    if vectorstore is None:
        return {"retrieved_docs": []}
    try:
        docs = vectorstore.similarity_search(state["customer_message"], k=RETRIEVAL_K)
        return {
            "retrieved_docs": [
                f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
                for doc in docs
            ]
        }
    except Exception as e:
        print(f"[WealthDesk] Documents Agent retrieval error: {e}")
        return {"retrieved_docs": []}
 
 
def _doc_respond(state: WealthDeskState) -> dict:
    """Generate response using policy context. No database tools needed."""
    history   = state.get("history", [])
    retrieved = state.get("retrieved_docs", [])
 
    context_block  = "\n\n---\n\n".join(retrieved) if retrieved else ""
    system_content = (
        DOCS_SYSTEM_PROMPT
        + (
            "\n\nThe following sections from BNB's policy documents are relevant "
            "to the customer's question. Use this information in your answer:\n\n"
            + context_block
            if context_block else ""
        )
    )
 
    messages = [SystemMessage(content=system_content)]
    for turn in history:
        messages.append(
            HumanMessage(content=turn["content"]) if turn["role"] == "user"
            else AIMessage(content=turn["content"]) # type: ignore
        )
    messages.append(HumanMessage(content=state["customer_message"])) # type: ignore
 
    try:
        result        = llm.invoke(messages)
        response_text = result.content
    except Exception as e:
        print(f"[WealthDesk] Documents Agent LLM error: {e}")
        response_text = "I am temporarily unavailable. Please try again in a moment."
 
    return {
        "response": response_text,
        "history":  history + [
            {"role": "user",      "content": state["customer_message"]},
            {"role": "assistant", "content": response_text},
        ],
    }
 
 
def _rates_respond(state: WealthDeskState) -> dict:
    """Generate response with MCP tool calls for live rates and branch data."""
    history  = state.get("history", [])
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for turn in history:
        messages.append(
            HumanMessage(content=turn["content"]) if turn["role"] == "user"
            else AIMessage(content=turn["content"]) # type: ignore
        )
    messages.append(HumanMessage(content=state["customer_message"])) # type: ignore
 
    try:
        result = llm_with_tools.invoke(messages)
 
        # Manual tool execution — see s05/nodes.py for the ToolNode alternative.
        if result.tool_calls:
            messages.append(result) # type: ignore
            for tc in result.tool_calls:
                tool_output = _run_tool(tc["name"], tc["args"])
                print(
                    f"[WealthDesk] Rates Agent MCP: {tc['name']}({tc['args']}) "
                    f"-> {str(tool_output)[:80]}"
                )
                messages.append(ToolMessage(content=str(tool_output), tool_call_id=tc["id"])) # type: ignore
            result = llm.invoke(messages)
 
        response_text = result.content
 
    except Exception as e:
        print(f"[WealthDesk] Rates Agent LLM error: {e}")
        response_text = "I am temporarily unavailable. Please try again in a moment."
 
    return {
        "response": response_text,
        "history":  history + [
            {"role": "user",      "content": state["customer_message"]},
            {"role": "assistant", "content": response_text},
        ],
    }
 
 
# ---------------------------------------------------------------------------
# TODO 1 of 5 -- create_documents_agent()
# ---------------------------------------------------------------------------
# Build and compile the Documents Agent as a standalone subgraph.
# It has two nodes: retrieve_docs → respond → END
#
#   def create_documents_agent():
#       builder = StateGraph(WealthDeskState)
#       builder.add_node("retrieve_docs", _doc_retrieve)
#       builder.add_node("respond",       _doc_respond)
#       builder.set_entry_point("retrieve_docs")
#       builder.add_edge("retrieve_docs", "respond")
#       builder.add_edge("respond",       END)
#       return builder.compile()
# ---------------------------------------------------------------------------
def create_documents_agent():
    builder = StateGraph(WealthDeskState)
    builder.add_node("retrieve_docs", _doc_retrieve)
    builder.add_node("document_respond",       _doc_respond)
    builder.set_entry_point("retrieve_docs")
    builder.add_edge("retrieve_docs", "document_respond")
    builder.add_edge("document_respond",       END)
    return builder.compile()
 
 
# ---------------------------------------------------------------------------
# TODO 2 of 5 -- create_rates_agent()
# ---------------------------------------------------------------------------
# Build and compile the Rates Agent as a standalone subgraph.
# It has one node: respond → END (tool calls happen inside _rates_respond)
#
#   def create_rates_agent():
#       builder = StateGraph(WealthDeskState)
#       builder.add_node("respond", _rates_respond)
#       builder.set_entry_point("respond")
#       builder.add_edge("respond", END)
#       return builder.compile()
# ---------------------------------------------------------------------------
def create_rates_agent():
    builder = StateGraph(WealthDeskState)
    builder.add_node("rates_respond", _rates_respond)
    builder.set_entry_point("rates_respond")
    builder.add_edge("rates_respond", END)
    return builder.compile()
 
 
 
_documents_agent = create_documents_agent()
_rates_agent     = create_rates_agent()
 
 
# ---------------------------------------------------------------------------
# Supervisor nodes
# ---------------------------------------------------------------------------
 
def classify(state: WealthDeskState) -> dict:
    """Classify into RATES, POLICY, COMPLEX, or OUT_OF_SCOPE. Provided."""
    messages = [SystemMessage(content=CLASSIFY_SYSTEM)]
    for turn in state.get("history", [])[-2:]:
        messages.append(
            HumanMessage(content=turn["content"]) if turn["role"] == "user"
            else AIMessage(content=turn["content"]) # type: ignore
        )
    messages.append(HumanMessage(content=state["customer_message"])) # type: ignore
    try:
        result     = classifier_llm.invoke(messages)
        query_type = result.content.strip().upper() # type: ignore
        if query_type not in {"RATES", "POLICY", "COMPLEX", "OUT_OF_SCOPE"}:
            query_type = "RATES"
    except Exception as e:
        print(f"[WealthDesk] Supervisor classification error: {e}")
        query_type = "RATES"
    return {"query_type": query_type}
 
 
# ---------------------------------------------------------------------------
# TODO 3 of 5 -- call_documents_agent()
# ---------------------------------------------------------------------------
# Supervisor node that invokes _documents_agent with a fresh state dict and
# merges the result back into the supervisor state.
#
#   def call_documents_agent(state: WealthDeskState) -> dict:
#       print("[WealthDesk] Supervisor → Documents Agent")
#       result = _documents_agent.invoke({
#           "customer_message": state["customer_message"],
#           "history":          state.get("history", []),
#           "response":         "",
#           "query_type":       state.get("query_type", "POLICY"),
#           "retrieved_docs":   [],
#           "specialist":       "",
#       })
#       return {
#           "response":       result["response"],
#           "retrieved_docs": result.get("retrieved_docs", []),
#           "history":        result.get("history", state.get("history", [])),
#           "specialist":     "documents_agent",
#       }
# ---------------------------------------------------------------------------
def call_documents_agent(state: WealthDeskState) -> dict:
    print("[WealthDesk] Supervisor → Documents Agent")
    result = _documents_agent.invoke({
        "customer_message": state["customer_message"],
        "history":          state.get("history", []),
        "response":         "",
        "query_type":       state.get("query_type", "POLICY"),
        "retrieved_docs":   [],
        "specialist":       "",
    })
    return {
        "response":       result["response"],
        "retrieved_docs": result.get("retrieved_docs", []),
        "history":        result.get("history", state.get("history", [])),
        "specialist":     "documents_agent",
    }
 
 
# ---------------------------------------------------------------------------
# TODO 4 of 5 -- call_rates_agent()
# ---------------------------------------------------------------------------
# Supervisor node that invokes _rates_agent with a fresh state dict.
#
#   def call_rates_agent(state: WealthDeskState) -> dict:
#       print("[WealthDesk] Supervisor → Rates Agent")
#       result = _rates_agent.invoke({
#           "customer_message": state["customer_message"],
#           "history":          state.get("history", []),
#           "response":         "",
#           "query_type":       state.get("query_type", "RATES"),
#           "retrieved_docs":   [],
#           "specialist":       "",
#       })
#       return {
#           "response":   result["response"],
#           "history":    result.get("history", state.get("history", [])),
#           "specialist": "rates_agent",
#       }
# ---------------------------------------------------------------------------
def call_rates_agent(state: WealthDeskState) -> dict:
    print("[WealthDesk] Supervisor → Rates Agent")
    result = _rates_agent.invoke({
        "customer_message": state["customer_message"],
        "history":          state.get("history", []),
        "response":         "",
        "query_type":       state.get("query_type", "RATES"),
        "retrieved_docs":   [],
        "specialist":       "",
    })
    return {
        "response":   result["response"],
        "history":    result.get("history", state.get("history", [])),
        "specialist": "rates_agent",
    }
 
 
def escalate(state: WealthDeskState) -> dict:
    """Provided -- no changes needed."""
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": ESCALATE_RESPONSE},
    ]
    return {"response": ESCALATE_RESPONSE, "history": new_history, "specialist": "escalated"}
 
 
def decline(state: WealthDeskState) -> dict:
    """Provided -- no changes needed."""
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": DECLINE_RESPONSE},
    ]
    return {"response": DECLINE_RESPONSE, "history": new_history, "specialist": "declined"}
 
 
def route_supervisor(state: WealthDeskState) -> str:
    """Routing function for the supervisor. Provided -- no changes needed."""
    qt = state.get("query_type", "RATES")
    if qt == "POLICY":
        return "call_documents_agent"
    if qt == "COMPLEX":
        return "escalate"
    if qt == "OUT_OF_SCOPE":
        return "decline"
    return "call_rates_agent"
 