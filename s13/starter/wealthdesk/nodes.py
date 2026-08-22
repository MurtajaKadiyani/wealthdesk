import re
import sqlite3
from typing import Callable, Optional
 
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langsmith import traceable
from langgraph.graph import END, StateGraph
 
from .config import (
    CLASSIFY_SYSTEM,
    DB_PATH,
    DECLINE_RESPONSE,
    DOCS_SYSTEM_PROMPT,
    EMBED_MODEL,
    ESCALATE_RESPONSE,
    RETRIEVAL_K,
    SAFE_COMPLIANCE_RESPONSE,
    SEBI_BANNED_PHRASES,
    SYSTEM_PROMPT,
    VECTORSTORE_DIR,
)
from .state import WealthDeskState
from .tools import _run_tool, classifier_llm, llm, llm_with_tools
 
# ---------------------------------------------------------------------------
# S13: Token streaming hook
#
# Set by app.py to a callable before graph.invoke() when the Streamlit UI
# wants to display tokens as they arrive. None = silent (tests, CLI).
# The two respond functions call it with each token from llm.stream().
# ---------------------------------------------------------------------------
_stream_callback: Optional[Callable[[str], None]] = None
 
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
# Compliance helpers
# ---------------------------------------------------------------------------
 
def _load_valid_rates() -> set:
    try:
        conn      = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        loan_rows = conn.execute("SELECT interest_rate FROM loan_products").fetchall()
        fd_rows   = conn.execute("SELECT interest_rate, senior_rate FROM fd_products").fetchall()
        conn.close()
        loan_rates = {row[0] for row in loan_rows}
        fd_base    = {row[0] for row in fd_rows}
        fd_senior  = {row[0] + row[1] for row in fd_rows}
        return loan_rates | fd_base | fd_senior
    except Exception:
        return set()
 
 
def _extract_rates(text: str) -> list:
    matches = re.findall(r"(\d+\.?\d*)\s*%\s*p\.a\.", text, re.IGNORECASE)
    return [float(m) for m in matches]
 
 
@traceable(name="sebi_compliance_check")
def _check_compliance_logic(draft: str) -> tuple:
    lower = draft.lower()
 
    for phrase in SEBI_BANNED_PHRASES:
        if phrase in lower:
            return False, f"banned phrase: '{phrase}'"
 
    mentioned_rates = _extract_rates(draft)
    if mentioned_rates:
        valid_rates = _load_valid_rates()
        if valid_rates:
            for rate in mentioned_rates:
                if rate not in valid_rates:
                    return False, f"hallucinated rate: {rate}% p.a. not in database"
 
    return True, "PASS"
 
 
# ---------------------------------------------------------------------------
# Compliance Agent node functions
# ---------------------------------------------------------------------------
 
def check_sebi(state: WealthDeskState) -> dict:
    draft          = state["response"]
    passed, reason = _check_compliance_logic(draft)
 
    if not passed:
        print(f"[WealthDesk] Compliance FAIL: {reason}")
        return {"compliance_status": f"FAIL: {reason}"}
 
    print("[WealthDesk] Compliance PASS")
    return {"compliance_status": "PASS"}
 
 
def revise_response(state: WealthDeskState) -> dict:
    draft  = state["response"]
    reason = state.get("compliance_status", "violation").replace("FAIL: ", "")
 
    prompt = (
        "You are a BNB compliance officer reviewing an AI banking response.\n\n"
        f"The response was flagged for: {reason}\n\n"
        "Rewrite it to fix the violation while keeping the response helpful.\n\n"
        "Rules:\n"
        "  1. Never use: 'guaranteed returns', 'guaranteed return', 'guaranteed interest', 'risk-free', 'assured profit', 'assured returns', 'no risk'\n"
        "  2. Only state interest rates that appeared in the original -- do not add new ones\n"
        "  3. Keep the rewritten response under 150 words\n"
        "  4. End with 'WealthDesk | Bharat National Bank'\n\n"
        f"Original response:\n{draft}\n\n"
        "Compliant rewrite:"
    )
 
    try:
        result       = llm.invoke([HumanMessage(content=prompt)])
        revised_text = result.content.strip() or SAFE_COMPLIANCE_RESPONSE # type: ignore
    except Exception as e:
        print(f"[WealthDesk] Compliance Agent revision error: {e}")
        revised_text = SAFE_COMPLIANCE_RESPONSE
 
    print("[WealthDesk] Compliance Agent: response revised")
    return {
        "response":          revised_text,
        "compliance_status": "REVISED",
    }
 
 
def route_compliance(state: WealthDeskState) -> str:
    return "revise" if state.get("compliance_status", "").startswith("FAIL") else END
 
 
def create_compliance_agent():
    builder = StateGraph(WealthDeskState)
 
    builder.add_node("check_sebi", check_sebi)
    builder.add_node("revise",     revise_response)
 
    builder.set_entry_point("check_sebi")
    builder.add_conditional_edges(
        "check_sebi",
        route_compliance,
        {"revise": "revise", END: END},
    )
    builder.add_edge("revise", END)
 
    return builder.compile()
 
 
_compliance_agent = create_compliance_agent()
 
 
# ---------------------------------------------------------------------------
# Specialist agent node functions
# ---------------------------------------------------------------------------
 
def _doc_retrieve(state: WealthDeskState) -> dict:
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
        # S13 STREAMING HOOK: when the Streamlit UI is active, app.py sets
        # _stream_callback to a _StreamingState instance before graph.invoke().
        # We use llm.stream() instead of llm.invoke() so tokens are emitted
        # one by one — _stream_callback(chunk.content) sends each token to the
        # UI placeholder, producing the typewriter effect.
        # When _stream_callback is None (tests, CLI), we fall back to the
        # simpler llm.invoke() which waits for the full response before returning.
        if _stream_callback is not None:
            response_text = ""
            for chunk in llm.stream(messages):
                if chunk.content:
                    response_text += chunk.content # type: ignore
                    _stream_callback(chunk.content) # type: ignore
        else:
            response_text = llm.invoke(messages).content
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
            # After tool results are appended, the LLM generates its final
            # answer. This is the response the customer sees — stream it
            # when the UI is watching, invoke() silently otherwise.
            # (The first llm_with_tools.invoke() above is never streamed
            # because tool-call detection needs the full structured output.)
            if _stream_callback is not None:
                response_text = ""
                for chunk in llm.stream(messages):
                    if chunk.content:
                        response_text += chunk.content # type: ignore
                        _stream_callback(chunk.content) # type: ignore
            else:
                response_text = llm.invoke(messages).content
        else:
            # Direct response (no tool call) — already computed, no re-invoke needed.
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
# Specialist agent factory functions
# ---------------------------------------------------------------------------
 
def create_documents_agent():
    builder = StateGraph(WealthDeskState)
    builder.add_node("retrieve_docs", _doc_retrieve)
    builder.add_node("respond",       _doc_respond)
    builder.set_entry_point("retrieve_docs")
    builder.add_edge("retrieve_docs", "respond")
    builder.add_edge("respond",       END)
    return builder.compile()
 
 
def create_rates_agent():
    builder = StateGraph(WealthDeskState)
    builder.add_node("respond", _rates_respond)
    builder.set_entry_point("respond")
    builder.add_edge("respond", END)
    return builder.compile()
 
 
_documents_agent = create_documents_agent()
_rates_agent     = create_rates_agent()
 
 
# ---------------------------------------------------------------------------
# Supervisor nodes
# ---------------------------------------------------------------------------
 
def classify(state: WealthDeskState) -> dict:
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
 
 
def call_documents_agent(state: WealthDeskState) -> dict:
    print("[WealthDesk] Supervisor → Documents Agent")
    result = _documents_agent.invoke({
        "customer_message":  state["customer_message"],
        "history":           state.get("history", []),
        "response":          "",
        "query_type":        state.get("query_type", "POLICY"),
        "retrieved_docs":    [],
        "specialist":        "",
        "compliance_status": "",
    })
    return {
        "response":       result["response"],
        "retrieved_docs": result.get("retrieved_docs", []),
        "history":        result.get("history", state.get("history", [])),
        "specialist":     "documents_agent",
    }
 
 
def call_rates_agent(state: WealthDeskState) -> dict:
    print("[WealthDesk] Supervisor → Rates Agent")
    result = _rates_agent.invoke({
        "customer_message":  state["customer_message"],
        "history":           state.get("history", []),
        "response":          "",
        "query_type":        state.get("query_type", "RATES"),
        "retrieved_docs":    [],
        "specialist":        "",
        "compliance_status": "",
    })
    return {
        "response":   result["response"],
        "history":    result.get("history", state.get("history", [])),
        "specialist": "rates_agent",
    }
 
 
def call_compliance_agent(state: WealthDeskState) -> dict:
    print("[WealthDesk] Supervisor → Compliance Agent")
    result = _compliance_agent.invoke({
        "customer_message":  state["customer_message"],
        "response":          state["response"],
        "history":           state.get("history", []),
        "query_type":        state.get("query_type", ""),
        "retrieved_docs":    state.get("retrieved_docs", []),
        "specialist":        state.get("specialist", ""),
        "compliance_status": "",
    })
    return {
        "response":          result["response"],
        "compliance_status": result.get("compliance_status", "PASS"),
    }
 
 
def escalate(state: WealthDeskState) -> dict:
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": ESCALATE_RESPONSE},
    ]
    return {"response": ESCALATE_RESPONSE, "history": new_history, "specialist": "escalated"}
 
 
def decline(state: WealthDeskState) -> dict:
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": DECLINE_RESPONSE},
    ]
    return {"response": DECLINE_RESPONSE, "history": new_history, "specialist": "declined"}
 
 
def route_supervisor(state: WealthDeskState) -> str:
    qt = state.get("query_type", "RATES")
    if qt == "POLICY":
        return "call_documents_agent"
    if qt == "COMPLEX":
        return "escalate"
    if qt == "OUT_OF_SCOPE":
        return "decline"
    return "call_rates_agent"
 