"""
wealthdesk/nodes.py
-------------------
Graph nodes and routing for WealthDesk.

Session 9 adds the compliance filter:
  _load_valid_rates()  -- load valid BNB rates from SQLite
  _check_compliance()  -- phrase check + rate check; returns (bool, str)
  check_compliance()   -- LangGraph node that calls _check_compliance()
                          and substitutes SAFE_COMPLIANCE_RESPONSE on failure
"""
import re
import sqlite3

from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langsmith import traceable

from .config import (
    CLASSIFY_SYSTEM, DB_PATH, DECLINE_RESPONSE, EMBED_MODEL,
    ESCALATE_RESPONSE, RETRIEVAL_K, SAFE_COMPLIANCE_RESPONSE,
    SEBI_BANNED_PHRASES, SYSTEM_PROMPT, VECTORSTORE_DIR,
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
# TODO 1 of 4 -- Implement _load_valid_rates()
# ---------------------------------------------------------------------------
# Load all valid BNB interest rates from the SQLite database.
# Returns a set of floats so _check_compliance() can verify rate accuracy.
#
#   try:
#       conn      = sqlite3.connect(str(DB_PATH), check_same_thread=False)
#       loan_rows = conn.execute("SELECT interest_rate FROM loan_products").fetchall()
#       fd_rows   = conn.execute("SELECT interest_rate, senior_rate FROM fd_products").fetchall()
#       conn.close()
#       loan_rates = {row[0] for row in loan_rows}
#       fd_base    = {row[0] for row in fd_rows}
#       fd_senior  = {row[0] + row[1] for row in fd_rows}
#       return loan_rates | fd_base | fd_senior
#   except Exception:
#       return set()
# ---------------------------------------------------------------------------
def _load_valid_rates() -> set:
    # TODO: implement this function
    return set()


def _extract_rates(text: str) -> list:
    """Extract all 'X% p.a.' values from text as floats. Provided -- no changes needed."""
    matches = re.findall(r"(\d+\.?\d*)\s*%\s*p\.a\.", text, re.IGNORECASE)
    return [float(m) for m in matches]


# ---------------------------------------------------------------------------
# TODO 2 of 4 -- Implement _check_compliance()
# ---------------------------------------------------------------------------
# Checks the LLM draft response for:
#   a) SEBI-banned phrases (e.g. "guaranteed returns", "risk-free")
#   b) Hallucinated rates not present in the database
#
# Returns (True, "PASS") if both checks pass,
#         (False, "banned phrase: '...'") or (False, "hallucinated rate: X% p.a.")
#
#   @traceable(name="sebi_compliance_check")
#   def _check_compliance(draft: str) -> tuple:
#       lower = draft.lower()
#
#       for phrase in SEBI_BANNED_PHRASES:
#           if phrase in lower:
#               return False, f"banned phrase: '{phrase}'"
#
#       mentioned_rates = _extract_rates(draft)
#       if mentioned_rates:
#           valid_rates = _load_valid_rates()
#           if valid_rates:
#               for rate in mentioned_rates:
#                   if rate not in valid_rates:
#                       return False, f"hallucinated rate: {rate}% p.a. not in database"
#
#       return True, "PASS"
# ---------------------------------------------------------------------------
def _check_compliance(draft: str) -> tuple:
    # TODO: implement this function (add @traceable decorator too)
    return True, "PASS"


def classify(state: WealthDeskState) -> dict:
    """Unchanged from Session 8."""
    messages = [
        SystemMessage(content=CLASSIFY_SYSTEM),
        HumanMessage(content=state["customer_message"]),
    ]
    try:
        result     = classifier_llm.invoke(messages)
        query_type = result.content.strip().upper()
        if query_type not in {"SIMPLE", "COMPLEX", "OUT_OF_SCOPE"}:
            query_type = "SIMPLE"
    except Exception as e:
        print(f"[WealthDesk] Classification error: {e}")
        query_type = "SIMPLE"
    return {"query_type": query_type}


def retrieve_docs(state: WealthDeskState) -> dict:
    """Unchanged from Session 8."""
    _init_vectorstore()
    if vectorstore is None:
        return {"retrieved_docs": []}
    try:
        docs      = vectorstore.similarity_search(state["customer_message"], k=RETRIEVAL_K)
        retrieved = [
            f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in docs
        ]
    except Exception as e:
        print(f"[WealthDesk] Retrieval error: {e}")
        retrieved = []
    return {"retrieved_docs": retrieved}


def respond(state: WealthDeskState) -> dict:
    """Unchanged from Session 8."""
    history   = state.get("history", [])
    retrieved = state.get("retrieved_docs", [])

    if retrieved:
        context_block  = "\n\n---\n\n".join(retrieved)
        system_content = (
            SYSTEM_PROMPT
            + "\n\nThe following sections from BNB's policy documents are relevant "
              "to the customer's question. Use this information in your answer:\n\n"
            + context_block
        )
    else:
        system_content = SYSTEM_PROMPT

    messages = [SystemMessage(content=system_content)]
    for turn in history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=state["customer_message"]))

    try:
        result = llm_with_tools.invoke(messages)

        # Manual tool execution — see s05/nodes.py for the ToolNode alternative.
        if result.tool_calls:
            messages.append(result)
            for tc in result.tool_calls:
                tool_output = _run_tool(tc["name"], tc["args"])
                print(
                    f"[WealthDesk] MCP tool: {tc['name']}({tc['args']}) "
                    f"-> {str(tool_output)[:80]}"
                )
                messages.append(ToolMessage(content=str(tool_output), tool_call_id=tc["id"]))
            result = llm.invoke(messages)

        response_text = result.content

    except Exception as e:
        print(f"[WealthDesk] LLM error: {e}")
        response_text = "I am temporarily unavailable. Please try again in a moment."

    new_history = history + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": response_text},
    ]
    return {"response": response_text, "history": new_history}


# ---------------------------------------------------------------------------
# TODO 3 of 4 -- Implement check_compliance() node
# ---------------------------------------------------------------------------
# This LangGraph node reads state["response"], calls _check_compliance(),
# and updates state depending on the result.
#
#   def check_compliance(state: WealthDeskState) -> dict:
#       draft          = state["response"]
#       passed, reason = _check_compliance(draft)
#
#       if not passed:
#           print(f"[WealthDesk] Compliance FAIL: {reason}")
#           return {
#               "response":          SAFE_COMPLIANCE_RESPONSE,
#               "compliance_status": f"FAIL: {reason}",
#           }
#
#       print("[WealthDesk] Compliance PASS")
#       return {"compliance_status": "PASS"}
# ---------------------------------------------------------------------------
def check_compliance(state: WealthDeskState) -> dict:
    # TODO: implement this node
    return {"compliance_status": "TODO: not implemented"}


def escalate(state: WealthDeskState) -> dict:
    """Unchanged from Session 8."""
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": ESCALATE_RESPONSE},
    ]
    return {"response": ESCALATE_RESPONSE, "history": new_history}


def decline(state: WealthDeskState) -> dict:
    """Unchanged from Session 8."""
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": DECLINE_RESPONSE},
    ]
    return {"response": DECLINE_RESPONSE, "history": new_history}


def route_query(state: WealthDeskState) -> str:
    """Unchanged from Session 8."""
    qt = state.get("query_type", "SIMPLE")
    if qt == "COMPLEX":
        return "escalate"
    if qt == "OUT_OF_SCOPE":
        return "decline"
    return "retrieve_docs"
