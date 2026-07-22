"""
wealthdesk/nodes.py
-------------------
Graph nodes and routing for WealthDesk.

Session 5: respond() gains tool-calling -- it calls query_rates and/or
query_branch when the LLM requests them, then makes a second LLM call
to synthesise the final answer.
"""
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_huggingface import HuggingFaceEmbeddings

from .config import (
    CLASSIFY_SYSTEM, DECLINE_RESPONSE, ESCALATE_RESPONSE,
    EMBED_MODEL, RETRIEVAL_K, SYSTEM_PROMPT, VECTORSTORE_DIR,
)
from .state import WealthDeskState
from .tools import classifier_llm, llm, llm_with_tools, _run_tool

vectorstore = None


def _init_vectorstore() -> None:
    """Load ChromaDB + embeddings. No-op if already initialised or mocked."""
    global vectorstore
    if vectorstore is not None:
        return
    try:
        embeddings  = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vectorstore = Chroma(
            persist_directory=str(VECTORSTORE_DIR),
            embedding_function=embeddings,
        )
        count = vectorstore._collection.count()
        if count == 0:
            print("[WealthDesk] Vectorstore opened but is EMPTY (0 chunks).")
            print("  Run 'python data/ingest.py' from the project root to populate it.")
        else:
            print(f"[WealthDesk] Vectorstore ready — {count} chunks loaded.")
    except Exception as e:
        print(f"[WealthDesk] Could not load vectorstore: {e}")
        print("  Run 'python data/ingest.py' to create it.")


def classify(state: WealthDeskState) -> dict:
    """Unchanged from Session 4."""
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
    return {"query_type": query_type, "retrieved_docs": []}


def retrieve_docs(state: WealthDeskState) -> dict:
    """Unchanged from Session 4."""
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
    """Handle SIMPLE queries with RAG context and database tool calls."""
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

        # -----------------------------------------------------------------------
        # TODO 4 of 4 -- Execute tool calls and make a second LLM call
        # -----------------------------------------------------------------------
        # If the LLM requested tool calls (result.tool_calls is non-empty):
        #
        # Step A: Append the assistant message to messages:
        #           messages.append(result)
        #
        # Step B: For each tc in result.tool_calls:
        #           tool_output = _run_tool(tc["name"], tc["args"])
        #           print(f"[WealthDesk] Tool: {tc['name']}({tc['args']}) -> {str(tool_output)[:80]}")
        #           messages.append(
        #               ToolMessage(content=str(tool_output), tool_call_id=tc["id"])
        #           )
        #
        # Step C: Make a second call (plain llm, no tools needed):
        #           result = llm.invoke(messages)
        #
        # After the if-block: response_text = result.content
        # -----------------------------------------------------------------------
        # TODO: add the if result.tool_calls block here

        # We execute tools manually here because WealthDeskState uses a custom
        # shape (history: list[dict]). In projects using LangGraph's standard
        # MessagesState you'd use the built-in ToolNode instead:
        #
        #   from langgraph.prebuilt import ToolNode
        #   builder.add_node("tools", ToolNode([query_rates, query_branch]))
        #   builder.add_edge("respond", "tools")
        #   builder.add_edge("tools", "respond")
        #
        # Both patterns are valid. ToolNode is less code; this gives more control.

        response_text = result.content

    except Exception as e:
        print(f"[WealthDesk] LLM error: {e}")
        response_text = "I am temporarily unavailable. Please try again in a moment."

    new_history = history + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": response_text},
    ]
    return {"response": response_text, "history": new_history}


def escalate(state: WealthDeskState) -> dict:
    """Unchanged from Session 4."""
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": ESCALATE_RESPONSE},
    ]
    return {"response": ESCALATE_RESPONSE, "history": new_history}


def decline(state: WealthDeskState) -> dict:
    """Unchanged from Session 4."""
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": DECLINE_RESPONSE},
    ]
    return {"response": DECLINE_RESPONSE, "history": new_history}


def route_query(state: WealthDeskState) -> str:
    """Unchanged from Session 4."""
    qt = state.get("query_type", "SIMPLE")
    if qt == "COMPLEX":
        return "escalate"
    if qt == "OUT_OF_SCOPE":
        return "decline"
    return "retrieve_docs"
