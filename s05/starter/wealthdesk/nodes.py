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
    CLASSIFY_SYSTEM_PROMPT, DECLINE_RESPONSE, ESCALATE_RESPONSE,
    EMBED_MODEL, RETRIEVAL_K, RETRIEVAL_SCORE_THRESHOLD, SYSTEM_PROMPT, VECTORSTORE_DIR,
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
    """Updated in Session 5: now classifies into SIMPLE / COMPLEX / OUT_OF_SCOPE."""
    messages = [
        SystemMessage(content=CLASSIFY_SYSTEM_PROMPT),
        HumanMessage(content=state["customer_message"]),
    ]
    try:
        result     = classifier_llm.invoke(messages)
        query_type = result.content.strip().upper() # type: ignore
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
        results   = vectorstore.similarity_search_with_relevance_scores(
            state["customer_message"], k=RETRIEVAL_K
        )
        retrieved = []
        for doc, score in results:
            if score >= RETRIEVAL_SCORE_THRESHOLD:
                retrieved.append(
                    f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
                )
            else:
                print(
                    f"[WealthDesk] Chunk skipped (score {score:.2f} < {RETRIEVAL_SCORE_THRESHOLD}): "
                    f"{doc.metadata.get('source', 'unknown')}"
                )
        print(f"[WealthDesk] Retrieval: {len(results)} candidates, {len(retrieved)} passed threshold ({RETRIEVAL_SCORE_THRESHOLD}).")
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
            messages.append(HumanMessage(content=turn["content"])) # type: ignore
        else:
            messages.append(AIMessage(content=turn["content"])) # type: ignore
    messages.append(HumanMessage(content=state["customer_message"])) # type: ignore
 
    try:
        result = llm_with_tools.invoke(messages)
        if result.invalid_tool_calls:
            for itc in result.invalid_tool_calls:
                print(f"[WealthDesk] Invalid tool call ignored: {itc.get('name', 'unknown')} — {itc.get('error', 'parse error')}")
        max_tool_rounds = 5
        tool_rounds     = 0
          # all return different raw formats; LangChain maps them to the same four keys):
        #   tc["name"] : str  -- tool name,  e.g. "query_rates"
        #   tc["args"] : dict -- arguments,  e.g. {"product_type": "loan"}
        #   tc["id"]   : str  -- unique ID,  e.g. "call_abc123" (used to match ToolMessage back)
        #   tc["type"] : str  -- always "tool_call" in this list
        while result.tool_calls and tool_rounds < max_tool_rounds:
            messages.append(result) # type: ignore
            for tc in result.tool_calls:
                tool_output = _run_tool(tc["name"], tc["args"])
                print(f"[WealthDesk] Tool: {tc['name']}({tc['args']}) -> {str(tool_output)[:80]}")
                # tool_call_id links this result back to the specific tool call in the AIMessage above.
                # When the LLM calls multiple tools in one turn, the second LLM call uses these IDs
                # to know which result belongs to which call. Most providers reject ToolMessages
                # that have no matching tool_call_id in the message history.
                messages.append(ToolMessage(content=str(tool_output), tool_call_id=tc["id"])) # type: ignore
            tool_rounds += 1
            result = llm_with_tools.invoke(messages)
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
    """Updated in Session 5: COMPLEX routes to escalate; SIMPLE routes to retrieve_docs."""
    qt = state.get("query_type", "SIMPLE")
    if qt == "COMPLEX":
        return "escalate"
    if qt == "OUT_OF_SCOPE":
        return "decline"
    return "retrieve_docs"
