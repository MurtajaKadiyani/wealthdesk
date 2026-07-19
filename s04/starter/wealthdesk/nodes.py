"""
wealthdesk/nodes.py
-------------------
Graph nodes and routing for WealthDesk.

Session 4 adds ChromaDB retrieval so SIMPLE queries get relevant
policy context before the LLM generates a response.
"""
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings

from .config import (
    CLASSIFY_SYSTEM, DECLINE_RESPONSE, ESCALATE_RESPONSE,
    EMBED_MODEL, RETRIEVAL_K, SYSTEM_PROMPT, VECTORSTORE_DIR,
)
from .state import WealthDeskState
from .tools import classifier_llm, llm

vectorstore = None  # shared across calls; initialised once by _init_vectorstore()


def _init_vectorstore() -> None:
    """Load ChromaDB + embeddings. No-op if already initialised or mocked."""
    global vectorstore
    if vectorstore is not None:  # already loaded — skip the 90 MB model reload
        return
    try:
        embeddings  = HuggingFaceEmbeddings(model_name=EMBED_MODEL)  # loads ~90 MB model from ~/.cache/huggingface/
        vectorstore = Chroma(
            persist_directory=str(VECTORSTORE_DIR),  # opens chroma.sqlite3 on disk — does NOT load all chunks into memory
            embedding_function=embeddings,            # same model used at ingest time — must match or retrieval breaks
        )
    except Exception as e:
        print(f"[WealthDesk] Could not load vectorstore: {e}")
        print("  Run 'python data/ingest.py' to create it.")


def classify(state: WealthDeskState) -> dict:
    """Classify the customer question. Provided -- no changes needed."""
    messages = [
        SystemMessage(content=CLASSIFY_SYSTEM),
        HumanMessage(content=state["customer_message"]),
    ]
    try:
        result     = classifier_llm.invoke(messages)
        query_type = result.content.strip().upper()
        if query_type not in {"IN_SCOPE", "OUT_OF_SCOPE"}:
            query_type = "IN_SCOPE"
    except Exception as e:
        print(f"[WealthDesk] Classification error: {e}")
        query_type = "IN_SCOPE"
    return {"query_type": query_type, "retrieved_docs": []}


def retrieve_docs(state: WealthDeskState) -> dict:
    """Query ChromaDB for policy chunks relevant to the customer's question.

    vectorstore.similarity_search() returns LangChain Document objects.
    Each Document has .page_content (the text) and .metadata (dict with 'source').
    """
    # -----------------------------------------------------------------------
    # TODO 2 of 4 -- Implement retrieve_docs()
    # -----------------------------------------------------------------------
    # 1. Call _init_vectorstore() to ensure the vectorstore is loaded.
    #
    # 2. If vectorstore is None (ingest.py not yet run), return {"retrieved_docs": []}.
    #
    # 3. Inside a try/except block:
    #      docs = vectorstore.similarity_search(state["customer_message"], k=RETRIEVAL_K)
    #      retrieved = [
    #          f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
    #          for doc in docs
    #      ]
    #    On exception: print the error, set retrieved = []
    #
    # 4. Return {"retrieved_docs": retrieved}
    # -----------------------------------------------------------------------
    # TODO: implement this node
    return {"retrieved_docs": []}


def respond(state: WealthDeskState) -> dict:
    """Handle SIMPLE queries, enriched with retrieved document context."""
    history   = state.get("history", [])
    retrieved = state.get("retrieved_docs", [])

    # -----------------------------------------------------------------------
    # TODO 3 of 4 -- Build system_content from retrieved docs
    # -----------------------------------------------------------------------
    # If retrieved is non-empty, prepend the chunks to the system prompt:
    #
    #   context_block  = "\n\n---\n\n".join(retrieved)
    #   system_content = (
    #       SYSTEM_PROMPT
    #       + "\n\nThe following sections from BNB's policy documents are relevant "
    #       "to the customer's question. Use this information in your answer:\n\n"
    #       + context_block
    #   )
    #
    # Otherwise:
    #   system_content = SYSTEM_PROMPT
    #
    # Then replace SYSTEM_PROMPT with system_content in the line below.
    # -----------------------------------------------------------------------
    # TODO: replace SYSTEM_PROMPT with system_content (built from retrieved)
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for turn in history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=state["customer_message"]))

    try:
        result        = llm.invoke(messages)
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
    """Handle COMPLEX queries. Provided -- no changes needed."""
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": ESCALATE_RESPONSE},
    ]
    return {"response": ESCALATE_RESPONSE, "history": new_history}


def decline(state: WealthDeskState) -> dict:
    """Handle OUT_OF_SCOPE queries. Provided -- no changes needed."""
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": DECLINE_RESPONSE},
    ]
    return {"response": DECLINE_RESPONSE, "history": new_history}


def route_query(state: WealthDeskState) -> str:
    """Route after classify(). Session 4: SIMPLE now routes to retrieve_docs."""
    qt = state.get("query_type", "SIMPLE")
    if qt == "COMPLEX":
        return "escalate"
    if qt == "OUT_OF_SCOPE":
        return "decline"
    return "respond"  # TODO 4: change "respond" to "retrieve_docs"
