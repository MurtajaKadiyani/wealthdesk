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
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from .config import (
    SYSTEM_PROMPT,CLASSIFY_SYSTEM_PROMPT,ESCALATE_RESPONSE,DECLINE_RESPONSE, # type: ignore
    EMBED_MODEL,VECTORSTORE_DIR, RETRIEVAL_K,RETRIEVAL_SCORE_THRESHOLD
)
from .state import WealthDeskState
from .tools import llm,classifier_llm
 
vectorstore = None  # shared across calls; initialised once by _init_vectorstore()

BLOCKLIST = [
    "ignore all previous",
    "forget everything",
    "you are now",
    "disregard your system",
    "act as",
    "jailbreak",
]
 
def _init_vectorstore() -> None:
    global vectorstore
    if vectorstore is not None:  # already loaded — skip the 90 MB model reload
        return
    try:
        embeddings  = HuggingFaceEmbeddings(model_name=EMBED_MODEL)  # loads ~90 MB model from ~/.cache/huggingface/
        vectorstore = Chroma(
            persist_directory=str(VECTORSTORE_DIR),  # opens chroma.sqlite3 on disk — does NOT load all chunks into memory
            embedding_function=embeddings,            # same model used at ingest time — must match or retrieval breaks
        )
        count = vectorstore._collection.count()
        if count == 0:
            print(f"[WealthDesk] Vectorstore opened but is EMPTY (0 chunks).")
            print(f"  Run 'python data/ingest.py' from the wealthdesk/ root to populate it.")
        else:
            print(f"[WealthDesk] Vectorstore ready — {count} chunks loaded.")
    except Exception as e:
        print(f"[WealthDesk] Could not load vectorstore: {e}")
        print("  Run 'python data/ingest.py' to create it.")
 
 
 
def classify(state: WealthDeskState) -> dict:
    """Call the LLM and return the agent's reply."""
    
    valid_types = {"IN_SCOPE", "OUT_OF_SCOPE"}
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
       if query_type not in valid_types:
          query_type = "IN_SCOPE"
    except Exception as e:
        print(f"[WealthDesk] Classification error: {e}")
        query_type = "IN_SCOPE"
 
    return {"query_type": query_type, "retrieved_docs": []}
 
def retrieve_docs(state: WealthDeskState) -> dict:
    _init_vectorstore()
    if vectorstore is None:
        return {"retrieved_docs": []}
    try:
        # similarity_search_with_relevance_scores returns a list of (doc, score) tuples.
        #   doc.page_content : the raw chunk text (e.g. "PAN card and Aadhaar are required...")
        #   doc.metadata     : dict — e.g. {"source": "home_loan_guide.md"}
        #   score            : cosine similarity 0–1; higher = more relevant to the query
        results   = vectorstore.similarity_search_with_relevance_scores(
            state["customer_message"], k=RETRIEVAL_K
        )
        retrieved = []
        for doc, score in results:  # doc = LangChain Document object; score = float 0–1
            if score >= RETRIEVAL_SCORE_THRESHOLD:
                retrieved.append(
                    f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
                )
            else:
                print(
                    f"[WealthDesk] Chunk skipped (score {score:.2f} < {RETRIEVAL_SCORE_THRESHOLD}): "
                    f"{doc.metadata.get('source', 'unknown')}"
                )
    except Exception as e:
        print(f"[WealthDesk] Retrieval error: {e}")
        retrieved = []
    return {"retrieved_docs": retrieved}
 
 
def respond(state: WealthDeskState) -> dict:
    """Call the LLM and return the agent's reply."""
    history = state.get("history",[])
    retrieved = state.get("retrieved_docs", [])
   
    if not retrieved:
        new_history = history + [
            {"role": "user",      "content": state["customer_message"]},
            {"role": "assistant", "content": ESCALATE_RESPONSE},
        ]
        return {"response": ESCALATE_RESPONSE, "history": new_history}
 
    context_block  = "\n\n---\n\n".join(retrieved)
    system_content = (
        SYSTEM_PROMPT
        + "\n\nThe following sections from BNB's policy documents are relevant "
        "to the customer's question. Use this information in your answer:\n\n"
        + context_block
    )
    messages = [
        SystemMessage(content=system_content)
    ]
    for turn in history:
       if turn["role"] == "user":
          messages.append(HumanMessage(content=turn["content"])) # type: ignore
       else:
          messages.append(AIMessage(content=turn["content"])) # type: ignore
    messages.append(HumanMessage(content=state["customer_message"])) # type: ignore
    try:
      result = llm.invoke(messages)
      response_text = result.content
    except Exception as e:
        print(f"[WealthDesk] LLM error: {e}")
        return {"response": "I am temporarily unavailable. Please try again in a moment."}
 
    new_history = history + [{"role": "user", "content": state["customer_message"]}, {"role": "assistant", "content": response_text}]
    return {"response": response_text, "history": new_history}
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
   query_type = state.get("query_type","IN_SCOPE")
   if query_type == "OUT_OF_SCOPE":
      return "decline"
   return "retrieve_docs"