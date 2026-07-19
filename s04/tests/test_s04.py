"""
s04/tests/test_s04.py
---------------------
Unit tests for Session 4: ChromaDB RAG + LangSmith Tracing.

Run from the wealthdesk/ directory:
    pytest s04/tests/ -v

All tests run without a live Groq API key, ChromaDB, or HuggingFace model.
The vectorstore and LLMs are mocked throughout.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver

SOLUTION_DIR = Path(__file__).parent.parent / "solution"
for _k in list(sys.modules):
    if _k == "wealthdesk" or _k.startswith("wealthdesk."):
        sys.modules.pop(_k)
sys.path.insert(0, str(SOLUTION_DIR))

from wealthdesk.config import DECLINE_RESPONSE, ESCALATE_RESPONSE, RETRIEVAL_K, RETRIEVAL_SCORE_THRESHOLD, SYSTEM_PROMPT  # noqa: E402
from wealthdesk.state import WealthDeskState  # noqa: E402
import wealthdesk.nodes as _nodes  # noqa: E402
from wealthdesk.nodes import classify, decline, escalate, respond, retrieve_docs, route_query  # noqa: E402
from wealthdesk.agent import build_graph  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_doc(page_content: str, source: str = "home_loan_guide.md") -> MagicMock:
    doc = MagicMock()
    doc.page_content = page_content
    doc.metadata = {"source": source}
    return doc


def _mock_vectorstore_with(docs: list, score: float = 0.8) -> MagicMock:
    """Return a mock vectorstore whose similarity_search_with_relevance_scores
    yields (doc, score) pairs. Default score 0.8 is above RETRIEVAL_SCORE_THRESHOLD."""
    mock_vs = MagicMock()
    mock_vs.similarity_search_with_relevance_scores.return_value = [
        (doc, score) for doc in docs
    ]
    return mock_vs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def memory_checkpointer():
    return MemorySaver()


@pytest.fixture
def mock_llm_inscope():
    """Classifier returns IN_SCOPE; main LLM returns a factual answer."""
    with patch.object(_nodes, "llm") as mock_main, \
         patch.object(_nodes, "classifier_llm") as mock_clf:
        mock_clf.invoke.return_value = MagicMock(content="IN_SCOPE")
        mock_main.invoke.return_value = MagicMock(
            content="The BNB home loan requires salary slips and a PAN card. WealthDesk | BNB"
        )
        yield mock_main, mock_clf


@pytest.fixture
def mock_vectorstore():
    doc = _make_mock_doc(
        "For a home loan, BNB requires: last 3 months' salary slips, "
        "PAN card, Aadhaar card, Form 16, and 6 months' bank statements.",
        source="home_loan_guide.md",
    )
    mock_vs = _mock_vectorstore_with([doc])
    with patch.object(_nodes, "vectorstore", mock_vs), \
         patch.object(_nodes, "_init_vectorstore"):
        yield mock_vs


# ---------------------------------------------------------------------------
# State structure tests
# ---------------------------------------------------------------------------

class TestWealthDeskState:
    def test_state_has_customer_message(self):
        assert "customer_message" in WealthDeskState.__annotations__

    def test_state_has_response(self):
        assert "response" in WealthDeskState.__annotations__

    def test_state_has_history(self):
        assert "history" in WealthDeskState.__annotations__

    def test_state_has_query_type(self):
        assert "query_type" in WealthDeskState.__annotations__

    def test_state_has_retrieved_docs(self):
        assert "retrieved_docs" in WealthDeskState.__annotations__, (
            "WealthDeskState must have a 'retrieved_docs' field (added in Session 4). "
            "Add it after 'query_type' with type hint list[str]."
        )

    def test_retrieved_docs_is_list_type(self):
        annotation = WealthDeskState.__annotations__["retrieved_docs"]
        origin = getattr(annotation, "__origin__", annotation)
        assert origin is list

    def test_state_instantiable_with_all_fields(self):
        state: WealthDeskState = {
            "customer_message": "What documents do I need?",
            "response":         "",
            "history":          [],
            "query_type":       "IN_SCOPE",
            "retrieved_docs":   [],
        }
        assert state["retrieved_docs"] == []

    def test_state_retrieved_docs_accepts_strings(self):
        state: WealthDeskState = {
            "customer_message": "test",
            "response":         "",
            "history":          [],
            "query_type":       "IN_SCOPE",
            "retrieved_docs":   ["[home_loan_guide.md]\nSome policy text."],
        }
        assert len(state["retrieved_docs"]) == 1


# ---------------------------------------------------------------------------
# retrieve_docs() node tests
# ---------------------------------------------------------------------------

class TestRetrieveDocsNode:
    def _state(self, question: str = "What documents do I need for a home loan?") -> WealthDeskState:
        return {
            "customer_message": question,
            "response":         "",
            "history":          [],
            "query_type":       "IN_SCOPE",
            "retrieved_docs":   [],
        }

    def test_retrieve_docs_returns_dict(self, mock_vectorstore):
        result = retrieve_docs(self._state())
        assert isinstance(result, dict)

    def test_retrieve_docs_returns_retrieved_docs_key(self, mock_vectorstore):
        result = retrieve_docs(self._state())
        assert "retrieved_docs" in result

    def test_retrieve_docs_returns_list(self, mock_vectorstore):
        result = retrieve_docs(self._state())
        assert isinstance(result["retrieved_docs"], list)

    def test_retrieve_docs_calls_similarity_search(self, mock_vectorstore):
        retrieve_docs(self._state("What documents do I need?"))
        mock_vectorstore.similarity_search_with_relevance_scores.assert_called_once()

    def test_retrieve_docs_passes_question_to_search(self, mock_vectorstore):
        question = "What documents do I need for a home loan?"
        retrieve_docs(self._state(question))
        call_args = mock_vectorstore.similarity_search_with_relevance_scores.call_args
        assert call_args[0][0] == question or call_args[1].get("query") == question

    def test_retrieve_docs_passes_k_parameter(self, mock_vectorstore):
        retrieve_docs(self._state())
        call_args = mock_vectorstore.similarity_search_with_relevance_scores.call_args
        k_value   = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("k")
        assert k_value == RETRIEVAL_K

    def test_retrieve_docs_filters_low_score_chunks(self):
        doc = _make_mock_doc("Some content.", source="faq.md")
        # Score below threshold — should be filtered out.
        low_score_vs = _mock_vectorstore_with([doc], score=RETRIEVAL_SCORE_THRESHOLD - 0.1)
        with patch.object(_nodes, "vectorstore", low_score_vs), \
             patch.object(_nodes, "_init_vectorstore"):
            result = retrieve_docs(self._state("Which"))
        assert result["retrieved_docs"] == []

    def test_retrieve_docs_keeps_high_score_chunks(self):
        doc = _make_mock_doc("Some content.", source="faq.md")
        high_score_vs = _mock_vectorstore_with([doc], score=RETRIEVAL_SCORE_THRESHOLD + 0.1)
        with patch.object(_nodes, "vectorstore", high_score_vs), \
             patch.object(_nodes, "_init_vectorstore"):
            result = retrieve_docs(self._state("What is the FD rate?"))
        assert len(result["retrieved_docs"]) == 1

    def test_retrieve_docs_formats_source_in_output(self, mock_vectorstore):
        result = retrieve_docs(self._state())
        assert len(result["retrieved_docs"]) > 0
        first = result["retrieved_docs"][0]
        assert "home_loan_guide.md" in first

    def test_retrieve_docs_includes_page_content(self, mock_vectorstore):
        result = retrieve_docs(self._state())
        combined = " ".join(result["retrieved_docs"])
        assert "salary slips" in combined or "home loan" in combined.lower()

    def test_retrieve_docs_returns_empty_when_vectorstore_is_none(self):
        with patch.object(_nodes, "vectorstore", None), \
             patch.object(_nodes, "_init_vectorstore"):
            result = retrieve_docs(self._state())
        assert result == {"retrieved_docs": []}

    def test_retrieve_docs_returns_empty_on_exception(self):
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_relevance_scores.side_effect = Exception("ChromaDB error")
        with patch.object(_nodes, "vectorstore", mock_vs), \
             patch.object(_nodes, "_init_vectorstore"):
            result = retrieve_docs(self._state())
        assert result == {"retrieved_docs": []}

    def test_retrieve_docs_does_not_call_llm(self, mock_vectorstore):
        with patch.object(_nodes, "llm") as mock_llm, \
             patch.object(_nodes, "classifier_llm") as mock_clf:
            retrieve_docs(self._state())
            mock_llm.invoke.assert_not_called()
            mock_clf.invoke.assert_not_called()


# ---------------------------------------------------------------------------
# respond() node tests -- context injection
# ---------------------------------------------------------------------------

class TestRespondWithContext:
    def _state_with_docs(self, docs: list[str]) -> WealthDeskState:
        return {
            "customer_message": "What documents do I need for a home loan?",
            "response":         "",
            "history":          [],
            "query_type":       "IN_SCOPE",
            "retrieved_docs":   docs,
        }

    def test_respond_includes_retrieved_docs_in_system_message(self):
        chunk = "[home_loan_guide.md]\nRequires: salary slips, PAN, Aadhaar."
        state = self._state_with_docs([chunk])
        with patch.object(_nodes, "llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Docs needed: salary slips.")
            respond(state)
        call_args   = mock_llm.invoke.call_args
        messages    = call_args[0][0]
        system_text = messages[0].content
        assert "salary slips" in system_text or "home_loan_guide.md" in system_text

    def test_respond_without_docs_returns_escalate_response(self):
        # Option B: when no docs are retrieved, respond() escalates directly
        # without calling the LLM. This is faster and deterministic.
        state = self._state_with_docs([])
        with patch.object(_nodes, "llm") as mock_llm:
            result = respond(state)
        mock_llm.invoke.assert_not_called()
        assert result["response"] == ESCALATE_RESPONSE

    def test_respond_without_docs_updates_history(self):
        state = self._state_with_docs([])
        result = respond(state)
        assert len(result["history"]) == 2
        assert result["history"][-1]["content"] == ESCALATE_RESPONSE

    def test_respond_context_contains_policy_keyword(self):
        chunk = "[home_loan_guide.md]\nMinimum age for home loan applicant is 21 years."
        state = self._state_with_docs([chunk])
        with patch.object(_nodes, "llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Min age is 21.")
            respond(state)
        call_args   = mock_llm.invoke.call_args
        messages    = call_args[0][0]
        system_text = messages[0].content
        assert "21" in system_text

    def test_respond_with_multiple_docs(self):
        chunks = [
            "[home_loan_guide.md]\nSalary slips required.",
            "[bnb_policy.md]\nPAN card mandatory for all loans.",
        ]
        state = self._state_with_docs(chunks)
        with patch.object(_nodes, "llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Both docs needed.")
            respond(state)
        call_args   = mock_llm.invoke.call_args
        messages    = call_args[0][0]
        system_text = messages[0].content
        assert "salary slips" in system_text.lower() or "Salary" in system_text
        assert "PAN" in system_text

    def test_respond_updates_history_with_docs(self):
        chunk = "[faq.md]\nFD can be opened online."
        state = self._state_with_docs([chunk])
        with patch.object(_nodes, "llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Yes, FD can be opened online.")
            result = respond(state)
        assert len(result["history"]) == 2


# ---------------------------------------------------------------------------
# route_query() tests
# ---------------------------------------------------------------------------

class TestRouteQuery:
    def _state(self, query_type: str) -> dict:
        return {"customer_message": "test", "response": "",
                "history": [], "query_type": query_type, "retrieved_docs": []}

    def test_in_scope_routes_to_retrieve_docs(self):
        # Option B: IN_SCOPE always goes to retrieval first.
        assert route_query(self._state("IN_SCOPE")) == "retrieve_docs"

    def test_out_of_scope_routes_to_decline(self):
        assert route_query(self._state("OUT_OF_SCOPE")) == "decline"

    def test_default_routes_to_retrieve_docs(self):
        # Unknown query_type defaults to retrieve_docs (safe fallback).
        state = {"customer_message": "test", "response": "", "history": [], "retrieved_docs": []}
        assert route_query(state) == "retrieve_docs"

    def test_classify_resets_retrieved_docs(self):
        # Stale retrieved_docs from a previous turn must be cleared by classify().
        state = {
            "customer_message": "What is the FD rate?",
            "response":         "",
            "history":          [],
            "query_type":       "",
            "retrieved_docs":   ["[old_source.md]\nStale content from last turn."],
        }
        with patch.object(_nodes, "classifier_llm") as mock_clf:
            mock_clf.invoke.return_value = MagicMock(content="IN_SCOPE")
            result = classify(state)
        assert result["retrieved_docs"] == []


# ---------------------------------------------------------------------------
# Graph routing tests
# ---------------------------------------------------------------------------

class TestGraphRouting:
    def test_in_scope_path_calls_vectorstore(self, mock_vectorstore, mock_llm_inscope, memory_checkpointer):
        graph  = build_graph(checkpointer=memory_checkpointer)
        config = {"configurable": {"thread_id": "route-inscope-rag"}}
        graph.invoke(
            {"customer_message": "What documents do I need for a home loan?", "response": ""},
            config=config,
        )
        mock_vectorstore.similarity_search_with_relevance_scores.assert_called_once()

    def test_in_scope_path_sets_retrieved_docs_in_result(self, mock_vectorstore, mock_llm_inscope, memory_checkpointer):
        graph  = build_graph(checkpointer=memory_checkpointer)
        config = {"configurable": {"thread_id": "route-inscope-docs"}}
        result = graph.invoke(
            {"customer_message": "What documents do I need?", "response": ""},
            config=config,
        )
        assert "retrieved_docs" in result
        assert isinstance(result["retrieved_docs"], list)
        assert len(result["retrieved_docs"]) > 0

    def test_no_docs_returns_escalate_response(self, memory_checkpointer):
        # Option B: when retrieval returns nothing, respond() escalates.
        # This replaces the Option A test that COMPLEX skips the vectorstore.
        empty_vs = _mock_vectorstore_with([])
        with patch.object(_nodes, "vectorstore", empty_vs), \
             patch.object(_nodes, "_init_vectorstore"), \
             patch.object(_nodes, "classifier_llm") as mock_clf, \
             patch.object(_nodes, "llm") as mock_llm:
            mock_clf.invoke.return_value = MagicMock(content="IN_SCOPE")
            graph  = build_graph(checkpointer=memory_checkpointer)
            config = {"configurable": {"thread_id": "no-docs-escalate"}}
            result = graph.invoke(
                {"customer_message": "Which loan should I take?", "response": ""},
                config=config,
            )
        mock_llm.invoke.assert_not_called()
        assert result["response"] == ESCALATE_RESPONSE

    def test_out_of_scope_path_skips_vectorstore(self, memory_checkpointer):
        mock_vs = MagicMock()
        with patch.object(_nodes, "vectorstore", mock_vs), \
             patch.object(_nodes, "_init_vectorstore"), \
             patch.object(_nodes, "classifier_llm") as mock_clf, \
             patch.object(_nodes, "llm"):
            mock_clf.invoke.return_value = MagicMock(content="OUT_OF_SCOPE")
            graph  = build_graph(checkpointer=memory_checkpointer)
            config = {"configurable": {"thread_id": "route-oos-no-rag"}}
            graph.invoke(
                {"customer_message": "Write me a poem.", "response": ""},
                config=config,
            )
        mock_vs.similarity_search.assert_not_called()

    def test_all_paths_produce_non_empty_response(self, mock_vectorstore, memory_checkpointer):
        for query_type, question in [
            ("IN_SCOPE",     "What documents do I need?"),
            ("OUT_OF_SCOPE", "Write a poem."),
        ]:
            with patch.object(_nodes, "classifier_llm") as mock_clf, \
                 patch.object(_nodes, "llm") as mock_llm:
                mock_clf.invoke.return_value = MagicMock(content=query_type)
                mock_llm.invoke.return_value = MagicMock(content="Some answer.")
                graph  = build_graph(checkpointer=MemorySaver())
                config = {"configurable": {"thread_id": f"all-paths-{query_type}"}}
                result = graph.invoke(
                    {"customer_message": question, "response": ""},
                    config=config,
                )
            assert isinstance(result["response"], str)
            assert len(result["response"]) > 0

    def test_in_scope_no_docs_also_produces_response(self, memory_checkpointer):
        empty_vs = _mock_vectorstore_with([])
        with patch.object(_nodes, "vectorstore", empty_vs), \
             patch.object(_nodes, "_init_vectorstore"), \
             patch.object(_nodes, "classifier_llm") as mock_clf:
            mock_clf.invoke.return_value = MagicMock(content="IN_SCOPE")
            graph  = build_graph(checkpointer=memory_checkpointer)
            config = {"configurable": {"thread_id": "inscope-nodocs"}}
            result = graph.invoke(
                {"customer_message": "Which loan is best for me?", "response": ""},
                config=config,
            )
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0


# ---------------------------------------------------------------------------
# Memory tests
# ---------------------------------------------------------------------------

class TestMemoryWithRAG:
    def test_history_accumulates_across_in_scope_turns(self, mock_vectorstore, mock_llm_inscope, memory_checkpointer):
        graph     = build_graph(checkpointer=memory_checkpointer)
        thread_id = "mem-rag-inscope"
        config    = {"configurable": {"thread_id": thread_id}}

        graph.invoke(
            {"customer_message": "What documents do I need?", "response": ""},
            config=config,
        )
        result = graph.invoke(
            {"customer_message": "And for a personal loan?", "response": ""},
            config=config,
        )
        assert len(result["history"]) == 4

    def test_different_threads_isolated(self, mock_vectorstore, mock_llm_inscope, memory_checkpointer):
        graph = build_graph(checkpointer=memory_checkpointer)

        graph.invoke(
            {"customer_message": "What is the FD rate?", "response": ""},
            config={"configurable": {"thread_id": "rag-thread-X"}},
        )
        result = graph.invoke(
            {"customer_message": "What documents do I need?", "response": ""},
            config={"configurable": {"thread_id": "rag-thread-Y"}},
        )
        assert len(result["history"]) == 2
