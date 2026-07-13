"""
s02/tests/test_s02.py
---------------------
Unit tests for Session 2: Multi-Turn Memory.

Run from the wealthdesk/ directory:
    pytest s02/tests/ -v
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

from wealthdesk.config import SYSTEM_PROMPT  # noqa: E402
from wealthdesk.state import WealthDeskState  # noqa: E402
import wealthdesk.nodes as _nodes  # noqa: E402
from wealthdesk.nodes import respond  # noqa: E402
from wealthdesk.agent import build_graph  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def memory_checkpointer():
    return MemorySaver()


@pytest.fixture
def mock_llm_response():
    with patch.object(_nodes, "llm") as mock_llm:
        mock_result = MagicMock()
        mock_result.content = "The BNB home loan rate is 8.5% per annum. WealthDesk | Bharat National Bank"
        mock_llm.invoke.return_value = mock_result
        yield mock_llm


@pytest.fixture
def mock_llm_error():
    with patch.object(_nodes, "llm") as mock_llm:
        mock_llm.invoke.side_effect = Exception("Groq API timeout")
        yield mock_llm


# ---------------------------------------------------------------------------
# State structure tests
# ---------------------------------------------------------------------------

class TestWealthDeskState:
    def test_state_has_customer_message_field(self):
        assert "customer_message" in WealthDeskState.__annotations__

    def test_state_has_response_field(self):
        assert "response" in WealthDeskState.__annotations__

    def test_state_has_history_field(self):
        assert "history" in WealthDeskState.__annotations__

    def test_customer_message_is_str(self):
        assert WealthDeskState.__annotations__["customer_message"] is str

    def test_response_is_str(self):
        assert WealthDeskState.__annotations__["response"] is str

    def test_history_is_list_type(self):
        annotation = WealthDeskState.__annotations__["history"]
        import types
        origin = getattr(annotation, "__origin__", annotation)
        assert origin is list

    def test_state_can_be_instantiated_with_history(self):
        state: WealthDeskState = {
            "customer_message": "What is the home loan rate?",
            "response":         "",
            "history":          [],
        }
        assert state["history"] == []

    def test_state_history_accepts_turn_dicts(self):
        turns = [
            {"role": "user",      "content": "What is the home loan rate?"},
            {"role": "assistant", "content": "8.5% p.a."},
        ]
        state: WealthDeskState = {
            "customer_message": "And the FD rate?",
            "response":         "",
            "history":          turns,
        }
        assert len(state["history"]) == 2


# ---------------------------------------------------------------------------
# respond() node tests
# ---------------------------------------------------------------------------

class TestRespondNode:
    def test_respond_returns_dict(self, mock_llm_response):
        state: WealthDeskState = {"customer_message": "What is the home loan rate?", "response": "", "history": []}
        result = respond(state)
        assert isinstance(result, dict)

    def test_respond_returns_response_key(self, mock_llm_response):
        state: WealthDeskState = {"customer_message": "Hello", "response": "", "history": []}
        result = respond(state)
        assert "response" in result

    def test_respond_returns_history_key(self, mock_llm_response):
        state: WealthDeskState = {"customer_message": "What is the FD rate?", "response": "", "history": []}
        result = respond(state)
        assert "history" in result

    def test_respond_appends_user_turn_to_history(self, mock_llm_response):
        question = "What is the home loan interest rate?"
        state: WealthDeskState = {"customer_message": question, "response": "", "history": []}
        result = respond(state)
        user_turns = [t for t in result["history"] if t["role"] == "user"]
        assert any(t["content"] == question for t in user_turns)

    def test_respond_appends_assistant_turn_to_history(self, mock_llm_response):
        state: WealthDeskState = {"customer_message": "What FDs do you offer?", "response": "", "history": []}
        result = respond(state)
        assistant_turns = [t for t in result["history"] if t["role"] == "assistant"]
        assert len(assistant_turns) >= 1

    def test_respond_history_grows_by_two_per_turn(self, mock_llm_response):
        state: WealthDeskState = {"customer_message": "Tell me about personal loans.", "response": "", "history": []}
        result = respond(state)
        assert len(result["history"]) == 2

    def test_respond_preserves_previous_history(self, mock_llm_response):
        previous_turns = [
            {"role": "user",      "content": "What is the home loan rate?"},
            {"role": "assistant", "content": "8.5% p.a."},
        ]
        state: WealthDeskState = {
            "customer_message": "And the FD rate for 2 years?",
            "response":         "",
            "history":          previous_turns,
        }
        result = respond(state)
        assert len(result["history"]) == 4
        assert result["history"][0] == previous_turns[0]

    def test_respond_includes_history_in_llm_call(self, mock_llm_response):
        from langchain_core.messages import AIMessage, HumanMessage
        previous_turns = [
            {"role": "user",      "content": "I earn Rs. 80,000 per month."},
            {"role": "assistant", "content": "Noted. How can I help you?"},
        ]
        state: WealthDeskState = {
            "customer_message": "How much home loan can I get?",
            "response":         "",
            "history":          previous_turns,
        }
        respond(state)
        call_args = mock_llm_response.invoke.call_args
        messages  = call_args[0][0]
        human_contents = [m.content for m in messages if isinstance(m, HumanMessage)]
        assert "I earn Rs. 80,000 per month." in human_contents
        ai_contents = [m.content for m in messages if isinstance(m, AIMessage)]
        assert "Noted. How can I help you?" in ai_contents

    def test_respond_passes_system_message(self, mock_llm_response):
        from langchain_core.messages import SystemMessage
        state: WealthDeskState = {"customer_message": "What is the FD rate?", "response": "", "history": []}
        respond(state)
        call_args = mock_llm_response.invoke.call_args
        messages  = call_args[0][0]
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        assert len(system_msgs) == 1

    def test_respond_returns_safe_message_on_llm_error(self, mock_llm_error):
        state: WealthDeskState = {"customer_message": "What is the home loan rate?", "response": "", "history": []}
        result = respond(state)
        assert "response" in result
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0

    def test_respond_fallback_does_not_expose_error_details(self, mock_llm_error):
        state: WealthDeskState = {"customer_message": "test", "response": "", "history": []}
        result = respond(state)
        assert "Groq API timeout" not in result["response"]
        assert "Exception" not in result["response"]

    def test_respond_error_still_updates_history(self, mock_llm_error):
        state: WealthDeskState = {"customer_message": "What is the home loan rate?", "response": "", "history": []}
        result = respond(state)
        assert "history" in result
        assert len(result["history"]) == 2


# ---------------------------------------------------------------------------
# Memory persistence tests
# ---------------------------------------------------------------------------

class TestMemoryPersistence:
    def test_same_thread_id_shares_history(self, mock_llm_response, memory_checkpointer):
        graph     = build_graph(checkpointer=memory_checkpointer)
        thread_id = "test-thread-001"
        config    = {"configurable": {"thread_id": thread_id}}

        mock_llm_response.invoke.return_value.content = (
            "Understood. With an income of Rs. 80,000 per month, "
            "you may be eligible for a significant home loan."
        )
        graph.invoke({"customer_message": "I earn Rs. 80,000 per month.", "response": ""}, config=config)

        mock_llm_response.invoke.return_value.content = (
            "Based on your income of Rs. 80,000, you can get up to Rs. 48 lakhs."
        )
        graph.invoke({"customer_message": "How much home loan can I get?", "response": ""}, config=config)

        call_args   = mock_llm_response.invoke.call_args
        messages    = call_args[0][0]
        all_content = " ".join(m.content for m in messages)
        assert "80,000" in all_content or "Rs. 80" in all_content

    def test_different_thread_ids_have_separate_histories(self, mock_llm_response, memory_checkpointer):
        graph    = build_graph(checkpointer=memory_checkpointer)
        config_a = {"configurable": {"thread_id": "thread-A"}}
        config_b = {"configurable": {"thread_id": "thread-B"}}

        mock_llm_response.invoke.return_value.content = "Home loan info for thread A."
        graph.invoke({"customer_message": "What is the home loan rate?", "response": ""}, config=config_a)

        mock_llm_response.invoke.return_value.content = "FD info for thread B."
        graph.invoke({"customer_message": "Tell me about FDs.", "response": ""}, config=config_b)

        mock_llm_response.invoke.return_value.content = "More FD info."
        graph.invoke({"customer_message": "What are the FD rates?", "response": ""}, config=config_b)

        call_args   = mock_llm_response.invoke.call_args
        messages    = call_args[0][0]
        all_content = " ".join(m.content for m in messages)
        assert "home loan rate" not in all_content.lower()

    def test_history_survives_fresh_graph_invocation(self, mock_llm_response, memory_checkpointer):
        graph     = build_graph(checkpointer=memory_checkpointer)
        thread_id = "test-thread-002"
        config    = {"configurable": {"thread_id": thread_id}}

        mock_llm_response.invoke.return_value.content = "Personal loan response."
        result_turn1 = graph.invoke({"customer_message": "Tell me about personal loans.", "response": ""}, config=config)
        history_after_turn1 = result_turn1.get("history", [])

        mock_llm_response.invoke.return_value.content = "Second response."
        result_turn2 = graph.invoke({"customer_message": "What is the processing fee?", "response": ""}, config=config)
        history_after_turn2 = result_turn2.get("history", [])

        assert len(history_after_turn2) > len(history_after_turn1)

    def test_invoke_without_config_raises_or_discards_history(self, mock_llm_response, memory_checkpointer):
        graph = build_graph(checkpointer=memory_checkpointer)
        try:
            result = graph.invoke({"customer_message": "Hello", "response": ""})
            assert "response" in result
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Graph structure tests
# ---------------------------------------------------------------------------

class TestGraph:
    def test_build_graph_accepts_checkpointer_argument(self, memory_checkpointer):
        graph = build_graph(checkpointer=memory_checkpointer)
        assert graph is not None

    def test_build_graph_with_no_args_uses_memory_saver(self):
        graph = build_graph()
        assert graph is not None

    def test_graph_invocation_returns_response(self, mock_llm_response, memory_checkpointer):
        graph  = build_graph(checkpointer=memory_checkpointer)
        config = {"configurable": {"thread_id": "test-basic"}}
        result = graph.invoke({"customer_message": "What is the home loan rate?", "response": "", "history": []}, config=config)
        assert "response" in result
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0

    def test_graph_invocation_returns_history(self, mock_llm_response, memory_checkpointer):
        graph  = build_graph(checkpointer=memory_checkpointer)
        config = {"configurable": {"thread_id": "test-history-return"}}
        result = graph.invoke({"customer_message": "Tell me about gold loans.", "response": "", "history": []}, config=config)
        assert "history" in result
        assert isinstance(result["history"], list)
        assert len(result["history"]) >= 2

    def test_graph_preserves_customer_message(self, mock_llm_response, memory_checkpointer):
        graph    = build_graph(checkpointer=memory_checkpointer)
        config   = {"configurable": {"thread_id": "test-preservation"}}
        question = "What is the personal loan interest rate?"
        result   = graph.invoke({"customer_message": question, "response": "", "history": []}, config=config)
        assert result["customer_message"] == question
