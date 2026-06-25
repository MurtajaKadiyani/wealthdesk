"""
s01/tests/test_s01.py
---------------------
Unit tests for Session 1: Basic Conversational Agent.

Run from the wealthdesk/ directory:
    pytest s01/tests/ -v

All tests run without a live Groq API key.
The LLM is mocked so tests are fast, deterministic, and safe to run in CI.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SOLUTION_DIR = Path(__file__).parent.parent / "solution"
for _k in list(sys.modules):
    if _k == "wealthdesk" or _k.startswith("wealthdesk."):
        sys.modules.pop(_k)
sys.path.insert(0, str(SOLUTION_DIR))

from wealthdesk.config import MAX_TOKENS, MODEL_NAME, SYSTEM_PROMPT  # noqa: E402
from wealthdesk.state import WealthDeskState  # noqa: E402
import wealthdesk.nodes as _nodes  # noqa: E402
from wealthdesk.nodes import respond  # noqa: E402
from wealthdesk.agent import build_graph  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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

    def test_customer_message_is_str_type(self):
        assert WealthDeskState.__annotations__["customer_message"] is str

    def test_response_is_str_type(self):
        assert WealthDeskState.__annotations__["response"] is str

    def test_state_can_be_instantiated(self):
        state: WealthDeskState = {
            "customer_message": "What is the home loan rate?",
            "response": "",
        }
        assert state["customer_message"] == "What is the home loan rate?"
        assert state["response"] == ""


# ---------------------------------------------------------------------------
# System prompt tests
# ---------------------------------------------------------------------------

class TestSystemPrompt:
    def test_prompt_identifies_wealthdesk(self):
        assert "WealthDesk" in SYSTEM_PROMPT

    def test_prompt_identifies_bnb(self):
        assert "Bharat National Bank" in SYSTEM_PROMPT or "BNB" in SYSTEM_PROMPT

    def test_prompt_includes_out_of_scope_rule(self):
        lower = SYSTEM_PROMPT.lower()
        has_scope_rule = (
            "decline" in lower
            or "out-of-scope" in lower
            or "only discuss bnb" in lower
            or "only help with bnb" in lower
        )
        assert has_scope_rule

    def test_prompt_includes_home_loan_rate(self):
        assert "8.5" in SYSTEM_PROMPT

    def test_prompt_includes_fd_rate(self):
        assert "6.8" in SYSTEM_PROMPT or "7.1" in SYSTEM_PROMPT

    def test_model_name_is_not_empty(self):
        assert len(MODEL_NAME) > 0

    def test_max_tokens_is_reasonable(self):
        assert 100 < MAX_TOKENS <= 500


# ---------------------------------------------------------------------------
# respond() node tests
# ---------------------------------------------------------------------------

class TestRespondNode:
    def test_respond_returns_dict(self, mock_llm_response):
        state: WealthDeskState = {"customer_message": "What is the home loan rate?", "response": ""}
        result = respond(state)
        assert isinstance(result, dict)

    def test_respond_returns_response_key(self, mock_llm_response):
        state: WealthDeskState = {"customer_message": "Hello", "response": ""}
        result = respond(state)
        assert "response" in result

    def test_respond_content_is_string(self, mock_llm_response):
        state: WealthDeskState = {"customer_message": "What FDs do you offer?", "response": ""}
        result = respond(state)
        assert isinstance(result["response"], str)

    def test_respond_content_is_non_empty(self, mock_llm_response):
        state: WealthDeskState = {"customer_message": "Tell me about personal loans.", "response": ""}
        result = respond(state)
        assert len(result["response"]) > 0

    def test_respond_calls_llm_once(self, mock_llm_response):
        state: WealthDeskState = {"customer_message": "What is the FD rate?", "response": ""}
        respond(state)
        mock_llm_response.invoke.assert_called_once()

    def test_respond_passes_system_message(self, mock_llm_response):
        from langchain_core.messages import SystemMessage
        state: WealthDeskState = {"customer_message": "What is the FD rate?", "response": ""}
        respond(state)
        call_args = mock_llm_response.invoke.call_args
        messages = call_args[0][0]
        system_messages = [m for m in messages if isinstance(m, SystemMessage)]
        assert len(system_messages) == 1

    def test_respond_passes_human_message_with_customer_text(self, mock_llm_response):
        from langchain_core.messages import HumanMessage
        customer_question = "How much can I borrow for a home loan?"
        state: WealthDeskState = {"customer_message": customer_question, "response": ""}
        respond(state)
        call_args = mock_llm_response.invoke.call_args
        messages = call_args[0][0]
        human_messages = [m for m in messages if isinstance(m, HumanMessage)]
        assert len(human_messages) == 1
        assert human_messages[0].content == customer_question

    def test_respond_returns_safe_message_on_llm_error(self, mock_llm_error):
        state: WealthDeskState = {"customer_message": "What is the home loan rate?", "response": ""}
        result = respond(state)
        assert "response" in result
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0

    def test_respond_fallback_does_not_expose_error_details(self, mock_llm_error):
        state: WealthDeskState = {"customer_message": "test", "response": ""}
        result = respond(state)
        assert "Groq API timeout" not in result["response"]
        assert "Exception" not in result["response"]


# ---------------------------------------------------------------------------
# Graph structure tests
# ---------------------------------------------------------------------------

class TestGraph:
    def test_build_graph_returns_compiled_graph(self):
        graph = build_graph()
        assert graph is not None

    def test_graph_can_invoke_with_valid_state(self, mock_llm_response):
        graph = build_graph()
        state: WealthDeskState = {"customer_message": "What is the home loan rate?", "response": ""}
        result = graph.invoke(state)
        assert "response" in result

    def test_graph_invocation_returns_string_response(self, mock_llm_response):
        graph = build_graph()
        state: WealthDeskState = {"customer_message": "Tell me about gold loans.", "response": ""}
        result = graph.invoke(state)
        assert isinstance(result["response"], str)

    def test_graph_preserves_customer_message(self, mock_llm_response):
        graph = build_graph()
        question = "What is the personal loan interest rate?"
        state: WealthDeskState = {"customer_message": question, "response": ""}
        result = graph.invoke(state)
        assert result["customer_message"] == question
