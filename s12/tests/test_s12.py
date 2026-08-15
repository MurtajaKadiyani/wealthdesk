"""
s12/tests/test_s12.py
---------------------
Tests for Session 12: Multi-Agent Architecture Part 2 (US-08 + US-09).

Run with:
    pytest s12/tests/ -v

All tests mock the LLM, vectorstore, and database -- no real Groq or I/O required.

Test groups:
  TestState                -- WealthDeskState has compliance_status + specialist
  TestComplianceHelpers    -- _load_valid_rates, _extract_rates, _check_compliance_logic
  TestCheckSebiNode        -- check_sebi node: PASS/FAIL scenarios, state updates
  TestReviseResponseNode   -- revise_response node: LLM call, REVISED status, fallback
  TestRouteCompliance      -- route_compliance: PASS→END, FAIL→revise
  TestComplianceAgentFactory -- factory, nodes, subgraph invocable with PASS and FAIL
  TestCallComplianceAgentNode -- supervisor node wraps _compliance_agent
  TestSupervisorGraph      -- full graph: specialists → compliance; escalate/decline bypass
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

from wealthdesk.config import SAFE_COMPLIANCE_RESPONSE, SEBI_BANNED_PHRASES  # noqa: E402
from wealthdesk.state import WealthDeskState  # noqa: E402
import wealthdesk.nodes as _nodes             # noqa: E402
from wealthdesk.nodes import (  # noqa: E402
    _check_compliance_logic, _extract_rates, _load_valid_rates,
    call_compliance_agent, call_documents_agent, call_rates_agent,
    check_sebi, classify,
    create_compliance_agent, create_documents_agent, create_rates_agent,
    decline, escalate, revise_response, route_compliance, route_supervisor,
)
from wealthdesk.agent import build_graph  # noqa: E402
from langgraph.graph import END  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(**kwargs) -> WealthDeskState:
    defaults = dict(
        customer_message="test",
        response="",
        history=[],
        query_type="RATES",
        retrieved_docs=[],
        specialist="",
        compliance_status="",
    )
    defaults.update(kwargs)
    return defaults  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# TestState
# ---------------------------------------------------------------------------

class TestState:
    def test_state_has_compliance_status(self):
        state = _state()
        assert "compliance_status" in state

    def test_state_has_specialist(self):
        state = _state()
        assert "specialist" in state

    def test_compliance_status_default_empty(self):
        state = _state()
        assert state["compliance_status"] == ""

    def test_compliance_status_accepts_pass(self):
        state = _state(compliance_status="PASS")
        assert state["compliance_status"] == "PASS"

    def test_compliance_status_accepts_fail(self):
        state = _state(compliance_status="FAIL: banned phrase: 'risk-free'")
        assert state["compliance_status"].startswith("FAIL")

    def test_compliance_status_accepts_revised(self):
        state = _state(compliance_status="REVISED")
        assert state["compliance_status"] == "REVISED"


# ---------------------------------------------------------------------------
# TestComplianceHelpers
# ---------------------------------------------------------------------------

class TestComplianceHelpers:
    def test_sebi_banned_phrases_not_empty(self):
        assert len(SEBI_BANNED_PHRASES) > 0

    def test_banned_phrases_include_guaranteed_returns(self):
        assert "guaranteed returns" in SEBI_BANNED_PHRASES

    def test_extract_rates_finds_decimal_rate(self):
        rates = _extract_rates("BNB home loan rate is 8.5% p.a.")
        assert 8.5 in rates

    def test_extract_rates_finds_integer_rate(self):
        rates = _extract_rates("FD rate is 7% p.a.")
        assert 7.0 in rates

    def test_extract_rates_ignores_non_pa_percentages(self):
        rates = _extract_rates("Up to 80% loan-to-value ratio.")
        assert len(rates) == 0

    def test_extract_rates_finds_multiple_rates(self):
        rates = _extract_rates("Home loan 8.5% p.a. and FD 6.8% p.a.")
        assert len(rates) == 2

    def test_load_valid_rates_returns_set(self):
        with patch.object(_nodes, "sqlite3") as mock_sqlite3:
            mock_conn = MagicMock()
            mock_sqlite3.connect.return_value = mock_conn
            mock_conn.execute.return_value.fetchall.side_effect = [
                [(8.5,), (9.5,)],
                [(6.0, 0.5), (7.0, 0.5)],
            ]
            result = _load_valid_rates()
        assert isinstance(result, set)

    def test_load_valid_rates_returns_empty_on_error(self):
        with patch.object(_nodes, "sqlite3") as mock_sqlite3:
            mock_sqlite3.connect.side_effect = Exception("DB unavailable")
            result = _load_valid_rates()
        assert result == set()

    def test_check_compliance_logic_passes_clean_text(self):
        passed, reason = _check_compliance_logic("BNB home loan rate is 8.5% p.a.")
        assert isinstance(passed, bool)
        assert isinstance(reason, str)

    def test_check_compliance_logic_fails_on_banned_phrase(self):
        passed, reason = _check_compliance_logic(
            "BNB FDs offer guaranteed returns of 7% p.a."
        )
        assert passed is False
        assert "guaranteed returns" in reason

    def test_check_compliance_logic_fails_on_risk_free(self):
        passed, reason = _check_compliance_logic("This is a risk-free investment.")
        assert passed is False
        assert "risk-free" in reason

    def test_check_compliance_logic_returns_pass_for_safe_text(self):
        with patch.object(_nodes, "_load_valid_rates", return_value=set()):
            passed, reason = _check_compliance_logic(
                "Please visit your BNB branch for details.\n\nWealthDesk | Bharat National Bank"
            )
        assert passed is True
        assert reason == "PASS"

    def test_check_compliance_logic_fails_hallucinated_rate(self):
        with patch.object(_nodes, "_load_valid_rates", return_value={8.5, 9.5, 6.0}):
            passed, reason = _check_compliance_logic(
                "BNB home loan rate is 15.0% p.a."
            )
        assert passed is False
        assert "hallucinated rate" in reason
        assert "15.0" in reason


# ---------------------------------------------------------------------------
# TestCheckSebiNode
# ---------------------------------------------------------------------------

class TestCheckSebiNode:
    def test_returns_pass_status_for_compliant_response(self):
        with patch.object(_nodes, "_check_compliance_logic", return_value=(True, "PASS")):
            result = check_sebi(_state(response="BNB home loan details.\nWealthDesk | BNB"))
        assert result["compliance_status"] == "PASS"

    def test_returns_fail_status_for_banned_phrase(self):
        with patch.object(_nodes, "_check_compliance_logic",
                          return_value=(False, "banned phrase: 'risk-free'")):
            result = check_sebi(_state(response="This is a risk-free product."))
        assert result["compliance_status"].startswith("FAIL")
        assert "risk-free" in result["compliance_status"]

    def test_does_not_modify_response(self):
        with patch.object(_nodes, "_check_compliance_logic", return_value=(True, "PASS")):
            result = check_sebi(_state(response="Original response."))
        assert "response" not in result

    def test_calls_check_compliance_logic_with_draft(self):
        draft = "BNB rate is 8.5% p.a."
        with patch.object(_nodes, "_check_compliance_logic", return_value=(True, "PASS")) as mock_check:
            check_sebi(_state(response=draft))
        mock_check.assert_called_once_with(draft)

    def test_fail_status_includes_reason(self):
        with patch.object(_nodes, "_check_compliance_logic",
                          return_value=(False, "hallucinated rate: 99.0% p.a. not in database")):
            result = check_sebi(_state(response="Rate is 99.0% p.a."))
        assert "99.0" in result["compliance_status"]


# ---------------------------------------------------------------------------
# TestReviseResponseNode
# ---------------------------------------------------------------------------

class TestReviseResponseNode:
    def test_returns_revised_response(self):
        with patch.object(_nodes, "llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Compliant rewrite.")
            result = revise_response(_state(
                response="Guaranteed returns await you.",
                compliance_status="FAIL: banned phrase: 'guaranteed returns'",
            ))
        assert result["response"] == "Compliant rewrite."

    def test_sets_compliance_status_to_revised(self):
        with patch.object(_nodes, "llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Safe response.")
            result = revise_response(_state(
                response="risk-free investment",
                compliance_status="FAIL: banned phrase: 'risk-free'",
            ))
        assert result["compliance_status"] == "REVISED"

    def test_falls_back_to_safe_response_on_llm_error(self):
        with patch.object(_nodes, "llm") as mock_llm:
            mock_llm.invoke.side_effect = Exception("LLM error")
            result = revise_response(_state(
                response="bad response",
                compliance_status="FAIL: banned phrase: 'no risk'",
            ))
        assert result["response"] == SAFE_COMPLIANCE_RESPONSE
        assert result["compliance_status"] == "REVISED"

    def test_calls_llm_with_original_draft_in_prompt(self):
        draft = "This is a risk-free product."
        with patch.object(_nodes, "llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Fixed.")
            revise_response(_state(
                response=draft,
                compliance_status="FAIL: banned phrase: 'risk-free'",
            ))
        call_args = mock_llm.invoke.call_args
        prompt_text = call_args[0][0][0].content
        assert draft in prompt_text


# ---------------------------------------------------------------------------
# TestRouteCompliance
# ---------------------------------------------------------------------------

class TestRouteCompliance:
    def test_pass_routes_to_end(self):
        result = route_compliance(_state(compliance_status="PASS"))
        assert result == END

    def test_fail_routes_to_revise(self):
        result = route_compliance(_state(compliance_status="FAIL: banned phrase"))
        assert result == "revise"

    def test_empty_status_routes_to_end(self):
        result = route_compliance(_state(compliance_status=""))
        assert result == END

    def test_revised_routes_to_end(self):
        result = route_compliance(_state(compliance_status="REVISED"))
        assert result == END


# ---------------------------------------------------------------------------
# TestComplianceAgentFactory
# ---------------------------------------------------------------------------

class TestComplianceAgentFactory:
    def test_factory_returns_compiled_graph(self):
        agent = create_compliance_agent()
        assert agent is not None

    def test_factory_returns_different_instances(self):
        a1 = create_compliance_agent()
        a2 = create_compliance_agent()
        assert a1 is not a2

    def test_agent_has_check_sebi_node(self):
        agent = create_compliance_agent()
        assert "check_sebi" in agent.get_graph().nodes

    def test_agent_has_revise_node(self):
        agent = create_compliance_agent()
        assert "revise" in agent.get_graph().nodes

    def test_agent_passes_compliant_response_unchanged(self):
        agent = create_compliance_agent()
        with patch.object(_nodes, "_check_compliance_logic", return_value=(True, "PASS")):
            result = agent.invoke(_state(response="Clean BNB response. WealthDesk | BNB"))
        assert result["compliance_status"] == "PASS"
        assert result["response"] == "Clean BNB response. WealthDesk | BNB"

    def test_agent_revises_non_compliant_response(self):
        agent = create_compliance_agent()
        with patch.object(_nodes, "_check_compliance_logic",
                          return_value=(False, "banned phrase: 'risk-free'")), \
             patch.object(_nodes, "llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Compliant version.")
            result = agent.invoke(_state(
                response="This is risk-free.",
                compliance_status="",
            ))
        assert result["compliance_status"] == "REVISED"
        assert result["response"] == "Compliant version."


# ---------------------------------------------------------------------------
# TestCallComplianceAgentNode
# ---------------------------------------------------------------------------

class TestCallComplianceAgentNode:
    def test_sets_compliance_status_from_agent(self):
        with patch.object(_nodes, "_compliance_agent") as mock_agent:
            mock_agent.invoke.return_value = {
                "response": "Clean response.", "compliance_status": "PASS"
            }
            result = call_compliance_agent(_state(response="Draft."))
        assert result["compliance_status"] == "PASS"

    def test_returns_response_from_agent(self):
        with patch.object(_nodes, "_compliance_agent") as mock_agent:
            mock_agent.invoke.return_value = {
                "response": "Revised response.", "compliance_status": "REVISED"
            }
            result = call_compliance_agent(_state(response="Bad draft."))
        assert result["response"] == "Revised response."

    def test_passes_current_response_to_agent(self):
        draft = "Original draft response."
        with patch.object(_nodes, "_compliance_agent") as mock_agent:
            mock_agent.invoke.return_value = {"response": draft, "compliance_status": "PASS"}
            call_compliance_agent(_state(response=draft))
        invoke_args = mock_agent.invoke.call_args[0][0]
        assert invoke_args["response"] == draft

    def test_defaults_compliance_status_if_agent_omits_it(self):
        with patch.object(_nodes, "_compliance_agent") as mock_agent:
            mock_agent.invoke.return_value = {"response": "Response without status."}
            result = call_compliance_agent(_state(response="Draft."))
        assert result["compliance_status"] == "PASS"


# ---------------------------------------------------------------------------
# TestSupervisorGraph
# ---------------------------------------------------------------------------

class TestSupervisorGraph:
    def test_build_graph_returns_compiled_graph(self):
        from langgraph.checkpoint.memory import MemorySaver
        graph = build_graph(checkpointer=MemorySaver())
        assert graph is not None

    def test_graph_has_classify_node(self):
        from langgraph.checkpoint.memory import MemorySaver
        graph = build_graph(checkpointer=MemorySaver())
        assert "classify" in graph.get_graph().nodes

    def test_graph_has_call_compliance_agent_node(self):
        from langgraph.checkpoint.memory import MemorySaver
        graph = build_graph(checkpointer=MemorySaver())
        assert "call_compliance_agent" in graph.get_graph().nodes

    def test_documents_agent_routes_to_compliance(self):
        from langgraph.checkpoint.memory import MemorySaver
        graph = build_graph(checkpointer=MemorySaver())
        edges = graph.builder.edges
        assert ("call_documents_agent", "call_compliance_agent") in edges

    def test_rates_agent_routes_to_compliance(self):
        from langgraph.checkpoint.memory import MemorySaver
        graph = build_graph(checkpointer=MemorySaver())
        edges = graph.builder.edges
        assert ("call_rates_agent", "call_compliance_agent") in edges

    def test_escalate_bypasses_compliance(self):
        from langgraph.checkpoint.memory import MemorySaver
        graph = build_graph(checkpointer=MemorySaver())
        edges = graph.builder.edges
        assert ("escalate", "call_compliance_agent") not in edges

    def test_decline_bypasses_compliance(self):
        from langgraph.checkpoint.memory import MemorySaver
        graph = build_graph(checkpointer=MemorySaver())
        edges = graph.builder.edges
        assert ("decline", "call_compliance_agent") not in edges

    def test_policy_query_ends_with_compliance_status(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf, \
             patch.object(_nodes, "_documents_agent") as mock_da, \
             patch.object(_nodes, "_compliance_agent") as mock_ca:
            mock_clf.invoke.return_value = MagicMock(content="POLICY")
            mock_da.invoke.return_value = {
                "response": "Policy info.", "history": [], "retrieved_docs": []
            }
            mock_ca.invoke.return_value = {
                "response": "Policy info.", "compliance_status": "PASS"
            }
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                _state(customer_message="What documents do I need?"),
                config={"configurable": {"thread_id": "test-policy-compliance"}},
            )
        assert result["compliance_status"] == "PASS"

    def test_rates_query_ends_with_compliance_status(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf, \
             patch.object(_nodes, "_rates_agent") as mock_ra, \
             patch.object(_nodes, "_compliance_agent") as mock_ca:
            mock_clf.invoke.return_value = MagicMock(content="RATES")
            mock_ra.invoke.return_value = {
                "response": "Home loan: 8.5% p.a.", "history": []
            }
            mock_ca.invoke.return_value = {
                "response": "Home loan: 8.5% p.a.", "compliance_status": "PASS"
            }
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                _state(customer_message="What is the home loan rate?"),
                config={"configurable": {"thread_id": "test-rates-compliance"}},
            )
        assert result["compliance_status"] == "PASS"

    def test_compliance_revised_status_propagates(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf, \
             patch.object(_nodes, "_rates_agent") as mock_ra, \
             patch.object(_nodes, "_compliance_agent") as mock_ca:
            mock_clf.invoke.return_value = MagicMock(content="RATES")
            mock_ra.invoke.return_value = {
                "response": "This is risk-free at 8.5% p.a.", "history": []
            }
            mock_ca.invoke.return_value = {
                "response": "BNB offers 8.5% p.a. subject to terms.",
                "compliance_status": "REVISED",
            }
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                _state(customer_message="What is the home loan rate?"),
                config={"configurable": {"thread_id": "test-revised"}},
            )
        assert result["compliance_status"] == "REVISED"

    def test_escalate_has_no_compliance_status(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf:
            mock_clf.invoke.return_value = MagicMock(content="COMPLEX")
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                _state(customer_message="Should I invest in FD or loan?"),
                config={"configurable": {"thread_id": "test-escalate-no-compliance"}},
            )
        assert result.get("compliance_status", "") == ""

    def test_decline_has_no_compliance_status(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf:
            mock_clf.invoke.return_value = MagicMock(content="OUT_OF_SCOPE")
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                _state(customer_message="What is the weather?"),
                config={"configurable": {"thread_id": "test-decline-no-compliance"}},
            )
        assert result.get("compliance_status", "") == ""

    def test_escalate_response_contains_relationship_manager(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf:
            mock_clf.invoke.return_value = MagicMock(content="COMPLEX")
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                _state(customer_message="Compare all BNB products."),
                config={"configurable": {"thread_id": "test-escalate-rm"}},
            )
        assert "Relationship Manager" in result["response"]

    def test_specialist_field_set_for_documents_agent(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf, \
             patch.object(_nodes, "_documents_agent") as mock_da, \
             patch.object(_nodes, "_compliance_agent") as mock_ca:
            mock_clf.invoke.return_value = MagicMock(content="POLICY")
            mock_da.invoke.return_value = {
                "response": "Policy.", "history": [], "retrieved_docs": []
            }
            mock_ca.invoke.return_value = {"response": "Policy.", "compliance_status": "PASS"}
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                _state(customer_message="Eligibility rules?"),
                config={"configurable": {"thread_id": "test-specialist-docs"}},
            )
        assert result["specialist"] == "documents_agent"

    def test_specialist_field_set_for_rates_agent(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf, \
             patch.object(_nodes, "_rates_agent") as mock_ra, \
             patch.object(_nodes, "_compliance_agent") as mock_ca:
            mock_clf.invoke.return_value = MagicMock(content="RATES")
            mock_ra.invoke.return_value = {"response": "Rate info.", "history": []}
            mock_ca.invoke.return_value = {"response": "Rate info.", "compliance_status": "PASS"}
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                _state(customer_message="Home loan rate?"),
                config={"configurable": {"thread_id": "test-specialist-rates"}},
            )
        assert result["specialist"] == "rates_agent"
