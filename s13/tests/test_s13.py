"""
s13/tests/test_s13.py
---------------------
Tests for Session 13: Streamlit UI + Human-in-the-Loop (WealthDesk).

Run with:
    pytest s13/tests/ -v

All tests are pure Python — no Streamlit context needed.
The app helper functions are imported directly from app.py.

Test groups:
  TestBuildInputState    -- build_input_state() returns correct graph input dict
  TestGetThreadConfig    -- get_thread_config() returns correct LangGraph config
  TestComplianceBadge    -- compliance_badge() returns correct display text
  TestNeedsHumanReview   -- needs_human_review() detects REVISED status (HITL trigger)
  TestFormatRouteLabel   -- format_route_label() formats routing info correctly
  TestAgentGraph         -- build_graph() compiles; S12 nodes are present
"""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

SOLUTION_DIR = Path(__file__).parent.parent / "solution"
for _k in list(sys.modules):
    if _k == "wealthdesk" or _k.startswith("wealthdesk."):
        sys.modules.pop(_k)
sys.path.insert(0, str(SOLUTION_DIR))

# Load app module without triggering Streamlit
_spec = importlib.util.spec_from_file_location("app", SOLUTION_DIR / "app.py")
_app  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_app)

build_input_state  = _app.build_input_state
get_thread_config  = _app.get_thread_config
compliance_badge   = _app.compliance_badge
needs_human_review = _app.needs_human_review
format_route_label = _app.format_route_label

from wealthdesk.agent import build_graph  # noqa: E402
import wealthdesk.nodes as _nodes         # noqa: E402


# ---------------------------------------------------------------------------
# TestBuildInputState
# ---------------------------------------------------------------------------

class TestBuildInputState:
    def test_has_customer_message(self):
        state = build_input_state("What is the home loan rate?")
        assert state["customer_message"] == "What is the home loan rate?"

    def test_has_empty_response(self):
        assert build_input_state("test")["response"] == ""

    def test_has_empty_specialist(self):
        assert build_input_state("test")["specialist"] == ""

    def test_has_empty_retrieved_docs(self):
        assert build_input_state("test")["retrieved_docs"] == []

    def test_has_empty_compliance_status(self):
        assert build_input_state("test")["compliance_status"] == ""

    def test_all_required_keys_present(self):
        state    = build_input_state("test")
        required = {"customer_message", "response", "specialist", "retrieved_docs", "compliance_status"}
        assert required.issubset(set(state.keys()))

    def test_different_messages_differ(self):
        assert build_input_state("FD")["customer_message"] != build_input_state("home loan")["customer_message"]

    def test_empty_message_accepted(self):
        assert build_input_state("")["customer_message"] == ""


# ---------------------------------------------------------------------------
# TestGetThreadConfig
# ---------------------------------------------------------------------------

class TestGetThreadConfig:
    def test_returns_dict(self):
        assert isinstance(get_thread_config("abc"), dict)

    def test_has_configurable_key(self):
        assert "configurable" in get_thread_config("abc")

    def test_configurable_has_thread_id(self):
        assert get_thread_config("my-thread")["configurable"]["thread_id"] == "my-thread"

    def test_different_ids_produce_different_configs(self):
        c1 = get_thread_config("t1")
        c2 = get_thread_config("t2")
        assert c1["configurable"]["thread_id"] != c2["configurable"]["thread_id"]


# ---------------------------------------------------------------------------
# TestComplianceBadge
# ---------------------------------------------------------------------------

class TestComplianceBadge:
    def test_pass_returns_checkmark(self):
        assert "✅" in compliance_badge("PASS")

    def test_revised_returns_warning(self):
        assert "⚠️" in compliance_badge("REVISED")

    def test_fail_returns_cross(self):
        assert "❌" in compliance_badge("FAIL: guaranteed returns")

    def test_empty_status_returns_empty_string(self):
        assert compliance_badge("") == ""

    def test_unknown_status_returns_empty_string(self):
        assert compliance_badge("UNKNOWN") == ""

    def test_pass_text(self):
        assert "Compliant" in compliance_badge("PASS")

    def test_revised_text(self):
        assert "Revised" in compliance_badge("REVISED")


# ---------------------------------------------------------------------------
# TestNeedsHumanReview
# ---------------------------------------------------------------------------

class TestNeedsHumanReview:
    def test_revised_returns_true(self):
        assert needs_human_review({"compliance_status": "REVISED"}) is True

    def test_pass_returns_false(self):
        assert needs_human_review({"compliance_status": "PASS"}) is False

    def test_fail_returns_false(self):
        assert needs_human_review({"compliance_status": "FAIL: guaranteed returns"}) is False

    def test_empty_returns_false(self):
        assert needs_human_review({"compliance_status": ""}) is False

    def test_missing_key_returns_false(self):
        assert needs_human_review({}) is False

    def test_escalated_specialist_is_not_hitl(self):
        assert needs_human_review({"compliance_status": "PASS", "specialist": "escalated"}) is False


# ---------------------------------------------------------------------------
# TestFormatRouteLabel
# ---------------------------------------------------------------------------

class TestFormatRouteLabel:
    def test_includes_specialist(self):
        result = {"specialist": "rates_agent", "compliance_status": "PASS"}
        assert "rates_agent" in format_route_label(result)

    def test_includes_badge_for_pass(self):
        result = {"specialist": "rates_agent", "compliance_status": "PASS"}
        assert "✅" in format_route_label(result)

    def test_no_badge_for_empty_status(self):
        result = {"specialist": "escalated", "compliance_status": ""}
        label  = format_route_label(result)
        assert "✅" not in label and "⚠️" not in label and "❌" not in label

    def test_dash_for_missing_keys(self):
        assert "—" in format_route_label({})

    def test_revised_badge_shown(self):
        result = {"specialist": "documents_agent", "compliance_status": "REVISED"}
        assert "⚠️" in format_route_label(result)

    def test_documents_agent_in_label(self):
        result = {"specialist": "documents_agent", "compliance_status": "PASS"}
        assert "documents_agent" in format_route_label(result)


# ---------------------------------------------------------------------------
# TestAgentGraph
# ---------------------------------------------------------------------------

class TestAgentGraph:
    def test_build_graph_compiles(self):
        from langgraph.checkpoint.memory import MemorySaver
        assert build_graph(checkpointer=MemorySaver()) is not None

    def test_graph_has_classify_node(self):
        from langgraph.checkpoint.memory import MemorySaver
        assert "classify" in build_graph(checkpointer=MemorySaver()).get_graph().nodes

    def test_graph_has_compliance_node(self):
        from langgraph.checkpoint.memory import MemorySaver
        assert "call_compliance_agent" in build_graph(checkpointer=MemorySaver()).get_graph().nodes

    def test_graph_invocable_returns_response(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf, \
             patch.object(_nodes, "_documents_agent") as mock_da, \
             patch.object(_nodes, "_compliance_agent") as mock_ca:
            mock_clf.invoke.return_value = MagicMock(content="DOCUMENTS")
            mock_da.invoke.return_value  = {
                "response":       "You need a passport and salary slips.",
                "history":        [],
                "retrieved_docs": [],
                "specialist":     "documents_agent",
            }
            mock_ca.invoke.return_value  = {
                "response":          "You need a passport and salary slips.",
                "compliance_status": "PASS",
            }
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                build_input_state("What documents do I need for a home loan?"),
                config=get_thread_config("test-s13-graph"),
            )
        assert "response" in result
        assert result["compliance_status"] == "PASS"

    def test_revised_result_triggers_hitl(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf, \
             patch.object(_nodes, "_rates_agent") as mock_ra, \
             patch.object(_nodes, "_compliance_agent") as mock_ca:
            mock_clf.invoke.return_value = MagicMock(content="RATES")
            mock_ra.invoke.return_value  = {
                "response":       "BNB guarantees the best FD rates in the market!",
                "history":        [],
                "retrieved_docs": [],
                "specialist":     "rates_agent",
            }
            mock_ca.invoke.return_value  = {
                "response":          "BNB offers competitive FD rates.",
                "compliance_status": "REVISED",
            }
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                build_input_state("What are the FD rates?"),
                config=get_thread_config("test-s13-hitl"),
            )
        assert needs_human_review(result) is True
