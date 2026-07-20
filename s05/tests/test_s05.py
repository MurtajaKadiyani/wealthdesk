"""
s05/tests/test_s05.py
---------------------
Tests for Session 5: SQLite tool calls.

Run with:
    pytest s05/tests/ -v

Test groups:
  TestWealthDeskState    -- state TypedDict has all five fields (unchanged from S04)
  TestQueryRatesTool     -- query_rates() SQL correctness, filtering, output format
  TestQueryBranchTool    -- query_branch() SQL correctness, city filter, parameterisation
  TestToolSQLSafety      -- SQL injection protection and parameterised-query enforcement
  TestToolsBinding       -- llm_with_tools exists; tools are @tool decorated; prompt updated
  TestRunToolDispatch    -- _run_tool dispatches correctly; handles unknown names
  TestRespondWithTools   -- respond() calls llm_with_tools; executes tool calls; calls llm again
  TestGraphRouting       -- SIMPLE goes through retrieve_docs -> respond; COMPLEX/OOS skip tools
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

import wealthdesk  # noqa: E402
import wealthdesk.nodes as _nodes  # noqa: E402
import wealthdesk.tools as _tools  # noqa: E402
from wealthdesk.config import SYSTEM_PROMPT  # noqa: E402
from wealthdesk.state import WealthDeskState  # noqa: E402
from wealthdesk.tools import _run_tool, query_branch, query_rates  # noqa: E402
from wealthdesk.nodes import respond  # noqa: E402
from wealthdesk.agent import build_graph  # noqa: E402


# ---------------------------------------------------------------------------
# TestWealthDeskState
# ---------------------------------------------------------------------------

class TestWealthDeskState:
    def test_state_has_customer_message_field(self):
        assert "customer_message" in WealthDeskState.__annotations__

    def test_state_has_response_field(self):
        assert "response" in WealthDeskState.__annotations__

    def test_state_has_history_field(self):
        assert "history" in WealthDeskState.__annotations__

    def test_state_has_query_type_field(self):
        assert "query_type" in WealthDeskState.__annotations__

    def test_state_has_retrieved_docs_field(self):
        assert "retrieved_docs" in WealthDeskState.__annotations__

    def test_state_has_exactly_five_fields(self):
        assert len(WealthDeskState.__annotations__) == 5


# ---------------------------------------------------------------------------
# TestQueryRatesTool
# ---------------------------------------------------------------------------

class TestQueryRatesTool:
    def test_query_rates_loan_returns_home_loan(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_rates.invoke({"product_type": "loan"})
        assert "Home Loan" in result

    def test_query_rates_loan_includes_rate(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_rates.invoke({"product_type": "loan"})
        assert "8.5" in result

    def test_query_rates_loan_includes_tenure(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_rates.invoke({"product_type": "loan"})
        assert "5" in result and "30" in result

    def test_query_rates_fd_returns_fd_data(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_rates.invoke({"product_type": "fd"})
        assert "FD" in result
        assert "6.8" in result or "7.1" in result

    def test_query_rates_fd_shows_senior_rate(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_rates.invoke({"product_type": "fd"})
        assert "7.3" in result

    def test_query_rates_all_contains_loans_and_fds(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_rates.invoke({"product_type": "all"})
        assert "Home Loan" in result
        assert "FD" in result

    def test_query_rates_default_is_all(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_rates.invoke({})
        assert "Home Loan" in result
        assert "FD" in result

    def test_query_rates_returns_string(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        assert isinstance(query_rates.invoke({"product_type": "loan"}), str)

    def test_query_rates_loan_does_not_include_fd(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_rates.invoke({"product_type": "loan"})
        assert "FD" not in result

    def test_query_rates_fd_does_not_include_loan_names(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_rates.invoke({"product_type": "fd"})
        assert "Home Loan" not in result


# ---------------------------------------------------------------------------
# TestQueryBranchTool
# ---------------------------------------------------------------------------

class TestQueryBranchTool:
    def test_query_branch_all_returns_multiple(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_branch.invoke({"city": "all"})
        assert "Bengaluru" in result
        assert "Mumbai" in result

    def test_query_branch_city_filter_returns_matching(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_branch.invoke({"city": "Mumbai"})
        assert "Mumbai" in result
        assert "Andheri" in result

    def test_query_branch_city_filter_excludes_others(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_branch.invoke({"city": "Mumbai"})
        assert "Bengaluru" not in result

    def test_query_branch_includes_ifsc(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_branch.invoke({"city": "Bengaluru"})
        assert "BNBI" in result

    def test_query_branch_includes_phone(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_branch.invoke({"city": "Bengaluru"})
        assert "080" in result

    def test_query_branch_no_match_returns_message(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_branch.invoke({"city": "Atlantis"})
        assert "No BNB branches found" in result

    def test_query_branch_default_is_all(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_branch.invoke({})
        assert "Bengaluru" in result
        assert "Mumbai" in result

    def test_query_branch_returns_string(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        assert isinstance(query_branch.invoke({"city": "all"}), str)


# ---------------------------------------------------------------------------
# TestToolSQLSafety
# ---------------------------------------------------------------------------

class TestToolSQLSafety:
    def test_query_branch_sql_injection_safe(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_branch.invoke({"city": "'; DROP TABLE branches; --"})
        assert isinstance(result, str)
        normal = query_branch.invoke({"city": "Mumbai"})
        assert "Mumbai" in normal

    def test_query_branch_uses_question_mark_placeholder(self):
        import inspect
        source = inspect.getsource(query_branch.func)
        assert "LIKE ?" in source, (
            "query_branch must use a ? placeholder for the city parameter. "
            "Never interpolate user input directly into a SQL string."
        )


# ---------------------------------------------------------------------------
# TestToolsBinding
# ---------------------------------------------------------------------------

class TestToolsBinding:
    def test_llm_with_tools_exists(self):
        assert hasattr(_tools, "llm_with_tools"), (
            "llm_with_tools not found. Create it with llm.bind_tools([query_rates, query_branch])."
        )

    def test_query_rates_is_tool_decorated(self):
        assert hasattr(query_rates, "name"), (
            "query_rates does not appear to be decorated with @tool."
        )

    def test_query_branch_is_tool_decorated(self):
        assert hasattr(query_branch, "name"), (
            "query_branch does not appear to be decorated with @tool."
        )

    def test_query_rates_tool_name(self):
        assert query_rates.name == "query_rates"

    def test_query_branch_tool_name(self):
        assert query_branch.name == "query_branch"

    def test_system_prompt_has_no_hardcoded_rates(self):
        assert "8.5%" not in SYSTEM_PROMPT, (
            "Session 5 removes the hardcoded rate table from SYSTEM_PROMPT. "
            "Rates now come from query_rates(). Remove the 'Product reference' block."
        )

    def test_system_prompt_mentions_tools(self):
        assert "tool" in SYSTEM_PROMPT.lower() or "database" in SYSTEM_PROMPT.lower(), (
            "SYSTEM_PROMPT should instruct the LLM to use database tools for rates."
        )


# ---------------------------------------------------------------------------
# TestRunToolDispatch
# ---------------------------------------------------------------------------

class TestRunToolDispatch:
    def test_run_tool_dispatches_query_rates(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = _run_tool("query_rates", {"product_type": "loan"})
        assert "Home Loan" in result

    def test_run_tool_dispatches_query_branch(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = _run_tool("query_branch", {"city": "Mumbai"})
        assert "Mumbai" in result

    def test_run_tool_unknown_name_returns_error_string(self):
        result = _run_tool("nonexistent_tool", {})
        assert "Unknown tool" in result
        assert "nonexistent_tool" in result

    def test_run_tool_returns_string(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = _run_tool("query_rates", {"product_type": "fd"})
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# TestRespondWithTools
# ---------------------------------------------------------------------------

class TestRespondWithTools:
    def _make_tool_call_result(self, tool_name, args, call_id="call_abc123"):
        result = MagicMock()
        result.content = ""
        result.tool_calls = [{"id": call_id, "name": tool_name, "args": args}]
        return result

    def _make_text_result(self, content):
        result = MagicMock()
        result.content = content
        result.tool_calls = []
        return result

    def _base_state(self):
        return {
            "customer_message": "What is the home loan rate?",
            "response": "", "history": [], "query_type": "SIMPLE", "retrieved_docs": [],
        }

    def test_respond_calls_llm_with_tools_first(self):
        with patch.object(_nodes, "llm_with_tools") as mock_wt, \
             patch.object(_nodes, "llm") as mock_llm:
            mock_wt.invoke.return_value = self._make_text_result("The rate is 8.5%.")
            respond(self._base_state())
        mock_wt.invoke.assert_called_once()
        mock_llm.invoke.assert_not_called()

    def test_respond_no_tool_calls_returns_first_result(self):
        expected = "BNB offers home loans, personal loans, and more."
        state    = {**self._base_state(), "customer_message": "What are BNB's loan products?"}
        with patch.object(_nodes, "llm_with_tools") as mock_wt, \
             patch.object(_nodes, "llm"):
            mock_wt.invoke.return_value = self._make_text_result(expected)
            result = respond(state)
        assert result["response"] == expected

    def test_respond_makes_second_call_when_tool_requested(self):
        with patch.object(_nodes, "llm_with_tools") as mock_wt, \
             patch.object(_nodes, "llm") as mock_llm, \
             patch.object(_nodes, "_run_tool", return_value="Home Loan: 8.5% p.a."):
            mock_wt.invoke.return_value = self._make_tool_call_result(
                "query_rates", {"product_type": "loan"}
            )
            mock_llm.invoke.return_value = self._make_text_result(
                "The home loan rate is 8.5% p.a. WealthDesk | BNB"
            )
            respond(self._base_state())
        mock_llm.invoke.assert_called_once()

    def test_respond_executes_tool_via_run_tool(self):
        state = {**self._base_state(), "customer_message": "Where are your Mumbai branches?"}
        with patch.object(_nodes, "llm_with_tools") as mock_wt, \
             patch.object(_nodes, "llm") as mock_llm, \
             patch.object(_nodes, "_run_tool", return_value="BNB Andheri West...") as mock_rt:
            mock_wt.invoke.return_value = self._make_tool_call_result(
                "query_branch", {"city": "Mumbai"}
            )
            mock_llm.invoke.return_value = self._make_text_result("Here are the Mumbai branches.")
            respond(state)
        mock_rt.assert_called_once_with("query_branch", {"city": "Mumbai"})

    def test_respond_uses_second_call_content_as_response(self):
        final_answer = "The car loan rate is 9.5% p.a. WealthDesk | BNB"
        state = {**self._base_state(), "customer_message": "What is the car loan rate?"}
        with patch.object(_nodes, "llm_with_tools") as mock_wt, \
             patch.object(_nodes, "llm") as mock_llm, \
             patch.object(_nodes, "_run_tool", return_value="Car Loan: 9.5% p.a."):
            mock_wt.invoke.return_value = self._make_tool_call_result(
                "query_rates", {"product_type": "loan"}
            )
            mock_llm.invoke.return_value = self._make_text_result(final_answer)
            result = respond(state)
        assert result["response"] == final_answer

    def test_respond_history_grows_by_two(self):
        state = {**self._base_state(), "customer_message": "What is the FD rate?"}
        with patch.object(_nodes, "llm_with_tools") as mock_wt, \
             patch.object(_nodes, "llm"):
            mock_wt.invoke.return_value = self._make_text_result("FD rates start at 6.8%.")
            result = respond(state)
        assert len(result["history"]) == 2
        assert result["history"][0]["role"] == "user"
        assert result["history"][1]["role"] == "assistant"

    def test_respond_appends_tool_message_to_conversation(self):
        captured_messages = []

        def capture_invoke(msgs):
            captured_messages.extend(msgs)
            return MagicMock(content="8.5% p.a. WealthDesk | BNB", tool_calls=[])

        with patch.object(_nodes, "llm_with_tools") as mock_wt, \
             patch.object(_nodes, "llm") as mock_llm, \
             patch.object(_nodes, "_run_tool", return_value="Home Loan: 8.5% p.a."):
            mock_wt.invoke.return_value = self._make_tool_call_result(
                "query_rates", {"product_type": "loan"}
            )
            mock_llm.invoke.side_effect = capture_invoke
            respond(self._base_state())

        from langchain_core.messages import ToolMessage as TM
        tool_messages = [m for m in captured_messages if isinstance(m, TM)]
        assert len(tool_messages) == 1
        assert "Home Loan" in tool_messages[0].content


# ---------------------------------------------------------------------------
# TestGraphRouting
# ---------------------------------------------------------------------------

class TestGraphRouting:
    def _mock_vectorstore(self):
        vs = MagicMock()
        vs.similarity_search.return_value = []
        return vs

    def test_simple_path_calls_llm_with_tools(self):
        from langgraph.checkpoint.memory import MemorySaver

        mock_vs = self._mock_vectorstore()
        with patch.object(_nodes, "llm_with_tools") as mock_wt, \
             patch.object(_nodes, "classifier_llm") as mock_cl, \
             patch.object(_nodes, "llm"), \
             patch.object(_nodes, "vectorstore", mock_vs), \
             patch.object(_nodes, "_init_vectorstore"):
            mock_cl.invoke.return_value = MagicMock(content="SIMPLE")
            mock_wt.invoke.return_value = MagicMock(
                content="The home loan rate is 8.5%.", tool_calls=[]
            )
            graph = build_graph(checkpointer=MemorySaver())
            config = {"configurable": {"thread_id": "test-simple"}}
            graph.invoke(
                {"customer_message": "What is the home loan rate?", "response": ""},
                config=config,
            )
        mock_wt.invoke.assert_called_once()

    def test_complex_path_skips_llm_with_tools(self):
        from langgraph.checkpoint.memory import MemorySaver

        mock_vs = self._mock_vectorstore()
        with patch.object(_nodes, "llm_with_tools") as mock_wt, \
             patch.object(_nodes, "classifier_llm") as mock_cl, \
             patch.object(_nodes, "vectorstore", mock_vs), \
             patch.object(_nodes, "_init_vectorstore"):
            mock_cl.invoke.return_value = MagicMock(content="COMPLEX")
            graph = build_graph(checkpointer=MemorySaver())
            config = {"configurable": {"thread_id": "test-complex"}}
            result = graph.invoke(
                {"customer_message": "Should I invest in FD or pay off my loan?", "response": ""},
                config=config,
            )
        mock_wt.invoke.assert_not_called()
        assert "Relationship Manager" in result["response"]

    def test_out_of_scope_path_skips_llm_with_tools(self):
        from langgraph.checkpoint.memory import MemorySaver

        mock_vs = self._mock_vectorstore()
        with patch.object(_nodes, "llm_with_tools") as mock_wt, \
             patch.object(_nodes, "classifier_llm") as mock_cl, \
             patch.object(_nodes, "vectorstore", mock_vs), \
             patch.object(_nodes, "_init_vectorstore"):
            mock_cl.invoke.return_value = MagicMock(content="OUT_OF_SCOPE")
            graph = build_graph(checkpointer=MemorySaver())
            config = {"configurable": {"thread_id": "test-oos"}}
            result = graph.invoke(
                {"customer_message": "What is the weather today?", "response": ""},
                config=config,
            )
        mock_wt.invoke.assert_not_called()
        assert "only help with BNB" in result["response"]
