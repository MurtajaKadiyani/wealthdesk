"""
s08/tests/test_s08.py
---------------------
Tests for Session 8: MCP Agent Integration (US-06 Part 2), using
langchain-mcp-adapters (MultiServerMCPClient) for tool loading.

Run with:
    pytest s08/tests/ -v

Importing wealthdesk.tools connects to the real Session 7 MCP server once,
at module load, to discover the tool schemas -- MultiServerMCPClient has no
way to bind_tools() without a real (fast, local, offline) round trip asking
the server what tools it exposes. This is NOT a live-LLM test: no Groq call
happens here, and the server only returns local SQLite data.

_run_tool()'s dispatch/error-handling logic is tested by swapping in fake
tool objects for the real ones, so those tests do not depend on database
contents.

Test groups:
  TestMCPServerPath    -- MCP_SERVER_PATH points to the Session 7 server
  TestMCPToolLoading   -- mcp_tools / _tool_registry loaded from the real server
  TestExtractText      -- MCP content-block list -> plain string
  TestRunTool          -- _run_tool dispatches correctly, handles unknown/errors
  TestGraphNodes       -- classify, escalate, decline produce correct state keys
  TestBuildGraph       -- graph compiles, invoke returns expected keys
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

SOLUTION_DIR = Path(__file__).parent.parent / "solution"
for _k in list(sys.modules):
    if _k == "wealthdesk" or _k.startswith("wealthdesk."):
        sys.modules.pop(_k)
sys.path.insert(0, str(SOLUTION_DIR))

from wealthdesk.config import MCP_SERVER_PATH  # noqa: E402
from wealthdesk.state import WealthDeskState   # noqa: E402
import wealthdesk.tools as _tools              # noqa: E402
import wealthdesk.nodes as _nodes              # noqa: E402
from wealthdesk.tools import _extract_text, _run_tool  # noqa: E402
from wealthdesk.nodes import classify, decline, escalate  # noqa: E402
from wealthdesk.agent import build_graph  # noqa: E402


# ---------------------------------------------------------------------------
# TestMCPServerPath
# ---------------------------------------------------------------------------

class TestMCPServerPath:
    def test_mcp_server_path_is_path_object(self):
        assert isinstance(MCP_SERVER_PATH, Path)

    def test_mcp_server_path_points_to_s07(self):
        assert "s07" in str(MCP_SERVER_PATH)

    def test_mcp_server_path_filename(self):
        assert MCP_SERVER_PATH.name == "mcp_server.py"

    def test_mcp_server_path_exists(self):
        assert MCP_SERVER_PATH.exists(), (
            f"S07 MCP server not found at {MCP_SERVER_PATH}. "
            "Complete Session 7 before running Session 8 tests."
        )


# ---------------------------------------------------------------------------
# TestMCPToolLoading
# ---------------------------------------------------------------------------

class TestMCPToolLoading:
    def test_mcp_tools_has_two_tools(self):
        assert len(_tools.mcp_tools) == 2

    def test_mcp_tools_contains_query_rates(self):
        names = [t.name for t in _tools.mcp_tools]
        assert "query_rates" in names

    def test_mcp_tools_contains_query_branch(self):
        names = [t.name for t in _tools.mcp_tools]
        assert "query_branch" in names

    def test_tool_registry_maps_names_to_tools(self):
        assert set(_tools._tool_registry.keys()) == {"query_rates", "query_branch"}

    def test_tools_have_descriptions(self):
        for t in _tools.mcp_tools:
            assert t.description and len(t.description) > 10

    def test_llm_with_tools_is_bound(self):
        assert _tools.llm_with_tools is not None
        assert _tools.llm_with_tools is not _tools.llm


# ---------------------------------------------------------------------------
# TestExtractText
# ---------------------------------------------------------------------------

class TestExtractText:
    def test_single_text_block(self):
        result = _extract_text([{"type": "text", "text": "Home Loan: 8.5%"}])
        assert result == "Home Loan: 8.5%"

    def test_multiple_text_blocks_joined_with_newline(self):
        result = _extract_text([
            {"type": "text", "text": "Block 1"},
            {"type": "text", "text": "Block 2"},
        ])
        assert result == "Block 1\nBlock 2"

    def test_empty_list_returns_empty_string(self):
        assert _extract_text([]) == ""

    def test_non_list_falls_back_to_str(self):
        assert _extract_text("already a string") == "already a string"

    def test_block_missing_text_key_defaults_to_empty(self):
        result = _extract_text([{"type": "text"}])
        assert result == ""


# ---------------------------------------------------------------------------
# TestRunTool
# ---------------------------------------------------------------------------

class TestRunTool:
    def _fake_tool(self, return_value=None, side_effect=None):
        fake = MagicMock()
        if side_effect is not None:
            fake.ainvoke = AsyncMock(side_effect=side_effect)
        else:
            fake.ainvoke = AsyncMock(return_value=return_value)
        return fake

    def test_dispatches_query_rates(self):
        fake = self._fake_tool(return_value=[{"type": "text", "text": "8.5%"}])
        with patch.dict(_tools._tool_registry, {"query_rates": fake}):
            result = _run_tool("query_rates", {"product_type": "loan"})
        assert result == "8.5%"
        fake.ainvoke.assert_called_once_with({"product_type": "loan"})

    def test_dispatches_query_branch(self):
        fake = self._fake_tool(return_value=[{"type": "text", "text": "BNB Bandra"}])
        with patch.dict(_tools._tool_registry, {"query_branch": fake}):
            result = _run_tool("query_branch", {"city": "Mumbai"})
        assert result == "BNB Bandra"

    def test_unknown_tool_returns_error_string(self):
        result = _run_tool("nonexistent_tool", {})
        assert "Unknown tool" in result
        assert "nonexistent_tool" in result

    def test_returns_string(self):
        fake = self._fake_tool(return_value=[{"type": "text", "text": "result"}])
        with patch.dict(_tools._tool_registry, {"query_rates": fake}):
            result = _run_tool("query_rates", {"product_type": "all"})
        assert isinstance(result, str)

    def test_tool_exception_returns_error_string(self):
        fake = self._fake_tool(side_effect=RuntimeError("crash"))
        with patch.dict(_tools._tool_registry, {"query_rates": fake}):
            result = _run_tool("query_rates", {"product_type": "loan"})
        assert "Tool error" in result
        assert "query_rates" in result


# ---------------------------------------------------------------------------
# TestGraphNodes
# ---------------------------------------------------------------------------

class TestGraphNodes:
    def _make_state(self, message="test", query_type="SIMPLE") -> WealthDeskState:
        return {
            "customer_message": message,
            "response":         "",
            "history":          [],
            "query_type":       query_type,
            "retrieved_docs":   [],
        }

    def test_escalate_returns_response_key(self):
        result = escalate(self._make_state())
        assert "response" in result

    def test_escalate_response_mentions_relationship_manager(self):
        result = escalate(self._make_state())
        assert "Relationship Manager" in result["response"]

    def test_escalate_response_includes_phone_number(self):
        result = escalate(self._make_state())
        assert "1800-103-1906" in result["response"]

    def test_escalate_updates_history(self):
        result = escalate(self._make_state("complex query"))
        assert len(result["history"]) == 2
        assert result["history"][0]["role"] == "user"
        assert result["history"][1]["role"] == "assistant"

    def test_decline_returns_response_key(self):
        result = decline(self._make_state())
        assert "response" in result

    def test_decline_response_mentions_bnb(self):
        result = decline(self._make_state())
        assert "BNB" in result["response"]

    def test_decline_updates_history(self):
        result = decline(self._make_state("off-topic query"))
        assert len(result["history"]) == 2


# ---------------------------------------------------------------------------
# TestBuildGraph
# ---------------------------------------------------------------------------

class TestBuildGraph:
    def test_build_graph_returns_compiled_graph(self):
        from langgraph.checkpoint.memory import MemorySaver
        graph = build_graph(checkpointer=MemorySaver())
        assert graph is not None

    def test_graph_invoke_complex_returns_escalation(self):
        from langgraph.checkpoint.memory import MemorySaver

        with patch.object(_nodes, "classifier_llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="COMPLEX")
            graph = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                {"customer_message": "Should I invest all my savings?", "response": ""},
                config={"configurable": {"thread_id": "test-complex"}},
            )
        assert "Relationship Manager" in result["response"]
        assert result["query_type"] == "COMPLEX"

    def test_graph_invoke_oos_returns_decline(self):
        from langgraph.checkpoint.memory import MemorySaver

        with patch.object(_nodes, "classifier_llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="OUT_OF_SCOPE")
            graph = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                {"customer_message": "Write me a poem", "response": ""},
                config={"configurable": {"thread_id": "test-oos"}},
            )
        assert "only help with BNB" in result["response"]
        assert result["query_type"] == "OUT_OF_SCOPE"

    def test_graph_invoke_returns_response_key(self):
        from langgraph.checkpoint.memory import MemorySaver

        with patch.object(_nodes, "classifier_llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="COMPLEX")
            graph = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                {"customer_message": "test", "response": ""},
                config={"configurable": {"thread_id": "test-keys"}},
            )
        assert "response" in result
        assert "query_type" in result
