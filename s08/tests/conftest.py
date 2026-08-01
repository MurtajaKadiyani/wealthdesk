"""
s08/tests/conftest.py
---------------------
Pytest configuration for Session 8 tests.

Sets dummy API keys so importing wealthdesk does not crash on missing env vars.
Session 8 tests mock _run_mcp_tool() so no real MCP server or database is
required to run the unit test suite.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("GROQ_API_KEY",      "test-key-not-real")
os.environ.setdefault("LANGSMITH_API_KEY",  "test-langsmith-key")
os.environ.setdefault("HF_HUB_VERBOSITY",  "error")

SOLUTION_DIR = Path(__file__).parent.parent / "solution"
for _k in list(sys.modules):
    if _k == "wealthdesk" or _k.startswith("wealthdesk."):
        sys.modules.pop(_k)
sys.path.insert(0, str(SOLUTION_DIR))
