"""
s10/tests/conftest.py
---------------------
Pytest configuration for Session 10 tests.
Sets dummy API keys so importing wealthdesk does not crash.
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
