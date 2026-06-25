"""
s01/tests/conftest.py
---------------------
Pytest configuration for Session 1 tests.

Sets a dummy GROQ_API_KEY in the environment before any test module is imported.
Without this, the module-level guard in wealthdesk/tools.py raises ValueError during
test collection and aborts pytest before a single test runs.
"""
import os
import sys

# Clear any cached wealthdesk modules from a previous session's import.
for _key in list(sys.modules):
    if _key == "wealthdesk" or _key.startswith("wealthdesk."):
        sys.modules.pop(_key)

os.environ.setdefault("GROQ_API_KEY", "test-key-not-real")
os.environ.setdefault("LANGSMITH_API_KEY", "test-langsmith-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
