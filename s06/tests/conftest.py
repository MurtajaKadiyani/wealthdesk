"""
s06/tests/conftest.py
---------------------
Pytest configuration for Session 6 tests.

Sets dummy API keys. Session 6 tests a separate module (evaluate.py),
not main.py, so no sys.modules.pop is needed.
"""
import os

os.environ.setdefault("GROQ_API_KEY", "test-key-not-real")
os.environ.setdefault("LANGSMITH_API_KEY", "test-langsmith-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
