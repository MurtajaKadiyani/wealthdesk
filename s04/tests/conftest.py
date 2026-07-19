"""
s04/tests/conftest.py
---------------------
Pytest configuration for Session 4 tests.

Sets dummy API keys and clears any stale wealthdesk modules at collection time.
HuggingFaceEmbeddings and Chroma are never initialised during tests because
the mock_vectorstore fixture patches wealthdesk.nodes.vectorstore with a mock
and patches _init_vectorstore to a no-op, so the real vectorstore path
is never reached.
"""
import os
import sys

for _key in list(sys.modules):
    if _key == "wealthdesk" or _key.startswith("wealthdesk."):
        sys.modules.pop(_key)

os.environ.setdefault("GROQ_API_KEY", "test-key-not-real")
os.environ.setdefault("LANGSMITH_API_KEY", "test-langsmith-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
