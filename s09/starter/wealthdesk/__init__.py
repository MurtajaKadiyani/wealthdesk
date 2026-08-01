"""
WealthDesk package -- Session 9: Compliance Filter + LangSmith Observability
=============================================================================

This file runs automatically when Python imports the wealthdesk package.
"""
import os

os.environ.setdefault("HF_HUB_VERBOSITY", "error")

# ---------------------------------------------------------------------------
# TODO 4 (part A) -- Uncomment these three lines to enable LangSmith tracing
# ---------------------------------------------------------------------------
# Setting LANGCHAIN_TRACING_V2=true tells LangGraph to send every run to
# LangSmith automatically -- no extra code needed per run.
# Values come from your .env file: LANGSMITH_API_KEY, LANGSMITH_TRACING,
# LANGSMITH_PROJECT.
# ---------------------------------------------------------------------------
# if os.getenv("LANGSMITH_API_KEY") and os.getenv("LANGSMITH_TRACING", "").lower() == "true":
#     os.environ["LANGCHAIN_TRACING_V2"] = "true"
#     os.environ.setdefault("LANGCHAIN_API_KEY", os.getenv("LANGSMITH_API_KEY", ""))
#     os.environ.setdefault("LANGCHAIN_PROJECT",  os.getenv("LANGSMITH_PROJECT", "batch1-wealthdesk"))

from dotenv import load_dotenv
load_dotenv()
