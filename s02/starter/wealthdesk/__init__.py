"""
WealthDesk package -- Session 1: Basic Conversational Agent (US-01)
====================================================================

This file runs automatically when Python imports the wealthdesk package.
Use it to set up the environment before any other module loads.
"""
import os

os.environ.setdefault("HF_HUB_VERBOSITY", "error")
from dotenv import load_dotenv
load_dotenv()

