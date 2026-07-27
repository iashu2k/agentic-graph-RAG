"""
app/eval/_vertexai_shim.py

Compatibility shim for a known ragas bug (ragas 0.4.3, still open as of
2026-07-27 - see https://github.com/vibrantlabsai/ragas/issues/2745):

ragas.llms.base unconditionally imports ChatVertexAI from
langchain_community.chat_models.vertexai, but that submodule was removed
in current langchain-community versions (Vertex AI support moved to the
separate langchain-google-vertexai package). This project doesn't use
Vertex AI at all, and pinning langchain-community to an older version that
still has the module is blocked by this project's own langchain-core>=1.5.1
and sqlalchemy>=2.0.51 requirements (same conflict class as the earlier
langchain-groq incompatibility).

Fix: register a fake langchain_community.chat_models.vertexai module in
sys.modules BEFORE ragas is imported, containing a placeholder ChatVertexAI
class. ragas's import succeeds, and since this project's ragas_judge.py
uses ChatOpenAI (pointed at Groq) instead of ChatVertexAI, the placeholder
is never actually instantiated or called - it only exists to satisfy the
broken import path.

Usage: import this module FIRST, before importing anything from ragas,
in any file that transitively imports ragas (e.g. app/eval/score_ragas.py):

    import app.eval._vertexai_shim  # noqa: F401 - must be first
    from ragas import evaluate
"""
import sys
import types

if "langchain_community.chat_models.vertexai" not in sys.modules:
  shim_module = types.ModuleType("langchain_community.chat_models.vertexai")

  class ChatVertexAI:
    """Placeholder - never actually used. See module docstring."""

    def __init__(self, *args, **kwargs):
      raise RuntimeError(
          "ChatVertexAI is a compatibility shim placeholder and was "
          "not meant to be instantiated. This project uses ChatOpenAI "
          "(pointed at Groq) via app/eval/ragas_judge.py instead."
      )

  shim_module.ChatVertexAI = ChatVertexAI
  sys.modules["langchain_community.chat_models.vertexai"] = shim_module
