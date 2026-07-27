# app/observability/tracing.py
from langfuse import get_client
from langfuse.langchain import CallbackHandler

langfuse = get_client()
langfuse_handler = CallbackHandler()


def traced_config(pipeline: str, route: str | None = None, retry_count: int = 0) -> dict:
  """Build a LangChain/LangGraph config dict with Langfuse callback + metadata attached."""
  return {
      "callbacks": [langfuse_handler],
      "run_name": f"deepfile-{pipeline}",
      "tags": [pipeline, route or "unrouted", f"retries:{retry_count}"],
      "metadata": {
          "langfuse_tags": [pipeline, route or "unrouted", f"retries:{retry_count}"],
          "pipeline": pipeline,
          "route": route,
          "retry_count": retry_count,
      },
  }


def flag_output(trace_id: str, answer: str, contexts: list[str] | None = None) -> None:
  """Async/observational guardrail scoring — does not block the response."""
  has_citation = any(f"[{i}]" in answer for i in range(1, 20))
  is_empty = not answer.strip()

  langfuse.create_score(
      trace_id=trace_id,
      name="has_citation",
      value=1.0 if has_citation else 0.0,
      data_type="NUMERIC",
  )
  langfuse.create_score(
      trace_id=trace_id,
      name="empty_answer",
      value=1.0 if is_empty else 0.0,
      data_type="NUMERIC",
  )
  if contexts is not None:
    langfuse.create_score(
        trace_id=trace_id,
        name="empty_context",
        value=1.0 if len(contexts) == 0 else 0.0,
        data_type="NUMERIC",
    )
