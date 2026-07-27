import json
from groq import Groq
from app.config import settings
from app.services import llm_client

client = Groq(api_key=settings.groq_api_key)

REWRITE_PROMPT = """This question returned NO results when queried against a SEC 10-Q knowledge graph/search system.

Original question: {question}
Route attempted: {route}

Known companies: Apple, Amazon, Intel, Microsoft, NVIDIA
Known fiscal periods: 2022 Q3, 2023 Q1, 2023 Q2, 2023 Q3 (ONLY these four — do not invent or assume any other quarter/year)
Known disclosure categories: risk_factor, restructuring, financial_metric, legal_proceeding, tax, segment, other

Rewrite the question to be more likely to match indexed content. Common failure causes:
- Company name mismatch (use full names, not tickers: 'Apple' not 'AAPL')
- Overly specific phrasing that won't match via keyword/substring search
- Jargon mismatch (e.g. "layoffs" -> "restructuring", "lawsuit" -> "legal proceeding")

CRITICAL: If the original question does not specify a fiscal quarter/year, do NOT add one — leave the time period unspecified. Never invent a fiscal quarter/year outside the four listed above.

Return ONLY valid JSON: {{"rewritten_question": "...", "reasoning": "one sentence explaining the change"}}

Original question: {question}
"""


def rewrite_query(question: str, route: str) -> dict:
  # Uses llm_client.get_model() rather than a hardcoded model string, so
  # app/eval/run_eval.py's fallback rotation (on Groq 429s) actually applies
  # to query rewriting, not just app/agent/generate.py's final answer call.
  resp = client.chat.completions.create(
      model=llm_client.get_model(),
      messages=[{"role": "user", "content": REWRITE_PROMPT.format(
        question=question, route=route)}],
      response_format={"type": "json_object"},
      temperature=0.3,
  )
  try:
    return json.loads(resp.choices[0].message.content)
  except json.JSONDecodeError:
    return {"rewritten_question": question, "reasoning": "rewrite failed, using original"}


def rewrite_node(state: dict) -> dict:
  result = rewrite_query(state["question"], state["route"])
  new_retry_count = state["retry_count"] + 1
  return {
      **state,
      "original_question": state.get("original_question", state["question"]),
      "question": result["rewritten_question"],
      "rewrite_reasoning": result.get("reasoning", ""),
      "retry_count": new_retry_count,
  }
