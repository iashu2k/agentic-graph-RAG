import json
from groq import Groq
from app.config import settings
from app.services import llm_client

client = Groq(api_key=settings.groq_api_key)

ROUTER_PROMPT = """Classify this question about SEC 10-Q filings into ONE route:

- "graph": needs relationship/multi-hop reasoning — comparing companies, shared risk factors/disclosures across entities, category-based aggregation
- "hybrid": needs specific factual/semantic lookup from filing text — revenue numbers, specific statements, narrative explanations
- "both": question has both a relational AND a specific-fact component

Known companies: Apple, Amazon, Intel, Microsoft, NVIDIA
Known disclosure categories: risk_factor, restructuring, financial_metric, legal_proceeding, tax, segment, other

{prior_attempt_note}

Return ONLY valid JSON: {{"route": "graph" | "hybrid" | "both", "reasoning": "one sentence"}}

Question: {question}
"""


def classify_route(question: str, prior_route: str | None = None) -> dict:
  prior_note = ""
  if prior_route:
    prior_note = (
        f"Note: route '{prior_route}' was already tried on an earlier phrasing of this "
        f"question and returned low-relevance/empty results. Consider a different route "
        f"if it seems more appropriate now."
    )

  # Uses llm_client.get_model() rather than a hardcoded model string, so
  # app/eval/run_eval.py's fallback rotation (on Groq 429s) actually applies
  # to routing classification, not just app/agent/generate.py's final answer call.
  resp = client.chat.completions.create(
      model=llm_client.get_model(),
      messages=[{"role": "user", "content": ROUTER_PROMPT.format(
          question=question, prior_attempt_note=prior_note
      )}],
      response_format={"type": "json_object"},
      temperature=0,
  )
  try:
    return json.loads(resp.choices[0].message.content)
  except json.JSONDecodeError:
    return {"route": prior_route or "hybrid", "reasoning": "fallback: router output unparseable"}


def router_node(state: dict) -> dict:
  prior_route = state.get("route")
  result = classify_route(state["question"], prior_route=prior_route)
  return {**state, "route": result.get("route", prior_route or "hybrid")}
