from groq import Groq
from app.config import settings

client = Groq(api_key=settings.groq_api_key)

# Module-level mutable model state, since app/config.py's Settings has no
# `model` field (unlike the aspirational snippet in early README drafts) -
# the model was previously hardcoded directly in generate_answer(). This
# allows app/eval/run_eval.py's fallback-rotation logic to swap models
# on a 429 without needing a Settings field that doesn't exist.
CURRENT_MODEL = "llama-3.3-70b-versatile"


def set_model(model_name: str) -> None:
  global CURRENT_MODEL
  CURRENT_MODEL = model_name


def get_model() -> str:
  return CURRENT_MODEL


def generate_answer(query: str, context_chunks: list[dict]):
  context_str = "\n\n".join(
      f"[Source: {c['company']} {c['filing_type']} {c['fiscal_quarter']} FY{c['fiscal_year']}, Section: {c['section']}]\n{c['content']}"
      for c in context_chunks
  )

  prompt = f"""Answer the question using ONLY the context below. Cite the source (company, filing type, quarter, fiscal year, section) for every claim.

Context:
{context_str}

Question: {query}

Answer:"""

  response = client.chat.completions.create(
      model=CURRENT_MODEL,
      messages=[{"role": "user", "content": prompt}],
      temperature=0.1,
  )
  return response.choices[0].message.content
