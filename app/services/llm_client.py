# app/services/llm_client.py
from groq import Groq
from langfuse import observe, get_client
from app.config import settings


client = Groq(api_key=settings.groq_api_key)
langfuse = get_client()


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


@observe(as_type="generation", name="groq-generate-answer")
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

  messages = [{"role": "user", "content": prompt}]

  # Set input + model metadata before the call, so it is captured even if
  # the request itself raises (e.g. 429 quota exhaustion mid fallback-rotation).
  langfuse.update_current_generation(
      input=messages,
      model=CURRENT_MODEL,
      model_parameters={"temperature": 0.1},
      metadata={"num_context_chunks": len(context_chunks)},
  )

  response = client.chat.completions.create(
      model=CURRENT_MODEL,
      messages=messages,
      temperature=0.1,
  )

  answer = response.choices[0].message.content
  usage = response.usage

  langfuse.update_current_generation(
      output=answer,
      usage_details={
          "input": usage.prompt_tokens if usage else None,
          "output": usage.completion_tokens if usage else None,
          "total": usage.total_tokens if usage else None,
      },
  )

  return answer
