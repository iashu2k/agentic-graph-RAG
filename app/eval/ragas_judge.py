"""
app/eval/ragas_judge.py

OpenRouter-backed judge LLM + local embeddings wrapper for RAGAS, since
RAGAS defaults to OpenAI for both the evaluator LLM and embeddings.

Switched from Groq to OpenRouter (2026-07-27) after repeatedly exhausting
Groq's daily TPD quota across multiple models (llama-3.3-70b-versatile,
then llama-3.1-8b-instant) during the same eval session - run_eval.py's
agent pipeline run and score_ragas.py's judge calls were both drawing from
the same Groq account's per-model daily quotas, compounding exhaustion.

OpenRouter is also OpenAI-API-compatible, so this uses the same
langchain-openai ChatOpenAI + base_url override pattern already used for
Groq - no new dependency needed, just a different base_url and API key.

Setup required:
1. Get an API key at https://openrouter.ai/keys
2. Add to .env: OPENROUTER_API_KEY=your-key-here
3. Add `openrouter_api_key: str` to app/config.py's Settings class

Model name format on OpenRouter uses a "provider/model" prefix, e.g.
"meta-llama/llama-3.1-8b-instruct" (note: "-instruct" not "-instant" -
OpenRouter's naming differs slightly from Groq's for the same underlying
model family). Full model list: https://openrouter.ai/models
"""
from functools import lru_cache
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from app.config import settings

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Defaults to Llama 3.1 8B via OpenRouter, per user request. Mutable at
# runtime via set_judge_model() so score_ragas.py can still rotate on a 429
# if OpenRouter's own rate limits are hit.
_judge_model = "meta-llama/llama-3.1-8b-instruct"

JUDGE_TEMPERATURE = 0
JUDGE_MAX_RETRIES = 3
JUDGE_TIMEOUT = 60

EMBEDDING_MODEL = settings.embedding_model  # BAAI/bge-small-en-v1.5


def set_judge_model(model_name: str) -> None:
  global _judge_model
  _judge_model = model_name
  get_ragas_llm.cache_clear()


def get_judge_model() -> str:
  return _judge_model


@lru_cache(maxsize=1)
def get_ragas_llm() -> Any:
  chat = ChatOpenAI(
      model=_judge_model,
      temperature=JUDGE_TEMPERATURE,
      max_retries=JUDGE_MAX_RETRIES,
      timeout=JUDGE_TIMEOUT,
      api_key=settings.openrouter_api_key,
      base_url=OPENROUTER_BASE_URL,
  )
  return LangchainLLMWrapper(chat)


@lru_cache(maxsize=1)
def get_ragas_embeddings() -> Any:
  hf_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
  return LangchainEmbeddingsWrapper(hf_embeddings)


# Fallback chain if OpenRouter itself rate-limits or a specific model is
# temporarily unavailable. OpenRouter routes to different underlying
# providers per model, so these draw from genuinely separate capacity,
# unlike Groq's per-model-but-same-account quota pools.
JUDGE_FALLBACK_CHAIN = [
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
    "google/gemma-2-9b-it",
]
