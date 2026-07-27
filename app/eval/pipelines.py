"""
app/eval/pipelines.py

Wraps each pipeline "version" behind a single run(question) -> dict interface
so the same eval loop can score baseline vector RAG (Phase 1), hybrid+rerank
(Phase 2), and the full agentic graph (Phase 4) without duplicating logic.

Each run() returns:
    {
        "answer": str,
        "contexts": list[str],          # retrieved chunk text, for RAGAS faithfulness/
                                         # context_precision/context_recall
        "retrieved_filing_ids": list[str],  # normalized filing_ids of retrieved chunks,
                                         # for exact-match precision/recall against
                                         # dataset_builder.py's ground_truth_docs
        "route": str | None,            # agentic pipeline only - "graph" or "hybrid"
    }

filing_id format matches the graph's Filing.filing_id: "{company}-{fiscal_year}-{fiscal_quarter}"
e.g. "Intel-2022-Q3". Vector/hybrid chunk metadata is assumed to carry
`company`, `fiscal_year`, `fiscal_quarter` fields already populated during
Phase 1/2 ingestion (see app/ingestion/loader.py).

NOTE on real function signatures (verified via grep against actual source,
2026-07-26 - do not assume signatures without checking, per the Phase 3
get_connection/get_conn lesson):
    app/retrieval/vector_search.py   -> vector_search(query: str, top_k: int = 20)
    app/retrieval/hybrid_search.py   -> hybrid_retrieve(query: str, fetch_k: int = 20,
                                                         fuse_k: int = 20, final_k: int = 5)
"""
from app.retrieval.vector_search import vector_search
from app.retrieval.hybrid_search import hybrid_retrieve
from app.services.llm_client import generate_answer
from app.agent.graph_builder import agent


def chunk_to_filing_id(chunk: dict) -> str | None:
  company = chunk.get("company")
  year = chunk.get("fiscal_year")
  quarter = chunk.get("fiscal_quarter")
  if company and year and quarter:
    return f"{company}-{year}-{quarter}"
  return None


def graph_result_to_filing_id(result: dict) -> str | None:
  """Knowledge graph results carry fiscal_quarter/fiscal_year directly
  (see app/agent/generate.py::graph_result_to_chunk) but company may be
  under a ticker-like key rather than a literal 'company' key.
  """
  company = result.get("company")
  if not company:
    ticker_keys = [k for k in result.keys() if "ticker" in k.lower()]
    if ticker_keys:
      company = result[ticker_keys[0]]
  year = result.get("fiscal_year")
  quarter = result.get("fiscal_quarter")
  if company and year and quarter:
    return f"{company}-{year}-{quarter}"
  return None


def run_baseline(question: str, top_k: int = 5) -> dict:
  results = vector_search(question, top_k=top_k)
  contexts = [r["content"] for r in results]
  filing_ids = [fid for r in results if (fid := chunk_to_filing_id(r))]
  answer = generate_answer(question, results)
  return {
      "answer": answer,
      "contexts": contexts,
      "retrieved_filing_ids": filing_ids,
      "route": None,
  }


def run_hybrid(question: str, final_k: int = 5) -> dict:
  results = hybrid_retrieve(question, final_k=final_k)
  contexts = [r["content"] for r in results]
  filing_ids = [fid for r in results if (fid := chunk_to_filing_id(r))]
  answer = generate_answer(question, results)
  return {
      "answer": answer,
      "contexts": contexts,
      "retrieved_filing_ids": filing_ids,
      "route": None,
  }


def run_agentic(question: str) -> dict:
  state = agent.invoke({
      "question": question,
      "original_question": None,
      "rewrite_reasoning": None,
      "route": None,
      "cypher_query": None,
      "graph_results": [],
      "hybrid_results": [],
      "context": "",
      "answer": "",
      "retry_count": 0,
      "max_retries": 2,
  })

  context_chunks = state.get("context", [])
  contexts = [c["content"] for c in context_chunks]

  filing_ids = []
  if state.get("route") == "graph":
    for r in state.get("graph_results", []):
      fid = graph_result_to_filing_id(r)
      if fid:
        filing_ids.append(fid)
  else:
    for r in state.get("hybrid_results", []):
      fid = chunk_to_filing_id(r)
      if fid:
        filing_ids.append(fid)

  return {
      "answer": state["answer"],
      "contexts": contexts,
      "retrieved_filing_ids": filing_ids,
      "route": state.get("route"),
      "retry_count": state.get("retry_count", 0),
  }


PIPELINES = {
    "baseline": run_baseline,
    "hybrid": run_hybrid,
    "agentic": run_agentic,
}
