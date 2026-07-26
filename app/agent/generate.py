from app.services.llm_client import generate_answer


def graph_result_to_chunk(r: dict) -> dict:
  if "company" in r:
    company = r["company"]
  else:
    ticker_keys = [k for k in r.keys() if "ticker" in k.lower()]
    if len(ticker_keys) > 1:
      company = r[ticker_keys[-1]]
    elif ticker_keys:
      company = r[ticker_keys[0]]
    else:
      company = "Unknown"

  content = " | ".join(f"{k}: {v}" for k, v in r.items())
  return {
      "company": company,
      "filing_type": "10-Q",
      "fiscal_quarter": r.get("fiscal_quarter", "N/A"),
      "fiscal_year": r.get("fiscal_year", "N/A"),
      "section": "Knowledge Graph",
      "content": content,
  }


def build_context_chunks(state: dict) -> list[dict]:
  chunks = []
  if state.get("graph_results"):
    chunks.extend(graph_result_to_chunk(r) for r in state["graph_results"])
  if state.get("hybrid_results"):
    chunks.extend(state["hybrid_results"])
  return chunks


def generate_node(state: dict) -> dict:
  context_chunks = build_context_chunks(state)
  if not context_chunks:
    return {**state, "context": [], "answer": "No relevant information found in the graph or document search."}

  question = state["question"]
  if state.get("graph_results"):
    question = (
        f"{question}\n\n"
        "Note: The knowledge graph context below already represents companies "
        "that match the relationship being asked about (e.g., if asked which companies "
        "share a trait with a reference company, the reference company is deliberately "
        "excluded from these results since the results ARE the answer)."
    )

  answer = generate_answer(question, context_chunks)
  return {**state, "context": context_chunks, "answer": answer}
