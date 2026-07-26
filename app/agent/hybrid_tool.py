from app.retrieval.hybrid_search import hybrid_retrieve


def hybrid_node(state: dict) -> dict:
  results = hybrid_retrieve(state["question"])
  return {**state, "hybrid_results": results}
