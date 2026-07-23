from sentence_transformers import CrossEncoder

_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query: str, candidates: list[dict], top_n: int = 5):
  if not candidates:
    return []

  pairs = [(query, c.get("search_text") or c["content"]) for c in candidates]
  scores = _reranker.predict(pairs)

  for c, score in zip(candidates, scores):
    c["rerank_score"] = float(score)

  return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_n]
