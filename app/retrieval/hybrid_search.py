from app.retrieval.vector_search import vector_search
from app.retrieval.keyword_search import keyword_search
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import rerank


def hybrid_retrieve(query: str, fetch_k: int = 20, fuse_k: int = 20, final_k: int = 5):
  vec_results = vector_search(query, top_k=fetch_k)
  kw_results = keyword_search(query, top_k=fetch_k)

  fused = reciprocal_rank_fusion([vec_results, kw_results], top_n=fuse_k)
  final = rerank(query, fused, top_n=final_k)

  return final
