def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60, top_n: int = 20):
  """Fuse ranked lists using RRF — position-based, no score normalization needed."""
  scores = {}
  chunk_data = {}

  for result_list in result_lists:
    for rank, item in enumerate(result_list, start=1):
      item_id = item["id"]
      scores[item_id] = scores.get(item_id, 0) + 1 / (k + rank)
      chunk_data[item_id] = item

  ranked_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
  fused = []
  for item_id, score in ranked_ids[:top_n]:
    chunk = chunk_data[item_id].copy()
    chunk["rrf_score"] = score
    fused.append(chunk)
  return fused
