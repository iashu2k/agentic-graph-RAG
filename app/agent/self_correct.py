def needs_retry(state: dict) -> bool:
  graph_empty = state["route"] in (
    "graph", "both") and not state.get("graph_results")
  hybrid_empty = state["route"] in (
    "hybrid", "both") and not state.get("hybrid_results")
  exhausted = state["retry_count"] >= state["max_retries"]
  return (graph_empty or hybrid_empty) and not exhausted


def self_correct_node(state: dict) -> dict:
  new_route = "both" if state["route"] != "both" else state["route"]
  return {**state, "route": new_route, "retry_count": state["retry_count"] + 1}
