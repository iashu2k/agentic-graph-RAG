from langgraph.graph import StateGraph, END
from app.agent.state import AgentState
from app.agent.router import router_node
from app.agent.graph_tool import graph_node
from app.agent.hybrid_tool import hybrid_node
from app.agent.rewrite import rewrite_node
from app.agent.generate import generate_node

RERANK_SCORE_THRESHOLD = 0.0


def retrieval_fanout(state: dict) -> dict:
  if state["route"] == "graph":
    return graph_node(state)
  elif state["route"] == "hybrid":
    return hybrid_node(state)
  else:
    state = graph_node(state)
    state = hybrid_node(state)
    return state


def needs_retry(state: dict) -> bool:
  exhausted = state["retry_count"] >= state["max_retries"]
  if exhausted:
    return False

  if state["route"] in ("graph", "both"):
    if not state.get("graph_results"):
      return True

  if state["route"] in ("hybrid", "both"):
    hybrid = state.get("hybrid_results") or []
    if not hybrid:
      return True
    top_score = max((r.get("rerank_score", -999)
                    for r in hybrid), default=-999)
    if top_score < RERANK_SCORE_THRESHOLD:
      return True

  return False


def correction_check(state: dict) -> str:
  return "retry" if needs_retry(state) else "generate"


def build_agent():
  workflow = StateGraph(AgentState)

  workflow.add_node("router", router_node)
  workflow.add_node("retrieve", retrieval_fanout)
  workflow.add_node("rewrite", rewrite_node)
  workflow.add_node("generate", generate_node)

  workflow.set_entry_point("router")
  workflow.add_edge("router", "retrieve")
  workflow.add_conditional_edges("retrieve", correction_check, {
      "retry": "rewrite",
      "generate": "generate",
  })
  workflow.add_edge("rewrite", "router")
  workflow.add_edge("generate", END)

  return workflow.compile()


agent = build_agent()
