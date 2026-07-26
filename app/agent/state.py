from typing import TypedDict, Literal, Optional


class AgentState(TypedDict):
  question: str
  original_question: Optional[str]
  rewrite_reasoning: Optional[str]
  route: Optional[Literal["graph", "hybrid", "both"]]
  cypher_query: Optional[str]
  graph_results: list[dict]
  hybrid_results: list[dict]
  context: str
  answer: str
  retry_count: int
  max_retries: int
