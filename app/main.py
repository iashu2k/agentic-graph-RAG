from fastapi import FastAPI
from pydantic import BaseModel
from app.agent.graph_builder import agent

app = FastAPI(title="DeepFile", version="0.4.0")


class QueryRequest(BaseModel):
  question: str


class QueryResponse(BaseModel):
  answer: str
  route: str
  retry_count: int
  original_question: str | None = None
  rewritten_question: str | None = None
  rewrite_reasoning: str | None = None


@app.get("/health")
def health():
  return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
  result = agent.invoke({
      "question": request.question,
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

  return QueryResponse(
      answer=result["answer"],
      route=result["route"],
      retry_count=result["retry_count"],
      original_question=result.get("original_question"),
      rewritten_question=result["question"] if result["retry_count"] > 0 else None,
      rewrite_reasoning=result.get("rewrite_reasoning"),
  )
