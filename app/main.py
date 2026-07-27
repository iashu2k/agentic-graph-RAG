from fastapi import FastAPI
from pydantic import BaseModel
from app.agent.graph_builder import agent
from app.observability.tracing import langfuse, traced_config, flag_output


app = FastAPI(title="DeepFile", version="0.5.0")


class QueryRequest(BaseModel):
  question: str


class QueryResponse(BaseModel):
  answer: str
  route: str
  retry_count: int
  original_question: str | None = None
  rewritten_question: str | None = None
  rewrite_reasoning: str | None = None
  trace_id: str | None = None


@app.get("/health")
def health():
  return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
  with langfuse.start_as_current_span(
      name="deepfile-query",
      input={"question": request.question},
  ) as span:
    config = traced_config(pipeline="agentic")

    result = agent.invoke(
        {
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
        },
        config=config,
    )

    trace_id = span.trace_id

    span.update(
        output={"answer": result["answer"], "route": result["route"]},
        metadata={
            "route": result["route"],
            "retry_count": result["retry_count"],
            "rewritten": result["retry_count"] > 0,
        },
    )

  contexts = result.get("graph_results", []) + result.get("hybrid_results", [])
  flag_output(trace_id=trace_id, answer=result["answer"], contexts=contexts)

  return QueryResponse(
      answer=result["answer"],
      route=result["route"],
      retry_count=result["retry_count"],
      original_question=result.get("original_question"),
      rewritten_question=result["question"] if result["retry_count"] > 0 else None,
      rewrite_reasoning=result.get("rewrite_reasoning"),
      trace_id=trace_id,
  )
