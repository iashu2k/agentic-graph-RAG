from fastapi import FastAPI
from pydantic import BaseModel
from app.retrieval.hybrid_search import hybrid_retrieve
from app.services.llm_client import generate_answer

app = FastAPI(title="DeepFile - Phase 2 Hybrid Search RAG")


class QueryRequest(BaseModel):
  question: str
  top_k: int = 5


@app.post("/query")
def query(req: QueryRequest):
  chunks = hybrid_retrieve(req.question, final_k=req.top_k)
  answer = generate_answer(req.question, chunks)
  return {
      "answer": answer,
      "sources": [
          {"company": c["company"], "filing_type": c["filing_type"],
           "fiscal_year": c["fiscal_year"], "fiscal_quarter": c["fiscal_quarter"],
           "section": c["section"], "rerank_score": round(c.get("rerank_score", 0), 3)}
          for c in chunks
      ]
  }


@app.get("/health")
def health():
  return {"status": "ok"}
