from fastapi import FastAPI
from pydantic import BaseModel
from app.retrieval.vector_search import search
from app.services.llm_client import generate_answer

app = FastAPI(title="DeepFile - Phase 1 Baseline RAG")


class QueryRequest(BaseModel):
  question: str
  top_k: int = 5


@app.post("/query")
def query(req: QueryRequest):
  chunks = search(req.question, top_k=req.top_k)
  answer = generate_answer(req.question, chunks)
  return {
      "answer": answer,
      "sources": [
          {"company": c["company"], "filing_type": c["filing_type"],
           "fiscal_year": c["fiscal_year"], "fiscal_quarter": c["fiscal_quarter"],
           "section": c["section"], "similarity": round(c["similarity"], 3)}
          for c in chunks
      ]
  }


@app.get("/health")
def health():
  return {"status": "ok"}
