from app.services.db import get_conn
from app.ingestion.embedder import embed_texts


def search(query: str, top_k: int = 5):
  query_emb = embed_texts([query])[0]

  conn = get_conn()
  cur = conn.cursor()
  cur.execute(
      """SELECT company, filing_type, fiscal_year, fiscal_quarter, section, content,
                  1 - (embedding <=> %s::vector) AS similarity
           FROM chunks
           ORDER BY embedding <=> %s::vector
           LIMIT %s""",
      (query_emb, query_emb, top_k)
  )
  rows = cur.fetchall()
  cur.close()
  conn.close()

  return [
      {
          "company": r[0], "filing_type": r[1], "fiscal_year": r[2], "fiscal_quarter": r[3],
          "section": r[4], "content": r[5], "similarity": float(r[6])
      }
      for r in rows
  ]
