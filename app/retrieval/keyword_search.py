from app.services.db import get_conn


def keyword_search(query: str, top_k: int = 20):
  conn = get_conn()
  cur = conn.cursor()
  cur.execute(
      """SELECT id, company, filing_type, fiscal_year, fiscal_quarter, section, content, search_text,
                  ts_rank(content_tsv, plainto_tsquery('english', %s)) AS rank
           FROM chunks
           WHERE content_tsv @@ plainto_tsquery('english', %s)
           ORDER BY rank DESC
           LIMIT %s""",
      (query, query, top_k)
  )
  rows = cur.fetchall()
  cur.close()
  conn.close()

  return [
      {"id": r[0], "company": r[1], "filing_type": r[2], "fiscal_year": r[3],
       "fiscal_quarter": r[4], "section": r[5], "content": r[6], "search_text": r[7],
       "rank_score": float(r[8])}
      for r in rows
  ]
