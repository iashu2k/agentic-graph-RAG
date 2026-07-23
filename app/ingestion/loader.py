from app.services.db import get_conn
from app.ingestion.embedder import embed_texts


def load_chunks(chunks: list[dict], company: str, filing_type: str, fiscal_year: int, fiscal_quarter: str):
  search_texts = [
      f"{company} {filing_type} {fiscal_quarter} {fiscal_year}. Section: {c['section']}. {c['content']}"
      for c in chunks
  ]
  embeddings = embed_texts(search_texts)

  conn = get_conn()
  cur = conn.cursor()
  for chunk, search_text, emb in zip(chunks, search_texts, embeddings):
    cur.execute(
        """INSERT INTO chunks (company, filing_type, fiscal_year, fiscal_quarter, section, content, search_text, embedding)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
        (company, filing_type, fiscal_year, fiscal_quarter,
         chunk["section"], chunk["content"], search_text, emb)
    )
  conn.commit()
  cur.close()
  conn.close()
  print(
    f"Loaded {len(chunks)} chunks for {company} {filing_type} {fiscal_year} {fiscal_quarter}")
