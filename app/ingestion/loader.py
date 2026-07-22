from app.services.db import get_conn
from app.ingestion.embedder import embed_texts


def load_chunks(chunks: list[dict], company: str, filing_type: str, fiscal_year: int, fiscal_quarter: str):
  texts = [c["content"] for c in chunks]
  embeddings = embed_texts(texts)

  conn = get_conn()
  cur = conn.cursor()
  for chunk, emb in zip(chunks, embeddings):
    cur.execute(
        """INSERT INTO chunks (company, filing_type, fiscal_year, fiscal_quarter, section, content, embedding)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
        (company, filing_type, fiscal_year, fiscal_quarter,
         chunk["section"], chunk["content"], emb)
    )
  conn.commit()
  cur.close()
  conn.close()
  print(
    f"Loaded {len(chunks)} chunks for {company} {filing_type} {fiscal_year} {fiscal_quarter}")
