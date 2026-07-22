import psycopg2
from app.config import settings


def get_conn():
  return psycopg2.connect(settings.database_url)


def init_db():
  conn = get_conn()
  cur = conn.cursor()
  cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
  cur.execute(f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            company TEXT,
            filing_type TEXT,
            fiscal_year INT,
            fiscal_quarter TEXT,
            section TEXT,
            content TEXT,
            embedding VECTOR({settings.embedding_dim})
        );
    """)
  cur.execute(
    "CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);")
  conn.commit()
  cur.close()
  conn.close()


if __name__ == "__main__":
  init_db()
  print("DB initialized.")
