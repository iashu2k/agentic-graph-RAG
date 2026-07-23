from app.config import settings
import psycopg2


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
        search_text TEXT,
        embedding VECTOR({settings.embedding_dim}),
        content_tsv TSVECTOR
    );
""")
  cur.execute(
    "CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);")
  cur.execute(
    "CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN(content_tsv);")
  cur.execute("""
        CREATE OR REPLACE FUNCTION chunks_tsv_trigger() RETURNS trigger AS $$
        BEGIN
          NEW.content_tsv := to_tsvector('english', NEW.content);
          RETURN NEW;
        END
        $$ LANGUAGE plpgsql;
    """)
  cur.execute("""
        DROP TRIGGER IF EXISTS chunks_tsv_update ON chunks;
        CREATE TRIGGER chunks_tsv_update BEFORE INSERT OR UPDATE ON chunks
        FOR EACH ROW EXECUTE FUNCTION chunks_tsv_trigger();
    """)
  conn.commit()
  cur.close()
  conn.close()


if __name__ == "__main__":
  init_db()
  print("DB initialized.")
