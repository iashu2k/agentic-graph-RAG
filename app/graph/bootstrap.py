from app.services.db import get_conn
from app.services.graph_db import graph_db


def bootstrap_from_postgres():
  graph_db.init_constraints()
  conn = get_conn()
  cur = conn.cursor()
  cur.execute("""
        SELECT DISTINCT company, filing_type, fiscal_year, fiscal_quarter, section
        FROM chunks
        WHERE company IS NOT NULL
    """)
  rows = cur.fetchall()

  for company, filing_type, fiscal_year, fiscal_quarter, section in rows:
    filing_id = f"{company}-{fiscal_year}-{fiscal_quarter}"
    graph_db.execute_write(
        """
            MERGE (c:Company {ticker: $company})
            MERGE (f:Filing {filing_id: $filing_id})
            SET f.fiscal_year = $fiscal_year,
                f.fiscal_quarter = $fiscal_quarter,
                f.filing_type = $filing_type
            MERGE (c)-[:FILED]->(f)
            MERGE (s:Section {filing_id: $filing_id, name: $section})
            MERGE (f)-[:HAS_SECTION]->(s)
            """,
        {
            "company": company,
            "filing_id": filing_id,
            "fiscal_year": fiscal_year,
            "fiscal_quarter": fiscal_quarter,
            "filing_type": filing_type,
            "section": section,
        },
    )

  cur.close()
  conn.close()
  print(f"Bootstrapped {len(rows)} distinct filing/section rows.")


if __name__ == "__main__":
  bootstrap_from_postgres()
  graph_db.close()
