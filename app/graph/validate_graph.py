
from pathlib import Path
from app.services.graph_db import graph_db


VALIDATION_PROBES = [
    {
        "question": "Intel restructuring Q3 2022",
        "cypher": """
            MATCH (c:Company {ticker:'Intel'})-[:FILED]->(f:Filing)-[:DISCLOSES]->(d:Disclosure)
            WHERE f.fiscal_quarter='Q3' AND f.fiscal_year=2022
              AND d.category='restructuring'
            RETURN d.description LIMIT 5
        """,
    },
    {
        "question": "Microsoft legal proceedings any quarter",
        "cypher": """
            MATCH (c:Company {ticker:'Microsoft'})-[:FILED]->(f:Filing)-[:DISCLOSES]->(d:Disclosure)
            WHERE d.category='legal_proceeding'
            RETURN f.filing_id, d.description LIMIT 5
        """,
    },
    {
        "question": "Apple tax disclosures 2023",
        "cypher": """
            MATCH (c:Company {ticker:'Apple'})-[:FILED]->(f:Filing)-[:DISCLOSES]->(d:Disclosure)
            WHERE f.fiscal_year=2023 AND d.category='tax'
            RETURN f.filing_id, d.description LIMIT 5
        """,
    },
    {
        "question": "NVIDIA segment reporting",
        "cypher": """
            MATCH (c:Company {ticker:'NVIDIA'})-[:FILED]->(f:Filing)-[:DISCLOSES]->(d:Disclosure)
            WHERE d.category='segment'
            RETURN f.filing_id, d.description LIMIT 5
        """,
    },
    {
        "question": "Amazon financial metrics Q1 2023",
        "cypher": """
            MATCH (c:Company {ticker:'Amazon'})-[:FILED]->(f:Filing)-[:DISCLOSES]->(d:Disclosure)
            WHERE f.fiscal_quarter='Q1' AND f.fiscal_year=2023
              AND d.category='financial_metric'
            RETURN d.description LIMIT 5
        """,
    },
]


def run_validation():
  hits, misses = 0, 0
  for probe in VALIDATION_PROBES:
    records = graph_db.execute_read(probe["cypher"])
    status = "HIT" if records else "MISS"
    print(f"\n=== {probe['question']} — {status} ({len(records)} results) ===")
    for r in records[:3]:
      print(dict(r))
    hits += 1 if records else 0
    misses += 1 if not records else 0

  print(
    f"\nSummary: {hits} hit, {misses} miss out of {len(VALIDATION_PROBES)}")


if __name__ == "__main__":
  run_validation()
  graph_db.close()
