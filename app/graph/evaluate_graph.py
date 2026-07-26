from app.services.graph_db import graph_db

TEST_CASES = [
    {
        "name": "Intel restructuring Q3 2022",
        "query": """
        MATCH (c:Company {ticker: 'Intel'})-[:FILED]->(f:Filing)-[:DISCLOSES]->(d:Disclosure)
        WHERE f.fiscal_quarter = 'Q3' AND f.fiscal_year = 2022
          AND d.category = 'restructuring'
        RETURN d.description
    """,
    },
    {
        "name": "Microsoft section titles Q2 2023",
        "query": """
            MATCH (c:Company {ticker: 'Microsoft'})-[:FILED]->(f:Filing)-[:HAS_SECTION]->(s:Section)
            WHERE f.fiscal_quarter = 'Q2' AND f.fiscal_year = 2023
            RETURN s.name
        """,
    },
]

if __name__ == "__main__":
  for case in TEST_CASES:
    records = graph_db.execute_read(case["query"])
    print(f"\n=== {case['name']} ===")
    print(f"Hits: {len(records)}" if records else "MISS: no results")
    for r in records[:5]:
      print(dict(r))
  graph_db.close()
