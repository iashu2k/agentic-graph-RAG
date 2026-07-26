from app.services.graph_db import graph_db


def audit():
  print("=== Node counts by label ===")
  for label in ["Company", "Filing", "Section", "Disclosure"]:
    result = graph_db.execute_read(f"MATCH (n:{label}) RETURN count(n) AS c")
    print(f"{label}: {result[0]['c']}")

  print("\n=== Disclosure count by category ===")
  result = graph_db.execute_read("""
        MATCH (d:Disclosure)
        RETURN d.category AS category, count(*) AS c
        ORDER BY c DESC
    """)
  for r in result:
    print(dict(r))

  print("\n=== Relationship counts ===")
  result = graph_db.execute_read("""
        MATCH ()-[r]->()
        RETURN type(r) AS rel_type, count(*) AS c
        ORDER BY c DESC
    """)
  for r in result:
    print(dict(r))

  print("\n=== Sample near-duplicate check (same filing, similar length descriptions) ===")
  result = graph_db.execute_read("""
        MATCH (f:Filing)-[:DISCLOSES]->(d:Disclosure)
        WITH f.filing_id AS filing, count(d) AS disclosure_count
        RETURN filing, disclosure_count
        ORDER BY disclosure_count DESC
        LIMIT 10
    """)
  for r in result:
    print(dict(r))


if __name__ == "__main__":
  audit()
  graph_db.close()
