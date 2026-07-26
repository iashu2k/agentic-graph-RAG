from app.graph.bootstrap import bootstrap_from_postgres
from app.graph.extract_disclosures import run_full_extraction
from app.services.graph_db import graph_db

if __name__ == "__main__":
  print("Step 1: Bootstrapping structural graph from Postgres metadata...")
  bootstrap_from_postgres()
  print("Step 2: Extracting disclosures via LLM (full corpus)...")
  run_full_extraction()
  graph_db.close()
  print("Graph build complete.")
