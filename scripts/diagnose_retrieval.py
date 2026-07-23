from app.retrieval.hybrid_search import hybrid_retrieve
from app.retrieval.vector_search import vector_search
import sys
import os
import csv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def diagnose(csv_path):
  with open(csv_path) as f:
    queries = list(csv.DictReader(f))

  for q in queries:
    query_text = q["query"]
    print(f"\n{'=' * 80}\nQUERY: {query_text}")
    print(f"Expected: {q['expected_company']} {q['expected_quarter']} FY{q['expected_year']} | keyword='{q['expected_section_keyword']}'")

    print("\n-- Vector-only top 5 --")
    for r in vector_search(query_text, top_k=5):
      print(
        f"  {r['company']} {r['fiscal_quarter']} FY{r['fiscal_year']} | {r['section']}")

    print("\n-- Hybrid+rerank top 5 --")
    for r in hybrid_retrieve(query_text, final_k=5):
      print(f"  {r['company']} {r['fiscal_quarter']} FY{r['fiscal_year']} | {r['section']} | score={r.get('rerank_score', 0):.3f}")


if __name__ == "__main__":
  print("\n\n########## BASIC QUERIES ##########")
  diagnose("data/eval/sample_queries.csv")
  print("\n\n########## HARD QUERIES ##########")
  diagnose("data/eval/sample_queries_hard.csv")
