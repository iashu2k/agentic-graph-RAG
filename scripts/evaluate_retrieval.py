from app.retrieval.hybrid_search import hybrid_retrieve
from app.retrieval.vector_search import vector_search
import sys
import os
import csv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def is_hit(results, expected_company, expected_quarter, expected_year, expected_keywords):
  keywords = [k.strip().lower() for k in expected_keywords.split("|")]
  for r in results[:5]:
    if (r["company"].lower() == expected_company.lower()
        and r["fiscal_quarter"] == expected_quarter
        and r["fiscal_year"] == int(expected_year)
            and any(kw in r["section"].lower() for kw in keywords)):
      return True
  return False


def run_eval(csv_path, label):
  with open(csv_path) as f:
    queries = list(csv.DictReader(f))

  vector_hits = 0
  hybrid_hits = 0
  rows = []

  for q in queries:
    query_text = q["query"]
    kw = q["expected_section_keyword"]

    vec_results = vector_search(query_text, top_k=5)
    v_hit = is_hit(vec_results, q["expected_company"],
                   q["expected_quarter"], q["expected_year"], kw)
    vector_hits += v_hit

    hyb_results = hybrid_retrieve(query_text, final_k=5)
    h_hit = is_hit(hyb_results, q["expected_company"],
                   q["expected_quarter"], q["expected_year"], kw)
    hybrid_hits += h_hit

    rows.append(
      {"query": query_text, "vector_only_hit": v_hit, "hybrid_hit": h_hit})
    print(f"{'✓' if h_hit else '✗'} (hybrid) | {'✓' if v_hit else '✗'} (vector) | {query_text}")

  n = len(queries)
  print(f"\n[{label}] Vector-only precision@5: {vector_hits}/{n} ({vector_hits / n * 100:.0f}%)")
  print(f"[{label}] Hybrid+rerank precision@5: {hybrid_hits}/{n} ({hybrid_hits / n * 100:.0f}%)\n")


def main():
  print("=== Basic queries (revenue/income by quarter) ===")
  run_eval("data/eval/sample_queries.csv", "basic")

  print("=== Hard queries (note numbers, EPS, legal terms) ===")
  run_eval("data/eval/sample_queries_hard.csv", "hard")


if __name__ == "__main__":
  main()
