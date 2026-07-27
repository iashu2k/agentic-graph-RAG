"""
app/eval/dataset_builder.py

Builds the Phase 5 evaluation set from two sources:

1. qna_data.csv (195 human-reviewed Q&A pairs) - provides "simple" and "hybrid"
   tier questions using the pre-existing, human-assigned `Question Type` column
   as ground truth for difficulty, rather than a keyword heuristic:

       Single-Doc Single-Chunk RAG  -> simple        (one fact, one filing)
       Single-Doc Multi-Chunk RAG   -> hybrid         (one filing, multiple sections)
       Multi-Doc RAG                -> multi_quarter  (same company, multiple quarters -
                                                        NOT a graph multi-hop case; still
                                                        answerable via hybrid search alone)

2. MULTI_HOP_QUESTIONS - a small, hand-written set of genuine cross-company
   relational questions that only the Phase 3 knowledge graph can answer
   (analogous to the validated "which companies share a risk factor with
   Intel" checkpoint case). qna_data.csv contains zero true multi-hop
   questions, so this tier was authored and verified separately against real
   Neo4j Disclosure nodes.

   Verification process: 10 candidate questions were run as live Cypher
   queries against the graph. 5 were dropped because they matched all 5
   companies trivially (foreign exchange risk, interest rate risk, credit
   risk, tax/jurisdiction risk, legal proceedings in every quarter) - these
   don't discriminate between pipeline quality, since even a naive keyword
   search would surface them. The legal-proceedings query specifically
   returned a false "NO RECORDS" due to a query bug (hardcoded quarter-count
   assumption); a corrected diagnostic confirmed all 5 companies genuinely
   have legal_proceeding disclosures in all 4 quarters, making it trivial
   rather than discriminating - so it was dropped for the same reason as
   the others, not re-included.

   The 5 surviving questions below are genuinely discriminating (partial,
   not universal, company matches) and are verified against live graph
   query results as of 2026-07-26.

Source Docs parsing:
    qna_data.csv's `Source Docs` column (e.g. "*AAPL*", "*2023 Q3 MSFT*") is
    parsed into a normalized list of filing_ids for use as RAGAS context_recall
    /context_precision ground truth. A bare ticker with no quarter (e.g. "*AAPL*")
    means "any/all of that company's filings" and is expanded to all 4 filing_ids
    present in the corpus for that company.
"""
import re
import random
import pandas as pd

QNA_PATH = "data/eval/qna_data.csv"
OUTPUT_PATH = "data/eval/eval_set_phase5.csv"

TIER_MAP = {
    "Single-Doc Single-Chunk RAG": "simple",
    "Single-Doc Multi-Chunk RAG": "hybrid",
    "Multi-Doc RAG": "multi_quarter",
}

TICKER_MAP = {
    "AAPL": "Apple",
    "AMZN": "Amazon",
    "INTC": "Intel",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
}

ALL_QUARTERS = ["2022-Q3", "2023-Q1", "2023-Q2", "2023-Q3"]

# Verified against live Neo4j Cypher queries on 2026-07-26. Each
# `verify_substring`/`verify_query_type` documents exactly how the answer
# was confirmed, so re-verification after any graph rebuild is trivial.
MULTI_HOP_QUESTIONS = [
    {
        "question": "Besides Intel, which other companies disclose supply chain risk?",
        "category": "risk_factor",
        "verify_substring": "supply chain",
        "verified_answer": "Amazon, Apple, Microsoft, NVIDIA",
    },
    {
        "question": "Which companies disclose risk from fluctuations in foreign currency exchange rates affecting intercompany balances?",
        "category": "risk_factor",
        "verify_substring": "intercompany",
        "verified_answer": "Amazon, Microsoft",
    },
    {
        "question": "Which companies mention losses from remeasurement of assets or liabilities denominated in non-functional currencies?",
        "category": "risk_factor",
        "verify_substring": "remeasurement",
        "verified_answer": "Apple, Amazon, Intel",
    },
    {
        "question": "Which companies disclose restructuring charges in more than one fiscal quarter?",
        "category": "restructuring",
        # per-company quarter-count aggregation, not a substring filter
        "verify_substring": None,
        "verified_answer": "Amazon, Intel, NVIDIA",
    },
    {
        "question": "Which companies disclose investing excess cash in AAA-rated money market funds or similar short-term instruments?",
        "category": "segment",
        "verify_substring": "money market",
        "verified_answer": "Intel",
    },
]

# Dropped candidates (not included - non-discriminating, matched all 5
# companies trivially): foreign exchange rate risk, interest rate risk on
# fixed income/marketable securities, credit risk on term debt, tax risk
# tied to jurisdictional interpretations, legal proceedings risk in every
# quarter (confirmed via diagnostic: all 5 companies have legal_proceeding
# disclosures in all 4 quarters with zero gaps - genuinely universal, not
# a query bug in the final diagnostic, so correctly excluded as trivial).


def parse_source_docs(raw: str) -> list[str]:
  """Normalize a Source Docs cell like '*2023 Q3 MSFT*' or '*AAPL*' into a
  list of filing_ids matching the graph's Filing.filing_id format
  ('{company}-{fiscal_year}-{fiscal_quarter}', e.g. 'Intel-2022-Q3').

  A bare ticker with no quarter expands to all 4 quarters for that company.
  """
  cleaned = raw.replace("*", "").strip()
  match = re.match(r"(\d{4})\s+(Q\d)\s+([A-Z]{3,5})", cleaned)
  if match:
    year, quarter, ticker = match.groups()
    company = TICKER_MAP.get(ticker, ticker)
    return [f"{company}-{year}-{quarter}"]

  ticker = cleaned
  company = TICKER_MAP.get(ticker, ticker)
  return [f"{company}-{q}" for q in ALL_QUARTERS]


def build_qna_tiers(qna_path: str = QNA_PATH, n_per_tier: int = 25, seed: int = 42) -> pd.DataFrame:
  df = pd.read_csv(qna_path)
  df = df.rename(columns={
      "Question": "question",
      "Answer": "ground_truth",
      "Source Docs": "source_docs_raw",
      "Question Type": "question_type",
  })
  df["tier"] = df["question_type"].map(TIER_MAP)
  df["ground_truth_docs"] = df["source_docs_raw"].apply(parse_source_docs)

  random.seed(seed)
  sampled = []
  for tier in ["simple", "hybrid", "multi_quarter"]:
    tier_df = df[df["tier"] == tier]
    n = min(n_per_tier, len(tier_df))
    sampled.append(tier_df.sample(n=n, random_state=seed))

  result = pd.concat(sampled).reset_index(drop=True)
  return result[["question", "ground_truth", "tier", "ground_truth_docs"]]


def build_multi_hop_tier() -> pd.DataFrame:
  """Returns the 5 verified multi-hop questions as a DataFrame.
  ground_truth_docs is left empty since multi-hop ground truth is a
  company list (verified_answer), not a set of filing_ids.
  """
  rows = []
  for item in MULTI_HOP_QUESTIONS:
    rows.append({
        "question": item["question"],
        "ground_truth": item["verified_answer"],
        "tier": "multi_hop",
        "ground_truth_docs": [],
    })
  return pd.DataFrame(rows)


def build_eval_set(n_per_tier: int = 25, seed: int = 42) -> pd.DataFrame:
  qna_tiers = build_qna_tiers(n_per_tier=n_per_tier, seed=seed)
  multi_hop = build_multi_hop_tier()
  return pd.concat([qna_tiers, multi_hop], ignore_index=True)


if __name__ == "__main__":
  eval_set = build_eval_set()
  print(eval_set["tier"].value_counts())
  eval_set.to_csv(OUTPUT_PATH, index=False)
  print(f"\\nWrote {len(eval_set)} questions to {OUTPUT_PATH}")
