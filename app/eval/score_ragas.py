"""
app/eval/score_ragas.py

Loads each pipeline's results CSV (from run_eval.py), scores it with RAGAS
(faithfulness, context_precision, context_recall, answer_correctness), and
produces:
    1. data/eval/ragas_scores.csv        - per-pipeline aggregate scores
    2. data/eval/ragas_scores_by_tier.csv - per-pipeline, per-tier scores
    3. data/eval/ragas_scores_raw.csv     - every scored row, for debugging

Resilience features:

1. Row-level chunking - each pipeline's dataset is scored in small batches
   (BATCH_SIZE rows at a time) rather than one evaluate() call over the
   whole dataset, so a quota exhaustion mid-run only loses the current
   batch's progress, not everything.

2. Checkpointing - completed batches are saved incrementally to a
   per-pipeline checkpoint CSV; reruns skip already-scored questions.
   IMPORTANT: RAGAS's result.to_pandas() renames columns internally
   (question -> user_input, answer -> response, contexts ->
   retrieved_contexts, in newer ragas versions) - checkpoints are
   normalized back to the original question/answer/contexts names
   BEFORE writing, so resume logic (`existing["question"]`) always works
   regardless of which ragas column-naming version produced the checkpoint.

3. Judge model fallback rotation - on a 429, rotates through
   ragas_judge.JUDGE_FALLBACK_CHAIN and retries the current batch once per
   model before giving up and recording NaN scores for that batch.
"""
import app.eval._vertexai_shim  # noqa: F401 - must be first, fixes ragas import bug
import ast
import time
import pandas as pd
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from app.eval import ragas_judge
from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall,
    answer_correctness,
)

RESULTS_DIR = Path("data/eval/results")
SCORES_DIR = Path("data/eval")
PIPELINES = ["baseline", "hybrid", "agentic"]
METRICS = [faithfulness, context_precision, context_recall, answer_correctness]
METRIC_COLS = ["faithfulness", "context_precision",
               "context_recall", "answer_correctness"]

BATCH_SIZE = 10  # rows per evaluate() call - keeps quota loss small on a 429

# Maps ragas's internal output column names (varies by version) back to
# this project's canonical names, so checkpoint files are always consistent
# regardless of which ragas version wrote them.
RAGAS_COLUMN_ALIASES = {
    "user_input": "question",
    "response": "answer",
    "retrieved_contexts": "contexts",
    "reference": "ground_truth",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
  rename_map = {k: v for k, v in RAGAS_COLUMN_ALIASES.items()
                if k in df.columns}
  return df.rename(columns=rename_map)


def is_rate_limit_error(e: Exception) -> bool:
  msg = str(e)
  return "429" in msg or "rate_limit_exceeded" in msg


def load_for_ragas(pipeline_name: str) -> pd.DataFrame:
  df = pd.read_csv(RESULTS_DIR / f"{pipeline_name}_results.csv")
  df["contexts"] = df["contexts"].apply(
      lambda x: ast.literal_eval(x) if isinstance(x, str) else x
  )
  return df[["question", "answer", "contexts", "ground_truth", "tier"]]


def score_batch_with_fallback(batch_df: pd.DataFrame, max_attempts: int = 3) -> pd.DataFrame:
  original_questions = batch_df["question"].tolist()
  original_tiers = batch_df["tier"].tolist()
  ds = Dataset.from_pandas(batch_df.reset_index(drop=True))
  last_error = None
  for attempt in range(max_attempts):
    try:
      result = evaluate(
          ds,
          metrics=METRICS,
          llm=ragas_judge.get_ragas_llm(),
          embeddings=ragas_judge.get_ragas_embeddings(),
          run_config=RunConfig(max_workers=1, timeout=300),
      )
      scored = normalize_columns(result.to_pandas())
      # ragas output rows may not preserve original tier column - reattach
      # positionally, since row order is preserved by evaluate().
      if "tier" not in scored.columns:
        scored["tier"] = original_tiers
      if "question" not in scored.columns:
        scored["question"] = original_questions
      return scored
    except Exception as e:
      last_error = e
      if is_rate_limit_error(e) and attempt < max_attempts - 1:
        next_model = ragas_judge.JUDGE_FALLBACK_CHAIN[
            (attempt + 1) % len(ragas_judge.JUDGE_FALLBACK_CHAIN)
        ]
        print(f"    Judge rate limited -> switching to {next_model}")
        ragas_judge.set_judge_model(next_model)
        time.sleep(3)
        continue
      break
  print(f"    Batch failed after fallbacks: {last_error}")
  failed = batch_df.copy()
  for col in METRIC_COLS:
    failed[col] = float("nan")
  return failed


def score_pipeline(pipeline_name: str) -> pd.DataFrame:
  checkpoint_path = SCORES_DIR / f"{pipeline_name}_ragas_checkpoint.csv"
  df = load_for_ragas(pipeline_name)

  done_questions = set()
  scored_rows = []
  if checkpoint_path.exists():
    existing = normalize_columns(pd.read_csv(checkpoint_path))
    if "question" in existing.columns:
      scored_rows = existing.to_dict("records")
      done_questions = set(existing["question"])
      print(f"[{pipeline_name}] Resuming - {len(done_questions)} already scored")
    else:
      print(
        f"[{pipeline_name}] Checkpoint unreadable (no question column) - starting fresh")

  remaining = df[~df["question"].isin(done_questions)].reset_index(drop=True)

  for start in range(0, len(remaining), BATCH_SIZE):
    batch = remaining.iloc[start:start + BATCH_SIZE]
    print(
      f"[{pipeline_name}] Scoring rows {start}-{start + len(batch)}/{len(remaining)}")
    scored_batch = score_batch_with_fallback(batch)
    scored_rows.extend(scored_batch.to_dict("records"))
    pd.DataFrame(scored_rows).to_csv(checkpoint_path, index=False)
    time.sleep(2)

  result_df = pd.DataFrame(scored_rows)
  result_df["pipeline"] = pipeline_name
  return result_df


def score_all() -> pd.DataFrame:
  all_scores = []
  for pipeline_name in PIPELINES:
    print(f"Scoring {pipeline_name}...")
    scored_df = score_pipeline(pipeline_name)
    all_scores.append(scored_df)

  combined = pd.concat(all_scores, ignore_index=True)
  combined.to_csv(SCORES_DIR / "ragas_scores_raw.csv", index=False)

  summary = (
      combined.groupby("pipeline")[METRIC_COLS]
      .mean()
      .round(3)
      .reindex(PIPELINES)
  )
  summary.to_csv(SCORES_DIR / "ragas_scores.csv")
  print(summary)

  tier_summary = (
      combined.groupby(["pipeline", "tier"])[METRIC_COLS]
      .mean()
      .round(3)
  )
  tier_summary.to_csv(SCORES_DIR / "ragas_scores_by_tier.csv")

  return summary


if __name__ == "__main__":
  score_all()
