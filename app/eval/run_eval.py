"""
app/eval/run_eval.py

Orchestrator: runs every question in the Phase 5 eval set through all three
pipeline versions, capturing answer + retrieved contexts for each, and writes
one CSV per pipeline version to data/eval/results/.

Two resilience features, mirroring the Phase 3 extraction script's approach
to Groq's per-model TPD quota:

1. Checkpointing - progress is saved incrementally to a partial CSV as each
   question completes, and a run resumes from the last completed row rather
   than restarting from scratch. Critical here because llama-3.3-70b-versatile's
   on-demand TPD limit (100,000 tokens/day) is much tighter than the lite
   model's 500K limit seen in Phase 3, and can be exhausted mid-run on a
   single ~80-question pass.

2. Model fallback rotation - on a 429 rate_limit_exceeded response, the
   generation model is rotated to the next model in FALLBACK_MODELS (each
   Groq model has its own separate TPD quota pool, per the Phase 3 finding),
   and the same question is retried once before giving up and recording an
   empty result.

Model swapping is done via app/services/llm_client.py's set_model()/
get_model() functions - NOT via settings.model, since app/config.py's
Settings class has no `model` field (the model was previously hardcoded
directly inside generate_answer()).

These CSVs are the direct input to app/eval/score_ragas.py.
"""
import pandas as pd
import time
from pathlib import Path
from app.eval.pipelines import PIPELINES
from app.services import llm_client

EVAL_SET_PATH = "data/eval/eval_set_phase5.csv"
OUTPUT_DIR = Path("data/eval/results")

# Same fallback chain used in Phase 3's extract_disclosures.py - each model
# draws from a separate Groq TPD quota pool, so rotating unblocks same-day
# rather than waiting for a ~24h reset.
FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",

]


def is_rate_limit_error(e: Exception) -> bool:
  msg = str(e)
  return "429" in msg or "rate_limit_exceeded" in msg


def run_fn_with_fallback(run_fn, question: str, max_model_attempts: int = len(FALLBACK_MODELS)) -> dict:
  last_error = None
  for attempt in range(max_model_attempts):
    try:
      return run_fn(question)
    except Exception as e:
      last_error = e
      if is_rate_limit_error(e) and attempt < max_model_attempts - 1:
        next_model = FALLBACK_MODELS[attempt + 1]
        print(
          f"    Rate limited on {llm_client.get_model()} -> switching to {next_model}")
        llm_client.set_model(next_model)
        time.sleep(2)
        continue
      else:
        break
  print(f"    ERROR (exhausted fallbacks): {last_error}")
  return {"answer": "", "contexts": [], "retrieved_filing_ids": [], "route": None}


def run_pipeline(pipeline_name: str, run_fn, eval_df: pd.DataFrame) -> None:
  out_path = OUTPUT_DIR / f"{pipeline_name}_results.csv"
  checkpoint_path = OUTPUT_DIR / f"{pipeline_name}_checkpoint.csv"

  done_questions = set()
  rows = []
  if checkpoint_path.exists():
    existing = pd.read_csv(checkpoint_path)
    rows = existing.to_dict("records")
    done_questions = set(existing["question"])
    print(f"[{pipeline_name}] Resuming - {len(done_questions)} already completed")

  for i, row in eval_df.iterrows():
    question = row["question"]
    if question in done_questions:
      continue

    print(f"[{pipeline_name}] {i + 1}/{len(eval_df)}: {question[:60]}")
    original_model = llm_client.get_model()
    result = run_fn_with_fallback(run_fn, question)
    llm_client.set_model(original_model)  # reset for next question

    rows.append({
        "question": question,
        "ground_truth": row["ground_truth"],
        "tier": row["tier"],
        "answer": result["answer"],
        "contexts": result["contexts"],
    })

    pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
    time.sleep(1.0)  # throttle for Groq TPM/TPD limits

  final_df = pd.DataFrame(rows)
  final_df.to_csv(out_path, index=False)
  print(f"[{pipeline_name}] Wrote {len(final_df)} rows to {out_path}")


def run_all():
  eval_df = pd.read_csv(EVAL_SET_PATH)
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

  for pipeline_name, run_fn in PIPELINES.items():
    run_pipeline(pipeline_name, run_fn, eval_df)


if __name__ == "__main__":
  run_all()
