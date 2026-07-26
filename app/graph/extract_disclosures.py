import json
import time
import re
from pathlib import Path
from groq import Groq
from app.config import settings
from app.services.db import get_conn
from app.services.graph_db import graph_db

client = Groq(api_key=settings.groq_api_key)

CHECKPOINT_PATH = Path(__file__).parent / ".processed_ids.txt"

EXTRACTION_PROMPT = """Extract distinct factual disclosures from this SEC 10-Q excerpt.
Classify each into ONE category: risk_factor, restructuring, financial_metric, legal_proceeding, tax, segment, other.

Return ONLY valid JSON in this exact schema, no extra text:
{{"disclosures": [{{"description": "short factual description", "category": "one of the categories above"}}]}}
If nothing material is disclosed, return {{"disclosures": []}}.

Excerpt:
{content}
"""


def load_processed_ids() -> set[int]:
  if not CHECKPOINT_PATH.exists():
    return set()
  with open(CHECKPOINT_PATH) as f:
    return set(int(line.strip()) for line in f if line.strip())


def mark_processed(chunk_id: int):
  with open(CHECKPOINT_PATH, "a") as f:
    f.write(f"{chunk_id}\n")


def extract_disclosures(content: str) -> list[dict]:
  resp = client.chat.completions.create(
      model="llama-3.3-70b-versatile",
      messages=[
        {"role": "user", "content": EXTRACTION_PROMPT.format(content=content[:3000])}],
      response_format={"type": "json_object"},
      temperature=0,
  )
  try:
    data = json.loads(resp.choices[0].message.content)
    return data.get("disclosures", [])
  except json.JSONDecodeError:
    return []


def write_disclosure(filing_id: str, description: str, category: str):
  graph_db.execute_write(
      """
        MATCH (f:Filing {filing_id: $filing_id})
        MERGE (dc:Disclosure {description: $description})
        SET dc.category = $category
        MERGE (f)-[:DISCLOSES]->(dc)
        """,
      {
          "filing_id": filing_id,
          "description": description,
          "category": category,
      },
  )


def parse_retry_seconds(error_message: str) -> float | None:
  match = re.search(r"try again in (\d+)m([\d.]+)s", error_message)
  if match:
    minutes, seconds = match.groups()
    return int(minutes) * 60 + float(seconds)
  match = re.search(r"try again in ([\d.]+)s", error_message)
  if match:
    return float(match.group(1))
  return None


def is_daily_quota_error(error_message: str) -> bool:
  return "tokens per day (TPD)" in error_message


def run_full_extraction():
  conn = get_conn()
  cur = conn.cursor()
  cur.execute("""
        SELECT id, company, fiscal_year, fiscal_quarter, content
        FROM chunks
        WHERE company IS NOT NULL
        ORDER BY id
    """)
  rows = cur.fetchall()
  cur.close()
  conn.close()

  processed = load_processed_ids()
  total = len(rows)
  skipped = 0
  done = 0

  for chunk_id, company, fiscal_year, fiscal_quarter, content in rows:
    if chunk_id in processed:
      skipped += 1
      continue

    filing_id = f"{company}-{fiscal_year}-{fiscal_quarter}"

    disclosures = None
    for attempt in range(3):
      try:
        disclosures = extract_disclosures(content)
        break
      except Exception as e:
        error_str = str(e)
        if is_daily_quota_error(error_str):
          wait = parse_retry_seconds(error_str)
          print(f"\nDaily token quota exhausted at chunk {chunk_id}.")
          print(
            f"Processed {done} new chunks this run ({skipped} previously done).")
          print(
            f"Groq says retry in ~{wait:.0f}s, but quota will likely stay tight all day.")
          print(
            "Stopping cleanly. Rerun this script later or tomorrow to resume from checkpoint.")
          graph_db.close()
          return
        print(f"Chunk {chunk_id} attempt {attempt + 1} failed: {e}")
        time.sleep(2 ** attempt)

    if disclosures is None:
      print(f"Chunk {chunk_id}: giving up after 3 attempts, skipping")
      continue

    for d in disclosures:
      description = d.get("description")
      category = d.get("category", "other")
      if not description:
        continue
      write_disclosure(filing_id, description, category)

    mark_processed(chunk_id)
    done += 1

    if done % 50 == 0:
      print(
        f"Progress: {done + skipped}/{total} chunks processed ({skipped} skipped)")

    time.sleep(0.6)

  print(
    f"\nExtraction complete. Processed {done} new chunks, skipped {skipped}, out of {total} total.")


if __name__ == "__main__":
  run_full_extraction()
  graph_db.close()
