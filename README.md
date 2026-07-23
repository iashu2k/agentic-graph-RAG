# DeepFile

**Agentic GraphRAG research assistant over SEC 10-Q filings**

DeepFile combines an LLM agent (LangGraph) that plans/routes/self-corrects retrieval with a knowledge graph (Neo4j) for multi-hop relational reasoning, layered on top of hybrid vector + keyword search (pgvector + Postgres full-text). The domain is SEC 10-Q filings for five major tech companies, where relationships between entities (companies, filings, notes, risk factors) require multi-hop reasoning that plain vector RAG cannot handle.

---

## Architecture Overview

- **Vector store:** pgvector via Supabase (free tier)
- **Keyword search:** Postgres full-text search (`tsvector`/`ts_rank`, BM25-style)
- **Reranker:** Cross-encoder (`ms-marco-MiniLM-L-6-v2`, free/local via sentence-transformers)
- **Knowledge graph:** Neo4j AuraDB (free tier) — planned, Phase 3
- **Agent orchestration:** LangGraph — planned, Phase 4
- **LLM:** Groq (Llama 3.3 70B / Llama 3.1 8B)
- **Evaluation:** RAGAS — planned, Phase 5
- **Observability:** Langfuse — planned, Phase 6
- **API layer:** FastAPI
- **Package manager:** uv

## Build Phases

| Phase | Description | Status |
|---|---|---|
| 0 | Corpus + infra setup (accounts, DBs, project scaffold) | ✅ Complete |
| 1 | Baseline RAG (plain vector search + generation) | ✅ Complete |
| 2 | Hybrid search + reranking (BM25 + RRF + cross-encoder) | ✅ Complete |
| 3 | Knowledge graph layer (entity/relationship extraction into Neo4j) | ⬜ Not started |
| 4 | Agentic router with self-correction loop (LangGraph) | ⬜ Not started |
| 5 | Evaluation harness (RAGAS benchmark) | ⬜ Not started |
| 6 | Observability + guardrails (Langfuse) | ⬜ Not started |
| 7 | Incremental indexing (stretch) | ⬜ Not started |

---

## Phase 0: Corpus and Infrastructure Setup

### Corpus

Instead of self-downloading EDGAR filings, this project uses the **docugami/KG-RAG-datasets** `sec-10-q` dataset directly:

- **Source:** https://github.com/docugami/KG-RAG-datasets (MIT licensed)
- **Contents:** 20 real 10-Q PDF filings for 5 tech companies — AAPL, AMZN, INTC, MSFT, NVDA
- **Coverage:** 4 quarters — 2022 Q3, 2023 Q1, 2023 Q2, 2023 Q3 (10-Q only, no 10-K)
- **Naming convention:** `YYYY QN TICKER.pdf` (e.g., `2023 Q2 AAPL.pdf`)
- **Evaluation set:** `qna_data.csv` — 195 human-reviewed question-answer pairs, used as RAGAS ground truth in Phase 5
- **Location in repo:** `data/raw/sec-10-q/docs/`

Notes:
- All 5 companies are big tech — no cross-sector relational questions are testable with this corpus alone.
- Only 10-Q filings are present — no full fiscal year (10-K) data, limiting certain temporal/annual questions.
- `raw_questions/questions_with_LLM_answers.csv` contains **unverified** GPT-4-Turbo draft answers and should NOT be used as ground truth — only `qna_data.csv` is human-reviewed.

### Free-Tier Infrastructure

**1. Postgres + pgvector — Supabase**
- Project created at supabase.com
- `vector` extension enabled (Database → Extensions)
- `pg_trgm` extension also enabled (optional, for future fuzzy text matching)
- Connection method: **Transaction pooler** (port 6543) — chosen since the FastAPI/agent workload is stateless and short-lived
- Connection string stored in `.env` as `DATABASE_URL`

**2. Knowledge Graph — Neo4j AuraDB**
- Instance `deepfile-kg` created at console.neo4j.io
- Credentials (URI, username, password) downloaded and saved at creation time
- Verified connection via Neo4j Browser with `RETURN 1`
- Credentials stored in `.env` as `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`

**3. LLM Inference — Groq**
- Account created at console.groq.com
- API key generated (Console → API Keys)
- Models used:
  - `llama-3.3-70b-versatile` — generation/reasoning steps
  - `llama-3.1-8b-instant` — lightweight tasks (future: query rewriting, extraction)
- Key stored in `.env` as `GROQ_API_KEY`


### Project Structure

```
deepfile/
├── data/
│ ├── raw/sec-10-q/docs/ # 20 docugami PDF filings
│ ├── processed/ # (future) cleaned/chunked text
│ └── eval/ # sample_queries.csv, sample_queries_hard.csv, qna_data.csv
├── app/
│ ├── main.py # FastAPI entry point
│ ├── config.py # Settings (pydantic-settings)
│ ├── ingestion/
│ │ ├── parser.py # PDF parsing (unstructured, hi_res strategy)
│ │ ├── chunker.py # Section-based chunking with overlap
│ │ ├── embedder.py # sentence-transformers embeddings
│ │ └── loader.py # Writes chunks + embeddings to pgvector
│ ├── retrieval/
│ │ ├── vector_search.py # Cosine similarity search
│ │ ├── keyword_search.py # BM25-style full-text search
│ │ ├── fusion.py # Reciprocal Rank Fusion
│ │ ├── reranker.py # Cross-encoder reranking
│ │ └── hybrid_search.py # Orchestrates vector + keyword + RRF + rerank
│ ├── graph/ # Phase 3 (not yet implemented)
│ ├── agent/ # Phase 4 (not yet implemented)
│ ├── eval/ # Phase 5 (not yet implemented)
│ ├── observability/ # Phase 6 (not yet implemented)
│ ├── api/ # (folder reserved; routes currently in main.py)
│ └── services/
│ ├── llm_client.py # Groq client wrapper
│ └── db.py # Postgres connection + schema init
├── scripts/
│ ├── run_ingestion.py # CLI: loops over all 20 PDFs, parses + embeds + loads
│ ├── evaluate_retrieval.py # Precision@5 comparison: vector-only vs hybrid+rerank
│ └── diagnose_retrieval.py # Dumps actual top-5 results per query for debugging
├── notebooks/
├── tests/
├── .env
├── .gitignore
└── pyproject.toml # uv-managed dependencies
```

### Environment Setup (uv)

```bash
uv init deepfile --python 3.12
cd deepfile

uv add "unstructured[pdf]" fastapi uvicorn psycopg2-binary sqlalchemy pgvector \
       sentence-transformers groq python-dotenv pydantic-settings pymupdf

mkdir -p app/ingestion app/retrieval app/graph app/agent app/eval app/observability app/api app/services
mkdir -p data/raw/sec-10-q/docs data/processed data/eval scripts notebooks tests
touch scripts/__init__.py
```

System dependencies (macOS, required for `unstructured`'s `hi_res` PDF strategy):

```bash
brew install tesseract poppler
```

`.env` file (not committed to git):

```
DATABASE_URL=postgresql://postgres.[project-ref]:[password]@aws-0-[region].pooler.supabase.com:6543/postgres
GROQ_API_KEY=your-groq-key
NEO4J_URI=neo4j+s://[instance-id].databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-neo4j-password
NEO4J_DATABASE=neo4j
AURA_INSTANCEID=your-instance-id
AURA_INSTANCENAME=deepfile-kg
```

---

`app/config.py` declares every field explicitly (pydantic-settings rejects undeclared `.env` variables by default):

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    groq_api_key: str
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    docs_dir: str = "data/raw/sec-10-q/docs"

    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str
    aura_instanceid: str
    aura_instancename: str

    model: str = "llama-3.3-70b-versatile"
    model_lite: str = "llama-3.1-8b-instant"

    class Config:
        env_file = ".env"

settings = Settings()
```

---

## Phase 1: Baseline RAG — COMPLETE ✅

**Goal:** Naive end-to-end pipeline — parse → chunk → embed → store → retrieve → generate — to establish a working demo and a baseline to measure later improvements against.

### Pipeline Steps

1. **Parsing** (`app/ingestion/parser.py`) — `unstructured.partition.pdf.partition_pdf` with `strategy="hi_res"` and `infer_table_structure=True`. Preserves section headers by tracking the most recent `Title`-category element and tagging subsequent text with it. Requires Tesseract (OCR) and Poppler installed locally.
2. **Chunking** (`app/ingestion/chunker.py`) — groups parsed elements by detected section, falling back to character-window splitting (max 1500 chars, ~12.5% overlap) only when a section exceeds the max length.
3. **Embedding** (`app/ingestion/embedder.py`) — `BAAI/bge-small-en-v1.5` via `sentence-transformers`, 384 dimensions, free and open-source.
4. **Storage** (`app/ingestion/loader.py`, `app/services/db.py`) — Postgres `chunks` table with an HNSW index on `embedding` (cosine distance).
5. **Ingestion orchestration** (`scripts/run_ingestion.py`) — auto-discovers all 20 PDFs, parses company/fiscal year/quarter from the docugami filename convention via regex, loops through all filings catching per-file errors without halting the batch.
6. **Retrieval** (`app/retrieval/vector_search.py`) — cosine similarity top-k search via pgvector's `<=>` operator.
7. **Generation** (`app/services/llm_client.py`) — Groq `llama-3.3-70b-versatile`, prompted to answer using ONLY retrieved context and cite company/filing type/quarter/year/section per claim.
8. **API layer** (`app/main.py`) — FastAPI with `POST /query` and `GET /health`.

### Running Phase 1

```bash
uv run python -m app.services.db          # one-time DB schema setup
uv run python -m scripts.run_ingestion    # ingest all 20 PDFs
uv run uvicorn app.main:app --reload      # start the API
```

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was Apple revenue in Q2 2023?"}'
```

### Documented Baseline Weakness

**Observation:** Asking specifically about Apple's Q2 2023 revenue returned Revenue-section chunks from Q3 2022, Q3 2023, and Q1 2023 — but not the correct Q2 2023 chunk, despite it being present in the corpus. The LLM correctly refused to answer rather than hallucinate.

**Root cause:** Pure cosine similarity on dense embeddings struggled to distinguish near-identical "Note 2 – Revenue" boilerplate across fiscal quarters — the embedding captured topical similarity but not enough quarter-specific precision to rank the correct chunk in the top-k.

**Resolution:** This became the motivating case for Phase 2 (hybrid BM25 + RRF) and was fixed there by prefixing company/quarter/year metadata directly into the text used for embedding and keyword indexing (see Phase 2 below).

### Phase 1 Deliverables

| Component | File |
|---|---|
| PDF parsing (hi_res + OCR) | `app/ingestion/parser.py` |
| Section-based chunking with overlap | `app/ingestion/chunker.py` |
| Embedding (bge-small-en-v1.5) | `app/ingestion/embedder.py` |
| pgvector storage | `app/ingestion/loader.py`, `app/services/db.py` |
| Ingestion orchestration (all 20 PDFs) | `scripts/run_ingestion.py` |
| Vector similarity retriever | `app/retrieval/vector_search.py` |
| Groq-based generation with citation prompting | `app/services/llm_client.py` |
| FastAPI `/query` and `/health` endpoints | `app/main.py` |

---

## Phase 2: Hybrid Search + Reranking — COMPLETE ✅

**Goal:** Fix the Phase 1 quarter/year retrieval precision gap by combining keyword search with vector search, then reranking the fused candidates.

### What Was Implemented

1. **Full-text search column** — `content_tsv` (`tsvector`/`GIN` index) added to `chunks`, auto-populated on insert via a Postgres trigger.
2. **Metadata-aware `search_text` field** — company, filing type, fiscal quarter/year, and section are prefixed directly into the text used for both embeddings and keyword indexing (not just stored as separate DB columns). This single fix resolved the original Phase 1 failure case by giving both retrieval legs explicit temporal signal.
3. **Keyword search** (`app/retrieval/keyword_search.py`) — `ts_rank`-based BM25-style scoring over `content_tsv`.
4. **Reciprocal Rank Fusion** (`app/retrieval/fusion.py`) — fuses vector and keyword rankings purely by rank position, avoiding score-scale mismatches between cosine similarity and `ts_rank`.
5. **Cross-encoder reranking** (`app/retrieval/reranker.py`) — `cross-encoder/ms-marco-MiniLM-L-6-v2` reranks the top ~20 fused candidates down to a final top-5.
6. **Hybrid orchestrator** (`app/retrieval/hybrid_search.py`) — vector search → keyword search → RRF fusion → reranking, wired into `/query`.

### Schema Changes

```sql
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_tsv tsvector;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS search_text TEXT;

CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN(content_tsv);

CREATE OR REPLACE FUNCTION chunks_tsv_trigger() RETURNS trigger AS $$
BEGIN
  NEW.content_tsv := to_tsvector('english', COALESCE(NEW.search_text, NEW.content));
  RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS chunks_tsv_update ON chunks;
CREATE TRIGGER chunks_tsv_update BEFORE INSERT OR UPDATE ON chunks
FOR EACH ROW EXECUTE FUNCTION chunks_tsv_trigger();
```

Existing rows required a full `TRUNCATE` + re-ingestion since the trigger only fires on new inserts/updates, not retroactively.

### Evaluation Methodology

Built a manual precision@5 harness (`scripts/evaluate_retrieval.py`) against two query sets:

- **Basic queries** (`data/eval/sample_queries.csv`, n=10) — straightforward "revenue/income by quarter" questions, similar to the original Phase 1 failure case.
- **Hard queries** (`data/eval/sample_queries_hard.csv`, n=10) — questions targeting exact terms (note names, Item numbers, legal/tax terminology) that dense embeddings tend to blur across semantically similar but factually wrong sections.

A "hit" requires the correct company, fiscal quarter, fiscal year, **and** a matching section keyword (pipe-separated OR match) to appear in the top-5 results — not just the right filing.

### Checkpoint Results

| Query Set | Vector-Only Precision@5 | Hybrid+Rerank Precision@5 |
|---|---|---|
| Basic (n=10) | 10/10 (100%) | 9/10 (90%) |
| Hard (n=10) | 8/10 (80%) | 9/10 (90%) |
| **Combined (n=20)** | **18/20 (90%)** | **18/20 (90%)** |

**Interpretation:** Aggregate precision is statistically tied, but the composition of results confirms the Phase 2 goal — hybrid+rerank wins specifically on hard queries requiring exact-term matching (e.g., correctly surfacing Amazon's "Item 3 – Quantitative and Qualitative Disclosures About Market Risk" at rank 1 vs. rank 4 for vector-only, and rescuing "Microsoft's legal proceedings," which vector-only missed entirely). The single hybrid regression (Intel revenue Q1 2023) is an isolated, low-stakes case on an already-easy query rather than a systemic flaw.

### Known Limitations (Left Unfixed — Candidates for Phase 3)

- **Microsoft section mis-titling:** `unstructured`'s `Title`-detection heuristic occasionally flags a units caption ("(In millions)") as the section header instead of the actual statement name (e.g., "CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS") for Microsoft's filings specifically — likely due to inconsistent font/bold styling in the source PDFs. A parsing quality gap, not a retrieval-ranking issue.
- **Intel restructuring query miss:** Both vector-only and hybrid+rerank failed to surface the correct chunk for "What does Intel disclose about restructuring in Q3 2022?" — a shared parsing/chunking gap rather than a fusion or reranking failure.
- **Rationale for not fixing now:** Both issues stem from fragile PDF layout-heuristic title detection, which a knowledge graph with explicit `Filing → Section → NoteType` relationships (Phase 3) would sidestep entirely by not depending on parser-inferred section titles for lookup precision.

### Phase 2 Deliverables

| Component | File |
|---|---|
| Full-text search column + trigger | `app/services/db.py` |
| Metadata-aware search_text field | `app/ingestion/loader.py` |
| Keyword search (BM25-style) | `app/retrieval/keyword_search.py` |
| Reciprocal Rank Fusion | `app/retrieval/fusion.py` |
| Cross-encoder reranker | `app/retrieval/reranker.py` |
| Hybrid retrieval orchestrator | `app/retrieval/hybrid_search.py` |
| Updated `/query` endpoint | `app/main.py` |
| Precision@5 evaluation harness | `scripts/evaluate_retrieval.py` |
| Retrieval diagnostic script | `scripts/diagnose_retrieval.py` |

### Running Phase 2

```bash
uv run python -m scripts.run_ingestion       # re-ingest with search_text metadata
uv run python -m scripts.evaluate_retrieval  # compare vector-only vs hybrid+rerank
uv run uvicorn app.main:app --reload
```

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was Apple revenue in Q2 2023?"}'
```

---

## Troubleshooting Log

- **zsh `no matches found: unstructured[html]`** — zsh treats `[...]` as glob syntax; fix by quoting: `uv add "unstructured[pdf]"`.
- **`ModuleNotFoundError: No module named 'app'`** — occurred running `python scripts/run_ingestion.py` directly; fixed by running as a module: `uv run python -m scripts.run_ingestion` (requires `scripts/__init__.py`).
- **pydantic `ValidationError: Extra inputs are not permitted`** — `pydantic-settings` rejects `.env` variables not declared as `Settings` fields; fixed by explicitly declaring all Neo4j/model fields in `app/config.py`.
- **`tesseract is not installed or it's not in your PATH`** — `unstructured`'s `hi_res` strategy requires OCR; fixed via `brew install tesseract poppler` (macOS).
- **Slow `hi_res` ingestion** — OCR + YOLOX layout inference per page is compute-heavy; a faster `strategy="fast"` alternative was evaluated but **not adopted**, since table extraction fidelity matters more than speed for financial statements in this project.
- **`content_tsv` NULL for existing rows** — the Postgres trigger only fires on new inserts/updates, not retroactively; fixed by truncating and re-ingesting after adding `search_text` and the trigger.
- **Evaluation metric too lenient (100%/100% false positive)** — initial `is_hit()` only checked company/quarter/year, so any chunk from the correct filing counted as a hit regardless of section; fixed by requiring a matching section keyword.
- **Evaluation metric too strict (60%/60% false negative)** — single hardcoded keywords per query didn't match real, inconsistently-titled section names across companies; fixed by switching to pipe-separated OR-matching keywords derived from actual `SELECT DISTINCT section` output.

---

## Ready for Phase 3

Phases 0–2 are complete: infrastructure is provisioned, the baseline vector RAG pipeline works end-to-end, and hybrid search + reranking has been validated to improve precision on exact-term queries over pure vector search. Two known parsing-heuristic limitations (Microsoft section mis-titling, Intel restructuring miss) are documented as motivating cases for Phase 3.

**Next up:** entity/relationship extraction into Neo4j (Company, Filing, Note, RiskFactor nodes; FILED_IN, DISCLOSES, RISK_OF edges) and Cypher-based multi-hop querying — designed to resolve retrieval precision issues that depend on unreliable PDF title heuristics by using explicit graph relationships instead.
