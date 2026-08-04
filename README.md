# DeepFile

**Agentic GraphRAG research assistant over SEC 10-Q filings** 📊

DeepFile combines an LLM agent (LangGraph) that plans/routes/self-corrects retrieval with a knowledge graph (Neo4j) for multi-hop relational reasoning, layered on top of hybrid vector + keyword search (pgvector + Postgres full-text). The domain is SEC 10-Q filings for five major tech companies, where relationships between entities (companies, filings, notes, disclosures) require multi-hop reasoning that plain vector RAG cannot handle.

---
## Architecture

<p align="center">
  <img
    src="./assets/architecture-diagram.png"
    alt="DeepFile Agentic GraphRAG architecture showing offline ingestion, graph and hybrid retrieval, LangGraph orchestration, self-correction, RAGAS evaluation, and Langfuse observability"
    width="1400"
  />
</p>

## Architecture Overview

- **Vector store:** pgvector via Supabase (free tier)
- **Keyword search:** Postgres full-text search (`tsvector`/`ts_rank`, BM25-style)
- **Reranker:** Cross-encoder (`ms-marco-MiniLM-L-6-v2`, free/local via sentence-transformers)
- **Knowledge graph:** Neo4j AuraDB (free tier) — built, Phase 3
- **Agent orchestration:** LangGraph — built, Phase 4
- **LLM:** Groq (Llama 3.3 70B / Llama 3.1 8B, with fallback models for quota exhaustion)
- **Evaluation:** RAGAS — built, Phase 5
- **Observability:** Langfuse (tracing, guardrails) — built, Phase 6
- **API layer:** FastAPI
- **Package manager:** uv

## Build Phases

| Phase | Description | Status |
|---|---|---|
| 0 | Corpus + infra setup (accounts, DBs, project scaffold) | ✅ Complete |
| 1 | Baseline RAG (plain vector search + generation) | ✅ Complete |
| 2 | Hybrid search + reranking (BM25 + RRF + cross-encoder) | ✅ Complete |
| 3 | Knowledge graph layer (entity/relationship extraction into Neo4j) | ✅ Complete |
| 4 | Agentic router with self-correction loop (LangGraph) | ✅ Complete |
| 5 | Evaluation harness (RAGAS benchmark) | ✅ Complete |
| 6 | Observability + guardrails (Langfuse) | ✅ Complete |

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
- **Important data caveat (discovered in Phase 3):** the `chunks.company` Postgres column (and the `Company.ticker` graph property) stores FULL COMPANY NAMES, not ticker symbols — `'Microsoft'`, `'Apple'`, `'Intel'`, `'Amazon'`, `'NVIDIA'`, not `'MSFT'`/`'AAPL'`/`'INTC'`/`'AMZN'`/`'NVDA'`. Any SQL or Cypher filtering by company must use these exact full-name strings.

### Free-Tier Infrastructure

**1. Postgres + pgvector — Supabase**
- Project created at supabase.com
- `vector` extension enabled (Database -> Extensions)
- `pg_trgm` extension also enabled (optional, for future fuzzy text matching)
- Connection method: **Transaction pooler** (port 6543) — chosen since the FastAPI/agent workload is stateless and short-lived
- Connection string stored in `.env` as `DATABASE_URL`

**2. Knowledge Graph — Neo4j AuraDB**
- Instance `deepfile-kg` created at console.neo4j.io
- Credentials (URI, username, password) downloaded and saved at creation time
- Verified connection via Neo4j Browser with `RETURN 1`
- Credentials stored in `.env` as `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`
- **Now populated** — see Phase 3 below for final node/relationship counts

**3. LLM Inference — Groq**
- Account created at console.groq.com
- API key generated (Console -> API Keys)
- Models used:
  - `llama-3.3-70b-versatile` — generation/reasoning steps
  - `llama-3.1-8b-instant` — lightweight tasks (query rewriting, entity/disclosure extraction)
  - `openai/gpt-oss-20b`, `meta-llama/llama-4-maverick-17b-128e-instruct` — fallback models used during Phase 3 when primary models hit their daily token quota (Groq tracks TPD limits per-model, not account-wide, so switching models unblocks same-day)
- Key stored in `.env` as `GROQ_API_KEY`

**4. Observability — Langfuse (added Phase 6)**
- Project created on Langfuse Cloud (free "Hobby" tier — 50k units/month, 30-day retention)
- Public/secret keys generated (Project Settings -> API Keys)
- Keys stored in `.env` as `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`
- See Phase 6 below for full tracing/guardrail implementation details

### Project Structure

```
deepfile/
├── data/
│ ├── raw/sec-10-q/docs/ # 20 docugami PDF filings
│ ├── processed/ # (future) cleaned/chunked text
│ └── eval/ # sample_queries.csv, sample_queries_hard.csv, qna_data.csv, ragas_scores_raw.csv, results/
├── app/
│ ├── main.py # FastAPI entry point — wired to the Phase 4 agent, Langfuse-traced (Phase 6)
│ ├── config.py # Settings (pydantic-settings) + Langfuse env bridge (Phase 6)
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
│ ├── graph/ # Phase 3 — COMPLETE
│ │ ├── bootstrap.py # Metadata-based bootstrap (Company/Filing/Section from Postgres)
│ │ ├── extract_disclosures.py # Full-corpus LLM extraction -> Disclosure nodes, checkpointed
│ │ ├── build_graph.py # Orchestrator: bootstrap -> extraction
│ │ ├── validate_phase3.py # Cypher validation probes vs sample_queries_hard.csv
│ │ ├── audit_graph.py # Node/relationship counts, duplicate/noise check
│ │ └── .processed_ids.txt # Checkpoint file for extraction resume (gitignored)
│ ├── agent/ # Phase 4 — COMPLETE
│ │ ├── graph_builder.py # LangGraph StateGraph definition, AgentState, compiled `agent`
│ │ ├── router.py # router_node — classifies question as graph vs hybrid
│ │ ├── retrieve.py # retrieve node — dispatches to graph_node / hybrid_node
│ │ ├── rewrite.py # rewrite_node — LLM-based query reformulation on retry
│ │ └── generate.py # generate_node — builds context chunks, calls llm_client
│ ├── eval/ # Phase 5 — COMPLETE (Langfuse-traced as of Phase 6)
│ │ ├── run_eval.py # Runs all 3 pipelines (baseline/hybrid/agentic) over qna_data.csv -> raw answers+contexts
│ │ └── score_ragas.py # RAGAS judge scoring (faithfulness, context precision/recall, answer correctness), checkpointed
│ ├── observability/ # Phase 6 — COMPLETE
│ │ └── tracing.py # Langfuse client, callback handler, traced_config(), flag_output() guardrails
│ ├── api/ # (folder reserved; routes currently in main.py)
│ └── services/
│ ├── llm_client.py # Groq client wrapper, generate_answer() traced as a Langfuse generation (Phase 6)
│ ├── db.py # Postgres connection (get_conn()) + schema init
│ └── graph_db.py # Neo4j connection wrapper (execute_read/execute_write via driver.execute_query)
├── scripts/
│ ├── run_ingestion.py # CLI: loops over all 20 PDFs, parses + embeds + loads
│ ├── evaluate_retrieval.py # Precision@5 comparison: vector-only vs hybrid+rerank
│ ├── diagnose_retrieval.py # Dumps actual top-5 results per query for debugging
│ └── test_reroute.py # Isolated unit test of the rewrite -> re-route loop
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
       sentence-transformers groq python-dotenv pydantic-settings pymupdf neo4j langgraph langfuse

mkdir -p app/ingestion app/retrieval app/graph app/agent app/eval app/observability app/api app/services
mkdir -p data/raw/sec-10-q/docs data/processed data/eval scripts notebooks tests
touch scripts/__init__.py app/graph/__init__.py app/agent/__init__.py
```

System dependencies (macOS, required for `unstructured`'s `hi_res` PDF strategy):

```bash
brew install tesseract poppler
```

`.env` file (not committed to git):

```
DATABASE_URL=postgresql://postgres.[project-ref]:[password]@aws-0-[region].pooler.supabase.com:6543/postgres
GROQ_API_KEY=your-groq-key
OPENROUTER_API_KEY=your-openrouter-key
NEO4J_URI=neo4j+s://[instance-id].databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-neo4j-password
NEO4J_DATABASE=neo4j
AURA_INSTANCEID=your-instance-id
AURA_INSTANCENAME=deepfile-kg
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

---

`app/config.py` declares every field explicitly (pydantic-settings rejects undeclared `.env` variables by default). Final version, updated through Phase 6:

```python
import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str
    groq_api_key: str
    openrouter_api_key: str
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    docs_dir: str = "data/raw/sec-10-q/docs"

    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str
    aura_instanceid: str
    aura_instancename: str

    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_host: str = "https://cloud.langfuse.com"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()

# Langfuse's get_client() reads raw os.environ, not this Settings object —
# bridge these three explicitly so the SDK picks them up regardless of
# which module imports app.config first.
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)
```

---

## Phase 1: Baseline RAG — COMPLETE

**Goal:** Naive end-to-end pipeline — parse -> chunk -> embed -> store -> retrieve -> generate — to establish a working demo and a baseline to measure later improvements against.

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

## Phase 2: Hybrid Search + Reranking — COMPLETE

**Goal:** Fix the Phase 1 quarter/year retrieval precision gap by combining keyword search with vector search, then reranking the fused candidates.

### What Was Implemented

1. **Full-text search column** — `content_tsv` (`tsvector`/`GIN` index) added to `chunks`, auto-populated on insert via a Postgres trigger.
2. **Metadata-aware `search_text` field** — company, filing type, fiscal quarter/year, and section are prefixed directly into the text used for both embeddings and keyword indexing (not just stored as separate DB columns). This single fix resolved the original Phase 1 failure case by giving both retrieval legs explicit temporal signal.
3. **Keyword search** (`app/retrieval/keyword_search.py`) — `ts_rank`-based BM25-style scoring over `content_tsv`.
4. **Reciprocal Rank Fusion** (`app/retrieval/fusion.py`) — fuses vector and keyword rankings purely by rank position, avoiding score-scale mismatches between cosine similarity and `ts_rank`.
5. **Cross-encoder reranking** (`app/retrieval/reranker.py`) — `cross-encoder/ms-marco-MiniLM-L-6-v2` reranks the top ~20 fused candidates down to a final top-5.
6. **Hybrid orchestrator** (`app/retrieval/hybrid_search.py`) — vector search -> keyword search -> RRF fusion -> reranking, wired into `/query`.

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

### Known Limitations (Left Unfixed at Phase 2 Close — Resolved in Phase 3)

- **Microsoft section mis-titling:** `unstructured`'s `Title`-detection heuristic occasionally flags a units caption ("(In millions)") as the section header instead of the actual statement name (e.g., "CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS") for Microsoft's filings specifically — likely due to inconsistent font/bold styling in the source PDFs. A parsing quality gap, not a retrieval-ranking issue.
- **Intel restructuring query miss:** Both vector-only and hybrid+rerank failed to surface the correct chunk for "What does Intel disclose about restructuring in Q3 2022?" — a shared parsing/chunking gap rather than a fusion or reranking failure.
- **Rationale for deferring to Phase 3:** Both issues stem from fragile PDF layout-heuristic title detection, which a knowledge graph with explicit `Filing -> Section/Disclosure` relationships sidesteps entirely by not depending on parser-inferred section titles for lookup precision. See Phase 3 for how both cases were ultimately resolved.

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

## Phase 3: Knowledge Graph Layer — COMPLETE

**Goal:** Extract entities and relationships from the ingested 10-Q chunks into Neo4j, enabling multi-hop relational queries that don't depend on fragile PDF section-title parsing — directly targeting the two limitations left open at the end of Phase 2.

### What Was Implemented

1. **Neo4j connection wrapper** (`app/services/graph_db.py`) — driver wrapper using the modern `driver.execute_query()` API (rather than manual session management) with `execute_read`/`execute_write` helpers and `init_constraints()` for schema setup.
2. **Metadata bootstrap** (`app/graph/bootstrap.py`) — deterministic, zero-LLM-cost pass pulling `DISTINCT (company, filing_type, fiscal_year, fiscal_quarter, section)` from Postgres `chunks`, creating `Company`/`Filing`/`Section` nodes plus `FILED`/`HAS_SECTION` relationships. Bootstrapped 2,272 distinct filing/section rows.
3. **Full-corpus disclosure extraction** (`app/graph/extract_disclosures.py`) — LLM-based extraction (Groq `llama-3.1-8b-instant`, with fallback models on quota exhaustion) run over **every** chunk regardless of section name, using a category-aware prompt (`risk_factor`, `restructuring`, `financial_metric`, `legal_proceeding`, `tax`, `segment`, `other`). Writes `Disclosure{description, category}` nodes and `DISCLOSES` relationships. Checkpointed via `.processed_ids.txt` so Groq daily-quota interruptions don't require restarting from scratch.
4. **Build orchestrator** (`app/graph/build_graph.py`) — runs bootstrap then extraction in one command.
5. **Validation script** (`app/graph/validate_phase3.py`) — Cypher probes against multiple companies/categories, going beyond the two original motivator cases.
6. **Audit script** (`app/graph/audit_graph.py`) — node/relationship counts and per-filing disclosure-count distribution to surface deduplication quality.

### Neo4j Graph Schema

**Nodes:**
- `Company {ticker}` — property name is `ticker` but the value is the full company name (see data caveat above)
- `Filing {filing_id, fiscal_year, fiscal_quarter, filing_type}` — `filing_id` format: `"{company}-{fiscal_year}-{fiscal_quarter}"`, e.g. `"Intel-2022-Q3"`
- `Section {filing_id, name}` — composite-unique per filing
- `Disclosure {description, category}` — category ∈ `{risk_factor, restructuring, financial_metric, legal_proceeding, tax, segment, other}`

**Relationships:**
- `(Company)-[:FILED]->(Filing)`
- `(Filing)-[:HAS_SECTION]->(Section)`
- `(Filing)-[:DISCLOSES]->(Disclosure)`

**Constraints:** `Company.ticker` unique, `Filing.filing_id` unique, `Section (filing_id, name)` composite unique.

### Design Decision: Metadata Bootstrap First, Then Full-Corpus LLM Extraction

Two open questions were resolved at the start of Phase 3:

1. **Bootstrap vs. LLM extraction:** chose to bootstrap structural nodes (Company/Filing/Section) from existing Postgres metadata columns first — free, deterministic, immediately queryable — then layer LLM-based `Disclosure` extraction as a separate pass. Keeping these decoupled made debugging significantly easier than a single combined step.
2. **Fix Microsoft/Intel directly vs. stretch goal:** treated as a natural validation target rather than a dedicated fix — the graph schema itself (querying by structured `Section`/`Disclosure` nodes instead of fuzzy section-title string matching) was expected to resolve both cases as a side effect, which it did (see Debugging Journey below).
3. **Agent routing stub:** deferred entirely to Phase 4, to avoid coupling graph construction to agent design decisions not yet made.

### Debugging Journey

- **Initial evaluation false negatives:** the first `validate_phase3`/eval queries hardcoded ticker symbols (`'INTC'`, `'MSFT'`) as the `Company.ticker` property value, but the graph (mirroring Postgres) actually stores full company names (`'Intel'`, `'Microsoft'`). This caused both the Intel and Microsoft validation cases to falsely report "MISS" even though the underlying data was correct — resolved by using full names in all graph queries. Same "verify what's actually in the DB before assuming a bug is in logic" lesson from Phase 2 recurred here.
- **Intel restructuring — real gap found:** the original extraction script (`extract_risk_factors.py`, since superseded) only processed chunks where `section ILIKE '%risk%'`. Intel's actual restructuring disclosures live in sections named "Note 5/6: Restructuring and Other Charges" and "Restructuring and Other Charges" — no "risk" substring — so they were never sent to the LLM for extraction at all. This was the true root cause of the persistent Intel miss, not a graph or Cypher problem.
- **Long-term fix — full-corpus extraction:** rather than continuing to patch the section-keyword filter one missing category at a time (tax, legal, segment, etc. all faced the same risk), the extraction was redesigned to run over **every** chunk in the corpus with a generalized `Disclosure{category}` schema, replacing the narrowly-scoped `RiskFactor` label entirely. This eliminated the whack-a-mole pattern and is the version now in production.
- **Groq daily token quota (TPD) exhaustion:** `llama-3.1-8b-instant`'s 500K tokens/day limit was hit mid-run during the full-corpus extraction (~3,669 chunks). Discovered that Groq tracks TPD per-model rather than account-wide, so switching to a different model (`openai/gpt-oss-20b`, then `meta-llama/llama-4-maverick-17b-128e-instruct`) drew from a separate quota pool and allowed the run to continue same-day rather than waiting ~24h. The `.processed_ids.txt` checkpoint file made every quota interruption non-destructive — the script simply resumed from the last completed chunk on rerun.
- **Both original motivator cases now pass:** "Intel restructuring Q3 2022" and "Microsoft section titles Q2 2023" both return correct results in `evaluate_graph.py` after (a) fixing ticker->full-name literals and (b) full-corpus disclosure extraction.

### Final Graph Statistics (from `audit_graph.py`)

**Node counts:**

| Label | Count |
|---|---|
| Company | 5 |
| Filing | 20 |
| Section | 2,272 |
| Disclosure | 16,608 |

**Disclosure count by category:**

| Category | Count |
|---|---|
| financial_metric | 6,920 |
| risk_factor | 5,456 |
| other | 1,583 |
| segment | 942 |
| legal_proceeding | 717 |
| tax | 572 |
| restructuring | 418 |

**Relationship counts:**

| Type | Count |
|---|---|
| DISCLOSES | 21,058 |
| HAS_SECTION | 2,272 |
| FILED | 20 |

**Source corpus:** 3,669 chunks across 20 filings (~183 chunks/filing average).

### Checkpoint: Multi-Hop Relational Query Validated

**Test:** "Which companies share a risk factor with Intel?" — a relationship question that pure vector search cannot structurally answer, since it requires traversing from one company to another *through* a shared disclosure, not just ranking chunks by similarity to the query text.

**Query:**

```cypher
MATCH (c1:Company {ticker: 'Intel'})-[:FILED]->(:Filing)-[:DISCLOSES]->(d1:Disclosure {category: 'risk_factor'})
WHERE toLower(d1.description) CONTAINS 'supply chain'
MATCH (c2:Company)-[:FILED]->(:Filing)-[:DISCLOSES]->(d2:Disclosure {category: 'risk_factor'})
WHERE c2.ticker <> 'Intel' AND toLower(d2.description) CONTAINS 'supply chain'
RETURN DISTINCT c2.ticker
```

**Result:** Amazon, Apple, Microsoft, and NVIDIA — all four other companies in the corpus — share supply-chain risk language with Intel.

**Why vector search can't do this:** a dense retriever embeds the query and returns chunks that *sound* related (most likely Intel's own risk section), but it has no mechanism to enforce a join condition like "return companies whose risk disclosures overlap with Intel's." That requires an explicit, traversable relationship between entities — exactly what the `(Company)-[:FILED]->(Filing)-[:DISCLOSES]->(Disclosure)` structure provides and flat similarity ranking cannot express. This is the core justification for Phase 3's existence, now empirically confirmed rather than just architecturally assumed.

### Known Limitation: Disclosure Node Deduplication (Not Blocking, Deferred)

The relationship-to-node ratio for `Disclosure` is 21,058 / 16,608 ≈ 1.27 — meaning each unique disclosure description is on average reused only ~1.27 times across the whole corpus, despite substantial expected topical overlap across 5 companies × 4 quarters (e.g., boilerplate legal/risk language, recurring segment descriptions). Per-filing disclosure counts as high as 1,698 (Microsoft-2023-Q1) confirm this: `MERGE` on exact-string `description` barely deduplicates, because the LLM rephrases the same underlying fact differently on nearly every call, so semantically identical disclosures land as distinct nodes.

**Why this wasn't fixed immediately:** it's a graph-quality issue, not a correctness one — every node is individually accurate, both eval cases pass, and multi-hop queries work. Fixing it preemptively (before confirming it actually hurts Phase 4 agent output quality) risks solving a problem that hasn't been validated as impactful yet, consistent with the "don't over-fix on assumptions" lesson from Phase 2.

**Status update from Phase 4:** the isolated re-routing test and end-to-end failure/rewrite/success test did not surface any observable degradation from this dedup gap, so the planned embedding-based post-hoc merge remains deferred rather than promoted to active work.

**Status update from Phase 6:** live Langfuse guardrail scoring on both offline eval and spot-checked live traces still shows no measurable quality degradation tied to this gap — remains deferred.

### Phase 3 Deliverables

| Component | File |
|---|---|
| Neo4j connection wrapper | `app/services/graph_db.py` |
| Metadata bootstrap (Company/Filing/Section) | `app/graph/bootstrap.py` |
| Full-corpus disclosure extraction (checkpointed) | `app/graph/extract_disclosures.py` |
| Build orchestrator | `app/graph/build_graph.py` |
| Cypher validation probes | `app/graph/validate_phase3.py` |
| Graph audit (counts, dedup check) | `app/graph/audit_graph.py` |

### Running Phase 3

```bash
uv add neo4j
uv run python -m app.graph.build_graph        # bootstrap + full-corpus extraction
uv run python -m app.graph.validate_graph     # broader Cypher validation
uv run python -m app.graph.audit_graph        # node/relationship counts, dedup check
```

---

## Phase 4: Agentic Router with Self-Correction Loop — COMPLETE

**Goal:** Replace the single-path retrieval used in Phases 1–3 with a LangGraph agent that classifies each question, retrieves via the right path (graph vs. hybrid), self-corrects on empty or low-confidence results by rewriting the query and re-routing, and only then generates the final answer.

### Graph Flow

```
router -> retrieve (graph_node | hybrid_node) -> needs_retry?
                                                      |-- yes --> rewrite -> router (loop)
                                                      |-- no  --> generate -> END
```

### What Was Implemented

1. **`router_node`** (`app/agent/router.py`) — classifies the question into `graph` (relationship/comparison queries answerable from the knowledge graph) or `hybrid` (general document search).
2. **`retrieve`** (`app/agent/retrieve.py`) — dispatches to `graph_node` (Cypher generation + execution) or `hybrid_node` (vector + keyword hybrid search) based on `route`.
3. **`needs_retry`** (`app/agent/graph_builder.py`) — conditional edge; returns `True` if `graph_results` and `hybrid_results` are both empty, or if the top hybrid rerank score falls below the relevance threshold.
4. **`rewrite_node`** (`app/agent/rewrite.py`) — on retry, asks the LLM to reformulate the question using more search-friendly phrasing, records `rewrite_reasoning`, increments `retry_count`, and stores the original question in `original_question`.
5. **`generate_node`** (`app/agent/generate.py`) — builds context chunks from whichever result set is populated (graph results are flattened into readable chunks via `graph_result_to_chunk`) and calls `generate_answer`.

Retry is capped by `max_retries` (default 2) to prevent infinite loops and unbounded Groq quota consumption.

### Validation Performed

1. **End-to-end failure -> rewrite -> success:** a vague question that failed hybrid retrieval was rewritten and re-answered successfully on retry.
2. **Isolated re-routing unit test** (`scripts/test_reroute.py`) — bypassing the graph's fixed `router` entry point by calling `needs_retry`, `rewrite_node`, and `router_node` directly:
   - Seeded a `hybrid` route with a deliberately low rerank score (-999).
   - Confirmed `needs_retry` returned `True`.
   - Confirmed `rewrite_node` reformulated the question ("Which companies share a risk factor with Intel?" -> "Which companies have a risk factor similar to Intel's?") and incremented `retry_count` from 0 to 1.
   - Confirmed `router_node`, given the rewritten question, correctly re-classified the route from `hybrid` to `graph`.

   This confirmed the agent doesn't just retry the same failed strategy — it can genuinely switch retrieval strategies mid-loop based on the rewritten question, not just repeat the original one.

### Debugging Journey

- **Testing re-routing via `agent.invoke()` initially gave misleading results:** seeding a mid-loop state (`route: "hybrid"`, fake low-score `hybrid_results`) directly into `agent.invoke()` didn't test what was intended — the compiled graph's entry point is always `router`, so it re-classified the question fresh before the seeded state was ever read, making the seeded `hybrid_results` dead data.
- **Fix — bypass the entry point for unit testing:** rather than invoking the compiled graph, `scripts/test_reroute.py` calls `needs_retry`, `rewrite_node`, and `router_node` directly as plain functions, in sequence, on a manually constructed state dict. This isolates the exact mechanism under test without the graph's fixed entry point interfering.
- **Confirmed this discrepancy doesn't affect production behavior:** every real `/query` call starts with `route: None`, so `router_node` legitimately runs first on every fresh question — the entry-point behavior is a test-harness-only concern, not a production bug.

### Known Issue (Non-Blocking)

Cypher generation via the LLM is not fully deterministic even at `temperature=0`; identical questions can occasionally return slightly different result sets across runs. Flagged for future investigation, does not block current functionality.

### API Integration

`POST /query` in `app/main.py` now invokes the compiled agent graph (`app.agent.graph_builder.agent`) directly, replacing the earlier direct `hybrid_retrieve` call used in Phases 1–3. The response schema surfaces the agent's internal trace — `route`, `retry_count`, and (when a retry occurred) `rewritten_question` and `rewrite_reasoning` — making the self-correction behavior visible to any API consumer rather than buried in server logs.

**Example:**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Did Intel fire people?"}'
```

```json
{
  "answer": "Yes, Intel underwent restructuring...",
  "route": "hybrid",
  "retry_count": 1,
  "original_question": "Did Intel fire people?",
  "rewritten_question": "Did Intel undergo restructuring?",
  "rewrite_reasoning": "Rephrased using a more specific, search-friendly term."
}
```

### Phase 4 Deliverables

| Component | File |
|---|---|
| LangGraph StateGraph + AgentState + compiled agent | `app/agent/graph_builder.py` |
| Router node (graph vs. hybrid classification) | `app/agent/router.py` |
| Retrieve node (graph_node / hybrid_node dispatch) | `app/agent/retrieve.py` |
| Rewrite node (query reformulation on retry) | `app/agent/rewrite.py` |
| Generate node (context assembly + answer generation) | `app/agent/generate.py` |
| Isolated re-routing unit test | `scripts/test_reroute.py` |
| Updated `/query` endpoint (agent-wired) | `app/main.py` |

### Running Phase 4

```bash
uv add langgraph
uv run python -m scripts.test_reroute      # isolated rewrite -> re-route unit test
uv run uvicorn app.main:app --reload
```

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Which companies share a risk factor with Intel?"}'
```

---

## Phase 5: Evaluation Harness (RAGAS Benchmark) — COMPLETE

**Goal:** Quantitatively measure whether Phase 3's knowledge graph and Phase 4's agentic self-correction loop actually improve end-to-end answer quality versus a plain vector baseline and a hybrid-only pipeline — rather than relying on the qualitative spot-checks used in Phases 1–4.

### What Was Implemented

1. **Eval runner** (`app/eval/run_eval.py`) — loops over all 80 sampled question-answer pairs (drawn from the 195 in `qna_data.csv`) through three separate pipelines — **baseline** (Phase 1 vector-only), **hybrid** (Phase 2 hybrid+rerank), and **agentic** (Phase 4 full LangGraph agent) — capturing each pipeline's generated `answer` and retrieved `contexts` per question into `data/eval/ragas_scores_raw.csv`.
2. **Retry/fallback wrapper** (`run_fn_with_fallback()`) — retries each pipeline call against fallback Groq models on rate-limit/quota errors, consistent with the fallback-model pattern established in Phase 3, so a single exhausted model doesn't produce empty answers for an entire pipeline's remaining rows.
3. **RAGAS scoring** (`app/eval/score_ragas.py`) — scores every (question, answer, contexts, ground_truth) row on four RAGAS metrics: **faithfulness**, **context precision**, **context recall**, and **answer correctness**. Uses an LLM-as-judge (Groq) with a configurable `timeout` and `max_workers`, and is **checkpoint-resumable** — any row with a missing/NaN metric is treated as unscored and retried on the next run, so partial failures never require a full re-score from scratch.

### Debugging Journey

- **Judge timeouts disproportionately hit the agentic pipeline:** the first full scoring pass left up to 22/80 agentic rows with NaN metrics (vs. 0–4/80 for baseline/hybrid). Diagnosed by checking whether the NaN rows had empty answers (a real pipeline failure) or non-empty answers (a pure judge-scoring timeout) — confirmed **0 empty answers** out of the 25 NaN rows, with a mean answer length of ~1,092 characters (up to 7,883). This proved the agent was answering successfully; the RAGAS judge call was simply timing out before scoring finished, most likely because agentic's context (combined graph + hybrid results) produces longer judge prompts than the single-source contexts used by baseline/hybrid.
- **Fix — checkpoint-resume rescoring, not exclusion:** because `score_ragas.py` already treats any NaN-metric row as "not fully scored," simply rerunning it against only the incomplete rows (skipping the 55+ already-scored agentic rows) resolved the gap without needing to drop or impute missing data, which would have biased the agentic average toward its easier-to-judge subset.
- **Lesson carried over from Phase 3:** the same "verify the actual cause before treating a symptom as ground truth" discipline applied — an early hypothesis that agentic was systematically failing 25%+ of questions was ruled out by inspecting the raw NaN rows directly rather than assuming the missing-data pattern reflected real pipeline quality.

### Final RAGAS Results (n=80 per pipeline, fully scored)

| Pipeline | Faithfulness | Context Precision | Context Recall | Answer Correctness |
|---|---|---|---|---|
| Baseline | 0.632 | 0.803 | 0.622 | 0.501 |
| Hybrid | 0.601 | 0.822 | 0.620 | 0.516 |
| Agentic | 0.641 | 0.692 | 0.600 | 0.549 |

**Interpretation:**

- **Answer correctness (the primary target metric):** Agentic scores highest (0.549) vs. hybrid (0.516) and baseline (0.501) — confirming the Phase 3 knowledge graph and Phase 4 self-correction loop translate into measurably better final answers, not just architecturally more sophisticated retrieval.
- **Faithfulness:** Agentic also leads (0.641) — its answers are the least prone to hallucinating beyond what their retrieved context supports, despite that context being noisier (see below).
- **Context precision:** Agentic trails both other pipelines (0.692 vs. 0.803 baseline, 0.822 hybrid) — its retrieved context contains proportionally more irrelevant material, consistent with agentic runs combining graph results and hybrid results into a single, larger context window rather than a single clean retrieval pass.
- **Context recall:** Roughly tied across all three pipelines (0.600–0.622) — none of the three retrieval strategies is meaningfully better at surfacing all necessary facts; they differ mainly in retrieval cleanliness (precision) and downstream answer quality (faithfulness/correctness), not coverage.
- **Net conclusion:** Agentic trades context precision for answer quality — its retrieval is noisier, but its iterative routing/rewrite/generation loop compensates well enough to produce the most accurate and most faithful final answers of the three pipelines, validating the added complexity of Phases 3–4.

### Phase 5 Deliverables

| Component | File |
|---|---|
| Multi-pipeline eval runner (baseline/hybrid/agentic) | `app/eval/run_eval.py` |
| Fallback-model retry wrapper | `app/eval/run_eval.py` (`run_fn_with_fallback()`) |
| RAGAS scoring harness (checkpoint-resumable) | `app/eval/score_ragas.py` |
| Raw per-question scores (answers + contexts + metrics) | `data/eval/ragas_scores_raw.csv` |
| Human-reviewed ground truth Q&A set | `data/eval/qna_data.csv` |

### Running Phase 5

```bash
uv add ragas
uv run python -m app.eval.run_eval        # generate answers+contexts for all 3 pipelines
uv run python -m app.eval.score_ragas     # score via RAGAS judge, checkpoint-resumable
```

Re-running `score_ragas` after a partial/interrupted run automatically skips already-scored rows and only retries rows with missing metrics:

```bash
caffeinate uv run python -m app.eval.score_ragas
```

---

## Phase 6: Observability + Guardrails (Langfuse) — COMPLETE ✅

**Goal:** Add production-grade observability across all three pipelines (baseline, hybrid, agentic) using Langfuse — tracing every retrieval/generation step, capturing token/cost data, and layering in lightweight guardrails — so that once the agentic pipeline sees real, unsampled traffic, drift from the Phase 5 RAGAS benchmark is detectable rather than invisible.

### What Was Implemented

1. **Langfuse Cloud provisioning** 🔑 — free "Hobby" tier (50k units/month, 30-day retention) chosen over self-hosting; sufficient for DeepFile's current benchmark + early live-traffic volume without standing up ClickHouse/Postgres/MinIO infrastructure.
2. **`app/config.py`** — three new declared fields (`langfuse_public_key`, `langfuse_secret_key`, `langfuse_host`), plus an explicit `os.environ.setdefault(...)` bridge after `Settings()` instantiation — required because the Langfuse SDK's `get_client()` reads raw OS environment variables directly, not the pydantic `Settings` object, so `.env` values need to be re-exported into `os.environ` for the SDK to see them.
3. **`app/observability/tracing.py`** (new module) —
   - `langfuse` — the singleton client instance (`get_client()`).
   - `langfuse_handler` — a `langfuse.langchain.CallbackHandler`, auto-traces LangGraph node execution (router → retrieve → rewrite → generate) once passed into `agent.invoke(config=...)`.
   - `traced_config(pipeline, route, retry_count)` — builds a `RunnableConfig` dict with callback, run name, tags, and metadata for filterable dashboards.
   - `flag_output(trace_id, answer, contexts)` — attaches async/observational guardrail scores (`has_citation`, `empty_answer`, `empty_context`) to a completed trace via `create_score()`, without blocking the response.
4. **`app/main.py`** — `/query` now wraps the agent invocation in `langfuse.start_as_current_span(...)`, passes `traced_config(...)` into `agent.invoke()`, and returns a `trace_id` field in the response for correlating a specific API call back to its Langfuse trace. Guardrail scoring runs after the response payload is built (non-blocking).
5. **`app/services/llm_client.py`** — `generate_answer()` decorated with `@observe(as_type="generation")`, since it calls the raw Groq SDK directly (not a LangChain-wrapped client), meaning the auto-tracing `CallbackHandler` alone cannot capture it. Input/model/token usage are explicitly reported via `update_current_generation(...)` before and after the API call, so partial data is still captured even on a mid-call exception (e.g., a 429 during fallback-rotation).
6. **`app/eval/run_eval.py`** — every offline eval question, across all three pipelines, now runs inside a Langfuse span tagged `(pipeline, tier, "offline-eval")`, with the resulting `langfuse_trace_id` written into each pipeline's results CSV. This is what makes an eventual "did production drift from the Phase 5 benchmark" comparison possible — offline and live traces live in the same Langfuse project, distinguished only by tags, and can be joined back to `ragas_scores_raw.csv` by trace ID.

### Key Design Decisions

- **All three pipelines traced, not just agentic** — keeps observability comparable across baseline/hybrid/agentic in case either non-agentic pipeline is ever exposed as a fallback mode, and lets offline eval runs and live traffic share one dashboard.
- **Guardrails: async/flag-only, not synchronous blocking** — consistent with the project's established "don't over-fix unobserved problems" discipline (Phase 3 dedup gap, Phase 4 Cypher non-determinism): with zero live-traffic data yet to calibrate thresholds, scoring traces for `has_citation`/`empty_answer`/`empty_context` first, then deciding on synchronous rejection later if real failure rates justify it.
- **Langfuse Cloud over self-hosted** — free tier's 50k units/month comfortably covers current benchmark + early production volume; self-hosting's operational overhead isn't justified yet.

### Debugging Journey

- **`pydantic_core.ValidationError`: all 11 fields "missing", not just the 3 new ones** — root cause was a plain Python indentation bug: `class Config: env_file = ".env"` had been left at module level (column 0) instead of nested inside `Settings`, making it a dead, disconnected class that pydantic-settings never read. Since pydantic-settings then found no `.env` source at all, *every* declared field failed validation, old and new alike — a good reminder that "all fields suddenly missing" points at the loader mechanism, not the individual fields. Fixed by switching to `model_config = SettingsConfigDict(env_file=".env", ...)`, the modern pydantic v2 syntax.
- **"Langfuse client initialized without public_key" despite `Settings` loading correctly** — `settings.langfuse_public_key` printed fine, but the Langfuse SDK's `get_client()` reads directly from `os.environ`, which pydantic-settings' `env_file` loading does **not** populate — it only fills the in-process `Settings` object. Fixed by explicitly bridging the three Langfuse values into `os.environ` at the bottom of `config.py`.
- **`ValidationError: Extra inputs are not permitted` for `langfuse_base_url`** — a stray `.env` key (`LANGFUSE_BASE_URL`) didn't match any declared `Settings` field (`langfuse_host`) or the SDK's expected env var name (`LANGFUSE_HOST`). Fixed by renaming the `.env` key to `LANGFUSE_HOST`, aligning both the pydantic field name and the SDK's own expected variable name in one fix.
- **Spot-check falsely appeared to do nothing ("Resuming — 80 already completed")** — running a 3-question slice against `run_pipeline()` produced zero new Langfuse traces, because those 3 questions were already present in the existing 80-row checkpoint from an earlier full run, so the loop body never executed. Rather than modify the checkpoint/resume logic (explicitly out of scope — it works correctly and wasn't the actual bug), the clean fix was calling each pipeline's `run_fn(question)` directly, bypassing `run_pipeline()`'s orchestration wrapper entirely. Checkpointing lives only in `run_pipeline()`, not in the underlying retrieval/generation logic, so this produced genuine fresh Groq calls and Langfuse traces with zero risk to existing eval data.
- **Verified via dashboard, not just code** — confirmed in Langfuse Cloud (Tracing → Observations, filtered by `type = generation`) that both `baseline` and `hybrid` pipelines produced populated generation spans (model name, token usage, cost) after the fix — closing the loop on whether `generate_answer()`'s shared `@observe` decorator actually covers every pipeline, not just agentic, since all three call the same centralized function (confirmed via `grep -n "generate_answer" app/eval/pipelines.py`).

### Phase 6 Deliverables

| Component | File |
|---|---|
| Langfuse client, callback handler, traced config builder, guardrail scorer | `app/observability/tracing.py` |
| Langfuse fields + OS-env bridge | `app/config.py` |
| Traced `/query` endpoint with `trace_id` in response | `app/main.py` |
| Groq generation calls instrumented as Langfuse generation spans | `app/services/llm_client.py` |
| Offline eval runs traced + tagged, trace IDs written to results CSVs | `app/eval/run_eval.py` |

### Running Phase 6

```bash
uv add langfuse
```

`.env` additions:

```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

```bash
uv run uvicorn app.main:app --reload
```

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Which companies share a risk factor with Intel?"}'
```

The response now includes a `trace_id` field — paste it into Langfuse Cloud's trace search to inspect the full router → retrieve → rewrite/generate span tree, token usage, and guardrail scores for that specific call.

Standalone spot-check (bypasses checkpoint/resume logic entirely, safe to run anytime without touching real eval data):

```bash
uv run python -c "
from app.eval.pipelines import PIPELINES
from app.observability.tracing import langfuse

test_question = 'What did Apple report about supply chain risk in Q2 2023?'
for name, run_fn in list(PIPELINES.items())[:2]:
    result = run_fn(test_question)
    print(name, '->', result['answer'][:150])

langfuse.flush()
"
```

### Open Items (Deferred, Non-Blocking)

- **Synchronous guardrail enforcement** — not yet implemented; current guardrails are observational (Langfuse scores) only. Revisit once real traffic surfaces actual failure rates worth blocking on.
- **RAGAS-vs-live comparison script** — `langfuse_trace_id` is now written into eval result CSVs specifically to enable this, but the actual comparison script (joining `ragas_scores_raw.csv` against live Langfuse trace metrics) has not yet been built.
- **Langfuse-native LLM-as-judge scoring** — still undecided whether this should supplement or eventually replace the offline RAGAS harness for ongoing monitoring; no decision made yet.

---

## Scaling Considerations (Design Discussion, Not Implemented)

DeepFile's corpus is intentionally small — 20 filings, ~3,669 chunks, 16,608 disclosure nodes — which is enough to prove the GraphRAG architecture works but not enough to force decisions about scale. This section documents how each scaling concern would be addressed if the corpus grew from ~100 filings to millions, without actually building any of it. The reasoning below is what real engineers reach for at that scale, not speculation — the goal is to show the trade-offs are understood, not to solve a problem that doesn't exist yet at 20 documents.

### 1. Incremental Indexing

**Problem:** Every pipeline script currently assumes a fixed, one-time batch of PDFs. Adding a new quarterly filing today means re-running full ingestion (`run_ingestion.py`) and full-corpus disclosure extraction (`extract_disclosures.py`) — reprocessing 20 filings to add 1.

**How it would work:**

1. **Content-hash-based change detection** — add an `ingestion_manifest` table (`filing_id`, `content_hash`, `status`, timestamps) to Postgres. On each run, hash every PDF's raw bytes (`sha256`) and diff against the stored hash. Only filings that are new or whose hash changed get reprocessed — this is the same pattern used by LangChain's `SQLRecordManager` and LlamaIndex's ingestion pipeline for hash-based incremental updates.

```sql
CREATE TABLE ingestion_manifest (
    filing_id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    status TEXT NOT NULL,           -- pending | chunked | embedded | graph_extracted | done
    last_ingested_at TIMESTAMPTZ
);
```

2. **Scope the pipeline to the delta, not the corpus** — refactor `run_ingestion.py`, `bootstrap.py`, and `extract_disclosures.py` to each accept a single `filing_id` rather than always looping over every file in `docs_dir`. For a changed filing, delete its existing `chunks` rows and `Disclosure` nodes before re-inserting, to avoid stale duplicates sitting alongside fresh ones:

```cypher
MATCH (f:Filing {filing_id: $filing_id})-[:DISCLOSES]->(d:Disclosure)
DETACH DELETE d
```

3. **Checkpoint at the filing level, not just the chunk level** — Phase 3's `.processed_ids.txt` checkpoint already resumes mid-extraction on quota exhaustion; incremental indexing extends this idea one level up, using the `ingestion_manifest.status` column so a full quarterly filing sweep can resume across process restarts, not just within one.

4. **Regression check before marking "done"** — since Phase 6 already wraps pipeline runs in Langfuse spans, a newly indexed filing's rollout would run 3–5 targeted spot-check questions through the agent before flipping its manifest status to `done`, catching a bad parse/extraction on the new PDF before it silently degrades retrieval for the rest of the corpus.

**Why this wasn't built:** DeepFile's corpus is static — 20 filings across a fixed 4-quarter window, sourced from a frozen dataset. There is no actual "new filing" event to react to, so building this now would be solving a problem the project doesn't have yet, which conflicts with the "don't over-fix unobserved problems" discipline applied consistently since Phase 3.

### 2. Semantic Caching

**Problem:** Every `/query` call re-runs the full retrieval + generation pipeline even if a near-identical question was answered five minutes ago (e.g., "What was Apple's Q2 2023 revenue?" and "What did Apple report for revenue in Q2 2023?"). Both cost a full Groq generation call and, for agentic, a full router/retrieve/rewrite loop.

**How it would work:**

- Embed each incoming question (reusing the existing `bge-small-en-v1.5` embedder — no new model needed) and compare it via cosine similarity against a cache of recent (question_embedding, answer) pairs, most simply stored as another pgvector table or, at higher scale, a dedicated cache store (Redis with a vector module, or a managed semantic-cache service).
- If similarity exceeds a tuned threshold (commonly ~0.85–0.90 in production semantic-cache implementations), return the cached answer directly, skipping retrieval and generation entirely.
- Cache invalidation ties naturally into incremental indexing above — when a filing is re-ingested, any cached answers referencing that `filing_id` in their contexts should be invalidated, since the underlying facts may have changed.
- Reported production results for this pattern: up to 68.8% reduction in LLM API calls, and latency/cost reductions in the 80–88% range for high cache-hit-rate workloads — the win is disproportionate to the implementation effort.

**Why this wasn't built:** At current traffic (manual testing + eval runs), there's no query-repetition pattern to actually cache against — this is explicitly a production-traffic optimization, not a development-time one.

### 3. Vector Index Sharding and Read Replicas

**Problem:** pgvector's HNSW index is memory-resident and rebuilt in full on major changes — fine at DeepFile's ~3,669-chunk scale, but HNSW indexes over a few million vectors can consume 10+ GB of RAM, and index-to-RAM ratio becomes the practical ceiling on a single Postgres instance well before 10 million documents.

**How it would work, at increasing scale:**

- **~100K–1M chunks:** switch from full-precision `vector(384)` to `halfvec(384)` (float16), roughly halving index memory footprint with under 1% recall degradation on normalized embeddings — a nearly free win that should ideally be adopted from day one rather than retrofitted later, since HNSW indexes can't be updated incrementally in the background and require a full rebuild to convert.
- **~1M–10M chunks:** partition the vector index by a natural key already present in the schema — company or fiscal year — so each shard only needs to be searched when a query's metadata filter (already used in Phase 2's `search_text` prefixing) narrows the candidate set to that partition, avoiding a full-corpus HNSW scan on every query.
- **Beyond a single instance's capacity:** move to horizontal sharding across multiple Postgres instances (or a purpose-built vector database with native sharding), with a routing layer that fans a query out to the relevant shard(s) based on metadata filters and merges results — the same RRF-style fusion already built in Phase 2 generalizes naturally to merging ranked results across shards, not just across vector/keyword search.
- **Read replicas** — since DeepFile's query pattern is read-heavy (retrieval) with comparatively rare writes (quarterly filing ingestion), a standard read-replica setup in front of the primary Postgres instance would absorb query load without touching the ingestion path at all.

**Why this wasn't built:** 20 filings fit comfortably in a single free-tier Supabase instance with room to spare — none of the above is a live constraint yet.

### 4. Zero-Downtime Index Updates

**Problem:** Re-embedding or re-indexing a large corpus in place risks serving degraded or inconsistent results mid-update — a query might hit a half-rebuilt HNSW index or a graph mid-`DETACH DELETE`.

**How it would work:** a blue-green (shadow index) strategy — build the new index/graph state in parallel alongside the currently-serving one, validate it (via the same Langfuse-traced spot-check idea from incremental indexing above), then atomically switch the application's read target from old to new, and only tear down the old index after a monitoring window confirms no regressions. This is the same "expand-and-contract" pattern used broadly for zero-downtime schema/database migrations, adapted to vector/graph indexes rather than relational tables.

**Why this wasn't built:** DeepFile's current update pattern is a full offline rebuild during active development, not a live production cutover — there's no serving traffic to protect during a rebuild yet.

### 5. Multi-Tenant Access Control

**Problem:** DeepFile currently has no notion of "tenants" — all 5 companies' filings are equally visible to every query. A real multi-tenant deployment (e.g., different clients each restricted to their own subset of filings) would need retrieval-time filtering enforced at the query layer, not just application-level convention.

**How it would work:** tag every `chunks` row and every `Filing`/`Disclosure` node with a `tenant_id`, and enforce it as a mandatory filter injected into every SQL/Cypher query the agent generates — not an optional parameter the LLM could omit or hallucinate around. For defense in depth, Postgres Row-Level Security policies would enforce this at the database layer as a second line of defense independent of application code correctness.

**Why this wasn't built:** DeepFile has one logical "tenant" (the research assistant itself, over one shared corpus) — there's no access-control boundary the current use case actually needs enforced.

---

## Troubleshooting Log

- **zsh `no matches found: unstructured[html]`** — zsh treats `[...]` as glob syntax; fix by quoting: `uv add "unstructured[pdf]"`.
- **`ModuleNotFoundError: No module named 'app'`** — occurred running `python scripts/run_ingestion.py` directly; fixed by running as a module: `uv run python -m scripts.run_ingestion` (requires `scripts/__init__.py`, and similarly `app/graph/__init__.py` for Phase 3 modules).
- **pydantic `ValidationError: Extra inputs are not permitted`** — `pydantic-settings` rejects `.env` variables not declared as `Settings` fields; fixed by explicitly declaring all Neo4j/model fields in `app/config.py`.
- **`tesseract is not installed or it's not in your PATH`** — `unstructured`'s `hi_res` strategy requires OCR; fixed via `brew install tesseract poppler` (macOS).
- **Slow `hi_res` ingestion** — OCR + YOLOX layout inference per page is compute-heavy; a faster `strategy="fast"` alternative was evaluated but **not adopted**, since table extraction fidelity matters more than speed for financial statements in this project.
- **`content_tsv` NULL for existing rows** — the Postgres trigger only fires on new inserts/updates, not retroactively; fixed by truncating and re-ingesting after adding `search_text` and the trigger.
- **Evaluation metric too lenient (100%/100% false positive)** — initial `is_hit()` only checked company/quarter/year, so any chunk from the correct filing counted as a hit regardless of section; fixed by requiring a matching section keyword.
- **Evaluation metric too strict (60%/60% false negative)** — single hardcoded keywords per query didn't match real, inconsistently-titled section names across companies; fixed by switching to pipe-separated OR-matching keywords derived from actual `SELECT DISTINCT section` output.
- **`ImportError: cannot import name 'get_connection' from 'app.services.db'`** — assumed a function name (`get_connection`) that didn't match the actual export (`get_conn`); fixed by checking the real `db.py` source before writing dependent modules, and updating imports to use `get_conn`.
- **Graph evaluation false "MISS" on both Intel and Microsoft cases** — hardcoded ticker symbols (`'INTC'`, `'MSFT'`) in Cypher `WHERE` clauses didn't match the graph's actual `Company.ticker` values (full names: `'Intel'`, `'Microsoft'`); fixed by verifying actual property values via `MATCH (c:Company) RETURN DISTINCT c.ticker` before writing eval queries.
- **Intel restructuring still missing after ticker fix** — root cause was the extraction script's `WHERE section ILIKE '%risk%'` filter excluding Intel's restructuring-related sections (no "risk" substring in their titles); fixed by redesigning extraction to run over the full corpus regardless of section name (`extract_disclosures.py`), rather than patching the keyword filter incrementally.
- **Groq `429 rate_limit_exceeded` on tokens-per-day (TPD)** — hit `llama-3.1-8b-instant`'s 500K daily token quota mid-extraction-run; the error's "try again in Xs" message was misleading near the ceiling since freed tokens get immediately re-consumed by the next request. Fixed short-term by switching to a different Groq model (separate quota pool per model, not account-wide) and long-term by making the extraction script checkpoint-resumable so daily quota resets don't require reprocessing.
- **Re-routing test via `agent.invoke()` gave dead-data results** — seeding a mid-loop state directly into `agent.invoke()` was silently overwritten because the compiled graph's entry point is always `router`; fixed by testing `needs_retry`, `rewrite_node`, and `router_node` as isolated functions in `scripts/test_reroute.py` instead of through the compiled graph.
- **RAGAS judge timeouts concentrated on the agentic pipeline** — up to 22/80 agentic rows returned NaN metrics after the first scoring pass, initially suspected to be real agentic answer failures; confirmed via answer-length inspection (0 empty answers, mean ~1,092 chars) that these were pure judge-scoring timeouts caused by agentic's longer combined-context judge prompts, not pipeline failures. Fixed by relying on `score_ragas.py`'s existing checkpoint-resume logic to rescore only the NaN rows.
- **`pydantic_core.ValidationError`: all 11 `.env` fields missing after adding Langfuse keys** — a stray top-level `class Config:` (left at column 0 instead of nested inside `Settings`) meant pydantic-settings never found an `env_file` source at all; fixed by switching to `model_config = SettingsConfigDict(env_file=".env", ...)`.
- **"Langfuse client initialized without public_key" despite `settings.langfuse_public_key` loading fine** — the Langfuse SDK's `get_client()` reads raw `os.environ`, not the pydantic `Settings` object; fixed by explicitly bridging the three Langfuse values into `os.environ` at the bottom of `config.py` via `os.environ.setdefault(...)`.
- **`ValidationError: Extra inputs are not permitted` for `langfuse_base_url`** — a stray `.env` key didn't match the declared `Settings` field (`langfuse_host`) or the SDK's expected variable name; fixed by renaming the `.env` key to `LANGFUSE_HOST`.
- **Spot-check appeared to do nothing ("Resuming — 80 already completed")** — testing Langfuse tracing against a small slice of `eval_set_phase5.csv` produced zero new traces because those questions were already checkpointed from an earlier full run; fixed by calling each pipeline's `run_fn(question)` directly, bypassing `run_pipeline()`'s checkpoint/resume wrapper entirely rather than modifying working checkpoint logic.

---

## Project Status 🚀

Phases 0–6 are complete: infrastructure is provisioned, baseline vector RAG works end-to-end, hybrid search + reranking improved precision on exact-term queries, a Neo4j knowledge graph (5 Company, 20 Filing, 2,272 Section, 16,608 Disclosure nodes) supports multi-hop relational queries, a LangGraph agent routes/retrieves/self-corrects/generates answers end-to-end through `/query`, a RAGAS evaluation harness has quantitatively confirmed that the agentic pipeline produces the most correct (0.549) and most faithful (0.641) answers of the three pipelines benchmarked, and Langfuse now traces every pipeline (baseline, hybrid, agentic) end-to-end with token/cost visibility and observational guardrail scoring on citations and empty answers/contexts.

Both original Phase 2 motivator cases (Microsoft section mis-titling, Intel restructuring miss) remain confirmed resolved via the graph. Three known non-blocking issues remain open: `Disclosure` node deduplication (~1.27 relationship-to-node ratio, deferred since Phase 3), non-deterministic Cypher generation at `temperature=0` (flagged since Phase 4), and synchronous guardrail enforcement (deferred at the close of Phase 6, pending real traffic to calibrate against) — none has yet been shown to measurably harm answer quality, but all remain candidates for follow-up investigation.

DeepFile is considered feature-complete at Phase 6 for its current scope — a research-assistant-scale corpus (20 filings) with production-grade retrieval, graph reasoning, self-correction, evaluation, and observability. Scale-specific concerns (incremental indexing, caching, sharding, zero-downtime updates, multi-tenancy) are deliberately not implemented, since the current corpus size doesn't create real pressure to solve them — see **Scaling Considerations** above for how each would be approached if that changed.
