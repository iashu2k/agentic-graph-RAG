# DeepFile

**Agentic GraphRAG research assistant over SEC 10-Q filings**

DeepFile combines an LLM agent (LangGraph) that plans/routes/self-corrects retrieval with a knowledge graph (Neo4j) for multi-hop relational reasoning, layered on top of hybrid vector + keyword search (pgvector + Postgres full-text). The domain is SEC 10-Q filings for five major tech companies, where relationships between entities (companies, filings, notes, disclosures) require multi-hop reasoning that plain vector RAG cannot handle.

---

## Architecture Overview

- **Vector store:** pgvector via Supabase (free tier)
- **Keyword search:** Postgres full-text search (`tsvector`/`ts_rank`, BM25-style)
- **Reranker:** Cross-encoder (`ms-marco-MiniLM-L-6-v2`, free/local via sentence-transformers)
- **Knowledge graph:** Neo4j AuraDB (free tier) — built, Phase 3
- **Agent orchestration:** LangGraph — built, Phase 4
- **LLM:** Groq (Llama 3.3 70B / Llama 3.1 8B, with fallback models for quota exhaustion)
- **Evaluation:** RAGAS — planned, Phase 5
- **Observability:** Langfuse — planned, Phase 6
- **API layer:** FastAPI
- **Package manager:** uv

## Build Phases

| Phase | Description | Status |
|---|---|---|
| 0 | Corpus + infra setup (accounts, DBs, project scaffold) | Complete |
| 1 | Baseline RAG (plain vector search + generation) | Complete |
| 2 | Hybrid search + reranking (BM25 + RRF + cross-encoder) | Complete |
| 3 | Knowledge graph layer (entity/relationship extraction into Neo4j) | Complete |
| 4 | Agentic router with self-correction loop (LangGraph) | Complete |
| 5 | Evaluation harness (RAGAS benchmark) | Not started |
| 6 | Observability + guardrails (Langfuse) | Not started |
| 7 | Incremental indexing (stretch) | Not started |

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

### Project Structure

```
deepfile/
├── data/
│ ├── raw/sec-10-q/docs/ # 20 docugami PDF filings
│ ├── processed/ # (future) cleaned/chunked text
│ └── eval/ # sample_queries.csv, sample_queries_hard.csv, qna_data.csv
├── app/
│ ├── main.py # FastAPI entry point — wired to the Phase 4 agent
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
│ ├── eval/ # Phase 5 (not yet implemented)
│ ├── observability/ # Phase 6 (not yet implemented)
│ ├── api/ # (folder reserved; routes currently in main.py)
│ └── services/
│ ├── llm_client.py # Groq client wrapper
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
       sentence-transformers groq python-dotenv pydantic-settings pymupdf neo4j langgraph

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

### Next Steps

- Investigate Cypher generation non-determinism.
- Consider surfacing intermediate `graph_results`/`hybrid_results` in the API response for debugging/demo purposes.
- Add integration tests covering the full `/query` endpoint (not just node-level unit tests).

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

---

## Ready for Phase 5

Phases 0–4 are complete: infrastructure is provisioned, baseline vector RAG works end-to-end, hybrid search + reranking improved precision on exact-term queries, a Neo4j knowledge graph (5 Company, 20 Filing, 2,272 Section, 16,608 Disclosure nodes) supports multi-hop relational queries, and a LangGraph agent now routes, retrieves, self-corrects, and generates answers end-to-end through `/query`. Both original Phase 2 motivator cases (Microsoft section mis-titling, Intel restructuring miss) remain confirmed resolved via the graph, and the Phase 4 self-correction loop has been validated both end-to-end and via an isolated re-routing unit test.

One known limitation — `Disclosure` node deduplication (~1.27 relationship-to-node ratio) — remains documented but unfixed; Phase 4 usage did not surface evidence that it degrades agent reasoning quality, so it stays deferred. A second known issue — non-deterministic Cypher generation at `temperature=0` — was newly surfaced in Phase 4 and is flagged for investigation but is non-blocking.

**Next up (Phase 5, not started):** build a RAGAS evaluation harness against the 195 human-reviewed question-answer pairs in `qna_data.csv`, benchmarking the full agentic pipeline (router + graph/hybrid retrieval + self-correction + generation) on faithfulness, answer relevance, and context precision/recall — establishing a quantitative measure of whether Phase 3's knowledge graph and Phase 4's agentic loop actually improved end-to-end answer quality versus the Phase 2 hybrid-only baseline.
