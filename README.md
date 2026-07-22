# DeepFile

**Agentic GraphRAG research assistant over SEC 10-Q filings**

DeepFile combines an LLM agent (LangGraph) that plans/routes/self-corrects retrieval with a knowledge graph (Neo4j) for multi-hop relational reasoning, layered on top of hybrid vector + keyword search (pgvector + Postgres full-text). The domain is SEC 10-Q filings for five major tech companies, where relationships between entities (companies, filings, risk factors) require multi-hop reasoning that plain vector RAG cannot handle.

---

## Architecture Overview

- **Vector store:** pgvector via Supabase (free tier)
- **Keyword search:** Postgres full-text / BM25 (planned, Phase 2)
- **Knowledge graph:** Neo4j AuraDB (free tier)
- **Agent orchestration:** LangGraph (planned, Phase 4)
- **Reranker:** BGE cross-encoder or Cohere Rerank (planned, Phase 2)
- **LLM:** Groq (Llama 3.3 70B / Llama 3.1 8B)
- **Evaluation:** RAGAS (planned, Phase 5)
- **Observability:** Langfuse (planned, Phase 6)
- **API layer:** FastAPI
- **Package manager:** uv

## Build Phases

| Phase | Description | Status |
|---|---|---|
| 0 | Corpus + infra setup (accounts, DBs, project scaffold) | ✅ Complete |
| 1 | Baseline RAG (plain vector search + generation) | 🔄 In progress |
| 2 | Hybrid search + reranking (BM25 + RRF + cross-encoder) | ⬜ Not started |
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
- `pg_trgm` extension also enabled (optional, for future fuzzy text matching — not currently used since Neo4j handles relational lookups)
- Connection method: **Transaction pooler** (port 6543) — chosen over Direct connection since the FastAPI/agent workload is stateless and short-lived
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
│   ├── raw/sec-10-q/docs/       # 20 docugami PDF filings
│   ├── processed/                # (future) cleaned/chunked text
│   └── eval/                     # (future) qna_data.csv, custom multi-hop questions
├── app/
│   ├── main.py                   # FastAPI entry point
│   ├── config.py                 # Settings (pydantic-settings)
│   ├── ingestion/                 # Phase 1
│   │   ├── parser.py              # PDF parsing (unstructured, hi_res strategy)
│   │   ├── chunker.py             # Section-based chunking with overlap
│   │   ├── embedder.py            # sentence-transformers embeddings
│   │   └── loader.py              # Writes chunks + embeddings to pgvector
│   ├── retrieval/                 # Phase 2 (vector_search.py implemented; keyword/fusion/rerank pending)
│   ├── graph/                      # Phase 3 (not yet implemented)
│   ├── agent/                      # Phase 4 (not yet implemented)
│   ├── eval/                       # Phase 5 (not yet implemented)
│   ├── observability/               # Phase 6 (not yet implemented)
│   ├── api/                         # (folder reserved; routes currently in main.py)
│   └── services/
│       ├── llm_client.py           # Groq client wrapper
│       └── db.py                    # Postgres connection + schema init
├── scripts/
│   └── run_ingestion.py            # CLI: loops over all 20 PDFs, parses + embeds + loads
├── notebooks/
├── tests/
├── .env
├── .gitignore
└── pyproject.toml                   # uv-managed dependencies
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

## Phase 1: Baseline RAG (In Progress)

**Goal:** Naive end-to-end pipeline — parse → chunk → embed → store → retrieve → generate — to establish a working demo and a baseline to measure later improvements (hybrid search, graph, agentic routing) against.

### Pipeline Steps Implemented

1. **Parsing** (`app/ingestion/parser.py`)
   - Uses `unstructured.partition.pdf.partition_pdf` with `strategy="hi_res"` and `infer_table_structure=True`
   - Preserves section headers by tracking the most recent `Title`-category element and tagging all subsequent text with it
   - Requires Tesseract (OCR) and Poppler installed locally for layout/table detection

2. **Chunking** (`app/ingestion/chunker.py`)
   - Groups parsed elements by detected section (semantic unit), not fixed token windows
   - Falls back to character-window splitting (max 1500 chars) with ~12.5% overlap only when a section exceeds the max length

3. **Embedding** (`app/ingestion/embedder.py`)
   - Model: `BAAI/bge-small-en-v1.5` (free, open-source, via `sentence-transformers`) — 384 dimensions
   - Chosen over OpenAI embeddings to stay within the free-tier-only constraint

4. **Storage** (`app/ingestion/loader.py`, `app/services/db.py`)
   - Postgres table `chunks` with columns: `id, company, filing_type, fiscal_year, fiscal_quarter, section, content, embedding VECTOR(384)`
   - HNSW index on `embedding` using cosine distance ops for fast approximate nearest-neighbor search

5. **Ingestion orchestration** (`scripts/run_ingestion.py`)
   - Auto-discovers all PDFs in `data/raw/sec-10-q/docs/`
   - Parses `company ticker`, `fiscal year`, and `fiscal quarter` directly from the docugami filename convention (`YYYY QN TICKER.pdf`) via regex — no reliance on EDGAR header metadata since no EDGAR files are used
   - Loops through all 20 filings, catching and logging per-file errors without halting the batch

6. **Retrieval** (`app/retrieval/vector_search.py`)
   - Embeds the incoming query with the same `bge-small-en-v1.5` model
   - Cosine similarity top-k search via pgvector's `<=>` operator
   - Returns company, filing type, fiscal year/quarter, section, content, and similarity score per result

7. **Generation** (`app/services/llm_client.py`)
   - Groq client using `llama-3.3-70b-versatile`
   - Prompt instructs the model to answer using ONLY the retrieved context and to cite company/filing type/quarter/fiscal year/section per claim

8. **API layer** (`app/main.py`)
   - FastAPI app with `POST /query` (accepts `question` + optional `top_k`) and `GET /health`
   - Returns generated answer plus a `sources` array listing every chunk used, with similarity scores

### Running Phase 1

```bash
# One-time DB schema setup
uv run python -m app.services.db

# Ingest all 20 PDFs (parse, chunk, embed, store)
uv run python -m scripts.run_ingestion

# Start the API
uv run uvicorn app.main:app --reload
```

Test:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was Apple revenue in Q2 2023?"}'
```

### Known Issues / Notes

- `hi_res` PDF parsing is slow (OCR + layout model inference per page) — currently kept for better table extraction fidelity on financial statements; a `"fast"` strategy alternative (pure text-layer extraction, no OCR) was evaluated as a faster option but not adopted, since table structure accuracy matters for financial data.
- First run downloads the `bge-small-en-v1.5` embedding model (~130MB) and `unstructured`'s YOLOX layout model (~217MB) from Hugging Face Hub — subsequent runs use the local cache.
- Anonymous Hugging Face Hub requests are rate-limited; setting an `HF_TOKEN` env var is optional but recommended if downloads are slow.

### Checkpoint

Ask a simple factual question via `/query` (e.g., "What was Apple's revenue in Q2 2023?") and receive a grounded answer citing the specific 10-Q chunk it was pulled from. This is the baseline against which Phase 2 (hybrid search + reranking) will be measured using RAGAS in Phase 5.

---

---

## Phase 1: Baseline RAG — COMPLETE ✅

**Status update:** Phase 1 is now considered wrapped up. All 20 docugami 10-Q PDFs were successfully ingested (parsed → chunked → embedded → stored), and the `/query` endpoint is live and demoable end-to-end.

### Verified Checkpoint

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was Apple revenue in Q2 2023?"}'
```

Returned a grounded, cited answer with 5 source chunks (all Apple, "Note 2 – Revenue" sections, similarity scores 0.82–0.84). This confirms the full pipeline — parse, chunk, embed, store, retrieve, generate, serve — is working end-to-end.

### Documented Baseline Weakness (Intentionally Not Fixed in Phase 1)

**Observation:** When asked specifically about Apple's Q2 2023 revenue, the retriever returned Revenue-section chunks from Q3 2022, Q3 2023 (x2), and Q1 2023 — but not the actual Q2 2023 chunk, despite it being present in the ingested corpus. The LLM correctly refused to answer rather than hallucinate, since the correct chunk wasn't in its retrieved context.

**Root cause:** Pure cosine similarity search on dense embeddings struggles to distinguish between near-identical "Note 2 – Revenue" boilerplate sections across different fiscal quarters — the embedding captures topical similarity ("this is a revenue table") but not the specific quarter/year mentioned in the query with enough precision to rank the correct chunk in the top-k.

**Why this is left unfixed for now:** This is a textbook example of the exact gap that plain vector RAG cannot reliably close — and it is the motivating failure case for Phase 2 (hybrid BM25 + RRF, which will catch exact quarter/year keyword matches) and Phase 3 (Neo4j `Filing` nodes tagged by quarter, enabling precise graph lookups instead of semantic guessing). This question has been logged as a candidate benchmark item for the Phase 5 RAGAS evaluation set, to quantify the before/after improvement once hybrid search and the graph layer are added.

**Quick diagnostic used to confirm the chunk existed in the DB (not an ingestion bug):**
```sql
SELECT company, fiscal_year, fiscal_quarter, section, similarity
FROM chunks
WHERE company = \'Apple\' AND fiscal_quarter = \'Q2\' AND fiscal_year = 2023;
```

### Phase 1 Deliverables Summary

| Component | File | Status |
|---|---|---|
| PDF parsing (hi_res + OCR) | `app/ingestion/parser.py` | ✅ |
| Section-based chunking with overlap | `app/ingestion/chunker.py` | ✅ |
| Embedding (bge-small-en-v1.5) | `app/ingestion/embedder.py` | ✅ |
| pgvector storage | `app/ingestion/loader.py`, `app/services/db.py` | ✅ |
| Ingestion orchestration (all 20 PDFs) | `scripts/run_ingestion.py` | ✅ |
| Vector similarity retriever | `app/retrieval/vector_search.py` | ✅ |
| Groq-based generation with citation prompting | `app/services/llm_client.py` | ✅ |
| FastAPI `/query` and `/health` endpoints | `app/main.py` | ✅ |

### Troubleshooting Log (for future reference)

- **zsh `no matches found: unstructured[html]`** — zsh treats `[...]` as glob syntax; fix by quoting: `uv add "unstructured[pdf]"`.
- **`ModuleNotFoundError: No module named 'app'`** — occurred when running `python scripts/run_ingestion.py` directly; fixed by running as a module instead: `uv run python -m scripts.run_ingestion` (requires `scripts/__init__.py`).
- **pydantic `ValidationError: Extra inputs are not permitted`** — `pydantic-settings` rejects `.env` variables not declared as `Settings` fields; fixed by explicitly declaring all Neo4j/model fields in `app/config.py`.
- **`tesseract is not installed or it's not in your PATH`** — `unstructured`'s `hi_res` PDF strategy requires OCR; fixed via `brew install tesseract poppler` (macOS).
- **Slow `hi_res` ingestion** — OCR + YOLOX layout inference per page is compute-heavy; a faster `strategy="fast"` (pure text-layer extraction, no OCR) alternative was evaluated but **not adopted**, since table extraction fidelity matters more than speed for financial statements in this project; kept `hi_res` deliberately.

---

## Ready for Phase 2

Phase 1 baseline is complete and its known weakness (quarter/year retrieval precision) is documented as the motivating case for the next phase. Next up: Postgres full-text/BM25 keyword search, reciprocal rank fusion with vector search, and cross-encoder reranking (BGE or Cohere Rerank).
