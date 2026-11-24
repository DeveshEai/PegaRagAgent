# 🦅 Pega CLSA Knowledge Assistant (Expert PEGA / PegaRagAgent)

> **An Adaptive, Evidence-Grounded Pega CLSA AI Architecture Assistant powered by Hybrid Search (Dense Vectors + Pega-Aware BM25 + Weighted RRF + Cross-Encoder Reranking), Autonomous Tool-Calling, Claim-Level Citation Verification, and Groq Cloud LLMs.**

[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-blue.svg)](https://www.trychroma.com/)
[![Groq](https://img.shields.io/badge/LLM%20Inference-Groq%20Cloud-orange.svg)](https://groq.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🌟 Project Overview & Purpose

**Expert PEGA (`PegaRagAgent`)** is a production-oriented Retrieval-Augmented Generation (RAG) and Autonomous Agent application designed to act as a **Senior Pega Certified Lead System Architect (CLSA)** colleague. It answers Pega architecture, implementation, Constellation UI, security, background processing, and troubleshooting questions strictly from indexed Pega Lead System Architect documentation and hands-on challenge guides.

### Why It Outperforms Generic Chatbots
* **Zero-Guessing & Claim-Level Grounding:** Every factual claim is backed by retrieved curriculum excerpts and verified inline citations (`[filename, Page X, Section: Y]`).
* **Hybrid Pega-Aware Retrieval:** Combines local `all-MiniLM-L6-v2` semantic embeddings in ChromaDB with a custom Pega-aware Okapi BM25 retriever (preserving identifiers such as `Rule-Obj-Class`, `pyWorkPage`, and `D_*`), weighted Reciprocal Rank Fusion (RRF), and Cross-Encoder reranking.
* **Two Execution Strategies + Opt-In Adaptive Routing:** Supports both **Fast Direct RAG** and a **4-Tool Autonomous CLSA Agent**, plus an opt-in **Adaptive RAG** pipeline with Pega-domain intent classification and bounded evidence-based escalation.
* **Natural Conversational UX:** Greetings, thanks, and capability questions bypass RAG retrieval cleanly without artificial citations, while technical comparisons default to Markdown tables and step-by-step requests produce numbered steps.

---

## ✅ Complete Feature Inventory

### Implemented Features
* **Hybrid Retrieval & Reranking (`retrieval/hybrid.py`):**
  * Local ChromaDB vector search (`all-MiniLM-L6-v2` via `SentenceTransformerEmbeddingFunction`).
  * Pega-aware tokenizer (`tokenize_pega_text`) preserving camelCase, hyphenated, and underscored identifiers (`pyWorkPage`, `D_pxGetPage`, `Rule-Obj-Class`) plus sub-tokens.
  * Okapi BM25 sparse retrieval (`BM25Retriever`, `k1=1.5`, `b=0.75`) with section/category metadata filtering.
  * Weighted Reciprocal Rank Fusion (`RRF_K=60`) with configurable `dense_weight` (`0.0` = pure BM25 to `1.0` = pure Vector) and Cross-Encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`, `MIN_RERANK_PROBABILITY=0.08`).
* **Autonomous CLSA Agent & Research Tools (`agents/research_agent.py`, `agents/tools.py`):**
  * Multi-step Groq function-calling loop (`run_agent`, `run_agent_stream`) with 4 local tools: `search_curriculum`, `search_challenges`, `read_full_page`, and `lookup_pega_terms`.
  * Automatic fallback to Direct Grounded Retrieval if a non-tool-capable model is selected or if Groq returns a tool-calling error.
* **Adaptive RAG Pipeline — Stages 1–4 (`agents/research_agent.py`, `agents/routing_eval.py`):** *(Disabled by default via `ENABLE_ADAPTIVE_ROUTING=False`; toggleable in UI or via `PEGA_ENABLE_ADAPTIVE_ROUTING=true`)*
  * **Stage 1 (Pega-Domain Router):** Deterministic fast-path taxonomy covering 12 Pega domains (`data_pages`, `clipboard`, `rule_resolution`, `circumstancing`, `case_management`, `sla`, `queue_processor`, `job_scheduler`, `integration`, `security`, `constellation`, `data_management`) and 6 complexity signals, paired with `qwen/qwen3.8-27b` LLM intent classification for ambiguous/follow-up queries.
  * **Stage 2 (Adaptive Orchestration):** `run_adaptive()` and `run_adaptive_stream()` unify routing into the existing Direct RAG and Autonomous Agent pipelines.
  * **Stage 3 (Evidence-Based Escalation):** Multi-signal `evaluate_evidence_sufficiency()` (quality, source diversity, resolved-query coverage, top-chunk relevance, retrieval confidence) with a strict cap of **1 escalation** (`MAX_ROUTING_ESCALATIONS=1`), non-tool-model protection, and honest knowledge-gap disclosure.
  * **Stage 4 (Evaluation & Observability):** Privacy-safe `RoutingMetricsTracker` (redacting API keys, Bearer tokens, cookies, and session IDs) and an offline 24-query `evaluate_routing_suite()` + Markdown report generator (`format_routing_eval_report()`).
* **Conversational UX, Formatting & Grounding (`agents/answer_generator.py`, `retrieval/grounding.py`):**
  * Shared execution-layer `detect_conversational_intent()` fast path for greetings, thanks, and capability questions (`"What can you help me with?"`) with zero retrieval or citations, while preserving short Pega queries (`"What is SLA?"`, `"Data Page?"`).
  * Multi-turn pronoun & subject resolution (`contextualize_query_with_history`) supporting `"it"`, `"its"`, `"they"`, `"their"`, and multi-turn antecedent lookup.
  * Explicit output-format detection (`detect_output_format`: `TABLE`, `NUMBERED_STEPS`, `BULLETS`, `EXAMPLE`, `PARAGRAPH`).
  * Pega CLSA colleague prompts distinguishing `Queue Processor` (async Kafka background processing) from `Work Queue` / `Workbasket` (human task assignment) and structuring Data Page troubleshooting across scope, refresh strategy, reload conditions, invalidation, source freshness, and node vs. cluster behavior.
  * Two-stage `verify_citations()` (file/page existence + lexical claim-to-evidence overlap) preserving Markdown hard line breaks, indentation, lists, and tables.
  * Completeness guards (`create_completion_with_completeness_guard`, `stream_completion_with_completeness_guard`, `trim_to_last_complete_sentence`) with context-length retry recovery and Markdown-safe trimming.
* **Document Ingestion & Knowledge Management (`ingestion/extractors.py`, `rag_engine.py`, `ui/upload_panel.py`, `ui/explorer_panel.py`):**
  * Page-aware PDF extraction (`PyMuPDF`), section- and table-aware DOCX extraction (`python-docx`, `DOCX_EXTRACTOR_VERSION=2`, `TABLE_ATOMIC_CHAR_LIMIT=1800`), and TXT/MD support.
  * `RecursiveCharacterTextSplitter` (`CHUNK_SIZE=950`, `CHUNK_OVERLAP=180`).
  * SHA-256 incremental sync (`indexed_manifest.json`), uploaded filename sanitization (`sanitize_uploaded_filename`), individual document deletion, and `Delete All Chunks` (with optional disk file deletion).
* **Bento Modular Streamlit UI (`app.py`, `ui/`):**
  * Streaming token output with live status updates, Light/Dark theme toggle, one-click demo questions, inline citation badges, collapsible `Sources (N)` previews, expandable `Execution Details` per assistant turn, and Configuration drawer with Adaptive RAG toggle and Routing Observability metrics.

### Partially Implemented / Opt-In
* **Adaptive RAG Default Activation:** Stages 1–4 are fully implemented and tested, with `ENABLE_ADAPTIVE_ROUTING` kept `False` by default in `config/settings.py` until live-corpus evaluation is complete. Users can enable it per session via the **Adaptive RAG Routing (Auto-Select)** toggle in Configuration or via `PEGA_ENABLE_ADAPTIVE_ROUTING=true`.
* **Routing Telemetry Persistence:** `RoutingMetricsTracker` stores up to 500 sanitized records in memory per process (`max_records=500`) rather than persisting to disk.

---

## 🧠 System Architecture

```
                         ┌──────────────────────────────┐
                         │      User Query (Streamlit)   │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │  detect_conversational_intent │──▶ Greeting/Thanks/Capability
                         │  (Shared Execution Layer)     │    (0 retrieval, 0 citations)
                         └──────────────┬───────────────┘
                                        │ Technical Pega Query
                                        ▼
          ┌─────────────────────────────────────────────────────────────┐
          │ Execution Mode (Manual Mode or Opt-In Adaptive RAG Router)  │
          │  • Tier 1: Pega-Domain Deterministic Rules (12 domains)     │
          │  • Tier 2: qwen/qwen3.8-27b LLM Intent Router (ambiguous)   │
          └───────────────┬─────────────────────────────┬───────────────┘
                          │ DIRECT_RAG                  │ AUTONOMOUS_AGENT
                          ▼                             ▼
        ┌──────────────────────────────────┐  ┌──────────────────────────────────┐
        │ Hybrid Search + Reranker         │  │ 4-Tool Groq Function-Calling     │
        │ • Dense (all-MiniLM-L6-v2)       │  │ • search_curriculum              │
        │ • Pega-Aware Okapi BM25          │  │ • search_challenges              │
        │ • Weighted RRF + Cross-Encoder   │  │ • read_full_page                 │
        └─────────────────┬────────────────┘  │ • lookup_pega_terms              │
                          │                   └─────────────────┬────────────────┘
                          ▼                                     │
        ┌──────────────────────────────────┐                    │
        │ Stage 3 Evidence Sufficiency     │                    │
        │ (if insufficient & tool-capable: │────────────────────┘
        │  max 1 escalation to Agent)      │  (1-escalation cap)
        └─────────────────┬────────────────┘
                          │
                          ▼
        ┌───────────────────────────────────────────────────────────────┐
        │ Grounded Generation + Streaming + Completeness Guard          │
        │ + Format Detection (TABLE / BULLETS / NUMBERED_STEPS / etc.)  │
        │ + Two-Stage verify_citations() + Expandable Execution Details │
        └───────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
PegaRagAgent/
├── app.py                        # Streamlit Bento UI entry point
├── rag_engine.py                 # Backward-compatible PegaRAGEngine facade & exports
├── requirements.txt              # Runtime dependencies
├── requirements-dev.txt          # Development/test dependency (pytest)
├── .env.example                  # Environment variable template
├── LAUNCH_AI_ASSISTANT.bat       # Windows launcher (auto-detects ..\PEGA-EXP-1 venv)
├── RUN_DOWNLOADER.bat            # Windows sequential launcher for curriculum & challenge scrapers
├── pega_downloader.py            # Playwright scraper for Pega Academy modules
├── download_challenges.py        # Playwright scraper for hands-on challenges
│
├── config/
│   ├── __init__.py
│   └── settings.py               # Paths, chunking, retrieval defaults, router flags, MODEL_REGISTRY
├── ingestion/
│   ├── __init__.py
│   └── extractors.py             # PyMuPDF PDF, section/table-aware DOCX, SHA-256 hashing
├── retrieval/
│   ├── __init__.py
│   ├── hybrid.py                 # tokenize_pega_text, BM25Retriever, weighted RRF, Cross-Encoder
│   └── grounding.py              # deduplicate_sources, verify_citations, contextualize_query_with_history
├── agents/
│   ├── __init__.py
│   ├── tools.py                  # 4 local research tool schemas & executors
│   ├── answer_generator.py       # Prompts, format detection, completeness guards, streaming/non-streaming
│   ├── research_agent.py         # Conversational fast path, router, run_agent*, run_adaptive*, escalation
│   └── routing_eval.py           # RoutingMetricsTracker, 24-case eval dataset, Markdown eval report
├── ui/
│   ├── __init__.py
│   ├── state.py                  # Session state initialization & chat queue helpers
│   ├── styles.py                 # Bento CSS & Lucide SVG icons
│   ├── chat_panel.py             # Streaming chat, citation badges, Sources, Execution Details
│   ├── config_panel.py           # API key, model selector, mode, Adaptive RAG toggle & observability
│   ├── upload_panel.py           # Sanitized file upload & Live Sync
│   └── explorer_panel.py         # Keyword/section chunk explorer & Delete/Delete-All workflows
├── tests/                        # 8 offline pytest modules (49 tests)
├── Lead System Architect/        # Curriculum & Challenge PDFs across 5 CLSA pillars
└── pega_rag_db/                  # Persistent ChromaDB (79 docs, 1,745 chunks) + indexed_manifest.json
```

---

## 📚 Pre-Indexed Knowledge Base

The local `pega_rag_db` comes pre-indexed with **79 documents (1,745 chunks)** across 5 Pega CLSA pillars:
* `01_LSA_Mindset`
* `02_Planning_and_Design`
* `03_Constellation_Adoption`
* `04_Infrastructure_and_Architecture`
* `05_Implementation_and_Deployment`

Catalog files `Pega_Course_Catalog.docx` and `README_Course_Catalog.docx` are also included in the repository root.

---

## ⚙️ Configuration, Models & Environment Variables

### Supported Groq Models (`config/settings.py`)
| Model ID | Context Window | Max Completion | Tool Calling | Role |
| :--- | :---: | :---: | :---: | :--- |
| `openai/gpt-oss-120b` | 131,072 | 2,048 | `True` (`Agent`) | Primary reasoning & agent model |
| `openai/gpt-oss-20b` | 131,072 | 1,500 | `True` (`Agent`) | Fallback tool-capable & router fallback model |
| `qwen/qwen3.8-27b` | 131,072 | 1,500 | `False` (`Direct`) | Primary Adaptive RAG intent router (`ROUTER_MODEL`) |
| `allam-2-7b` | 4,096 | 600 | `False` (`Direct`) | Compact direct-grounded model |

### Environment Variables
Copy `.env.example` to `.env` (or enter `GROQ_API_KEY` in the UI Configuration drawer):
* `GROQ_API_KEY` — Groq Cloud API key ([console.groq.com/keys](https://console.groq.com/keys)).
* `PEGA_ENABLE_ADAPTIVE_ROUTING` — `"false"` (default) or `"true"` to enable Adaptive RAG routing at startup.
* `PEGA_ROUTER_MODEL` — Intent router model (default: `qwen/qwen3.8-27b`).
* `PEGA_ROUTER_TIMEOUT` — Router timeout in seconds (default: `4.0`).
* `PEGA_RERANKER_MODEL` — Cross-encoder model (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`).
* `PEGA_ENABLE_RERANKER` — `"true"` (default) or `"false"`.
* `PEGA_SESSION_COOKIE`, `PEGA_SESSION_COOKIE_NAME`, `PEGA_TA_SESSION_ID`, `PEGA_HELLO_DATA` — Optional Pega Academy credentials used only by `pega_downloader.py` / `download_challenges.py`.

---

## 🚀 Installation, Running & Testing

### 1. Setup
```bash
git clone https://github.com/DeveshEai/PegaRagAgent.git
cd PegaRagAgent
pip install -r requirements.txt
```
*(For development/testing, also install `pip install -r requirements-dev.txt`.)*

### 2. Run the Application
* **Windows One-Click Launcher:** Double-click `LAUNCH_AI_ASSISTANT.bat` (automatically uses `..\PEGA-EXP-1\Scripts\python.exe` if present, otherwise falls back to `python`).
* **CLI (Windows / macOS / Linux):**
  ```bash
  streamlit run app.py --server.port=8501 --server.fileWatcherType=none
  ```
* **Optional Curriculum Downloader (Windows):** Run `RUN_DOWNLOADER.bat` to execute `pega_downloader.py` and `download_challenges.py` sequentially.

### 3. Run Automated Verification
```bash
python -m compileall -q .
python -m pytest tests/ -v
```
The offline test suite (`tests/`) covers ingestion, hybrid BM25/RRF/reranking, citation verification, follow-up contextualization, agent streaming/guards, adaptive routing (Stages 1–4), UX formatting/fast-path regressions, document management, and Streamlit UI startup.

---

## 🔒 Security & Privacy Considerations
* **Local Embeddings & Vector Storage:** Embeddings (`all-MiniLM-L6-v2`) and ChromaDB storage (`pega_rag_db/`) run locally on your machine.
* **Upload Sanitization:** `sanitize_uploaded_filename()` strips directory traversal paths and unsafe characters before saving uploaded files.
* **Telemetry Redaction:** `sanitize_routing_record()` strips API keys (`gsk_*`, `sk-*`), `Bearer` tokens, and session/cookie patterns before recording in-memory routing observability.
* **Secret Hygiene:** `.env` is excluded via `.gitignore`; never commit `.env` or Pega session cookies.

---

## 📄 License
This project is licensed under the MIT License — see [LICENSE](LICENSE) for details. Pega curriculum references belong to Pegasystems Inc.
