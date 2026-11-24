"""
Centralized configuration, paths, retrieval defaults, and Groq model registry
for PegaRagAgent. Independent of runtime RAG, ChromaDB, Groq, and Streamlit modules.
"""

import os
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv

# Base Directories (resolved relative to PegaRagAgent/ root)
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = WORKSPACE_DIR / "Lead System Architect"
DB_DIR = WORKSPACE_DIR / "pega_rag_db"
MANIFEST_PATH = DB_DIR / "indexed_manifest.json"

# Load local .env if present
load_dotenv(WORKSPACE_DIR / ".env")

# Database & Embedding Configuration
COLLECTION_NAME = "pega_lsa_curriculum"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Chunking & Extraction Configuration
CHUNK_SIZE = 950
CHUNK_OVERLAP = 180
DOCX_EXTRACTOR_VERSION = 2
TABLE_ATOMIC_CHAR_LIMIT = 1800

# Hybrid Retrieval & Reranking Configuration
RERANKER_MODEL_NAME = os.environ.get("PEGA_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
ENABLE_RERANKER = os.environ.get("PEGA_ENABLE_RERANKER", "true").lower() in ("1", "true", "yes")
RRF_K = 60
DEFAULT_CANDIDATE_K = 15
DEFAULT_FINAL_EVIDENCE_K = 4
MIN_RERANK_PROBABILITY = 0.08
BM25_K1 = 1.5
BM25_B = 0.75

# ---------------------------------------------------------
# Centralized Model Registry & Token Limits (Groq Nov 2026)
# ---------------------------------------------------------

PRIMARY_GROQ_MODEL = "openai/gpt-oss-120b"
FALLBACK_GROQ_MODEL = "openai/gpt-oss-20b"

# Adaptive Query Routing Configuration (feature flag disabled until Stage 5 passes)
ROUTER_MODEL = os.environ.get("PEGA_ROUTER_MODEL", "qwen/qwen3.8-27b")
ENABLE_ADAPTIVE_ROUTING = os.environ.get("PEGA_ENABLE_ADAPTIVE_ROUTING", "false").lower() in ("1", "true", "yes")
MAX_ROUTING_ESCALATIONS = 1
ROUTER_TIMEOUT_SECONDS = float(os.environ.get("PEGA_ROUTER_TIMEOUT", "4.0"))
INTENT_CATEGORIES = (
    "SIMPLE_LOOKUP",
    "TECHNICAL_EXPLANATION",
    "COMPARISON",
    "COMPLEX_ARCHITECTURE",
    "MULTI_DOCUMENT_RESEARCH",
    "STEP_BY_STEP_CHALLENGE",
)

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Primary & Fallback Production Tool-Capable Models (131K context)
    "openai/gpt-oss-120b": {
        "context_window": 131072,
        "max_completion_tokens": 2048,
        "tools_supported": True,
        "prompt_budget": 12000,
        "priority": 0
    },
    "openai/gpt-oss-20b": {
        "context_window": 131072,
        "max_completion_tokens": 1500,
        "tools_supported": True,
        "prompt_budget": 10000,
        "priority": 1
    },
    # Secondary Active Text Models
    "qwen/qwen3.8-27b": {
        "context_window": 131072,
        "max_completion_tokens": 1500,
        "tools_supported": False,
        "prompt_budget": 10000,
        "priority": 2
    },
    # Small / Specialized Context Models
    "allam-2-7b": {
        "context_window": 4096,
        "max_completion_tokens": 600,
        "tools_supported": False,
        "prompt_budget": 2200,
        "priority": 3
    }
}

DEFAULT_UNKNOWN_SPEC: Dict[str, Any] = {
    "context_window": 4096,
    "max_completion_tokens": 600,
    "tools_supported": False,
    "prompt_budget": 2200,
    "priority": 4
}


def get_model_spec(model_name: str) -> Dict[str, Any]:
    """Retrieves verified model specifications or returns a conservative default."""
    if not model_name:
        return MODEL_REGISTRY[PRIMARY_GROQ_MODEL]

    clean_name = model_name.strip()
    if clean_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[clean_name]

    low = clean_name.lower()
    if "gpt-oss-120b" in low:
        return MODEL_REGISTRY[PRIMARY_GROQ_MODEL]
    if "gpt-oss-20b" in low or "gpt-oss" in low:
        return MODEL_REGISTRY[FALLBACK_GROQ_MODEL]
    if "qwen" in low:
        return {
            "context_window": 131072,
            "max_completion_tokens": 1500,
            "tools_supported": False,
            "prompt_budget": 10000,
            "priority": 2
        }

    return dict(DEFAULT_UNKNOWN_SPEC)


def is_tool_capable_model(model_name: str) -> bool:
    """Returns True only if the model is verified to support function/tool calling."""
    return get_model_spec(model_name).get("tools_supported", False)
