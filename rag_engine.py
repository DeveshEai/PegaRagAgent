"""
Pega Architecture RAG Engine
=============================
- High-precision page-aware document parsing (PDF via PyMuPDF, DOCX via python-docx)
- Semantic chunking with rich context headers (Document name, section, category, page)
- Incremental indexing using persistent ChromaDB + SentenceTransformers (all-MiniLM-L6-v2)
- Fast generation via Groq API (openai/gpt-oss-120b / openai/gpt-oss-20b)
"""

import os
import re
import math
import json
import hashlib
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv

# Load local .env if present
load_dotenv(Path(__file__).parent / ".env")

import fitz  # PyMuPDF
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

# Base Directories
WORKSPACE_DIR = Path(__file__).parent.resolve()
DOCS_DIR = WORKSPACE_DIR / "Lead System Architect"
DB_DIR = WORKSPACE_DIR / "pega_rag_db"
MANIFEST_PATH = DB_DIR / "indexed_manifest.json"

COLLECTION_NAME = "pega_lsa_curriculum"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Hybrid Retrieval & Reranking Configuration
RERANKER_MODEL_NAME = os.environ.get("PEGA_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
ENABLE_RERANKER = os.environ.get("PEGA_ENABLE_RERANKER", "true").lower() in ("1", "true", "yes")
RRF_K = 60
DEFAULT_CANDIDATE_K = 15
DEFAULT_FINAL_EVIDENCE_K = 4
MIN_RERANK_PROBABILITY = 0.08
DOCX_EXTRACTOR_VERSION = 2
TABLE_ATOMIC_CHAR_LIMIT = 1800

# ---------------------------------------------------------
# Centralized Model Registry & Token Limits (Groq Nov 2026)
# ---------------------------------------------------------

PRIMARY_GROQ_MODEL = "openai/gpt-oss-120b"
FALLBACK_GROQ_MODEL = "openai/gpt-oss-20b"

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


def estimate_tokens(text: str) -> int:
    """
    Lightweight, conservative token estimator (~3.6 characters per token).
    Avoids introducing heavy tokenizer dependencies or network calls.
    """
    if not text:
        return 0
    return max(1, int(len(text) / 3.6))


def deduplicate_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicates a list of retrieved source dicts while preserving order.
    Uses source filename, page, and a SHA-256 hash of the normalized full chunk body
    so distinct table chunks and passages from the same section are never collapsed.
    """
    unique_sources = []
    seen = set()
    for s in sources:
        body = clean_chunk_content(s.get("document", ""))
        normalized_body = re.sub(r"\s+", " ", body).strip()
        content_hash = hashlib.sha256(normalized_body.encode("utf-8")).hexdigest()
        key = (s.get("source", "Unknown"), str(s.get("page", "1")), content_hash)
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)
    return unique_sources


def clean_chunk_content(text: str) -> str:
    """
    Removes redundant pre-pended metadata headers from chunk text if present.
    In ChromaDB, chunks are stored with 'Document: ... Section: ... Content:\n'.
    Stripping this header avoids duplicate metadata inside the final prompt.
    """
    if not text:
        return ""
    if text.startswith("Document: ") and "\nContent:\n" in text:
        parts = text.split("\nContent:\n", 1)
        if len(parts) == 2:
            return parts[1].strip()
    return text.strip()


def build_grounded_context_prompt(
    sources: List[Dict[str, Any]],
    max_token_budget: int = 5000
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Constructs a clean, deduplicated context string within the allocated token budget.
    Returns the formatted context string and the list of retained unique sources.
    """
    unique_sources = deduplicate_sources(sources)
    context_blocks = []
    retained_sources = []
    current_tokens = 0

    for i, src in enumerate(unique_sources, 1):
        source_name = src.get("source", "Unknown")
        page_num = str(src.get("page", "1"))
        section_name = src.get("section", "General")
        category_name = src.get("category", "Master Module")
        clean_text = clean_chunk_content(src.get("document", ""))
        
        block = (
            f"--- SOURCE [{i}] ---\n"
            f"File: {source_name} (Page {page_num})\n"
            f"Section: {section_name} ({category_name})\n"
            f"Content:\n{clean_text}\n"
        )
        block_tokens = estimate_tokens(block)
        
        # Stop if adding this block would exceed the available token budget
        if retained_sources and (current_tokens + block_tokens > max_token_budget):
            break
            
        context_blocks.append(block)
        retained_sources.append(src)
        current_tokens += block_tokens

    context_str = "\n".join(context_blocks)
    return context_str, retained_sources


def is_context_length_error(err_msg: str) -> bool:
    """Checks if a Groq error message indicates a context_length_exceeded condition."""
    low = str(err_msg).lower()
    return any(x in low for x in [
        "context_length_exceeded",
        "reduce the length of the messages",
        "context length",
        "maximum context length"
    ])


GROUNDED_SYSTEM_PROMPT = (
    "You are a Senior Pega Certified Lead System Architect (CLSA) advising a fellow architect in a natural, conversational peer-to-peer style.\n\n"
    "RESPONSE STYLE & COMPLEXITY:\n"
    "1. Answer directly from the first sentence without mandatory or formulaic headings (never use boilerplate headers such as 'Short answer', 'What the excerpts actually tell us', 'Implication for architecture and governance', or repetitive 'Bottom line' wrap-ups).\n"
    "2. Match length to question complexity:\n"
    "   - Simple definitions (e.g., 'What is X?'): 80–150 words.\n"
    "   - Technical explanations or comparisons: 150–300 words.\n"
    "   - Complex architecture scenarios: as detailed as necessary.\n"
    "3. Include practical Pega examples (e.g., `pyWorkPage`, `Queue Processor`, `Job Scheduler`, `Rule-Obj-Class`, `Work-`, `Data-`, `Int-`) and explain architectural implications (performance, locking, reusability, governance) when genuinely relevant.\n\n"
    "STRICT GROUNDING & CITATION INTEGRITY:\n"
    "4. Clearly separate retrieved evidence from general Pega knowledge: cite `[filename, Page X, Section: Y]` only for claims directly supported by the provided excerpts.\n"
    "5. Never present general Pega knowledge as if it came from the retrieved documents, and never invent citations or attach unrelated sources to a claim.\n"
    "6. If the retrieved excerpts lack sufficient detail, explicitly state that gap (for example: 'The retrieved curriculum excerpts do not cover ...') and still provide helpful general Pega CLSA guidance clearly marked as general knowledge without citations."
)

AGENT_SYSTEM_PROMPT = (
    "You are a Senior Pega Certified Lead System Architect (CLSA) advising a fellow architect in a natural, conversational peer-to-peer style.\n\n"
    "TOOL USE & RESEARCH:\n"
    "1. Call your tools (`search_curriculum`, `search_challenges`, `read_full_page`, `lookup_pega_terms`) to retrieve verified Pega excerpts before answering. Query both curriculum and challenges when a topic spans design theory and hands-on configuration, and use `read_full_page` if an excerpt is truncated.\n\n"
    "RESPONSE STYLE & COMPLEXITY:\n"
    "2. Answer directly without mandatory or formulaic headings (avoid boilerplate sections like 'Short answer', 'What the excerpts actually tell us', 'Implication for architecture and governance', or repetitive 'Bottom line' conclusions).\n"
    "3. Match depth to complexity: 80–150 words for simple definitions, 150–300 words for technical explanations/comparisons, and as detailed as necessary for complex architecture scenarios.\n"
    "4. Include practical Pega examples and architectural implications where useful.\n\n"
    "STRICT GROUNDING & CITATION INTEGRITY:\n"
    "5. Clearly distinguish retrieved curriculum facts (cited inline as `[filename, Page X, Section: Y]` using only files/pages returned by your tools) from general Pega knowledge. Never invent citations, attach unrelated sources, or present general knowledge as retrieved evidence.\n"
    "6. If retrieved excerpts are insufficient after searching, explicitly note that limitation and provide helpful general Pega guidance clearly labeled as general knowledge without citations."
)


# ---------------------------------------------------------
# Hybrid Retrieval (BM25 + RRF), Reranking & Citation Guard
# ---------------------------------------------------------

def tokenize_pega_text(text: str) -> List[str]:
    """
    Pega-aware tokenizer preserving exact technical identifiers (e.g., Rule-Obj-Class,
    pyWorkPage, D_pxGetPage) while also emitting lowercase sub-tokens for broad recall.
    """
    if not text:
        return []
    cleaned = clean_chunk_content(text)
    raw_tokens = re.findall(r"[A-Za-z0-9_.\-]+", cleaned)
    tokens: List[str] = []
    for tok in raw_tokens:
        low = tok.lower().strip("._-")
        if not low:
            continue
        tokens.append(low)
        # Emit sub-tokens for hyphenated/underscored/camelCase identifiers
        if "-" in low or "_" in low or "." in low:
            parts = [p for p in re.split(r"[-_.]", low) if p]
            if len(parts) > 1:
                tokens.extend(parts)
        else:
            camel_parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", tok)
            if len(camel_parts) > 1:
                tokens.extend(p.lower() for p in camel_parts if p)
    return tokens


class BM25Retriever:
    """Lightweight in-memory Okapi BM25 retriever built over existing ChromaDB chunks."""

    def __init__(self, documents: List[str], metadatas: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.metadatas = metadatas
        self.k1 = k1
        self.b = b
        self.num_docs = len(documents)
        self.doc_freqs: List[Counter] = []
        self.doc_lengths: List[int] = []
        self.idf: Dict[str, float] = {}
        self.avgdl: float = 0.0
        self._build_index()

    def _build_index(self) -> None:
        df: Counter = Counter()
        total_len = 0
        for doc in self.documents:
            tokens = tokenize_pega_text(doc)
            tf = Counter(tokens)
            self.doc_freqs.append(tf)
            dl = len(tokens)
            self.doc_lengths.append(dl)
            total_len += dl
            for term in tf.keys():
                df[term] += 1

        self.avgdl = (total_len / self.num_docs) if self.num_docs > 0 else 1.0
        for term, freq in df.items():
            # Standard Okapi BM25 IDF with +1 floor to avoid negative weights on frequent terms
            self.idf[term] = math.log(1.0 + (self.num_docs - freq + 0.5) / (freq + 0.5))

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_CANDIDATE_K,
        section_filter: Optional[str] = None,
        category_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if self.num_docs == 0:
            return []

        query_tokens = tokenize_pega_text(query)
        if not query_tokens:
            return []

        scored_indices: List[Tuple[int, float]] = []
        for idx in range(self.num_docs):
            meta = self.metadatas[idx] or {}
            if section_filter and section_filter != "All Sections":
                if meta.get("section") != section_filter:
                    continue
            if category_filter and meta.get("category") != category_filter:
                continue

            tf = self.doc_freqs[idx]
            dl = self.doc_lengths[idx]
            score = 0.0
            for q_tok in query_tokens:
                if q_tok not in tf:
                    continue
                freq = tf[q_tok]
                idf = self.idf.get(q_tok, 0.0)
                denom = freq + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
                score += idf * ((freq * (self.k1 + 1.0)) / denom)

            if score > 0.0:
                scored_indices.append((idx, score))

        scored_indices.sort(key=lambda x: x[1], reverse=True)
        top_items = scored_indices[:top_k]
        max_bm25 = top_items[0][1] if top_items else 1.0

        results: List[Dict[str, Any]] = []
        for idx, raw_score in top_items:
            meta = self.metadatas[idx] or {}
            norm_score = round((raw_score / max_bm25) * 100.0, 1) if max_bm25 > 0 else 0.0
            results.append({
                "document": self.documents[idx],
                "source": meta.get("source", "Unknown"),
                "page": str(meta.get("page", "1")),
                "section": meta.get("section", "General"),
                "category": meta.get("category", "Document"),
                "relative_path": meta.get("relative_path", ""),
                "score": norm_score,
                "bm25_score": round(raw_score, 4)
            })
        return results


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    rrf_k: int = RRF_K,
    top_k: int = DEFAULT_CANDIDATE_K,
    dense_weight: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Combines dense vector search and sparse BM25 rankings using weighted Reciprocal Rank Fusion (RRF).
    Preserves document metadata and tracks individual retrieval signals.
    """
    fused_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    rrf_scores: Dict[Tuple[str, str, str], float] = {}
    sparse_weight = max(0.0, min(1.0, 1.0 - dense_weight))
    dense_w = max(0.0, min(1.0, dense_weight))

    def _chunk_key(item: Dict[str, Any]) -> Tuple[str, str, str]:
        cleaned = clean_chunk_content(item.get("document", ""))
        return (item.get("source", "Unknown"), str(item.get("page", "1")), cleaned[:100])

    for rank, item in enumerate(dense_results, 1):
        key = _chunk_key(item)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (dense_w / (rrf_k + rank))
        if key not in fused_map:
            entry = dict(item)
            entry["dense_score"] = item.get("score", 0.0)
            entry["bm25_score"] = 0.0
            fused_map[key] = entry

    for rank, item in enumerate(sparse_results, 1):
        key = _chunk_key(item)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (sparse_weight / (rrf_k + rank))
        if key not in fused_map:
            entry = dict(item)
            entry["dense_score"] = 0.0
            fused_map[key] = entry
        else:
            fused_map[key]["bm25_score"] = item.get("bm25_score", 0.0)

    ranked = sorted(fused_map.items(), key=lambda kv: rrf_scores[kv[0]], reverse=True)[:top_k]
    max_rrf = rrf_scores[ranked[0][0]] if ranked else 1.0

    results: List[Dict[str, Any]] = []
    for key, entry in ranked:
        raw_rrf = rrf_scores[key]
        entry["rrf_score"] = round(raw_rrf, 5)
        entry["score"] = round((raw_rrf / max_rrf) * 100.0, 1) if max_rrf > 0 else entry.get("score", 0.0)
        entry["retrieval_method"] = "Hybrid (Dense + BM25 RRF)"
        results.append(entry)
    return results


class CrossEncoderReranker:
    """
    Configurable Cross-Encoder reranker powered by sentence_transformers.CrossEncoder.
    Degrades gracefully to RRF ordering if the model is unavailable or disabled.
    """

    def __init__(self, model_name: str = RERANKER_MODEL_NAME, enabled: bool = ENABLE_RERANKER):
        self.model_name = model_name
        self.enabled = enabled
        self._model = None
        self._unavailable = False

    def _load_model(self) -> bool:
        if not self.enabled or self._unavailable:
            return False
        if self._model is not None:
            return True
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            return True
        except Exception as e:
            print(f"Cross-encoder reranker unavailable ({self.model_name}): {e}")
            self._unavailable = True
            return False

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = DEFAULT_FINAL_EVIDENCE_K
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        if not self._load_model():
            return candidates[:top_k]

        try:
            pairs = [(query, clean_chunk_content(c.get("document", ""))) for c in candidates]
            raw_logits = self._model.predict(pairs)

            scored_items: List[Tuple[Dict[str, Any], float]] = []
            for item, logit in zip(candidates, raw_logits):
                # Clamp logit to avoid overflow in math.exp
                clamped = max(-20.0, min(20.0, float(logit)))
                prob = 1.0 / (1.0 + math.exp(-clamped))
                enriched = dict(item)
                enriched["rerank_score"] = round(prob, 4)
                enriched["score"] = round(prob * 100.0, 1)
                enriched["retrieval_method"] = "Hybrid + Cross-Encoder Reranked"
                scored_items.append((enriched, prob))

            scored_items.sort(key=lambda x: x[1], reverse=True)
            max_prob = scored_items[0][1] if scored_items else 0.0

            # Filter clearly irrelevant tail chunks when strong matches exist
            if max_prob >= 0.30:
                filtered = [item for item, prob in scored_items if prob >= MIN_RERANK_PROBABILITY]
                if filtered:
                    return filtered[:top_k]

            return [item for item, _ in scored_items[:top_k]]
        except Exception as e:
            print(f"Cross-encoder scoring failed, falling back to RRF ranking: {e}")
            return candidates[:top_k]


_CITATION_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "can", "may", "might", "must", "this", "that", "these", "those", "it", "its",
    "they", "their", "them", "you", "your", "we", "our", "when", "where", "which",
    "who", "how", "what", "why", "if", "then", "than", "so", "not", "no", "yes",
    "use", "used", "using", "based", "also", "such", "only", "more", "most", "into",
    "pega", "page", "section", "pdf", "document", "architecture", "system"
}


def _extract_meaningful_tokens(text: str) -> set:
    """Extracts meaningful non-stopword Pega/technical tokens for lexical claim checking."""
    tokens = tokenize_pega_text(text)
    return {t for t in tokens if len(t) > 1 and t not in _CITATION_STOPWORDS}


def verify_citations(
    answer_text: str,
    retrieved_sources: List[Dict[str, Any]]
) -> Tuple[str, Dict[str, Any]]:
    """
    Two-stage citation verification against retrieved_sources:
    1. Existence check: confirms the cited (filename, page) pair exists in retrieved_sources.
    2. Lexical claim-to-evidence check: verifies that the surrounding claim sentence shares
       meaningful technical/content token overlap with the cited page's retrieved text.
       (Note: uses lexical token overlap, not NLI semantic entailment.)
    """
    if not answer_text:
        return answer_text, {"verified": 0, "removed": 0, "mismatched_claims": 0}

    valid_pairs = set()
    evidence_tokens_by_pair: Dict[Tuple[str, str], set] = {}
    for src in retrieved_sources:
        fname = str(src.get("source", "")).strip()
        page = str(src.get("page", "")).strip()
        if fname:
            pair = (fname.lower(), page.lower())
            valid_pairs.add(pair)
            tokens = _extract_meaningful_tokens(src.get("document", ""))
            if pair in evidence_tokens_by_pair:
                evidence_tokens_by_pair[pair] |= tokens
            else:
                evidence_tokens_by_pair[pair] = tokens

    citation_pattern = re.compile(
        r"\[([^\[\]]+?\.(?:pdf|docx?|txt|md))[^\[\]]*?Page\s+([^\s,\]\)]+)[^\[\]]*\]",
        re.IGNORECASE
    )

    verified_count = 0
    removed_count = 0
    mismatched_claims = 0

    def _resolve_pair(cited_file: str, cited_page: str) -> Optional[Tuple[str, str]]:
        if (cited_file, cited_page) in valid_pairs:
            return (cited_file, cited_page)
        for vf, vp in valid_pairs:
            if (cited_file in vf or vf in cited_file) and cited_page == vp:
                return (vf, vp)
        return None

    def _check_match(match: re.Match) -> str:
        nonlocal verified_count, removed_count, mismatched_claims
        full_cite = match.group(0)
        cited_file = match.group(1).strip().lower()
        cited_page = match.group(2).strip().rstrip(",;.").lower()

        resolved = _resolve_pair(cited_file, cited_page)
        if resolved is None:
            removed_count += 1
            return ""

        # Lexical claim-to-evidence check on the sentence preceding this citation
        start_idx = match.start()
        preceding = answer_text[max(0, start_idx - 350):start_idx]
        # Take the current sentence/bullet fragment
        last_break = max(preceding.rfind("."), preceding.rfind("\n"), preceding.rfind("!"), preceding.rfind("?"))
        claim_fragment = preceding[last_break + 1:] if last_break != -1 else preceding
        claim_tokens = _extract_meaningful_tokens(claim_fragment)

        if len(claim_tokens) >= 3:
            ev_tokens = evidence_tokens_by_pair.get(resolved, set())
            overlap_tokens = claim_tokens & ev_tokens
            overlap_ratio = len(overlap_tokens) / len(claim_tokens)
            if len(overlap_tokens) < 2 and overlap_ratio < 0.12:
                mismatched_claims += 1
                return ""

        verified_count += 1
        return full_cite

    verified_text = citation_pattern.sub(_check_match, answer_text)
    verified_text = re.sub(r"  +", " ", verified_text).strip()
    return verified_text, {
        "verified": verified_count,
        "removed": removed_count,
        "mismatched_claims": mismatched_claims
    }


def contextualize_query_with_history(
    query: str,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Expands short or referential follow-up questions with the previous user topic
    so Hybrid BM25 + Vector retrieval maintains thread context.
    """
    if not chat_history:
        return query
    q_low = query.lower().strip()
    referential_signals = (
        "that", "this", "it", "they", "them", "those", "these",
        "step ", "elaborate", "explain more", "apply", "example", "how about", "what about"
    )
    is_short = len(q_low.split()) <= 10
    if is_short or any(sig in q_low for sig in referential_signals):
        prev_user_msgs = [
            m.get("content", "").strip()
            for m in chat_history
            if m.get("role") == "user" and m.get("content", "").strip()
        ]
        if prev_user_msgs:
            last_topic = prev_user_msgs[-1][:200]
            return f"{last_topic} — {query}"
    return query


def should_use_agent_loop(query: str) -> Tuple[bool, str]:
    """
    Lightweight query router:
    - Straightforward conceptual or comparison questions route to fast Hybrid + Reranked Direct Retrieval.
    - Multi-part, cross-pillar, or hands-on + theory investigative queries route to the multi-step Agent Loop.
    """
    q_low = query.lower().strip()
    investigative_signals = [
        "challenge", "walkthrough", "step-by-step", "step by step",
        "end-to-end", "end to end", "and also", "along with",
        "compare and implement", "design and configure", "full page", "page "
    ]
    multi_clause = (len(q_low.split()) >= 20 and (" and " in q_low or ";" in q_low or "?" in q_low[:-1]))
    if any(sig in q_low for sig in investigative_signals) or multi_clause:
        return True, "Multi-step investigative query detected; using autonomous tool-calling agent."
    return False, "Straightforward architectural query; using Hybrid Retrieval + Cross-Encoder Reranking."

# ---------------------------------------------------------
# Document Extraction with Rich Page-Level Metadata
# ---------------------------------------------------------

def extract_pdf_pages(file_path: Path) -> List[Dict[str, Any]]:
    """Extracts text page-by-page from a PDF, preserving exact page numbers."""
    pages_data = []
    try:
        doc = fitz.open(file_path)
        for page_idx, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                pages_data.append({
                    "page_number": page_idx + 1,
                    "text": text
                })
        doc.close()
    except Exception as e:
        print(f"Error reading PDF {file_path.name}: {e}")
    return pages_data


def _normalize_docx_table_markdown(table: Any) -> Dict[str, Any]:
    """
    Converts a python-docx Table into a clean Markdown table representation,
    normalizing whitespace and multiline/merged cells.
    Returns dict with keys: markdown, header_lines, data_lines, rows, cols, is_empty.
    """
    raw_rows: List[List[str]] = []
    for row in table.rows:
        cells_text: List[str] = []
        for cell in row.cells:
            # Collapse internal newlines/tabs/multiple spaces in merged or multiline cells
            cleaned = re.sub(r"\s+", " ", cell.text or "").strip()
            # Escape literal pipes so Markdown column alignment stays valid
            cleaned = cleaned.replace("|", "\\|")
            cells_text.append(cleaned)
        if any(c for c in cells_text):
            raw_rows.append(cells_text)

    if not raw_rows:
        return {
            "markdown": "",
            "header_lines": [],
            "data_lines": [],
            "rows": 0,
            "cols": len(table.columns) if hasattr(table, "columns") else 0,
            "is_empty": True,
        }

    num_cols = max(len(r) for r in raw_rows)
    padded_rows = [r + [""] * (num_cols - len(r)) for r in raw_rows]

    header = padded_rows[0]
    header_line = "| " + " | ".join(header) + " |"
    sep_line = "| " + " | ".join(["---"] * num_cols) + " |"
    data_lines = ["| " + " | ".join(r) + " |" for r in padded_rows[1:]]

    all_lines = [header_line, sep_line] + data_lines
    return {
        "markdown": "\n".join(all_lines),
        "header_lines": [header_line, sep_line],
        "data_lines": data_lines,
        "rows": len(padded_rows),
        "cols": num_cols,
        "is_empty": False,
    }


def _split_markdown_table_by_rows(
    header_lines: List[str],
    data_lines: List[str],
    prefix_context: str,
    max_chars: int = TABLE_ATOMIC_CHAR_LIMIT
) -> List[str]:
    """
    Splits an oversized Markdown table at row boundaries while repeating
    the header + separator rows and context prefix in every chunk.
    """
    header_block = "\n".join(header_lines)
    if not data_lines:
        return [f"{prefix_context}\n{header_block}".strip()]

    chunks: List[str] = []
    current_rows: List[str] = []
    base_len = len(prefix_context) + 1 + len(header_block) + 1

    for row_line in data_lines:
        projected_len = base_len + sum(len(r) + 1 for r in current_rows) + len(row_line)
        if current_rows and projected_len > max_chars:
            chunk_body = f"{prefix_context}\n{header_block}\n" + "\n".join(current_rows)
            chunks.append(chunk_body.strip())
            current_rows = [row_line]
        else:
            current_rows.append(row_line)

    if current_rows:
        chunk_body = f"{prefix_context}\n{header_block}\n" + "\n".join(current_rows)
        chunks.append(chunk_body.strip())

    return chunks


def extract_docx_sections(file_path: Path) -> List[Dict[str, Any]]:
    """
    Extracts structured sections from DOCX documents in true XML document order,
    preserving Heading 1/2/3 hierarchy, paragraphs, and Markdown-formatted tables.
    """
    sections_data: List[Dict[str, Any]] = []
    para_count = 0
    heading_count = 0
    table_count = 0
    empty_table_count = 0
    table_dims: List[str] = []

    try:
        from docx.oxml.text.paragraph import CT_P
        from docx.oxml.table import CT_Tbl
        from docx.text.paragraph import Paragraph
        from docx.table import Table

        doc = docx.Document(file_path)
        heading_stack: Dict[int, Optional[str]] = {1: None, 2: None, 3: None}
        current_heading = "General Overview"
        current_heading_path = "General Overview"
        current_blocks: List[Dict[str, Any]] = []
        prose_buffer: List[str] = []

        def _flush_prose() -> None:
            if prose_buffer:
                current_blocks.append({
                    "type": "prose",
                    "text": "\n".join(prose_buffer)
                })
                prose_buffer.clear()

        def _flush_section() -> None:
            _flush_prose()
            if current_blocks:
                flat_parts = [
                    b["text"] if b["type"] == "prose" else b["markdown"]
                    for b in current_blocks
                ]
                sections_data.append({
                    "page_number": current_heading,
                    "heading_path": current_heading_path,
                    "text": "\n\n".join(flat_parts),
                    "blocks": list(current_blocks),
                })
                current_blocks.clear()

        def _build_heading_path() -> str:
            parts = [heading_stack[lvl] for lvl in (1, 2, 3) if heading_stack[lvl]]
            return " > ".join(parts) if parts else current_heading

        for child in doc.element.body.iterchildren():
            if isinstance(child, CT_P):
                p = Paragraph(child, doc)
                text = re.sub(r"\s+", " ", p.text or "").strip()
                if not text:
                    continue
                style_name = (p.style.name if p.style and p.style.name else "").strip()
                if style_name.startswith("Heading") or style_name == "Title":
                    _flush_section()
                    heading_count += 1
                    lvl_match = re.search(r"Heading\s+(\d+)", style_name)
                    if lvl_match:
                        lvl = min(3, max(1, int(lvl_match.group(1))))
                        heading_stack[lvl] = text
                        for sub_lvl in range(lvl + 1, 4):
                            heading_stack[sub_lvl] = None
                    else:
                        heading_stack[1] = text
                        heading_stack[2] = None
                        heading_stack[3] = None
                    current_heading = text
                    current_heading_path = _build_heading_path()
                else:
                    para_count += 1
                    prose_buffer.append(text)

            elif isinstance(child, CT_Tbl):
                _flush_prose()
                tbl = Table(child, doc)
                tbl_info = _normalize_docx_table_markdown(tbl)
                if tbl_info["is_empty"]:
                    empty_table_count += 1
                    print(f"[DOCX Extract WARNING] {file_path.name}: skipped empty table under '{current_heading_path}'")
                    continue
                table_count += 1
                table_dims.append(f"{tbl_info['rows']}x{tbl_info['cols']}")
                current_blocks.append({
                    "type": "table",
                    "markdown": tbl_info["markdown"],
                    "header_lines": tbl_info["header_lines"],
                    "data_lines": tbl_info["data_lines"],
                    "rows": tbl_info["rows"],
                    "cols": tbl_info["cols"],
                })

        _flush_section()
        print(
            f"[DOCX Extract] {file_path.name}: paragraphs={para_count}, "
            f"headings={heading_count}, tables={table_count} ({', '.join(table_dims) if table_dims else 'none'}), "
            f"empty_tables={empty_table_count}"
        )
    except Exception as e:
        print(f"Error reading DOCX {file_path.name}: {e}")
    return sections_data


def compute_file_hash(file_path: Path) -> str:
    """Computes SHA-256 content hash of a file's bytes."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)
    return sha256.hexdigest()


def _legacy_md5_hash(file_path: Path) -> str:
    """Legacy filename/size/mtime MD5 hash used only for backward-compatible manifest migration."""
    stat = file_path.stat()
    hash_str = f"{file_path.name}_{stat.st_size}_{stat.st_mtime}"
    return hashlib.md5(hash_str.encode()).hexdigest()


def compute_similarity_score(distance: float, metric: str = "l2") -> float:
    """
    Converts ChromaDB distance to a [0, 100] similarity percentage matching the collection metric.
    - cosine: Chroma returns 1 - cos_sim in [0, 2] -> cos_sim = 1 - distance
    - ip:     Chroma returns 1 - dot_product       -> sim = 1 - distance
    - l2:     For unit-normalized SentenceTransformer embeddings (||u||=||v||=1),
              ||u - v||^2 = 2 * (1 - cos_sim) in [0, 4] -> cos_sim = 1 - (distance / 2.0)
    """
    m = (metric or "l2").lower()
    if m in ("cosine", "ip"):
        sim = 1.0 - float(distance)
    else:
        sim = 1.0 - (float(distance) / 2.0)
    return round(max(0.0, min(100.0, sim * 100.0)), 1)


# ---------------------------------------------------------
# RAG Engine Class
# ---------------------------------------------------------

class PegaRAGEngine:
    def __init__(self, db_dir: Path = DB_DIR):
        self.db_dir = db_dir
        self.db_dir.mkdir(parents=True, exist_ok=True)
        
        # Local SentenceTransformer embedding function (100% offline, free, fast)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
        
        # Persistent ChromaDB client (preserve existing collection metadata/index; new DBs use cosine)
        self.chroma_client = chromadb.PersistentClient(path=str(self.db_dir))
        existing_names = {c.name for c in self.chroma_client.list_collections()}
        if COLLECTION_NAME in existing_names:
            self.collection = self.chroma_client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=self.embedding_fn
            )
        else:
            self.collection = self.chroma_client.create_collection(
                name=COLLECTION_NAME,
                embedding_function=self.embedding_fn,
                metadata={
                    "description": "Pega Academy LSA Curriculum & Challenges",
                    "hnsw:space": "cosine"
                }
            )
        self.distance_metric = ((self.collection.metadata or {}).get("hnsw:space") or "l2").lower()
        
        # Recursive text splitter tuned for technical enterprise docs
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=950,
            chunk_overlap=180,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""]
        )

        # Lazy-initialized in-memory BM25 retriever & Cross-Encoder reranker
        self._bm25_retriever: Optional[BM25Retriever] = None
        self.reranker = CrossEncoderReranker(
            model_name=RERANKER_MODEL_NAME,
            enabled=ENABLE_RERANKER
        )

    def invalidate_bm25_cache(self) -> None:
        """Forces BM25 index refresh on the next search after new files are indexed."""
        self._bm25_retriever = None

    def _ensure_bm25_retriever(self) -> Optional[BM25Retriever]:
        """Builds or returns the cached in-memory BM25 index from existing ChromaDB chunks."""
        if self._bm25_retriever is not None:
            return self._bm25_retriever
        try:
            data = self.collection.get(include=["documents", "metadatas"])
            docs = data.get("documents") or []
            metas = data.get("metadatas") or []
            if docs:
                self._bm25_retriever = BM25Retriever(docs, metas)
            return self._bm25_retriever
        except Exception as e:
            print(f"BM25 index initialization failed, falling back to dense search: {e}")
            return None

    def load_manifest(self) -> Dict[str, Any]:
        if MANIFEST_PATH.exists():
            try:
                with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save_manifest(self, manifest: Dict[str, Any]):
        try:
            with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            print(f"Error saving manifest: {e}")

    def chunk_and_index_file(self, file_path: Path) -> int:
        """
        Parses a single file, applies structure-aware & table-aware contextual chunking,
        and upserts into ChromaDB without leaving stale chunks behind.
        """
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            raw_pages = extract_pdf_pages(file_path)
        elif suffix in [".docx", ".doc"]:
            raw_pages = extract_docx_sections(file_path)
        elif suffix in [".txt", ".md"]:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            raw_pages = [{"page_number": 1, "text": content}]
        else:
            return 0

        if not raw_pages:
            return 0

        # Infer section and category from folder structure
        parent_name = file_path.parent.name
        is_challenge = "challenge" in str(file_path).lower() or parent_name.lower() == "challenges"
        category = "Hands-on Challenge" if is_challenge else "Master Module"

        # Find section name (e.g., 02_Planning_and_Design)
        section = "General"
        for part in file_path.parts:
            if any(part.startswith(prefix) for prefix in ["01_", "02_", "03_", "04_", "05_"]):
                section = part
                break

        rel_path = str(file_path.relative_to(WORKSPACE_DIR))
        # Ensure re-indexing a modified/upgraded file removes any prior chunks for this relative_path
        try:
            existing = self.collection.get(where={"relative_path": rel_path}, include=[])
            old_ids = list(existing.get("ids") or [])
            if old_ids:
                for i in range(0, len(old_ids), 200):
                    self.collection.delete(ids=old_ids[i:i + 200])
        except Exception as cleanup_err:
            print(f"Warning: pre-index cleanup failed for {rel_path}: {cleanup_err}")

        chunks_to_add = []
        ids_to_add = []
        metadatas_to_add = []

        chunk_counter = 0
        prose_chunk_count = 0
        table_chunk_count = 0

        def _append_chunk(chunk_body: str, page_num: str, content_type: str, extra_meta: Optional[Dict[str, Any]] = None) -> None:
            nonlocal chunk_counter, prose_chunk_count, table_chunk_count
            chunk_counter += 1
            chunk_id = f"{file_path.stem}_p{page_num}_c{chunk_counter}"
            enriched_text = (
                f"Document: {file_path.name}\n"
                f"Section: {section} | Category: {category} | Page: {page_num}\n"
                f"Content:\n{chunk_body}"
            )
            meta = {
                "source": file_path.name,
                "relative_path": rel_path,
                "section": section,
                "category": category,
                "page": page_num,
                "chunk_index": chunk_counter,
                "content_type": content_type,
            }
            if extra_meta:
                meta.update(extra_meta)
            chunks_to_add.append(enriched_text)
            ids_to_add.append(chunk_id)
            metadatas_to_add.append(meta)
            if content_type == "table":
                table_chunk_count += 1
            else:
                prose_chunk_count += 1

        for page_data in raw_pages:
            page_num = str(page_data["page_number"])
            heading_path = str(page_data.get("heading_path") or page_num)
            blocks = page_data.get("blocks")

            if blocks:
                for block in blocks:
                    if block["type"] == "prose":
                        sub_chunks = self.text_splitter.split_text(block["text"])
                        for chunk_text in sub_chunks:
                            body_with_section = f"Section Context: {heading_path}\n{chunk_text}"
                            _append_chunk(body_with_section, page_num, "prose")
                    elif block["type"] == "table":
                        prefix_ctx = f"Document: {file_path.name} | Section Context: {heading_path}"
                        full_table = f"{prefix_ctx}\n{block['markdown']}"
                        extra = {
                            "table_rows": int(block.get("rows", 0)),
                            "table_cols": int(block.get("cols", 0)),
                        }
                        if len(full_table) <= TABLE_ATOMIC_CHAR_LIMIT:
                            _append_chunk(full_table, page_num, "table", extra)
                        else:
                            row_chunks = _split_markdown_table_by_rows(
                                header_lines=block["header_lines"],
                                data_lines=block["data_lines"],
                                prefix_context=prefix_ctx,
                                max_chars=TABLE_ATOMIC_CHAR_LIMIT,
                            )
                            for rc in row_chunks:
                                _append_chunk(rc, page_num, "table", extra)
            else:
                page_text = page_data["text"]
                sub_chunks = self.text_splitter.split_text(page_text)
                for chunk_text in sub_chunks:
                    _append_chunk(chunk_text, page_num, "prose")

        # Batch upsert into ChromaDB
        if chunks_to_add:
            batch_size = 200
            for i in range(0, len(chunks_to_add), batch_size):
                self.collection.upsert(
                    ids=ids_to_add[i:i+batch_size],
                    documents=chunks_to_add[i:i+batch_size],
                    metadatas=metadatas_to_add[i:i+batch_size]
                )
            self.invalidate_bm25_cache()

        print(
            f"[Chunking] {file_path.name}: prose_chunks={prose_chunk_count}, "
            f"table_chunks={table_chunk_count}, total_chunks={len(chunks_to_add)}"
        )
        return len(chunks_to_add)

    def sync_database(self, base_dir: Path = WORKSPACE_DIR) -> Dict[str, Any]:
        """
        Scans all supported documents in the directory.
        - Skips unchanged PDFs and up-to-date DOCX files (SHA-256 + extractor_version).
        - Selectively re-indexes DOCX files created with an older extractor_version.
        - Indexes NEW or MODIFIED files incrementally.
        """
        manifest = self.load_manifest()
        new_indexed = 0
        reindexed_count = 0
        skipped = 0
        total_chunks_added = 0

        # Scan for PDFs and Word documents
        supported_exts = {".pdf", ".docx", ".doc", ".txt", ".md"}
        all_files = []
        for ext in supported_exts:
            all_files.extend(list(base_dir.rglob(f"*{ext}")))

        # Filter out ChromaDB directory itself
        all_files = [f for f in all_files if "pega_rag_db" not in str(f)]

        for file_path in all_files:
            rel_path = str(file_path.relative_to(base_dir))
            current_hash = compute_file_hash(file_path)
            is_docx = file_path.suffix.lower() in {".docx", ".doc"}
            target_version = DOCX_EXTRACTOR_VERSION if is_docx else 1

            is_selective_reindex = False
            preserved_doc_name = None

            if rel_path in manifest:
                entry = manifest[rel_path]
                stored_hash = entry.get("hash", "")
                stored_version = int(entry.get("extractor_version", 1))
                preserved_doc_name = entry.get("document_name")

                hash_matches = (stored_hash == current_hash) or (
                    len(stored_hash) == 32 and stored_hash == _legacy_md5_hash(file_path)
                )

                if hash_matches:
                    if len(stored_hash) == 32:
                        # Upgrade legacy MD5 entry to SHA-256 without re-chunking PDFs
                        entry["hash"] = current_hash
                    if is_docx and stored_version < DOCX_EXTRACTOR_VERSION:
                        is_selective_reindex = True
                    else:
                        skipped += 1
                        continue

            # Process & index new, modified, or selectively re-indexed file
            chunks_count = self.chunk_and_index_file(file_path)
            new_entry = {
                "hash": current_hash,
                "file_name": file_path.name,
                "chunks_count": chunks_count,
                "last_indexed": str(file_path.stat().st_mtime),
                "extractor_version": target_version,
            }
            if preserved_doc_name:
                new_entry["document_name"] = preserved_doc_name
            manifest[rel_path] = new_entry

            if is_selective_reindex:
                reindexed_count += 1
            else:
                new_indexed += 1
            total_chunks_added += chunks_count

        self.save_manifest(manifest)
        total_docs = self.collection.count()
        print(
            f"[Sync] new_indexed={new_indexed}, selectively_reindexed={reindexed_count}, "
            f"skipped={skipped}, total_chunks_in_db={total_docs}"
        )

        return {
            "new_indexed": new_indexed,
            "reindexed": reindexed_count,
            "skipped": skipped,
            "total_files": len(all_files),
            "chunks_added": total_chunks_added,
            "total_chunks_in_db": total_docs
        }

    def list_indexed_documents(self) -> List[Dict[str, Any]]:
        """
        Returns a consolidated list of all indexed documents combining ChromaDB metadata
        and indexed_manifest.json, safely handling stale manifest entries and zero-chunk docs.
        """
        manifest = self.load_manifest()
        chroma_stats: Dict[str, Dict[str, Any]] = {}

        try:
            data = self.collection.get(include=["metadatas"])
            for meta in (data.get("metadatas") or []):
                if not meta:
                    continue
                rel = meta.get("relative_path") or meta.get("source") or "Unknown"
                if rel not in chroma_stats:
                    chroma_stats[rel] = {
                        "file_name": meta.get("source", Path(rel).name),
                        "relative_path": rel,
                        "category": meta.get("category", "Document"),
                        "section": meta.get("section", "General"),
                        "chunks_count": 0
                    }
                chroma_stats[rel]["chunks_count"] += 1
        except Exception as e:
            print(f"Error inspecting ChromaDB metadata: {e}")

        docs_map: Dict[str, Dict[str, Any]] = {}
        # Populate from manifest first
        for rel_p, m_info in manifest.items():
            fname = m_info.get("file_name", Path(rel_p).name)
            doc_display_name = m_info.get("document_name") or fname
            live_stat = chroma_stats.get(rel_p)
            live_count = live_stat["chunks_count"] if live_stat else 0
            cat = live_stat["category"] if live_stat else (
                "Hands-on Challenge" if "challenge" in rel_p.lower() else "Master Module"
            )
            sec = live_stat["section"] if live_stat else "General"
            disk_path = WORKSPACE_DIR / rel_p
            docs_map[rel_p] = {
                "document_name": doc_display_name,
                "file_name": fname,
                "relative_path": rel_p,
                "category": cat,
                "section": sec,
                "chunks_count": live_count,
                "manifest_chunks": m_info.get("chunks_count", 0),
                "file_exists_on_disk": disk_path.exists()
            }

        # Include any ChromaDB documents missing from manifest
        for rel_p, c_stat in chroma_stats.items():
            if rel_p not in docs_map:
                disk_path = WORKSPACE_DIR / rel_p
                docs_map[rel_p] = {
                    "document_name": c_stat["file_name"],
                    "file_name": c_stat["file_name"],
                    "relative_path": rel_p,
                    "category": c_stat["category"],
                    "section": c_stat["section"],
                    "chunks_count": c_stat["chunks_count"],
                    "manifest_chunks": 0,
                    "file_exists_on_disk": disk_path.exists()
                }

        return sorted(docs_map.values(), key=lambda d: (d["file_name"].lower(), d["relative_path"].lower()))

    def delete_document_chunks(
        self,
        relative_path: Optional[str] = None,
        source_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Deletes ONLY the selected document's chunks from ChromaDB, removes its entry from
        indexed_manifest.json, and rebuilds the in-memory BM25 corpus.
        Does NOT delete the physical file from disk.
        """
        rel_clean = (relative_path or "").strip()
        src_clean = (source_name or "").strip()

        if not rel_clean and not src_clean:
            return {
                "status": "error",
                "message": "A valid relative_path or source_name is required for targeted deletion.",
                "deleted_chunks": 0
            }

        matched_ids: List[str] = []
        try:
            if rel_clean:
                res = self.collection.get(where={"relative_path": rel_clean}, include=[])
                matched_ids = list(res.get("ids") or [])

            # Fallback to source metadata only if relative_path yielded nothing
            if not matched_ids and src_clean:
                res = self.collection.get(where={"source": src_clean}, include=["metadatas"])
                ids = res.get("ids") or []
                metas = res.get("metadatas") or []
                for cid, meta in zip(ids, metas):
                    m_rel = (meta or {}).get("relative_path", "")
                    if not rel_clean or m_rel == rel_clean or not m_rel:
                        matched_ids.append(cid)

            if matched_ids:
                # Batch delete matched IDs only (never broad collection deletion)
                batch_size = 200
                for i in range(0, len(matched_ids), batch_size):
                    self.collection.delete(ids=matched_ids[i:i + batch_size])
        except Exception as e:
            return {
                "status": "error",
                "message": f"ChromaDB targeted deletion failed: {e}",
                "deleted_chunks": 0
            }

        # Remove from indexed_manifest.json
        manifest = self.load_manifest()
        removed_manifest_keys = []
        if rel_clean and rel_clean in manifest:
            del manifest[rel_clean]
            removed_manifest_keys.append(rel_clean)
        elif src_clean:
            for k in list(manifest.keys()):
                if manifest[k].get("file_name") == src_clean or k == src_clean:
                    del manifest[k]
                    removed_manifest_keys.append(k)
                    break

        if removed_manifest_keys:
            self.save_manifest(manifest)

        # Invalidate and rebuild BM25 corpus immediately so deleted content cannot be retrieved
        self.invalidate_bm25_cache()
        self._ensure_bm25_retriever()

        return {
            "status": "success",
            "deleted_chunks": len(matched_ids),
            "manifest_removed": bool(removed_manifest_keys),
            "remaining_chunks_in_db": self.collection.count(),
            "remaining_docs_in_manifest": len(manifest)
        }

    def delete_all_chunks(
        self,
        delete_disk_files: bool = False,
        docs_dir: Optional[Path] = None,
        workspace_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Deletes every indexed record from the existing ChromaDB collection using safe batch deletion
        (without deleting or recreating the collection), clears indexed_manifest.json after
        successful deletion, invalidates BM25 cache, and optionally deletes original source files
        strictly contained inside DOCS_DIR.
        """
        target_docs_dir = (docs_dir or DOCS_DIR).resolve()
        target_workspace_dir = (workspace_dir or WORKSPACE_DIR).resolve()

        manifest = self.load_manifest()
        previous_doc_count = len(self.list_indexed_documents())

        try:
            existing = self.collection.get(include=["metadatas"] if delete_disk_files else [])
            all_ids = list(existing.get("ids") or [])
            all_metas = list(existing.get("metadatas") or []) if delete_disk_files else []
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to inspect ChromaDB collection before deletion: {e}",
                "deleted_chunks": 0,
                "previous_doc_count": previous_doc_count,
                "remaining_chunks": self.collection.count(),
                "deleted_disk_files": 0
            }

        try:
            if all_ids:
                batch_size = 200
                for i in range(0, len(all_ids), batch_size):
                    self.collection.delete(ids=all_ids[i:i + batch_size])
        except Exception as e:
            return {
                "status": "error",
                "message": f"ChromaDB batch deletion failed: {e}",
                "deleted_chunks": 0,
                "previous_doc_count": previous_doc_count,
                "remaining_chunks": self.collection.count(),
                "deleted_disk_files": 0
            }

        # Clear manifest only after ChromaDB deletion succeeded
        self.save_manifest({})

        # Invalidate BM25 cache so empty-collection searches never return stale results
        self.invalidate_bm25_cache()

        deleted_disk_files = 0
        if delete_disk_files:
            candidate_rel_paths = set(manifest.keys())
            for meta in all_metas:
                if isinstance(meta, dict) and meta.get("relative_path"):
                    candidate_rel_paths.add(str(meta["relative_path"]))

            for rel_p in candidate_rel_paths:
                try:
                    candidate_path = (target_workspace_dir / rel_p).resolve()
                    # Restrict disk deletion strictly to files inside target_docs_dir
                    candidate_path.relative_to(target_docs_dir)
                    if candidate_path != target_docs_dir and candidate_path.exists() and candidate_path.is_file():
                        candidate_path.unlink()
                        deleted_disk_files += 1
                except Exception:
                    # Skip paths outside DOCS_DIR, missing files, or missing directories safely
                    continue

        return {
            "status": "success",
            "deleted_chunks": len(all_ids),
            "previous_doc_count": previous_doc_count,
            "remaining_chunks": self.collection.count(),
            "deleted_disk_files": deleted_disk_files
        }

    def _dense_search(
        self,
        query: str,
        top_k: int = 15,
        section_filter: Optional[str] = None,
        category_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Performs dense vector search against ChromaDB with optional section/category filters."""
        total_count = self.collection.count()
        if total_count <= 0:
            return []
        effective_k = min(top_k, total_count)

        clauses = []
        if section_filter and section_filter != "All Sections":
            clauses.append({"section": section_filter})
        if category_filter:
            clauses.append({"category": category_filter})

        where_filter = None
        if len(clauses) == 1:
            where_filter = clauses[0]
        elif len(clauses) > 1:
            where_filter = {"$and": clauses}

        results = self.collection.query(
            query_texts=[query],
            n_results=effective_k,
            where=where_filter
        )

        formatted_results = []
        if results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(docs)

            for doc_text, meta, dist in zip(docs, metas, distances):
                similarity_score = compute_similarity_score(dist, self.distance_metric)
                formatted_results.append({
                    "document": doc_text,
                    "source": meta.get("source", "Unknown"),
                    "page": str(meta.get("page", "1")),
                    "section": meta.get("section", "General"),
                    "category": meta.get("category", "Document"),
                    "relative_path": meta.get("relative_path", ""),
                    "score": similarity_score
                })

        return formatted_results

    def search(
        self,
        query: str,
        top_k: int = 5,
        section_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
        use_hybrid: bool = True,
        use_reranker: bool = True,
        dense_weight: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval pipeline:
        1. Dense retrieval (ChromaDB) + Sparse lexical retrieval (BM25)
        2. Weighted Reciprocal Rank Fusion (RRF)
        3. Cross-Encoder reranking and relevance filtering
        """
        if not use_hybrid:
            return self._dense_search(
                query=query,
                top_k=top_k,
                section_filter=section_filter,
                category_filter=category_filter
            )

        candidate_k = max(DEFAULT_CANDIDATE_K, top_k * 3)
        dense_hits = self._dense_search(
            query=query,
            top_k=candidate_k,
            section_filter=section_filter,
            category_filter=category_filter
        )

        bm25 = self._ensure_bm25_retriever()
        sparse_hits = (
            bm25.search(
                query=query,
                top_k=candidate_k,
                section_filter=section_filter,
                category_filter=category_filter
            )
            if bm25 is not None
            else []
        )

        fused_candidates = reciprocal_rank_fusion(
            dense_results=dense_hits,
            sparse_results=sparse_hits,
            rrf_k=RRF_K,
            top_k=candidate_k,
            dense_weight=dense_weight
        )

        if use_reranker:
            return self.reranker.rerank(query=query, candidates=fused_candidates, top_k=top_k)

        return fused_candidates[:top_k]

    # ---------------------------------------------------------
    # Autonomous Agent Tools Implementation
    # ---------------------------------------------------------

    def tool_search_curriculum(self, query: str, section: Optional[str] = None, top_k: int = 4) -> Dict[str, Any]:
        """Tool: Search theoretical concepts & design patterns in the 28 Master Modules."""
        results = self.search(
            query=query,
            top_k=top_k,
            section_filter=section,
            category_filter="Master Module"
        )
        if not results:
            results = self.search(query=query, top_k=top_k, section_filter=section)

        formatted = []
        for r in results[:top_k]:
            formatted.append({
                "document": r["source"],
                "section": r["section"],
                "page": r["page"],
                "relevance_score": f"{r['score']}%",
                "excerpt": r["document"]
            })
        return {"status": "success", "results_count": len(formatted), "results": formatted}

    def tool_search_challenges(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        """Tool: Search hands-on implementation steps and solutions in the 40 Challenges."""
        results = self.search(
            query=query,
            top_k=top_k,
            category_filter="Hands-on Challenge"
        )
        if not results:
            results = self.search(query=query, top_k=top_k)

        formatted = []
        for r in results[:top_k]:
            formatted.append({
                "document": r["source"],
                "section": r["section"],
                "page": r["page"],
                "relevance_score": f"{r['score']}%",
                "excerpt": r["document"]
            })
        return {"status": "success", "results_count": len(formatted), "results": formatted}

    def tool_read_full_page(self, document_name: str, page_number: int) -> Dict[str, Any]:
        """Tool: Read the complete, untruncated content of a specific page in any Pega PDF."""
        # Find document on disk
        matches = list(DOCS_DIR.rglob(document_name))
        if not matches:
            matches = [f for f in DOCS_DIR.rglob("*.pdf") if document_name.lower() in f.name.lower()]
        
        if not matches:
            return {"status": "error", "message": f"Document '{document_name}' not found in curriculum."}

        target_path = matches[0]
        try:
            doc = fitz.open(target_path)
            total_pages = len(doc)
            if 1 <= page_number <= total_pages:
                page_text = doc[page_number - 1].get_text("text").strip()
                doc.close()
                return {
                    "status": "success",
                    "document": target_path.name,
                    "page": page_number,
                    "total_pages": total_pages,
                    "full_page_text": page_text
                }
            else:
                doc.close()
                return {
                    "status": "error",
                    "message": f"Requested page {page_number} is out of bounds (Document has {total_pages} pages)."
                }
        except Exception as e:
            return {"status": "error", "message": f"Failed reading page: {str(e)}"}

    def tool_lookup_pega_terms(self, term: str, top_k: int = 4) -> Dict[str, Any]:
        """Tool: Specialized lookup for exact Pega rules, classes, APIs, or architectural terms."""
        results = self.search(query=term, top_k=top_k)
        formatted = []
        for r in results:
            formatted.append({
                "document": r["source"],
                "section": r["section"],
                "page": r["page"],
                "relevance_score": f"{r['score']}%",
                "excerpt": r["document"]
            })
        return {"status": "success", "term": term, "results": formatted}

    # ---------------------------------------------------------
    # Agent Tool Schemas for Groq Function Calling
    # ---------------------------------------------------------

    @staticmethod
    def get_agent_tools_schema() -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_curriculum",
                    "description": "Searches theoretical architectural concepts, principles, rulesets, and design patterns in the 28 Pega Master Modules.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Specific search query focused on architectural principles, design patterns, or system governance."
                            },
                            "section": {
                                "type": "string",
                                "enum": ["All Sections", "01_LSA_Mindset", "02_Planning_and_Design", "03_Constellation_Adoption", "04_Infrastructure_and_Architecture", "05_Implementation_and_Deployment"],
                                "description": "Optional curriculum section to filter search."
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_challenges",
                    "description": "Searches hands-on implementation steps, configuration walkthroughs, and solutions in the 40 Pega Challenges.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Practical configuration or exercise query (e.g. 'how to configure SAML in App Studio', 'creating field groups in Constellation')."
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_full_page",
                    "description": "Reads the entire, complete text of a specific page from any Pega PDF document. Use this when a search chunk is truncated or when you need full step-by-step instructions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "document_name": {
                                "type": "string",
                                "description": "The exact name or partial name of the PDF file (e.g. '01_fundamental-concepts-constellation.pdf')."
                            },
                            "page_number": {
                                "type": "integer",
                                "description": "The exact 1-indexed page number to read."
                            }
                        },
                        "required": ["document_name", "page_number"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "lookup_pega_terms",
                    "description": "Looks up exact Pega technical rules, classes (e.g. 'Queue Processor', 'Job Scheduler', 'ABAC', 'BIX', 'Rule-Obj-Class', 'DX API').",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "term": {
                                "type": "string",
                                "description": "The exact technical Pega rule or class name to search."
                            }
                        },
                        "required": ["term"]
                    }
                }
            }
        ]

    def find_best_alternative_model(self, client: Groq, current_model: str) -> Optional[str]:
        """Dynamically finds an active text chat model from the user's specific Groq account."""
        try:
            live_models = client.models.list()
            raw_ids = [m.id for m in live_models.data]
            valid_ids = []
            for m_id in raw_ids:
                m_low = m_id.lower()
                if any(x in m_low for x in ["whisper", "orpheus", "tts", "audio", "guard", "safeguard", "canopylabs"]):
                    continue
                if m_id != current_model:
                    valid_ids.append(m_id)

            priority = [
                FALLBACK_GROQ_MODEL,
                PRIMARY_GROQ_MODEL,
                "qwen/qwen3.8-27b",
                "allam-2-7b"
            ]
            for p in priority:
                if p in valid_ids:
                    return p
            return valid_ids[0] if valid_ids else None
        except Exception:
            return None

    def resolve_available_model(self, client: Groq, requested_model: Optional[str]) -> Tuple[str, Optional[str]]:
        """
        Validates the requested model against Groq's live Models API.
        If unavailable or retired, automatically falls back to openai/gpt-oss-20b
        (or the best active text model on the account).
        """
        target = (requested_model or "").strip() or PRIMARY_GROQ_MODEL
        try:
            live_models = client.models.list()
            active_ids = [
                m.id for m in live_models.data
                if not any(x in m.id.lower() for x in ["whisper", "orpheus", "tts", "audio", "guard", "safeguard", "canopylabs"])
            ]
            if not active_ids:
                return target, None
            if target in active_ids:
                return target, None
            if FALLBACK_GROQ_MODEL in active_ids:
                return FALLBACK_GROQ_MODEL, f"Requested model `{target}` is unavailable on Groq; using fallback `{FALLBACK_GROQ_MODEL}`."
            if PRIMARY_GROQ_MODEL in active_ids:
                return PRIMARY_GROQ_MODEL, f"Requested model `{target}` is unavailable on Groq; using `{PRIMARY_GROQ_MODEL}`."
            alt = active_ids[0]
            return alt, f"Requested model `{target}` is unavailable on Groq; using `{alt}`."
        except Exception:
            return target, None

    @staticmethod
    def _trim_to_last_complete_sentence(text: str) -> str:
        """Trims a length-truncated response cleanly to the last complete sentence boundary."""
        if not text:
            return ""
        stripped = text.rstrip()
        last_end = max(stripped.rfind("."), stripped.rfind("!"), stripped.rfind("?"))
        if last_end != -1 and last_end > len(stripped) * 0.4:
            return stripped[:last_end + 1]
        last_nl = stripped.rfind("\n")
        if last_nl != -1 and last_nl > len(stripped) * 0.4:
            return stripped[:last_nl]
        return stripped

    def _create_completion_with_completeness_guard(
        self,
        client: Groq,
        model_name: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int
    ) -> Tuple[str, Optional[str]]:
        """
        Executes a Groq chat completion and inspects finish_reason.
        If finish_reason == 'length', performs at most ONE controlled retry with a concise
        completion instruction and expanded token budget (within model context_window).
        If still truncated, trims cleanly at the last complete sentence and appends a notice.
        """
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        choice = response.choices[0]
        content = choice.message.content or ""
        finish_reason = getattr(choice, "finish_reason", None)

        if finish_reason != "length":
            return content, finish_reason

        spec = get_model_spec(model_name)
        ctx_win = spec.get("context_window", 4096)
        prompt_tokens_est = sum(estimate_tokens(m.get("content", "")) for m in messages)
        headroom = ctx_win - prompt_tokens_est - 256
        expanded_tokens = min(2200, max(max_tokens + 400, int(max_tokens * 1.5)), headroom)

        if expanded_tokens > max_tokens + 100:
            retry_messages = list(messages[:-1]) + [{
                "role": messages[-1]["role"],
                "content": messages[-1]["content"] + "\n\n(Be concise and complete your response cleanly within the token limit.)"
            }]
            try:
                retry_resp = client.chat.completions.create(
                    model=model_name,
                    messages=retry_messages,
                    temperature=temperature,
                    max_tokens=expanded_tokens
                )
                retry_choice = retry_resp.choices[0]
                retry_content = retry_choice.message.content or ""
                retry_reason = getattr(retry_choice, "finish_reason", None)
                if retry_reason != "length" and retry_content:
                    return retry_content, retry_reason
                if retry_content:
                    content = retry_content
            except Exception:
                pass

        trimmed = self._trim_to_last_complete_sentence(content)
        notice = "\n\n> *Note: Response reached the model's output length limit and was trimmed at the last complete sentence.*"
        return (trimmed + notice) if trimmed else "No complete response generated.", "length"

    def _execute_direct_grounded_fallback(
        self,
        query: str,
        client: Groq,
        model_name: str,
        temperature: float = 0.0,
        chat_history: Optional[List[Dict[str, str]]] = None,
        execution_trace: Optional[List[Dict[str, Any]]] = None,
        reason: str = "",
        top_k: int = DEFAULT_FINAL_EVIDENCE_K,
        use_reranker: bool = True,
        dense_weight: float = 0.5,
        section_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Direct Grounded Retrieval Fallback when a model does not support Groq function/tool calling
        or when a straightforward query is routed directly.
        Retrieves verified excerpts from ChromaDB + BM25 and sends a strict zero-hallucination prompt
        WITHOUT the 'tools' parameter, enforcing token budgeting and recovery.
        """
        if execution_trace is None:
            execution_trace = []

        spec = get_model_spec(model_name)
        prompt_budget = spec.get("prompt_budget", 2200)
        max_completion_tokens = spec.get("max_completion_tokens", 600)

        # 1. Contextualize query with prior thread history for multi-turn follow-ups
        search_query = contextualize_query_with_history(query, chat_history)

        # Retrieve top reranked hybrid chunks from curriculum
        retrieved_chunks = self.search(
            query=search_query,
            top_k=top_k,
            section_filter=section_filter,
            use_reranker=use_reranker,
            dense_weight=dense_weight
        )

        # Also retrieve challenges if query appears practical/implementation oriented
        query_lower = search_query.lower()
        if any(w in query_lower for w in ["how to", "configure", "implement", "step", "challenge", "exercise", "create", "setup"]):
            try:
                challenge_res = self.tool_search_challenges(query=search_query, top_k=2)
                if "results" in challenge_res:
                    for r in challenge_res["results"]:
                        if not any(c.get("document") == r.get("excerpt") for c in retrieved_chunks):
                            retrieved_chunks.append({
                                "source": r.get("document", "Challenge"),
                                "page": str(r.get("page", "1")),
                                "section": r.get("section", "Hands-on Challenges"),
                                "category": "Hands-on Challenge",
                                "score": float(r.get("relevance_score", "0%").replace("%", "")),
                                "document": r.get("excerpt", "")
                            })
            except Exception:
                pass

        # Build clean, budgeted context
        context_str, retained_sources = build_grounded_context_prompt(
            retrieved_chunks, max_token_budget=prompt_budget
        )

        execution_trace.append({
            "step": len(execution_trace) + 1,
            "tool": "direct_grounded_retrieval",
            "args": {
                "model": model_name,
                "sources_retrieved": len(retained_sources),
                "prompt_budget": prompt_budget,
                "reason": reason or f"Model '{model_name}' operates with direct grounded retrieval."
            },
            "summary": f"Retrieved {len(retained_sources)} top-ranked Pega curriculum excerpts via Hybrid Search (Dense + BM25 RRF + Reranker)."
        })

        system_prompt = GROUNDED_SYSTEM_PROMPT

        user_content = (
            f"USER QUESTION:\n{query}\n\n"
            f"OFFICIAL PEGA ACADEMY CONTEXT EXCERPTS:\n"
            f"{context_str}\n\n"
            f"Answer clearly and naturally using only the verified excerpts above, with compact inline citations `[filename, Page X, Section: Y]`:"
        )

        messages = [{"role": "system", "content": system_prompt}]
        # Include prior thread history scaled to the model's context window
        if chat_history:
            ctx_win = spec.get("context_window", 4096)
            hist_limit = 6 if ctx_win >= 16384 else 2
            char_limit = 700 if ctx_win >= 16384 else 350
            for h in chat_history[-hist_limit:]:
                role = h.get("role", "user")
                if role in ("user", "assistant") and h.get("content"):
                    messages.append({"role": role, "content": h["content"][:char_limit]})
        messages.append({"role": "user", "content": user_content})

        try:
            raw_answer, finish_reason = self._create_completion_with_completeness_guard(
                client=client,
                model_name=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_completion_tokens
            )
            final_sources = deduplicate_sources(retained_sources)
            verified_answer, _ = verify_citations(raw_answer or "No response generated.", final_sources)
            return {
                "answer": verified_answer,
                "trace": execution_trace,
                "sources": final_sources
            }
        except Exception as e:
            err_str = str(e)

            # Context Length Error Recovery: single controlled retry with halved budget and no history
            if is_context_length_error(err_str):
                reduced_budget = max(800, prompt_budget // 2)
                reduced_completion = max(300, max_completion_tokens // 2)
                reduced_context_str, reduced_sources = build_grounded_context_prompt(
                    retrieved_chunks, max_token_budget=reduced_budget
                )
                retry_user_content = (
                    f"USER QUESTION:\n{query}\n\n"
                    f"OFFICIAL PEGA ACADEMY CONTEXT EXCERPTS:\n"
                    f"{reduced_context_str}\n\n"
                    f"Answer clearly and naturally using only the verified excerpts above, with compact inline citations `[filename, Page X, Section: Y]`:"
                )
                retry_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": retry_user_content}
                ]
                execution_trace.append({
                    "step": len(execution_trace) + 1,
                    "tool": "context_recovery_retry",
                    "args": {
                        "model": model_name,
                        "reduced_budget": reduced_budget,
                        "reduced_completion": reduced_completion
                    },
                    "summary": f"Reduced context budget to {reduced_budget} tokens to fit model limits."
                })
                try:
                    retry_res = client.chat.completions.create(
                        model=model_name,
                        messages=retry_messages,
                        temperature=temperature,
                        max_tokens=reduced_completion
                    )
                    final_reduced = deduplicate_sources(reduced_sources)
                    raw_retry = retry_res.choices[0].message.content or "No response generated."
                    verified_retry, _ = verify_citations(raw_retry, final_reduced)
                    return {
                        "answer": verified_retry,
                        "trace": execution_trace,
                        "sources": final_reduced
                    }
                except Exception:
                    return {
                        "answer": f"⚠️ **Context Length Notice:** The query and evidence exceed the maximum token capacity ({spec.get('context_window', 4096)} tokens) for model `{model_name}`. Please switch to a higher-capacity model like `openai/gpt-oss-120b` in Configuration.",
                        "trace": execution_trace,
                        "sources": deduplicate_sources(reduced_sources)
                    }

            # Check if model is deprecated, terms required, or decommissioned
            err_low = err_str.lower()
            is_model_issue = any(x in err_low for x in [
                "model_not_found", "model_terms_required", "model_decommissioned",
                "decommissioned", "deprecated", "does not exist", "not have access"
            ])
            if is_model_issue:
                alt_model = self.find_best_alternative_model(client, model_name)
                if alt_model and alt_model != model_name:
                    alt_spec = get_model_spec(alt_model)
                    alt_tokens = alt_spec.get("max_completion_tokens", 1000)
                    try:
                        response = client.chat.completions.create(
                            model=alt_model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=alt_tokens
                        )
                        final_sources = deduplicate_sources(retained_sources)
                        raw_alt = response.choices[0].message.content or ""
                        verified_alt, _ = verify_citations(raw_alt, final_sources)
                        return {
                            "answer": f"> *Note: Requested model `{model_name}` was unavailable. Auto-switched to `{alt_model}`.*\n\n" + verified_alt,
                            "trace": execution_trace,
                            "sources": final_sources
                        }
                    except Exception as retry_err:
                        return {
                            "answer": f"❌ **Groq API Error:** {str(retry_err)}",
                            "trace": execution_trace,
                            "sources": deduplicate_sources(retained_sources)
                        }
            return {
                "answer": f"❌ **Groq API Error:** {str(e)}",
                "trace": execution_trace,
                "sources": deduplicate_sources(retained_sources)
            }

    # ---------------------------------------------------------
    # Autonomous Agent Execution Loop
    # ---------------------------------------------------------

    def run_agent(
        self,
        query: str,
        groq_api_key: str,
        model_name: str = PRIMARY_GROQ_MODEL,
        temperature: float = 0.0,
        max_iterations: int = 4,
        chat_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = DEFAULT_FINAL_EVIDENCE_K,
        use_reranker: bool = True,
        dense_weight: float = 0.5,
        section_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs an autonomous agent loop powered by Groq Function Calling:
        1. Agent plans which tools to call (curriculum search, challenge search, or full-page read).
        2. Executes tools locally against ChromaDB & PDFs.
        3. Feeds verified evidence back to the model.
        4. If a model does not support tool calling (or encounters a Groq 400), automatically
           falls back to Direct Grounded Retrieval without failing.
        """
        if not groq_api_key or not groq_api_key.strip():
            return {
                "answer": "⚠️ **Groq API Key Required:** Please provide your Groq API key in the sidebar to activate the Agent. You can get one free at [console.groq.com/keys](https://console.groq.com/keys).",
                "trace": [],
                "sources": []
            }

        client = Groq(api_key=groq_api_key.strip())
        model_name, model_notice = self.resolve_available_model(client, model_name)
        execution_trace = []
        if model_notice:
            execution_trace.append({"type": "note", "message": model_notice})
        collected_sources = []

        # Check if the requested model supports function/tool calling
        if not is_tool_capable_model(model_name):
            return self._execute_direct_grounded_fallback(
                query=query,
                client=client,
                model_name=model_name,
                temperature=temperature,
                chat_history=chat_history,
                execution_trace=execution_trace,
                reason=f"Model '{model_name}' does not support function/tool calling. Directly routed to Grounded Retrieval.",
                top_k=top_k,
                use_reranker=use_reranker,
                dense_weight=dense_weight,
                section_filter=section_filter
            )

        # Smart Query Routing: straightforward questions use fast Hybrid + Reranked Direct Retrieval;
        # multi-part or hands-on investigative questions use the multi-step Agent Loop.
        use_agent, route_reason = should_use_agent_loop(query)
        if not use_agent:
            return self._execute_direct_grounded_fallback(
                query=query,
                client=client,
                model_name=model_name,
                temperature=temperature,
                chat_history=chat_history,
                execution_trace=execution_trace,
                reason=route_reason,
                top_k=top_k,
                use_reranker=use_reranker,
                dense_weight=dense_weight,
                section_filter=section_filter
            )

        tools_schema = self.get_agent_tools_schema()
        spec = get_model_spec(model_name)
        loop_max_tokens = min(2000, spec.get("max_completion_tokens", 1500) + 500)

        system_prompt = AGENT_SYSTEM_PROMPT

        messages = [{"role": "system", "content": system_prompt}]

        # Append recent conversation history if provided
        if chat_history:
            for h in chat_history[-6:]:
                role = h.get("role", "user")
                if role in ("user", "assistant") and h.get("content"):
                    messages.append({"role": role, "content": h["content"][:700]})

        messages.append({"role": "user", "content": query})

        current_iteration = 0

        while current_iteration < max_iterations:
            current_iteration += 1

            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=tools_schema,
                    tool_choice="auto",
                    temperature=temperature,
                    max_tokens=loop_max_tokens
                )
            except Exception as e:
                err_str = str(e).lower()

                # Check 1: Tool calling is not supported with this model
                is_tool_err = any(x in err_str for x in [
                    "tool calling is not supported",
                    "tool calling",
                    "tools is not supported",
                    "does not support tools",
                    "param': 'tool calling'",
                    "param': 'tools'"
                ])
                if is_tool_err:
                    return self._execute_direct_grounded_fallback(
                        query=query,
                        client=client,
                        model_name=model_name,
                        temperature=temperature,
                        chat_history=chat_history,
                        execution_trace=execution_trace,
                        reason=f"Groq API reported tool calling unsupported for model '{model_name}'. Auto-switched to Direct Grounded Retrieval."
                    )

                # Check 2: Model decommissioned, not found, or requires terms
                is_model_issue = any(x in err_str for x in [
                    "model_not_found", "model_terms_required", "model_decommissioned",
                    "decommissioned", "deprecated", "does not exist", "not have access"
                ])
                if is_model_issue:
                    alt_model = self.find_best_alternative_model(client, model_name)
                    if alt_model and alt_model != model_name:
                        model_name = alt_model
                        # If alternative model is known non-tool, fallback immediately
                        if not is_tool_capable_model(alt_model):
                            return self._execute_direct_grounded_fallback(
                                query=query,
                                client=client,
                                model_name=model_name,
                                temperature=temperature,
                                chat_history=chat_history,
                                execution_trace=execution_trace,
                                reason=f"Auto-switched from unavailable model to '{model_name}' (Direct Grounded Retrieval)."
                            )
                        try:
                            response = client.chat.completions.create(
                                model=model_name,
                                messages=messages,
                                tools=tools_schema,
                                tool_choice="auto",
                                temperature=temperature,
                                max_tokens=loop_max_tokens
                            )
                        except Exception as retry_err:
                            retry_err_str = str(retry_err).lower()
                            if any(x in retry_err_str for x in ["tool calling", "tools"]):
                                return self._execute_direct_grounded_fallback(
                                    query=query,
                                    client=client,
                                    model_name=model_name,
                                    temperature=temperature,
                                    chat_history=chat_history,
                                    execution_trace=execution_trace,
                                    reason=f"Model '{model_name}' does not support tools. Switched to Direct Grounded Retrieval."
                                )
                            return {
                                "answer": f"❌ **Groq API Error:** {str(retry_err)}",
                                "trace": execution_trace,
                                "sources": deduplicate_sources(collected_sources)
                            }
                    else:
                        return {
                            "answer": f"❌ **Groq API Error:** {str(e)}",
                            "trace": execution_trace,
                            "sources": deduplicate_sources(collected_sources)
                        }
                else:
                    return {
                        "answer": f"❌ **Groq API Error:** {str(e)}",
                        "trace": execution_trace,
                        "sources": deduplicate_sources(collected_sources)
                    }

            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            # If the model produced a final answer without calling tools
            if not tool_calls:
                final_sources = deduplicate_sources(collected_sources)
                raw_ans = response_message.content or "No response generated."
                if getattr(response.choices[0], "finish_reason", None) == "length":
                    raw_ans = self._trim_to_last_complete_sentence(raw_ans) + "\n\n> *Note: Response reached the model's output length limit and was trimmed at the last complete sentence.*"
                verified_ans, _ = verify_citations(raw_ans, final_sources)
                return {
                    "answer": verified_ans,
                    "trace": execution_trace,
                    "sources": final_sources
                }

            # Convert response message to dict and append to message history
            # The tool_calls must be preserved in the assistant message
            messages.append(response_message)

            # Execute each tool call
            for tool_call in tool_calls:
                func_name = tool_call.function.name
                func_args_str = tool_call.function.arguments or "{}"
                
                try:
                    parsed_args = json.loads(func_args_str or "{}")
                    func_args = parsed_args if isinstance(parsed_args, dict) else {}
                except Exception:
                    func_args = {}

                # Execute appropriate tool
                tool_output_dict: Dict[str, Any] = {}
                if func_name == "search_curriculum":
                    tool_output_dict = self.tool_search_curriculum(
                        query=func_args.get("query") or query,
                        section=func_args.get("section")
                    )
                elif func_name == "search_challenges":
                    tool_output_dict = self.tool_search_challenges(
                        query=func_args.get("query") or query
                    )
                elif func_name == "read_full_page":
                    try:
                        page_num_arg = int(func_args.get("page_number") or 1)
                    except (TypeError, ValueError):
                        page_num_arg = 1
                    tool_output_dict = self.tool_read_full_page(
                        document_name=func_args.get("document_name") or "",
                        page_number=page_num_arg
                    )
                elif func_name == "lookup_pega_terms":
                    tool_output_dict = self.tool_lookup_pega_terms(
                        term=func_args.get("term") or query
                    )
                else:
                    tool_output_dict = {"status": "error", "message": f"Unknown tool '{func_name}'"}

                if not isinstance(tool_output_dict, dict):
                    tool_output_dict = {}

                # Collect sources for UI display
                if isinstance(tool_output_dict.get("results"), list):
                    for r in tool_output_dict["results"]:
                        if not isinstance(r, dict):
                            continue
                        doc_label = str(r.get("document") or "Unknown")
                        rel_score = str(r.get("relevance_score") or "0%").replace("%", "")
                        try:
                            score_val = float(rel_score)
                        except ValueError:
                            score_val = 0.0
                        collected_sources.append({
                            "source": doc_label,
                            "page": str(r.get("page") or "1"),
                            "section": r.get("section") or "General",
                            "category": "Hands-on Challenge" if "challenge" in doc_label.lower() else "Master Module",
                            "score": score_val,
                            "document": r.get("excerpt") or ""
                        })
                elif "full_page_text" in tool_output_dict:
                    full_text = tool_output_dict.get("full_page_text") or ""
                    collected_sources.append({
                        "source": tool_output_dict.get("document") or "Unknown",
                        "page": str(tool_output_dict.get("page") or "1"),
                        "section": "Direct Page Read",
                        "category": "Full Page Read",
                        "score": 100.0,
                        "document": full_text,
                        "preview": full_text[:600] + ("..." if len(full_text) > 600 else "")
                    })

                # Log clean human-readable trace step
                if func_name == "search_curriculum":
                    step_summary = f"Searched curriculum modules for \"{func_args.get('query', query)}\""
                elif func_name == "search_challenges":
                    step_summary = f"Searched hands-on challenges for \"{func_args.get('query', query)}\""
                elif func_name == "read_full_page":
                    step_summary = f"Read full page {func_args.get('page_number', 1)} of {func_args.get('document_name', 'document')}"
                elif func_name == "lookup_pega_terms":
                    step_summary = f"Looked up Pega rule/term \"{func_args.get('term', query)}\""
                else:
                    step_summary = f"Executed `{func_name}`"

                execution_trace.append({
                    "step": current_iteration,
                    "tool": func_name,
                    "args": func_args,
                    "summary": step_summary
                })

                # Append tool response message
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": json.dumps(tool_output_dict, ensure_ascii=False)
                })

        # Final synthesis if loop exhausted
        try:
            if collected_sources:
                # Deduplicate evidence while preserving original order
                unique_sources = []
                seen_keys = set()
                for s in collected_sources:
                    doc_snippet = s.get("document", "").strip()[:120]
                    dedup_key = (s.get("source"), str(s.get("page")), doc_snippet)
                    if dedup_key not in seen_keys:
                        seen_keys.add(dedup_key)
                        unique_sources.append(s)

                final_answer = self.generate_answer(
                    query=query,
                    retrieved_chunks=unique_sources,
                    groq_api_key=groq_api_key,
                    model_name=model_name,
                    temperature=temperature,
                    chat_history=chat_history
                )

                # Check if generate_answer returned a Groq generation error string
                if isinstance(final_answer, str) and final_answer.startswith("❌"):
                    return self._execute_direct_grounded_fallback(
                        query=query,
                        client=client,
                        model_name=model_name,
                        temperature=temperature,
                        chat_history=chat_history,
                        execution_trace=execution_trace,
                        reason=f"Synthesis fallback triggered: {final_answer}"
                    )

                return {
                    "answer": final_answer,
                    "trace": execution_trace,
                    "sources": unique_sources
                }
            else:
                return self._execute_direct_grounded_fallback(
                    query=query,
                    client=client,
                    model_name=model_name,
                    temperature=temperature,
                    chat_history=chat_history,
                    execution_trace=execution_trace,
                    reason="Research iterations completed without retrieving evidence."
                )
        except Exception as e:
            try:
                return self._execute_direct_grounded_fallback(
                    query=query,
                    client=client,
                    model_name=model_name,
                    temperature=temperature,
                    chat_history=chat_history,
                    execution_trace=execution_trace,
                    reason=f"Final pass recovery: {type(e).__name__}"
                )
            except Exception:
                return {
                    "answer": "❌ **Generation Error:** Unable to synthesize response from retrieved sources. Please verify your Groq connection and try again.",
                    "trace": execution_trace,
                    "sources": collected_sources
                }

    # ---------------------------------------------------------
    # Legacy Direct Generator (Backwards-compatible)
    # ---------------------------------------------------------


    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        groq_api_key: str,
        model_name: str = PRIMARY_GROQ_MODEL,
        temperature: float = 0.2,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Sends context, conversation history, and question to Groq LLM for a grounded response with citations."""
        if not groq_api_key or not groq_api_key.strip():
            return (
                "⚠️ **Groq API Key Required:** Please provide your Groq API key in the sidebar to generate answers. "
                "You can get a free key instantly at [https://console.groq.com/keys](https://console.groq.com/keys)."
            )

        client = Groq(api_key=groq_api_key.strip())
        model_name, model_notice = self.resolve_available_model(client, model_name)

        spec = get_model_spec(model_name)
        prompt_budget = spec.get("prompt_budget", 4500)
        max_completion_tokens = spec.get("max_completion_tokens", 1200)

        # Build clean, deduplicated context within prompt budget
        context_str, retained_sources = build_grounded_context_prompt(
            retrieved_chunks, max_token_budget=prompt_budget
        )

        system_prompt = GROUNDED_SYSTEM_PROMPT

        user_prompt = (
            f"USER QUESTION:\n{query}\n\n"
            f"RELEVANT PEGA ACADEMY CONTEXT EXCERPTS:\n"
            f"{context_str}\n\n"
            f"Answer clearly and naturally using only the verified excerpts above, with compact inline citations `[filename, Page X, Section: Y]`:"
        )

        messages = [{"role": "system", "content": system_prompt}]
        if chat_history:
            ctx_win = spec.get("context_window", 4096)
            hist_limit = 6 if ctx_win >= 16384 else 2
            char_limit = 700 if ctx_win >= 16384 else 350
            for h in chat_history[-hist_limit:]:
                role = h.get("role", "user")
                if role in ("user", "assistant") and h.get("content"):
                    messages.append({"role": role, "content": h["content"][:char_limit]})
        messages.append({"role": "user", "content": user_prompt})

        try:
            client = Groq(api_key=groq_api_key.strip())
            try:
                raw_ans, _ = self._create_completion_with_completeness_guard(
                    client=client,
                    model_name=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_completion_tokens
                )
                verified_ans, _ = verify_citations(raw_ans, retained_sources)
                return verified_ans
            except Exception as model_err:
                err_str = str(model_err)

                # Context Length Error Recovery: single retry with halved budget and completion
                if is_context_length_error(err_str):
                    reduced_budget = max(800, prompt_budget // 2)
                    reduced_completion = max(300, max_completion_tokens // 2)
                    reduced_context_str, reduced_sources = build_grounded_context_prompt(
                        retrieved_chunks, max_token_budget=reduced_budget
                    )
                    retry_user_prompt = (
                        f"USER QUESTION:\n{query}\n\n"
                        f"RELEVANT PEGA ACADEMY CONTEXT EXCERPTS:\n"
                        f"{reduced_context_str}\n\n"
                        f"Answer clearly and naturally using only the verified excerpts above, with compact inline citations `[filename, Page X, Section: Y]`:"
                    )
                    try:
                        retry_response = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": retry_user_prompt}
                            ],
                            temperature=temperature,
                            max_tokens=reduced_completion
                        )
                        retry_choice = retry_response.choices[0]
                        raw_retry = retry_choice.message.content or ""
                        if getattr(retry_choice, "finish_reason", None) == "length":
                            raw_retry = self._trim_to_last_complete_sentence(raw_retry) + "\n\n> *Note: Response reached the model's output length limit and was trimmed at the last complete sentence.*"
                        verified_retry, _ = verify_citations(raw_retry, reduced_sources)
                        return verified_retry
                    except Exception:
                        return f"⚠️ **Context Length Notice:** The query and evidence exceed the maximum token capacity for model `{model_name}`. Please switch to a higher-capacity model like `openai/gpt-oss-120b` in Configuration."

                err_lower = err_str.lower()
                is_model_err = any(x in err_lower for x in [
                    "model_not_found", "model_terms_required", "model_decommissioned",
                    "decommissioned", "deprecated", "does not exist", "not have access"
                ])
                if is_model_err:
                    alt_model = self.find_best_alternative_model(client, model_name)
                    if alt_model and alt_model != model_name:
                        alt_spec = get_model_spec(alt_model)
                        alt_tokens = alt_spec.get("max_completion_tokens", 1000)
                        raw_alt, _ = self._create_completion_with_completeness_guard(
                            client=client,
                            model_name=alt_model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=alt_tokens
                        )
                        verified_alt, _ = verify_citations(raw_alt, retained_sources)
                        return f"> *Note: Requested model `{model_name}` was unavailable. Auto-switched to `{alt_model}`.*\n\n" + verified_alt
                raise model_err
        except Exception as e:
            return f"❌ **Error generating response from Groq:** {str(e)}"

    # ---------------------------------------------------------
    # Real-Time Streaming Generation & Agent Pipeline
    # ---------------------------------------------------------

    def _stream_completion_with_completeness_guard(
        self,
        client: Groq,
        model_name: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int
    ):
        """
        Streams tokens from Groq (stream=True) and inspects final finish_reason.
        Yields:
          - {"type": "token", "text": str}
          - {"type": "complete", "text": str, "finish_reason": Optional[str]}
        If finish_reason == 'length', performs at most ONE bounded recovery or trims cleanly
        at the last complete sentence boundary.
        """
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        accumulated = []
        finish_reason = None

        try:
            for chunk in stream:
                if not chunk or not getattr(chunk, "choices", None):
                    continue
                choice = chunk.choices[0]
                delta = getattr(choice, "delta", None)
                delta_text = getattr(delta, "content", None) if delta else None
                if delta_text:
                    accumulated.append(delta_text)
                    yield {"type": "token", "text": delta_text}
                if getattr(choice, "finish_reason", None):
                    finish_reason = choice.finish_reason
        except Exception as stream_err:
            partial = "".join(accumulated).strip()
            if not partial:
                raise stream_err
            trimmed = self._trim_to_last_complete_sentence(partial)
            yield {
                "type": "complete",
                "text": trimmed + "\n\n> *Note: Stream was interrupted and trimmed at the last complete sentence.*",
                "finish_reason": "interrupted"
            }
            return

        full_text = "".join(accumulated)
        if finish_reason != "length":
            yield {"type": "complete", "text": full_text, "finish_reason": finish_reason}
            return

        # Single bounded completeness recovery on length truncation
        spec = get_model_spec(model_name)
        ctx_win = spec.get("context_window", 4096)
        prompt_tokens_est = sum(estimate_tokens(m.get("content", "")) for m in messages)
        headroom = ctx_win - prompt_tokens_est - 256
        expanded_tokens = min(2200, max(max_tokens + 400, int(max_tokens * 1.5)), headroom)

        if expanded_tokens > max_tokens + 100:
            retry_messages = list(messages[:-1]) + [{
                "role": messages[-1]["role"],
                "content": messages[-1]["content"] + "\n\n(Be concise and complete your response cleanly within the token limit.)"
            }]
            try:
                retry_text, retry_reason = self._create_completion_with_completeness_guard(
                    client=client,
                    model_name=model_name,
                    messages=retry_messages,
                    temperature=temperature,
                    max_tokens=expanded_tokens
                )
                if retry_text:
                    yield {"type": "complete", "text": retry_text, "finish_reason": retry_reason}
                    return
            except Exception:
                pass

        trimmed = self._trim_to_last_complete_sentence(full_text)
        notice = "\n\n> *Note: Response reached the model's output length limit and was trimmed at the last complete sentence.*"
        yield {
            "type": "complete",
            "text": (trimmed + notice) if trimmed else "No complete response generated.",
            "finish_reason": "length"
        }

    def stream_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        groq_api_key: str,
        model_name: str = PRIMARY_GROQ_MODEL,
        temperature: float = 0.0,
        chat_history: Optional[List[Dict[str, str]]] = None
    ):
        """
        Streams a grounded answer progressively while preserving token budgeting,
        context-length recovery, finish_reason completeness guard, and citation verification.
        """
        if not groq_api_key or not groq_api_key.strip():
            msg = (
                "⚠️ **Groq API Key Required:** Please provide your Groq API key in Configuration to generate answers. "
                "You can get a free key at [console.groq.com/keys](https://console.groq.com/keys)."
            )
            yield {"type": "final", "answer": msg, "sources": [], "model_used": model_name or PRIMARY_GROQ_MODEL}
            return

        client = Groq(api_key=groq_api_key.strip())
        model_name, model_notice = self.resolve_available_model(client, model_name)

        spec = get_model_spec(model_name)
        prompt_budget = spec.get("prompt_budget", 4500)
        max_completion_tokens = spec.get("max_completion_tokens", 1200)

        context_str, retained_sources = build_grounded_context_prompt(
            retrieved_chunks, max_token_budget=prompt_budget
        )
        final_sources = deduplicate_sources(retained_sources)

        user_prompt = (
            f"USER QUESTION:\n{query}\n\n"
            f"RELEVANT PEGA ACADEMY CONTEXT EXCERPTS:\n"
            f"{context_str}\n\n"
            f"Answer clearly and naturally using only the verified excerpts above, with compact inline citations `[filename, Page X, Section: Y]`:"
        )

        messages = [{"role": "system", "content": GROUNDED_SYSTEM_PROMPT}]
        if chat_history:
            ctx_win = spec.get("context_window", 4096)
            hist_limit = 6 if ctx_win >= 16384 else 2
            char_limit = 700 if ctx_win >= 16384 else 350
            for h in chat_history[-hist_limit:]:
                role = h.get("role", "user")
                if role in ("user", "assistant") and h.get("content"):
                    messages.append({"role": role, "content": h["content"][:char_limit]})
        messages.append({"role": "user", "content": user_prompt})

        try:
            try:
                completed_text = ""
                for ev in self._stream_completion_with_completeness_guard(
                    client=client,
                    model_name=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_completion_tokens
                ):
                    if ev["type"] == "token":
                        yield ev
                    elif ev["type"] == "complete":
                        completed_text = ev["text"]
                verified_ans, _ = verify_citations(completed_text, final_sources)
                prefix_note = f"> *Note: {model_notice}*\n\n" if model_notice else ""
                yield {"type": "final", "answer": prefix_note + verified_ans, "sources": final_sources, "model_used": model_name}
                return
            except Exception as model_err:
                err_str = str(model_err)
                if is_context_length_error(err_str):
                    reduced_budget = max(800, prompt_budget // 2)
                    reduced_completion = max(300, max_completion_tokens // 2)
                    reduced_context_str, reduced_sources = build_grounded_context_prompt(
                        retrieved_chunks, max_token_budget=reduced_budget
                    )
                    retry_user_prompt = (
                        f"USER QUESTION:\n{query}\n\n"
                        f"RELEVANT PEGA ACADEMY CONTEXT EXCERPTS:\n"
                        f"{reduced_context_str}\n\n"
                        f"Answer clearly and naturally using only the verified excerpts above, with compact inline citations `[filename, Page X, Section: Y]`:"
                    )
                    retry_messages = [
                        {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
                        {"role": "user", "content": retry_user_prompt}
                    ]
                    completed_retry = ""
                    for ev in self._stream_completion_with_completeness_guard(
                        client=client,
                        model_name=model_name,
                        messages=retry_messages,
                        temperature=temperature,
                        max_tokens=reduced_completion
                    ):
                        if ev["type"] == "token":
                            yield ev
                        elif ev["type"] == "complete":
                            completed_retry = ev["text"]
                    final_reduced = deduplicate_sources(reduced_sources)
                    verified_retry, _ = verify_citations(completed_retry, final_reduced)
                    yield {"type": "final", "answer": verified_retry, "sources": final_reduced}
                    return

                err_lower = err_str.lower()
                is_model_err = any(x in err_lower for x in [
                    "model_not_found", "model_terms_required", "model_decommissioned",
                    "decommissioned", "deprecated", "does not exist", "not have access"
                ])
                if is_model_err:
                    alt_model = self.find_best_alternative_model(client, model_name)
                    if alt_model and alt_model != model_name:
                        alt_spec = get_model_spec(alt_model)
                        alt_tokens = alt_spec.get("max_completion_tokens", 1000)
                        completed_alt = ""
                        for ev in self._stream_completion_with_completeness_guard(
                            client=client,
                            model_name=alt_model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=alt_tokens
                        ):
                            if ev["type"] == "token":
                                yield ev
                            elif ev["type"] == "complete":
                                completed_alt = ev["text"]
                        verified_alt, _ = verify_citations(completed_alt, final_sources)
                        note = f"> *Note: Requested model `{model_name}` was unavailable. Auto-switched to `{alt_model}`.*\n\n"
                        yield {"type": "final", "answer": note + verified_alt, "sources": final_sources, "model_used": alt_model}
                        return
                raise model_err
        except Exception as e:
            yield {
                "type": "final",
                "answer": f"❌ **Error generating response from Groq:** {str(e)}",
                "sources": final_sources,
                "model_used": model_name
            }

    def run_agent_stream(
        self,
        query: str,
        groq_api_key: str,
        model_name: str = PRIMARY_GROQ_MODEL,
        temperature: float = 0.0,
        max_iterations: int = 4,
        chat_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = DEFAULT_FINAL_EVIDENCE_K,
        use_reranker: bool = True,
        dense_weight: float = 0.5,
        section_filter: Optional[str] = None
    ):
        """
        Streaming-enabled Agentic & Hybrid RAG pipeline.
        Yields:
          - {"type": "status", "message": str}
          - {"type": "token", "text": str}
          - {"type": "final", "answer": str, "sources": list, "trace": list}
        """
        if not groq_api_key or not groq_api_key.strip():
            yield {
                "type": "final",
                "answer": "⚠️ **Groq API Key Required:** Please provide your Groq API key in Configuration to activate the Assistant.",
                "sources": [],
                "trace": [],
                "model_used": model_name or PRIMARY_GROQ_MODEL
            }
            return

        client = Groq(api_key=groq_api_key.strip())
        model_name, model_notice = self.resolve_available_model(client, model_name)
        execution_trace: List[Dict[str, Any]] = []
        if model_notice:
            execution_trace.append({"type": "note", "message": model_notice})
        collected_sources: List[Dict[str, Any]] = []

        # Route non-tool models and straightforward queries to fast Hybrid + Reranked Direct Streaming
        use_agent, route_reason = should_use_agent_loop(query)
        if (not is_tool_capable_model(model_name)) or (not use_agent):
            yield {"type": "status", "message": "Retrieving Pega curriculum excerpts (Hybrid BM25 + Vector + Reranker)..."}
            search_query = contextualize_query_with_history(query, chat_history)
            retrieved_chunks = self.search(
                query=search_query,
                top_k=top_k,
                section_filter=section_filter,
                use_reranker=use_reranker,
                dense_weight=dense_weight
            )
            if any(w in search_query.lower() for w in ["how to", "configure", "implement", "step", "challenge", "exercise", "create", "setup"]):
                try:
                    ch_res = self.tool_search_challenges(query=search_query, top_k=2)
                    for r in ch_res.get("results", []):
                        if not any(c.get("document") == r.get("excerpt") for c in retrieved_chunks):
                            retrieved_chunks.append({
                                "source": r.get("document", "Challenge"),
                                "page": str(r.get("page", "1")),
                                "section": r.get("section", "Hands-on Challenges"),
                                "category": "Hands-on Challenge",
                                "score": float(r.get("relevance_score", "0%").replace("%", "")),
                                "document": r.get("excerpt", "")
                            })
                except Exception:
                    pass

            if not retrieved_chunks:
                yield {
                    "type": "final",
                    "answer": "The provided Pega curriculum documentation does not contain relevant excerpts for this query.",
                    "sources": [],
                    "trace": execution_trace
                }
                return

            yield {"type": "status", "message": "Generating grounded response..."}
            for ev in self.stream_answer(
                query=query,
                retrieved_chunks=retrieved_chunks,
                groq_api_key=groq_api_key,
                model_name=model_name,
                temperature=temperature,
                chat_history=chat_history
            ):
                if ev["type"] == "token":
                    yield ev
                elif ev["type"] == "final":
                    yield {
                        "type": "final",
                        "answer": ev["answer"],
                        "sources": ev["sources"],
                        "trace": execution_trace
                    }
            return

        # Multi-step Agentic Tool-Calling Loop
        client = Groq(api_key=groq_api_key.strip())
        tools_schema = self.get_agent_tools_schema()
        spec = get_model_spec(model_name)
        loop_max_tokens = min(2000, spec.get("max_completion_tokens", 1500) + 500)

        system_prompt = AGENT_SYSTEM_PROMPT

        messages: List[Any] = [{"role": "system", "content": system_prompt}]
        if chat_history:
            for h in chat_history[-6:]:
                role = h.get("role", "user")
                if role in ("user", "assistant") and h.get("content"):
                    messages.append({"role": role, "content": h["content"][:700]})
        messages.append({"role": "user", "content": query})

        current_iteration = 0
        while current_iteration < max_iterations:
            current_iteration += 1
            yield {"type": "status", "message": f"Researching Pega curriculum (step {current_iteration}/{max_iterations})..."}

            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=tools_schema,
                    tool_choice="auto",
                    temperature=temperature,
                    max_tokens=loop_max_tokens
                )
            except Exception as e:
                err_str = str(e).lower()
                if any(x in err_str for x in ["tool calling", "tools is not supported", "does not support tools", "param': 'tools'"]):
                    yield {"type": "status", "message": "Model does not support tool calling; switching to Direct Grounded Streaming..."}
                    chunks = self.search(query=contextualize_query_with_history(query, chat_history), top_k=top_k, section_filter=section_filter, use_reranker=use_reranker, dense_weight=dense_weight)
                    for ev in self.stream_answer(query=query, retrieved_chunks=chunks, groq_api_key=groq_api_key, model_name=model_name, temperature=temperature, chat_history=chat_history):
                        if ev["type"] == "token":
                            yield ev
                        elif ev["type"] == "final":
                            yield {"type": "final", "answer": ev["answer"], "sources": ev["sources"], "trace": execution_trace}
                    return
                yield {"type": "final", "answer": f"❌ **Groq API Error:** {str(e)}", "sources": deduplicate_sources(collected_sources), "trace": execution_trace}
                return

            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            if not tool_calls:
                final_sources = deduplicate_sources(collected_sources)
                raw_ans = response_message.content or "No response generated."
                if getattr(response.choices[0], "finish_reason", None) == "length":
                    raw_ans = self._trim_to_last_complete_sentence(raw_ans) + "\n\n> *Note: Response reached the model's output length limit and was trimmed at the last complete sentence.*"
                verified_ans, _ = verify_citations(raw_ans, final_sources)
                yield {"type": "final", "answer": verified_ans, "sources": final_sources, "trace": execution_trace}
                return

            messages.append(response_message)
            for tool_call in tool_calls:
                func_name = tool_call.function.name
                try:
                    parsed_args = json.loads(tool_call.function.arguments or "{}")
                    func_args = parsed_args if isinstance(parsed_args, dict) else {}
                except Exception:
                    func_args = {}

                if func_name == "search_curriculum":
                    q_arg = func_args.get("query") or query
                    yield {"type": "status", "message": f"Searching curriculum for \"{q_arg}\"..."}
                    tool_output_dict = self.tool_search_curriculum(query=q_arg, section=func_args.get("section"))
                elif func_name == "search_challenges":
                    q_arg = func_args.get("query") or query
                    yield {"type": "status", "message": f"Searching hands-on challenges for \"{q_arg}\"..."}
                    tool_output_dict = self.tool_search_challenges(query=q_arg)
                elif func_name == "read_full_page":
                    try:
                        page_num_arg = int(func_args.get("page_number") or 1)
                    except (TypeError, ValueError):
                        page_num_arg = 1
                    doc_arg = func_args.get("document_name") or ""
                    yield {"type": "status", "message": f"Reading page {page_num_arg} of {doc_arg or 'PDF'}..."}
                    tool_output_dict = self.tool_read_full_page(document_name=doc_arg, page_number=page_num_arg)
                elif func_name == "lookup_pega_terms":
                    term_arg = func_args.get("term") or query
                    yield {"type": "status", "message": f"Looking up Pega term \"{term_arg}\"..."}
                    tool_output_dict = self.tool_lookup_pega_terms(term=term_arg)
                else:
                    tool_output_dict = {"status": "error", "message": f"Unknown tool '{func_name}'"}

                if not isinstance(tool_output_dict, dict):
                    tool_output_dict = {}

                if isinstance(tool_output_dict.get("results"), list):
                    for r in tool_output_dict["results"]:
                        if not isinstance(r, dict):
                            continue
                        doc_label = str(r.get("document") or "Unknown")
                        rel_score = str(r.get("relevance_score") or "0%").replace("%", "")
                        try:
                            score_val = float(rel_score)
                        except ValueError:
                            score_val = 0.0
                        collected_sources.append({
                            "source": doc_label,
                            "page": str(r.get("page") or "1"),
                            "section": r.get("section") or "General",
                            "category": "Hands-on Challenge" if "challenge" in doc_label.lower() else "Master Module",
                            "score": score_val,
                            "document": r.get("excerpt") or ""
                        })
                elif "full_page_text" in tool_output_dict:
                    full_text = tool_output_dict.get("full_page_text") or ""
                    collected_sources.append({
                        "source": tool_output_dict.get("document") or "Unknown",
                        "page": str(tool_output_dict.get("page") or "1"),
                        "section": "Direct Page Read",
                        "category": "Full Page Read",
                        "score": 100.0,
                        "document": full_text
                    })

                execution_trace.append({
                    "step": current_iteration,
                    "tool": func_name,
                    "args": func_args,
                    "summary": f"Executed {func_name}"
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": json.dumps(tool_output_dict, ensure_ascii=False)
                })

        # Stream final synthesis from collected evidence
        unique_sources = deduplicate_sources(collected_sources)
        if not unique_sources:
            unique_sources = self.search(
                query=contextualize_query_with_history(query, chat_history),
                top_k=top_k,
                section_filter=section_filter,
                use_reranker=use_reranker,
                dense_weight=dense_weight
            )

        yield {"type": "status", "message": "Synthesizing verified architectural response..."}
        for ev in self.stream_answer(
            query=query,
            retrieved_chunks=unique_sources,
            groq_api_key=groq_api_key,
            model_name=model_name,
            temperature=temperature,
            chat_history=chat_history
        ):
            if ev["type"] == "token":
                yield ev
            elif ev["type"] == "final":
                yield {"type": "final", "answer": ev["answer"], "sources": ev["sources"], "trace": execution_trace}
