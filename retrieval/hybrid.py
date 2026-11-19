"""
Hybrid retrieval primitives: Pega-aware tokenization, Okapi BM25 retriever,
weighted Reciprocal Rank Fusion (RRF), Cross-Encoder reranking, and similarity scoring.
"""

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from config.settings import (
    BM25_B,
    BM25_K1,
    DEFAULT_CANDIDATE_K,
    DEFAULT_FINAL_EVIDENCE_K,
    ENABLE_RERANKER,
    MIN_RERANK_PROBABILITY,
    RERANKER_MODEL_NAME,
    RRF_K,
)
from retrieval.grounding import clean_chunk_content


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

    def __init__(self, documents: List[str], metadatas: List[Dict[str, Any]], k1: float = BM25_K1, b: float = BM25_B):
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
