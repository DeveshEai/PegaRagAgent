"""
Grounding, citation verification, deduplication, token-budgeted context construction,
and conversation contextualization for PegaRagAgent.
"""

import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple


def estimate_tokens(text: str) -> int:
    """
    Lightweight, conservative token estimator (~3.6 characters per token).
    Avoids introducing heavy tokenizer dependencies or network calls.
    """
    if not text:
        return 0
    return max(1, int(len(text) / 3.6))


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
    from retrieval.hybrid import tokenize_pega_text

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
