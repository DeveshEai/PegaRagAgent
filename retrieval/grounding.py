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
    # Collapse only inline multi-spaces between non-space chars; preserve Markdown hard breaks ('  \n') and list indentation
    verified_text = re.sub(r"(?<=\S) {2,}(?=\S)", " ", verified_text)
    verified_text = re.sub(r"(?<! ) \n", "\n", verified_text)
    verified_text = re.sub(r" +([.,;:!?])", r"\1", verified_text).strip()
    return verified_text, {
        "verified": verified_count,
        "removed": removed_count,
        "mismatched_claims": mismatched_claims
    }


_FOLLOWUP_PHRASES = (
    "step ", "elaborate", "explain more", "apply", "example",
    "how about", "what about", "more details", "tell me more",
)
_FOLLOWUP_PRONOUN_RE = re.compile(r"\b(it|its|they|them|their|theirs|this|that|these|those)\b", re.IGNORECASE)
_GREETING_ONLY_RE = re.compile(r"^(hi|hello|hey|thanks|thank you|ok|okay|cool|got it)[!. ]*$", re.IGNORECASE)


def _extract_recent_subject(chat_history: List[Dict[str, str]]) -> str:
    """Extract the most recent substantive topic from recent user/assistant turns."""
    # Prefer the most recent substantive user question (skipping greetings or bare pronoun follow-ups)
    for msg in reversed(chat_history[-6:]):
        if msg.get("role") != "user":
            continue
        content = (msg.get("content") or "").strip()
        if not content or _GREETING_ONLY_RE.match(content):
            continue
        # If this prior user turn is itself a bare pronoun follow-up (e.g. "What about retries?"), keep looking back
        words = content.split()
        if len(words) <= 6 and (_FOLLOWUP_PRONOUN_RE.search(content) or content.lower().startswith(("what about", "how about"))):
            continue
        return content[:200]

    # Fallback: extract the first substantive sentence from the most recent assistant message
    for msg in reversed(chat_history[-4:]):
        if msg.get("role") != "assistant":
            continue
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        first_line = content.splitlines()[0].strip()
        first_sentence = re.split(r"(?<=[.!?])\s+", first_line)[0].strip()
        if len(first_sentence) >= 12:
            return first_sentence[:200]
    return ""


def contextualize_query_with_history(
    query: str,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Expands short or referential follow-up questions (including 'its'/'their')
    with the most recent substantive conversation subject so Hybrid BM25 + Vector
    retrieval maintains thread context without attaching the full history.
    """
    if not chat_history:
        return query
    q_low = (query or "").lower().strip()
    if not q_low:
        return query

    is_short = len(q_low.split()) <= 10
    has_referential = bool(_FOLLOWUP_PRONOUN_RE.search(q_low)) or any(sig in q_low for sig in _FOLLOWUP_PHRASES)
    if is_short or has_referential:
        last_topic = _extract_recent_subject(chat_history)
        if last_topic and last_topic.lower() != q_low:
            return f"{last_topic} — {query}"
    return query
