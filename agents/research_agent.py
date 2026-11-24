"""
Autonomous research agent orchestration (non-streaming and streaming)
for PegaRagAgent.
"""

import json
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from groq import Groq

from agents.answer_generator import (
    detect_output_format,
    execute_direct_grounded_fallback,
    find_best_alternative_model,
    generate_answer,
    is_context_length_error,
    resolve_available_model,
    stream_answer,
    trim_to_last_complete_sentence,
)
from agents.tools import collect_tool_sources, get_agent_tools_schema
from config.settings import (
    DEFAULT_FINAL_EVIDENCE_K,
    INTENT_CATEGORIES,
    MAX_ROUTING_ESCALATIONS,
    PRIMARY_GROQ_MODEL,
    ROUTER_MODEL,
    ROUTER_TIMEOUT_SECONDS,
    get_model_spec,
    is_tool_capable_model,
)
from retrieval.grounding import (
    contextualize_query_with_history,
    deduplicate_sources,
    verify_citations,
)

AGENT_SYSTEM_PROMPT = (
    "You are a Senior Pega Certified Lead System Architect (CLSA) advising a fellow architect in a direct, confident, peer-to-peer colleague style.\n\n"
    "TOOL USE & RESEARCH:\n"
    "1. Call your tools (`search_curriculum`, `search_challenges`, `read_full_page`, `lookup_pega_terms`) to retrieve verified Pega excerpts before answering. Query both curriculum and challenges when a topic spans design theory and hands-on configuration, and use `read_full_page` if an excerpt is truncated.\n\n"
    "PEGA CLSA COMMUNICATION & FORMATTING:\n"
    "2. Lead immediately with a clear, practical answer—skip generic AI greetings, filler introductions, and formulaic headings ('Short answer', 'What the excerpts actually tell us', 'Implication for architecture and governance', 'Bottom line').\n"
    "3. Connect business impact with technical Pega implementation using concrete artifacts (`pyWorkPage`, `D_*` Data Pages, `Queue Processor`, `Job Scheduler`, `Case Type`, etc.) when relevant.\n"
    "4. Maintain strict Pega terminology accuracy: never conflate `Queue Processor` (asynchronous Kafka-backed background processing) with `Work Queue` / `Workbasket` (human task assignment and routing).\n"
    "5. For stale Data Page or caching troubleshooting, separate configuration-dependent causes—scope (`Thread`, `Requestor`, `Node`), refresh strategy, reload conditions, explicit invalidation, source freshness, and node vs. cluster behavior—without assuming unstated configuration.\n"
    "6. Strictly honor requested output formats: comparisons (`compare`, `vs`, `difference`) MUST default to a Markdown comparison table; explicit requests for a table, bulleted list, numbered steps, or example MUST use that exact format.\n"
    "7. Follow-up policy: When genuinely helpful to clarify Pega version, UI (`Constellation` vs `Theme Cosmos`), scope, or next implementation step, end with one brief, relevant follow-up question. Never force generic closing statements.\n\n"
    "CONDITIONAL GROUNDING & CITATION INTEGRITY:\n"
    "8. Attach inline citations `[filename, Page X, Section: Y]` (using only files/pages returned by your tools, including inside table cells or lists) ONLY to claims supported by retrieved excerpts. Never invent citations or present general knowledge as retrieved evidence.\n"
    "9. If retrieved excerpts remain insufficient after searching, state the knowledge gap concisely and stop—do NOT pad with speculative explanations or unsupported general-knowledge filler."
)


_CONVERSATIONAL_GREETINGS = frozenset({
    "hi", "hello", "hey", "hii", "heyy", "hiya", "yo",
    "good morning", "good afternoon", "good evening",
    "hi there", "hello there", "hey there",
})
_CONVERSATIONAL_THANKS = frozenset({
    "thanks", "thank you", "thx", "ty", "thanks a lot",
    "thank you very much", "appreciate it", "much appreciated",
})
_CONVERSATIONAL_SMALLTALK = frozenset({
    "how are you", "how are you doing", "who are you",
    "what can you do", "what can you help me with", "what can you help with",
    "how can you help", "how can you help me", "what do you do",
    "help", "bye", "goodbye", "see you",
    "ok", "okay", "got it", "cool", "nice", "sounds good", "great",
})
_PEGA_ANCHOR_WORDS = frozenset({
    "pega", "clsa", "lsa", "sla", "rule", "rules", "case", "data",
    "clipboard", "constellation", "queue", "scheduler", "integration",
    "security", "abac", "rbac", "circumstanc", "pyworkpage", "dx",
})


def detect_conversational_intent(query: str) -> Optional[Dict[str, Any]]:
    """
    Deterministically detect pure greetings, thanks, and casual conversation
    in the shared execution layer before retrieval, while preserving short Pega
    knowledge queries (e.g., 'What is SLA?', 'Data Page?').
    """
    raw = (query or "").strip()
    if not raw:
        return None
    q_clean = re.sub(r"[^\w\s']", " ", raw.lower())
    q_clean = " ".join(q_clean.split())
    if not q_clean:
        return None

    # Never hijack queries containing Pega domain terms, anchor words, or question prefixes
    if q_clean.startswith(("what is ", "define ", "what are ", "explain ", "compare ", "how to ", "how do ", "how does ")):
        if q_clean not in _CONVERSATIONAL_SMALLTALK:
            return None
    words = set(q_clean.split())
    if words & _PEGA_ANCHOR_WORDS:
        return None
    if analyze_pega_query_domain(raw)["matched_domains"]:
        return None

    if q_clean in _CONVERSATIONAL_GREETINGS:
        answer = (
            "Hey! Ready whenever you are—what Pega architecture, rule design, "
            "or troubleshooting question are we tackling today?"
        )
    elif q_clean in _CONVERSATIONAL_THANKS:
        answer = (
            "You're welcome! Let me know if you want to dig deeper into any Pega rule, "
            "Constellation pattern, or design trade-off."
        )
    elif q_clean in _CONVERSATIONAL_SMALLTALK:
        if q_clean in (
            "who are you", "what can you do", "help",
            "what can you help me with", "what can you help with",
            "how can you help", "how can you help me", "what do you do",
        ):
            answer = (
                "I'm your Pega CLSA architecture assistant. Ask me about Data Pages, "
                "Rule Resolution, Case Management, Queue Processors, Constellation, "
                "integrations, or hands-on challenges from your indexed Pega guides."
            )
        elif q_clean in ("bye", "goodbye", "see you"):
            answer = "Sounds good—catch you later! Ping me anytime you have a Pega question."
        else:
            answer = "Got it! Whenever you're ready, share the Pega topic or scenario you'd like to explore."
    else:
        return None

    return {
        "intent": "CONVERSATIONAL",
        "route": "CONVERSATIONAL",
        "answer": answer,
        "reason": "Conversational greeting, thanks, or small-talk detected before retrieval.",
    }


def _build_execution_meta(
    execution_strategy: str,
    detected_intent: str,
    output_format: str = "PARAGRAPH",
    router_model: Optional[str] = None,
    evidence_status: str = "Not applicable",
    escalation_status: str = "Not escalated",
    trace: Optional[List[Dict[str, Any]]] = None,
    sources: Optional[List[Dict[str, Any]]] = None,
    citation_verification: Any = "Not applicable",
    retrieval_performed: bool = True,
) -> Dict[str, Any]:
    """Assemble sanitized execution metadata from actual pipeline outputs."""
    trace = trace or []
    sources = sources or []
    tools_used = [
        t.get("tool")
        for t in trace
        if t.get("tool") and t.get("tool") not in ("direct_grounded_retrieval", "conversational_fast_path")
    ]
    search_count = sum(
        1 for t in trace if t.get("tool") in ("direct_grounded_retrieval", "search_curriculum", "search_challenges")
    )
    if retrieval_performed and search_count == 0 and sources:
        search_count = 1
    docs = []
    seen_docs = set()
    for s in sources:
        src = str(s.get("source") or "").strip()
        pg = str(s.get("page") or "").strip()
        if src and (src, pg) not in seen_docs:
            seen_docs.add((src, pg))
            docs.append({"source": src, "page": pg, "section": str(s.get("section") or "General")})
    return {
        "execution_strategy": execution_strategy,
        "detected_intent": detected_intent,
        "output_format": output_format,
        "router_model": router_model or "deterministic",
        "evidence_status": evidence_status,
        "escalation_status": escalation_status,
        "tools_used": tools_used,
        "search_count": search_count,
        "documents_consulted": docs,
        "citation_verification": citation_verification,
        "retrieval_performed": retrieval_performed,
    }


PEGA_DOMAIN_TERMS: Dict[str, Tuple[str, ...]] = {
    "data_pages": (
        "data page", "data pages", "d_", "declare page", "page list",
        "page group", "keyed data page", "load management", "refresh strategy",
        "savable data page",
    ),
    "clipboard": (
        "clipboard", "pyworkpage", "pyworkparty", "primary page", "step page",
        "embedded page", "named page", "pxresults",
    ),
    "rule_resolution": (
        "rule resolution", "ruleset", "rule availability", "final rule",
        "withdrawn", "blocked", "branching", "skimming", "class inheritance",
        "pattern inheritance", "directed inheritance",
    ),
    "circumstancing": (
        "circumstancing", "circumstanced", "multivariate circumstance",
        "single-value circumstance", "template circumstance", "date circumstance",
        "as-of date",
    ),
    "case_management": (
        "case type", "case management", "case lifecycle", "stage", "subcase",
        "child case", "parent case", "case locking", "optimistic locking",
        "pessimistic locking", "wait shape", "duplicate search",
    ),
    "sla": (
        "sla", "service-level", "service level", "goal", "deadline",
        "passed deadline", "escalation action", "urgency",
    ),
    "queue_processor": (
        "queue processor", "standard queue processor", "dedicated queue processor",
        "delayed item", "broken-process", "broken queue", "kafka",
        "queue-for-processing",
    ),
    "job_scheduler": (
        "job scheduler", "scheduled job", "node classification",
        "background processing", "agent schedule",
    ),
    "integration": (
        "connect-rest", "connect-soap", "service-rest", "service-soap",
        "dx api", "connector", "rest integration", "oauth 2.0",
        "auth profile", "mapping rule", "json transform", "integration",
    ),
    "security": (
        "abac", "rbac", "access control", "access group", "access role",
        "privilege", "access of role to object", "aro", "access deny",
        "client-based access", "authentication", "authorization", "keystore",
        "security",
    ),
    "constellation": (
        "constellation", "dx component", "cosmos react", "landing page",
        "ui-kit", "theme cosmos", "app studio",
    ),
    "data_management": (
        "data transform", "data model", "system of record", "sor",
        "local data storage", "data type", "report definition",
        "declare expression", "declare index", "validate rule",
        "decision table", "decision tree", "when rule",
    ),
}

PEGA_COMPLEXITY_SIGNALS: Dict[str, Tuple[str, ...]] = {
    "architecture_design": (
        "architect", "architecture", "design an", "design a",
        "enterprise design", "class structure", "multi-application", "ecs",
        "scalability", "high availability", "production design",
    ),
    "troubleshooting": (
        "troubleshoot", "debug", "failing", "not working", "error",
        "exception", "stuck", "broken", "timeout", "deadlock", "tracer",
    ),
    "root_cause_analysis": (
        "root cause", "why does", "why is", "investigate why", "diagnose",
        "trace the issue",
    ),
    "dependency_impact": (
        "impact of", "impact analysis", "downstream impact", "dependenc",
        "blast radius", "side effect", "migration impact",
    ),
    "multi_rule_investigation": (
        "across multiple", "multiple rules", "interact across", "end-to-end",
        "end to end", "cross-pillar", "cross pillar", "combined with",
        "how do all", "relationship between", "relationships between",
        "interaction between", "work together",
    ),
    "implementation_walkthrough": (
        "challenge", "walkthrough", "step-by-step", "step by step",
        "full page", "page ", "hands-on", "configure end",
    ),
}


def _term_matches(q_low: str, term: str) -> bool:
    """Match a Pega domain or complexity term without substring false positives."""
    if term == "d_":
        return re.search(r"\bd_[a-z0-9_]+", q_low) is not None
    if term == "page ":
        return re.search(r"\bpage\s+\d+", q_low) is not None
    if len(term) <= 4 and term.isalnum():
        return re.search(rf"\b{re.escape(term)}\b", q_low) is not None
    return term in q_low


def analyze_pega_query_domain(query: str) -> Dict[str, Any]:
    """
    Inspect a Pega query for domain categories, task complexity signals,
    and whether it represents a simple definition or straightforward comparison.
    """
    q_low = (query or "").lower().strip()
    words = q_low.split()

    matched_domains = [
        domain
        for domain, terms in PEGA_DOMAIN_TERMS.items()
        if any(_term_matches(q_low, t) for t in terms)
    ]
    matched_complexity = [
        sig_type
        for sig_type, signals in PEGA_COMPLEXITY_SIGNALS.items()
        if any(_term_matches(q_low, s) for s in signals)
    ]

    has_multi_question = (";" in q_low) or (q_low.count("?") >= 2)
    comparison_markers = ("compare ", " vs ", "versus", "difference between", "differences between")
    is_comparison_phrase = any(m in q_low for m in comparison_markers)

    is_simple_definition = (
        not matched_complexity
        and not has_multi_question
        and not is_comparison_phrase
        and len(words) <= 12
        and q_low.startswith(("what is ", "define ", "what are ", "explain ", "how does ", "what does "))
    )

    is_straightforward_comparison = (
        not matched_complexity
        and not has_multi_question
        and is_comparison_phrase
        and len(words) <= 14
        and len(matched_domains) <= 2
    )

    return {
        "matched_domains": matched_domains,
        "matched_complexity": matched_complexity,
        "has_multi_question": has_multi_question,
        "is_simple_definition": is_simple_definition,
        "is_straightforward_comparison": is_straightforward_comparison,
        "word_count": len(words),
    }


def should_use_agent_loop(query: str) -> Tuple[bool, str]:
    """
    Lightweight Pega-domain-aware query router:
    - Simple Pega definitions, individual rule explanations, and straightforward comparisons
      route to fast Hybrid + Reranked Direct Retrieval.
    - Architecture design, troubleshooting, root-cause/impact analysis, multi-rule investigations,
      and implementation walkthroughs route to the multi-step Agent Loop.
    """
    analysis = analyze_pega_query_domain(query)
    if analysis["is_simple_definition"] or analysis["is_straightforward_comparison"]:
        return False, "Straightforward Pega conceptual or comparison query; using Hybrid Retrieval + Cross-Encoder Reranking."

    matched_complexity = analysis["matched_complexity"]
    matched_domains = analysis["matched_domains"]
    high_complexity_types = {
        "implementation_walkthrough",
        "root_cause_analysis",
        "dependency_impact",
        "multi_rule_investigation",
    }

    if (
        any(c in high_complexity_types for c in matched_complexity)
        or analysis["has_multi_question"]
        or ("investigate" in (query or "").lower() and len(matched_domains) >= 2)
        or (
            ("architecture_design" in matched_complexity or "troubleshooting" in matched_complexity)
            and (len(matched_domains) >= 2 or analysis["word_count"] >= 14)
        )
    ):
        return True, "Multi-step Pega investigative or architectural query detected; using autonomous tool-calling agent."

    return False, "Straightforward architectural query; using Hybrid Retrieval + Cross-Encoder Reranking."


_ROUTER_SYSTEM_PROMPT = (
    "You are an intent classifier for a Pega CLSA RAG system. "
    "Classify the user's question (considering conversation history for follow-ups) into JSON ONLY:\n"
    '{"intent":"<CATEGORY>","route":"DIRECT_RAG"|"AUTONOMOUS_AGENT","reason":"<under 15 words>"}\n\n'
    "Valid categories:\n"
    "- SIMPLE_LOOKUP: Single-term definition or direct fact lookup -> DIRECT_RAG\n"
    "- TECHNICAL_EXPLANATION: Focused explanation of one Pega concept/rule -> DIRECT_RAG\n"
    "- COMPARISON: Comparing options; use DIRECT_RAG if simple/single-doc, AUTONOMOUS_AGENT if multi-doc or architectural trade-off\n"
    "- COMPLEX_ARCHITECTURE: Multi-pillar design, governance, or troubleshooting scenario -> AUTONOMOUS_AGENT\n"
    "- MULTI_DOCUMENT_RESEARCH: Explicitly spans multiple documents, modules, or challenges -> AUTONOMOUS_AGENT\n"
    "- STEP_BY_STEP_CHALLENGE: Hands-on challenge, configuration walkthrough, or full-page reading -> AUTONOMOUS_AGENT\n\n"
    "Output valid JSON only. No markdown fences or extra text."
)


def _extract_json_object(raw_text: str) -> Optional[Dict[str, Any]]:
    """Extracts and parses the first JSON object from an LLM response string."""
    if not raw_text:
        return None
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None


def classify_query_intent(
    query: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    groq_api_key: str = "",
    answer_model_name: str = PRIMARY_GROQ_MODEL,
    router_model: str = ROUTER_MODEL,
    groq_factory: Callable[..., Any] = Groq,
    should_use_agent_fn: Callable[[str], Tuple[bool, str]] = should_use_agent_loop,
) -> Dict[str, Any]:
    """
    Hybrid 2-tier query intent classifier (deterministic fast path + lightweight LLM router):
    - Respects answer_model_name tool-calling capability.
    - Preserves conversation history for follow-up questions.
    - Records routing latency and token usage; falls back deterministically on errors.
    """
    t0 = time.perf_counter()
    effective_query = contextualize_query_with_history(query, chat_history)

    # 0. Conversational fast path (greetings, thanks, small-talk) before retrieval
    conv = detect_conversational_intent(query)
    if conv is not None:
        return {
            "intent": "CONVERSATIONAL",
            "route": "CONVERSATIONAL",
            "reason": conv["reason"],
            "tier": "deterministic",
            "model_used": None,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "tokens_used": 0,
        }

    # 1. Protect non-tool-capable answer models (e.g., qwen/qwen3.8-27b, allam-2-7b)
    if not is_tool_capable_model(answer_model_name):
        return {
            "intent": "TECHNICAL_EXPLANATION",
            "route": "DIRECT_RAG",
            "reason": f"Selected answer model '{answer_model_name}' does not support tool calling.",
            "tier": "deterministic",
            "model_used": None,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "tokens_used": 0,
        }

    # 2. Pega-domain-aware deterministic fast paths (0 extra API calls)
    q_low = (query or "").lower().strip()
    has_history = bool(chat_history)
    analysis = analyze_pega_query_domain(query)
    matched_complexity = analysis["matched_complexity"]
    matched_domains = analysis["matched_domains"]

    high_complexity_types = {
        "implementation_walkthrough",
        "root_cause_analysis",
        "dependency_impact",
        "multi_rule_investigation",
    }
    is_deterministic_agent = (
        any(c in high_complexity_types for c in matched_complexity)
        or analysis["has_multi_question"]
        or ("investigate" in q_low and len(matched_domains) >= 2)
        or (
            ("architecture_design" in matched_complexity or "troubleshooting" in matched_complexity)
            and (len(matched_domains) >= 2 or analysis["word_count"] >= 14)
        )
    )
    if is_deterministic_agent:
        if "implementation_walkthrough" in matched_complexity:
            det_intent = "STEP_BY_STEP_CHALLENGE"
        elif "troubleshooting" in matched_complexity or "root_cause_analysis" in matched_complexity:
            det_intent = "TROUBLESHOOTING"
        else:
            det_intent = "COMPLEX_ARCHITECTURE"
        return {
            "intent": det_intent,
            "route": "AUTONOMOUS_AGENT",
            "reason": "Multi-step Pega investigative, troubleshooting, or architectural task detected.",
            "tier": "deterministic",
            "model_used": None,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "tokens_used": 0,
        }

    if not has_history and analysis["is_simple_definition"]:
        is_lookup = q_low.startswith(("what is ", "define ", "what are "))
        return {
            "intent": "SIMPLE_LOOKUP" if is_lookup else "TECHNICAL_EXPLANATION",
            "route": "DIRECT_RAG",
            "reason": "Standalone Pega definition or single-rule explanation.",
            "tier": "deterministic",
            "model_used": None,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "tokens_used": 0,
        }

    if not has_history and analysis["is_straightforward_comparison"]:
        return {
            "intent": "COMPARISON",
            "route": "DIRECT_RAG",
            "reason": "Straightforward Pega concept or rule comparison.",
            "tier": "deterministic",
            "model_used": None,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "tokens_used": 0,
        }

    # 3. Offline/no-key deterministic fallback
    if not groq_api_key or not groq_api_key.strip():
        use_agent, det_reason = should_use_agent_fn(effective_query)
        return {
            "intent": "COMPLEX_ARCHITECTURE" if use_agent else "TECHNICAL_EXPLANATION",
            "route": "AUTONOMOUS_AGENT" if use_agent else "DIRECT_RAG",
            "reason": det_reason,
            "tier": "fallback",
            "model_used": None,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "tokens_used": 0,
        }

    # 4. Lightweight LLM Intent Classification with validated router model
    try:
        client = groq_factory(api_key=groq_api_key.strip())
        resolved_router, _ = resolve_available_model(client, router_model)

        history_lines = []
        if chat_history:
            for msg in chat_history[-4:]:
                role = msg.get("role", "")
                content = (msg.get("content") or "").strip()
                if role in ("user", "assistant") and content:
                    history_lines.append(f"{role}: {content[:200]}")

        user_payload = f"Question: {query}"
        if history_lines:
            user_payload = "Recent conversation:\n" + "\n".join(history_lines) + f"\n\nFollow-up question: {query}"

        resp = client.chat.completions.create(
            model=resolved_router,
            messages=[
                {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_payload},
            ],
            temperature=0.0,
            max_tokens=100,
            timeout=ROUTER_TIMEOUT_SECONDS,
        )
        raw_out = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        tokens_used = int(getattr(usage, "total_tokens", 0) or 0)

        parsed = _extract_json_object(raw_out)
        if parsed:
            intent = str(parsed.get("intent", "")).strip().upper()
            route = str(parsed.get("route", "")).strip().upper()
            reason = str(parsed.get("reason", "")).strip() or "Classified by LLM router."
            if intent in INTENT_CATEGORIES and route in ("DIRECT_RAG", "AUTONOMOUS_AGENT"):
                return {
                    "intent": intent,
                    "route": route,
                    "reason": reason,
                    "tier": "llm",
                    "model_used": resolved_router,
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                    "tokens_used": tokens_used,
                }
    except Exception:
        pass

    # 5. Safe deterministic fallback on timeout, API error, or malformed JSON
    use_agent, det_reason = should_use_agent_fn(effective_query)
    return {
        "intent": "COMPLEX_ARCHITECTURE" if use_agent else "TECHNICAL_EXPLANATION",
        "route": "AUTONOMOUS_AGENT" if use_agent else "DIRECT_RAG",
        "reason": f"Router fallback: {det_reason}",
        "tier": "fallback",
        "model_used": None,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        "tokens_used": 0,
    }


def run_agent(
    engine: Any,
    query: str,
    groq_api_key: str,
    model_name: str = PRIMARY_GROQ_MODEL,
    temperature: float = 0.0,
    max_iterations: int = 4,
    chat_history: Optional[List[Dict[str, str]]] = None,
    top_k: int = DEFAULT_FINAL_EVIDENCE_K,
    use_reranker: bool = True,
    dense_weight: float = 0.5,
    section_filter: Optional[str] = None,
    groq_factory: Callable[..., Any] = Groq,
    should_use_agent_fn: Callable[[str], Tuple[bool, str]] = should_use_agent_loop,
) -> Dict[str, Any]:
    """
    Runs an autonomous agent loop powered by Groq Function Calling:
    1. Agent plans which tools to call (curriculum search, challenge search, or full-page read).
    2. Executes tools locally against ChromaDB & PDFs.
    3. Feeds verified evidence back to the model.
    4. If a model does not support tool calling (or encounters a Groq 400), automatically
       falls back to Direct Grounded Retrieval without failing.
    """
    conv = detect_conversational_intent(query)
    if conv is not None:
        conv_trace = [{
            "step": 1,
            "tool": "conversational_fast_path",
            "args": {},
            "summary": "Conversational message detected; bypassed retrieval and citations.",
        }]
        return {
            "answer": conv["answer"],
            "trace": conv_trace,
            "sources": [],
            "execution_meta": _build_execution_meta(
                execution_strategy="Conversation",
                detected_intent="CONVERSATIONAL",
                output_format="Not applicable",
                router_model="deterministic",
                evidence_status="Not applicable",
                escalation_status="Not escalated",
                trace=conv_trace,
                sources=[],
                citation_verification="Not applicable",
                retrieval_performed=False,
            ),
        }

    if not groq_api_key or not groq_api_key.strip():
        return {
            "answer": "⚠️ **Groq API Key Required:** Please provide your Groq API key in the sidebar to activate the Agent. You can get one free at [console.groq.com/keys](https://console.groq.com/keys).",
            "trace": [],
            "sources": []
        }

    client = groq_factory(api_key=groq_api_key.strip())
    model_name, model_notice = resolve_available_model(client, model_name)
    execution_trace: List[Dict[str, Any]] = []
    if model_notice:
        execution_trace.append({"type": "note", "message": model_notice})
    collected_sources: List[Dict[str, Any]] = []

    if not is_tool_capable_model(model_name):
        return execute_direct_grounded_fallback(
            engine=engine,
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

    use_agent, route_reason = should_use_agent_fn(query)
    if not use_agent:
        return execute_direct_grounded_fallback(
            engine=engine,
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

    tools_schema = get_agent_tools_schema()
    spec = get_model_spec(model_name)
    loop_max_tokens = min(2000, spec.get("max_completion_tokens", 1500) + 500)

    messages: List[Any] = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
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
            is_tool_err = any(x in err_str for x in [
                "tool calling is not supported",
                "tool calling",
                "tools is not supported",
                "does not support tools",
                "param': 'tool calling'",
                "param': 'tools'"
            ])
            if is_tool_err:
                return execute_direct_grounded_fallback(
                    engine=engine,
                    query=query,
                    client=client,
                    model_name=model_name,
                    temperature=temperature,
                    chat_history=chat_history,
                    execution_trace=execution_trace,
                    reason=f"Groq API reported tool calling unsupported for model '{model_name}'. Auto-switched to Direct Grounded Retrieval.",
                    top_k=top_k,
                    use_reranker=use_reranker,
                    dense_weight=dense_weight,
                    section_filter=section_filter
                )

            is_model_issue = any(x in err_str for x in [
                "model_not_found", "model_terms_required", "model_decommissioned",
                "decommissioned", "deprecated", "does not exist", "not have access"
            ])
            if is_model_issue:
                alt_model = find_best_alternative_model(client, model_name)
                if alt_model and alt_model != model_name:
                    model_name = alt_model
                    execution_trace.append({
                        "step": current_iteration,
                        "tool": "model_auto_switch",
                        "args": {"new_model": alt_model},
                        "summary": f"Requested model unavailable; auto-switched to `{alt_model}`."
                    })
                    if not is_tool_capable_model(alt_model):
                        return execute_direct_grounded_fallback(
                            engine=engine,
                            query=query,
                            client=client,
                            model_name=alt_model,
                            temperature=temperature,
                            chat_history=chat_history,
                            execution_trace=execution_trace,
                            reason=f"Auto-switched to `{alt_model}` using Direct Grounded Retrieval.",
                            top_k=top_k,
                            use_reranker=use_reranker,
                            dense_weight=dense_weight,
                            section_filter=section_filter
                        )
                    continue

            if is_context_length_error(err_str):
                return execute_direct_grounded_fallback(
                    engine=engine,
                    query=query,
                    client=client,
                    model_name=model_name,
                    temperature=temperature,
                    chat_history=chat_history,
                    execution_trace=execution_trace,
                    reason=f"Agent history exceeded context window for '{model_name}'; switched to Direct Grounded Retrieval.",
                    top_k=top_k,
                    use_reranker=use_reranker,
                    dense_weight=dense_weight,
                    section_filter=section_filter
                )

            if "rate_limit_exceeded" in err_str or "413" in err_str or "too large" in err_str:
                return execute_direct_grounded_fallback(
                    engine=engine,
                    query=query,
                    client=client,
                    model_name=model_name,
                    temperature=temperature,
                    chat_history=chat_history,
                    execution_trace=execution_trace,
                    reason="Payload size exceeded Groq rate limit; switched to compact Direct Grounded Retrieval.",
                    top_k=top_k,
                    use_reranker=use_reranker,
                    dense_weight=dense_weight,
                    section_filter=section_filter
                )

            return {
                "answer": f"❌ **Groq API Error during Agent Execution:** {str(e)}",
                "trace": execution_trace,
                "sources": collected_sources
            }

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if not tool_calls:
            final_sources = deduplicate_sources(collected_sources)
            raw_ans = response_message.content or "No response generated."
            if getattr(response.choices[0], "finish_reason", None) == "length":
                raw_ans = trim_to_last_complete_sentence(raw_ans) + "\n\n> *Note: Response reached the model's output length limit and was trimmed at the last complete sentence.*"
            verified_ans, _ = verify_citations(raw_ans, final_sources)
            return {
                "answer": verified_ans,
                "trace": execution_trace,
                "sources": final_sources
            }

        messages.append(response_message)
        for tool_call in tool_calls:
            func_name = tool_call.function.name
            func_args_str = tool_call.function.arguments
            try:
                parsed_args = json.loads(func_args_str or "{}")
                func_args = parsed_args if isinstance(parsed_args, dict) else {}
            except Exception:
                func_args = {}

            tool_output_dict: Dict[str, Any] = {}
            if func_name == "search_curriculum":
                tool_output_dict = engine.tool_search_curriculum(
                    query=func_args.get("query") or query,
                    section=func_args.get("section")
                )
                step_summary = f"Searched curriculum modules for \"{func_args.get('query', query)}\""
            elif func_name == "search_challenges":
                tool_output_dict = engine.tool_search_challenges(
                    query=func_args.get("query") or query
                )
                step_summary = f"Searched hands-on challenges for \"{func_args.get('query', query)}\""
            elif func_name == "read_full_page":
                try:
                    page_num_arg = int(func_args.get("page_number") or 1)
                except (TypeError, ValueError):
                    page_num_arg = 1
                tool_output_dict = engine.tool_read_full_page(
                    document_name=func_args.get("document_name") or "",
                    page_number=page_num_arg
                )
                step_summary = f"Read full page {func_args.get('page_number', 1)} of {func_args.get('document_name', 'document')}"
            elif func_name == "lookup_pega_terms":
                tool_output_dict = engine.tool_lookup_pega_terms(
                    term=func_args.get("term") or query
                )
                step_summary = f"Looked up Pega rule/term \"{func_args.get('term', query)}\""
            else:
                tool_output_dict = {"status": "error", "message": f"Unknown tool '{func_name}'"}
                step_summary = f"Executed `{func_name}`"

            if not isinstance(tool_output_dict, dict):
                tool_output_dict = {}

            collect_tool_sources(tool_output_dict, collected_sources, include_preview=True)

            execution_trace.append({
                "step": current_iteration,
                "tool": func_name,
                "args": func_args,
                "summary": step_summary
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": func_name,
                "content": json.dumps(tool_output_dict, ensure_ascii=False)
            })

    try:
        if collected_sources:
            unique_sources = deduplicate_sources(collected_sources)
            final_answer = generate_answer(
                query=query,
                retrieved_chunks=unique_sources,
                groq_api_key=groq_api_key,
                model_name=model_name,
                temperature=temperature,
                chat_history=chat_history,
                groq_factory=groq_factory
            )
            if isinstance(final_answer, str) and final_answer.startswith("❌"):
                return execute_direct_grounded_fallback(
                    engine=engine,
                    query=query,
                    client=client,
                    model_name=model_name,
                    temperature=temperature,
                    chat_history=chat_history,
                    execution_trace=execution_trace,
                    reason=f"Synthesis fallback triggered: {final_answer}",
                    top_k=top_k,
                    use_reranker=use_reranker,
                    dense_weight=dense_weight,
                    section_filter=section_filter
                )
            return {
                "answer": final_answer,
                "trace": execution_trace,
                "sources": unique_sources
            }
        return execute_direct_grounded_fallback(
            engine=engine,
            query=query,
            client=client,
            model_name=model_name,
            temperature=temperature,
            chat_history=chat_history,
            execution_trace=execution_trace,
            reason="Research iterations completed without retrieving evidence.",
            top_k=top_k,
            use_reranker=use_reranker,
            dense_weight=dense_weight,
            section_filter=section_filter
        )
    except Exception as e:
        try:
            return execute_direct_grounded_fallback(
                engine=engine,
                query=query,
                client=client,
                model_name=model_name,
                temperature=temperature,
                chat_history=chat_history,
                execution_trace=execution_trace,
                reason=f"Final pass recovery: {type(e).__name__}",
                top_k=top_k,
                use_reranker=use_reranker,
                dense_weight=dense_weight,
                section_filter=section_filter
            )
        except Exception:
            return {
                "answer": "❌ **Generation Error:** Unable to synthesize response from retrieved sources. Please verify your Groq connection and try again.",
                "trace": execution_trace,
                "sources": collected_sources
            }


def run_agent_stream(
    engine: Any,
    query: str,
    groq_api_key: str,
    model_name: str = PRIMARY_GROQ_MODEL,
    temperature: float = 0.0,
    max_iterations: int = 4,
    chat_history: Optional[List[Dict[str, str]]] = None,
    top_k: int = DEFAULT_FINAL_EVIDENCE_K,
    use_reranker: bool = True,
    dense_weight: float = 0.5,
    section_filter: Optional[str] = None,
    groq_factory: Callable[..., Any] = Groq,
    should_use_agent_fn: Callable[[str], Tuple[bool, str]] = should_use_agent_loop,
):
    """
    Streaming-enabled Agentic & Hybrid RAG pipeline.
    Yields:
      - {"type": "status", "message": str}
      - {"type": "token", "text": str}
      - {"type": "final", "answer": str, "sources": list, "trace": list}
    """
    conv = detect_conversational_intent(query)
    if conv is not None:
        conv_trace = [{
            "step": 1,
            "tool": "conversational_fast_path",
            "args": {},
            "summary": "Conversational message detected; bypassed retrieval and citations.",
        }]
        yield {
            "type": "final",
            "answer": conv["answer"],
            "sources": [],
            "trace": conv_trace,
            "execution_meta": _build_execution_meta(
                execution_strategy="Conversation",
                detected_intent="CONVERSATIONAL",
                output_format="Not applicable",
                router_model="deterministic",
                evidence_status="Not applicable",
                escalation_status="Not escalated",
                trace=conv_trace,
                sources=[],
                citation_verification="Not applicable",
                retrieval_performed=False,
            ),
        }
        return

    if not groq_api_key or not groq_api_key.strip():
        yield {
            "type": "final",
            "answer": "⚠️ **Groq API Key Required:** Please provide your Groq API key in Configuration to activate the Assistant.",
            "sources": [],
            "trace": [],
            "model_used": model_name or PRIMARY_GROQ_MODEL
        }
        return

    client = groq_factory(api_key=groq_api_key.strip())
    model_name, model_notice = resolve_available_model(client, model_name)
    execution_trace: List[Dict[str, Any]] = []
    if model_notice:
        execution_trace.append({"type": "note", "message": model_notice})
    collected_sources: List[Dict[str, Any]] = []

    use_agent, _ = should_use_agent_fn(query)
    if (not is_tool_capable_model(model_name)) or (not use_agent):
        yield {"type": "status", "message": "Retrieving Pega curriculum excerpts (Hybrid BM25 + Vector + Reranker)..."}
        search_query = contextualize_query_with_history(query, chat_history)
        retrieved_chunks = engine.search(
            query=search_query,
            top_k=top_k,
            section_filter=section_filter,
            use_reranker=use_reranker,
            dense_weight=dense_weight
        )
        if any(w in search_query.lower() for w in ["how to", "configure", "implement", "step", "challenge", "exercise", "create", "setup"]):
            try:
                ch_res = engine.tool_search_challenges(query=search_query, top_k=2)
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
        for ev in stream_answer(
            query=query,
            retrieved_chunks=retrieved_chunks,
            groq_api_key=groq_api_key,
            model_name=model_name,
            temperature=temperature,
            chat_history=chat_history,
            groq_factory=groq_factory
        ):
            if ev["type"] == "token":
                yield ev
            elif ev["type"] == "final":
                exec_meta = _build_execution_meta(
                    execution_strategy="Direct RAG",
                    detected_intent="TECHNICAL_EXPLANATION",
                    output_format=ev.get("output_format") or detect_output_format(query),
                    router_model="deterministic",
                    evidence_status="Sufficient" if ev.get("sources") else "Insufficient",
                    escalation_status="Not escalated",
                    trace=execution_trace,
                    sources=ev.get("sources") or [],
                    citation_verification=ev.get("citation_stats") or {"verified": 0, "removed": 0, "mismatched_claims": 0},
                    retrieval_performed=True,
                )
                yield {
                    "type": "final",
                    "answer": ev["answer"],
                    "sources": ev["sources"],
                    "trace": execution_trace,
                    "citation_stats": ev.get("citation_stats"),
                    "output_format": ev.get("output_format"),
                    "execution_meta": exec_meta,
                }
        return

    tools_schema = get_agent_tools_schema()
    spec = get_model_spec(model_name)
    loop_max_tokens = min(2000, spec.get("max_completion_tokens", 1500) + 500)

    messages: List[Any] = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
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
                chunks = engine.search(
                    query=contextualize_query_with_history(query, chat_history),
                    top_k=top_k,
                    section_filter=section_filter,
                    use_reranker=use_reranker,
                    dense_weight=dense_weight
                )
                for ev in stream_answer(
                    query=query,
                    retrieved_chunks=chunks,
                    groq_api_key=groq_api_key,
                    model_name=model_name,
                    temperature=temperature,
                    chat_history=chat_history,
                    groq_factory=groq_factory
                ):
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
                raw_ans = trim_to_last_complete_sentence(raw_ans) + "\n\n> *Note: Response reached the model's output length limit and was trimmed at the last complete sentence.*"
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
                tool_output_dict = engine.tool_search_curriculum(query=q_arg, section=func_args.get("section"))
            elif func_name == "search_challenges":
                q_arg = func_args.get("query") or query
                yield {"type": "status", "message": f"Searching hands-on challenges for \"{q_arg}\"..."}
                tool_output_dict = engine.tool_search_challenges(query=q_arg)
            elif func_name == "read_full_page":
                try:
                    page_num_arg = int(func_args.get("page_number") or 1)
                except (TypeError, ValueError):
                    page_num_arg = 1
                doc_arg = func_args.get("document_name") or ""
                yield {"type": "status", "message": f"Reading page {page_num_arg} of {doc_arg or 'PDF'}..."}
                tool_output_dict = engine.tool_read_full_page(document_name=doc_arg, page_number=page_num_arg)
            elif func_name == "lookup_pega_terms":
                term_arg = func_args.get("term") or query
                yield {"type": "status", "message": f"Looking up Pega term \"{term_arg}\"..."}
                tool_output_dict = engine.tool_lookup_pega_terms(term=term_arg)
            else:
                tool_output_dict = {"status": "error", "message": f"Unknown tool '{func_name}'"}

            if not isinstance(tool_output_dict, dict):
                tool_output_dict = {}

            collect_tool_sources(tool_output_dict, collected_sources, include_preview=False)

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

    unique_sources = deduplicate_sources(collected_sources)
    if not unique_sources:
        unique_sources = engine.search(
            query=contextualize_query_with_history(query, chat_history),
            top_k=top_k,
            section_filter=section_filter,
            use_reranker=use_reranker,
            dense_weight=dense_weight
        )

    yield {"type": "status", "message": "Synthesizing verified architectural response..."}
    for ev in stream_answer(
        query=query,
        retrieved_chunks=unique_sources,
        groq_api_key=groq_api_key,
        model_name=model_name,
        temperature=temperature,
        chat_history=chat_history,
        groq_factory=groq_factory
    ):
        if ev["type"] == "token":
            yield ev
        elif ev["type"] == "final":
            exec_meta = _build_execution_meta(
                execution_strategy="Autonomous CLSA Agent",
                detected_intent="COMPLEX_ARCHITECTURE",
                output_format=ev.get("output_format") or detect_output_format(query),
                router_model="manual",
                evidence_status="Sufficient" if ev.get("sources") else "Insufficient",
                escalation_status="Not escalated",
                trace=execution_trace,
                sources=ev.get("sources") or [],
                citation_verification=ev.get("citation_stats") or {"verified": 0, "removed": 0, "mismatched_claims": 0},
                retrieval_performed=True,
            )
            yield {
                "type": "final",
                "answer": ev["answer"],
                "sources": ev["sources"],
                "trace": execution_trace,
                "citation_stats": ev.get("citation_stats"),
                "output_format": ev.get("output_format"),
                "execution_meta": exec_meta,
            }


def _resolve_adaptive_route(
    query: str,
    chat_history: Optional[List[Dict[str, str]]],
    groq_api_key: str,
    model_name: str,
    router_model: str,
    execution_mode: str,
    groq_factory: Callable[..., Any],
    classify_fn: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Resolve routing metadata for adaptive, manual direct, or manual agent mode."""
    mode_norm = (execution_mode or "adaptive").strip().lower()
    if mode_norm in ("direct", "direct_rag", "fast_direct_rag"):
        return {
            "intent": "TECHNICAL_EXPLANATION",
            "route": "DIRECT_RAG",
            "reason": "Manual Fast Direct RAG mode selected.",
            "tier": "manual",
            "model_used": None,
            "latency_ms": 0.0,
            "tokens_used": 0,
        }
    if mode_norm in ("agent", "autonomous", "autonomous_agent"):
        return {
            "intent": "COMPLEX_ARCHITECTURE",
            "route": "AUTONOMOUS_AGENT",
            "reason": "Manual Autonomous CLSA Agent mode selected.",
            "tier": "manual",
            "model_used": None,
            "latency_ms": 0.0,
            "tokens_used": 0,
        }

    classifier = classify_fn or classify_query_intent
    try:
        return classifier(
            query=query,
            chat_history=chat_history,
            groq_api_key=groq_api_key,
            answer_model_name=model_name,
            router_model=router_model,
            groq_factory=groq_factory,
        )
    except Exception as exc:
        use_agent, det_reason = should_use_agent_loop(query)
        return {
            "intent": "COMPLEX_ARCHITECTURE" if use_agent else "TECHNICAL_EXPLANATION",
            "route": "AUTONOMOUS_AGENT" if use_agent else "DIRECT_RAG",
            "reason": f"Router error ({exc}); {det_reason}",
            "tier": "fallback",
            "model_used": None,
            "latency_ms": 0.0,
            "tokens_used": 0,
        }


_COVERAGE_STOPWORDS = frozenset({
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "can", "could", "should", "would", "will", "shall", "may", "might",
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that",
    "this", "these", "those", "in", "on", "at", "to", "for", "from", "of",
    "by", "with", "about", "into", "between", "across", "during", "after",
    "before", "pega", "explain", "define", "compare", "difference", "differences",
    "work", "works", "use", "used", "using", "tell", "me", "please",
})


def _parse_chunk_score(raw_score: Any) -> float:
    """Parse numeric score from float/int or percentage string."""
    if isinstance(raw_score, (int, float)):
        return float(raw_score)
    if isinstance(raw_score, str):
        cleaned = raw_score.strip().rstrip("%").strip()
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    return 0.0


def evaluate_evidence_sufficiency(
    query: str,
    chunks: List[Dict[str, Any]],
    intent: str = "TECHNICAL_EXPLANATION",
) -> Dict[str, Any]:
    """
    Evaluate whether retrieved Direct RAG evidence is sufficient to answer reliably.
    Combines evidence quality, source diversity, query coverage, and retrieval confidence
    (never relying on source count or cross-encoder score alone).
    """
    chunks = chunks or []
    substantive = [
        c for c in chunks
        if len((c.get("document") or c.get("excerpt") or "").strip()) >= 40
    ]
    combined_text = " ".join(
        (c.get("document") or c.get("excerpt") or "").strip().lower()
        for c in substantive
    )
    total_chars = len(combined_text)

    unique_sources = len({
        str(c.get("source") or c.get("document") or "").strip()
        for c in substantive
        if str(c.get("source") or c.get("document") or "").strip()
    })
    unique_locations = len({
        (str(c.get("source") or "").strip(), str(c.get("page") or "").strip())
        for c in substantive
    })

    raw_scores = [_parse_chunk_score(c.get("score", c.get("relevance_score", 0.0))) for c in substantive]
    nonzero_scores = [s for s in raw_scores if s > 0.0]
    if nonzero_scores and max(nonzero_scores) <= 1.0:
        nonzero_scores = [s * 100.0 for s in nonzero_scores]
    top_score = round(max(nonzero_scores), 2) if nonzero_scores else 0.0
    avg_score = round(sum(nonzero_scores) / len(nonzero_scores), 2) if nonzero_scores else 0.0

    raw_tokens = re.findall(r"[a-z0-9][a-z0-9_-]{2,}", (query or "").lower())
    query_terms: List[str] = []
    for tok in raw_tokens:
        if tok not in _COVERAGE_STOPWORDS and tok not in query_terms:
            query_terms.append(tok)

    if query_terms:
        matched_terms = sum(1 for t in query_terms if t in combined_text)
        query_coverage = round(matched_terms / len(query_terms), 3)
    else:
        query_coverage = 1.0 if substantive else 0.0

    metrics = {
        "chunk_count": len(chunks),
        "substantive_chunks": len(substantive),
        "unique_sources": unique_sources,
        "unique_locations": unique_locations,
        "top_score": top_score,
        "avg_score": avg_score,
        "query_coverage": query_coverage,
        "total_chars": total_chars,
    }

    if len(substantive) == 0 or total_chars < 40:
        return {"sufficient": False, "reason": "No substantive curriculum excerpts retrieved.", "metrics": metrics}

    if query_terms:
        top_chunk_text = (substantive[0].get("document") or substantive[0].get("excerpt") or "").strip().lower()
        top_chunk_matches = sum(1 for t in query_terms if t in top_chunk_text)
        if top_chunk_matches == 0:
            return {
                "sufficient": False,
                "reason": "Top-ranked excerpt is off-topic relative to key query terms.",
                "metrics": metrics,
            }

    if top_score > 0.0 and top_score < 35.0 and query_coverage < 0.5:
        return {"sufficient": False, "reason": "Low retrieval confidence and weak query coverage.", "metrics": metrics}

    if query_terms and query_coverage < 0.35:
        return {"sufficient": False, "reason": "Retrieved excerpts do not sufficiently cover key query terms.", "metrics": metrics}

    if intent == "COMPARISON" and len(query_terms) >= 2 and query_coverage < 0.5:
        return {"sufficient": False, "reason": "Comparison query lacks balanced coverage across compared concepts.", "metrics": metrics}

    if intent in ("COMPLEX_ARCHITECTURE", "MULTI_DOCUMENT_RESEARCH") and unique_locations < 2 and query_coverage < 0.75:
        return {"sufficient": False, "reason": "Multi-faceted query requires broader source diversity.", "metrics": metrics}

    return {"sufficient": True, "reason": "Evidence quality, coverage, diversity, and confidence are sufficient.", "metrics": metrics}


def _prioritize_subject_relevant_chunks(query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Re-rank retrieved chunks lightly by core query-subject presence so unrelated
    lexical matches (e.g., Work Queue/WorkBasket for a circumstancing query) do not
    outrank or dominate genuine subject chunks.
    """
    if not chunks:
        return chunks
    subject_tokens = [
        t for t in re.findall(r"[a-z0-9][a-z0-9_-]{4,}", (query or "").lower())
        if t not in _COVERAGE_STOPWORDS and t not in {"advantages", "benefits", "using", "relationship", "explain"}
    ]
    if not subject_tokens:
        return chunks

    def _stem_hits(chunk: Dict[str, Any]) -> int:
        text = (chunk.get("document") or chunk.get("excerpt") or "").lower()
        return sum(1 for t in subject_tokens if t[:7] in text)

    matching = [c for c in chunks if _stem_hits(c) > 0]
    if not matching:
        return chunks
    non_matching = [c for c in chunks if _stem_hits(c) == 0]
    # Keep matching chunks first; retain non-matching only if fewer than 2 matching chunks exist
    if len(matching) >= 2:
        return matching
    return matching + non_matching


def _retrieve_direct_rag_chunks(
    engine: Any,
    query: str,
    chat_history: Optional[List[Dict[str, str]]],
    top_k: int,
    section_filter: Optional[str],
    use_reranker: bool,
    dense_weight: float,
) -> List[Dict[str, Any]]:
    """Retrieve Direct RAG chunks once via hybrid search + optional challenge lookup."""
    search_query = contextualize_query_with_history(query, chat_history)
    retrieved_chunks = list(
        engine.search(
            query=search_query,
            top_k=top_k,
            section_filter=section_filter,
            use_reranker=use_reranker,
            dense_weight=dense_weight,
        ) or []
    )
    retrieved_chunks = _prioritize_subject_relevant_chunks(search_query, retrieved_chunks)
    if any(w in search_query.lower() for w in ["how to", "configure", "implement", "step", "challenge", "exercise", "create", "setup"]):
        try:
            ch_res = engine.tool_search_challenges(query=search_query, top_k=2)
            for r in ch_res.get("results", []):
                if not any(c.get("document") == r.get("excerpt") for c in retrieved_chunks):
                    retrieved_chunks.append({
                        "source": r.get("document", "Challenge"),
                        "page": str(r.get("page", "1")),
                        "section": r.get("section", "Hands-on Challenges"),
                        "category": "Hands-on Challenge",
                        "score": _parse_chunk_score(r.get("relevance_score", 0.0)),
                        "document": r.get("excerpt", ""),
                    })
        except Exception:
            pass
    return retrieved_chunks


def _build_knowledge_gap_answer(
    query: str,
    ev_eval: Dict[str, Any],
    escalated: bool = False,
    model_name: str = PRIMARY_GROQ_MODEL,
) -> str:
    """Return an honest, non-hallucinated answer explaining the indexed documentation gap."""
    reason = ev_eval.get("reason", "Retrieved excerpts did not sufficiently cover the question.")
    if escalated:
        prefix = "I searched both the direct curriculum index and ran a deeper autonomous multi-tool investigation, but "
    elif not is_tool_capable_model(model_name):
        prefix = (
            f"The selected model (`{model_name}`) does not support autonomous tool-calling escalation, and "
        )
    else:
        prefix = ""
    return (
        f"**Insufficient Verified Evidence:** {prefix}"
        f"the indexed Pega documentation does not contain enough verified detail to answer "
        f"\"{query}\" reliably ({reason}).\n\n"
        "Please upload or index the relevant Pega Academy guide/PDF covering this topic, or rephrase with specific rule names."
    )


def _emit_routing_telemetry(query: str, routing: Dict[str, Any]) -> None:
    """Safely record sanitized routing observability metrics without affecting execution."""
    try:
        from agents.routing_eval import record_routing_event
        record_routing_event(query, routing)
    except Exception:
        pass


def run_adaptive(
    engine: Any,
    query: str,
    groq_api_key: str,
    model_name: str = PRIMARY_GROQ_MODEL,
    temperature: float = 0.0,
    max_iterations: int = 4,
    chat_history: Optional[List[Dict[str, str]]] = None,
    top_k: int = DEFAULT_FINAL_EVIDENCE_K,
    use_reranker: bool = True,
    dense_weight: float = 0.5,
    section_filter: Optional[str] = None,
    router_model: str = ROUTER_MODEL,
    execution_mode: str = "adaptive",
    groq_factory: Callable[..., Any] = Groq,
    classify_fn: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Stage 2 + Stage 3 Adaptive RAG orchestrator (non-streaming).
    Connects classify_query_intent() to Direct RAG and Autonomous CLSA Agent,
    with single-step evidence-based escalation when Direct RAG evidence is insufficient.
    """
    if not groq_api_key or not groq_api_key.strip():
        result = run_agent(
            engine=engine,
            query=query,
            groq_api_key=groq_api_key,
            model_name=model_name,
            temperature=temperature,
            max_iterations=max_iterations,
            chat_history=chat_history,
            top_k=top_k,
            use_reranker=use_reranker,
            dense_weight=dense_weight,
            section_filter=section_filter,
            groq_factory=groq_factory,
        )
        result["routing"] = {
            "intent": "SIMPLE_LOOKUP",
            "route": "DIRECT_RAG",
            "reason": "Missing Groq API key.",
            "tier": "fallback",
            "model_used": None,
            "latency_ms": 0.0,
            "tokens_used": 0,
            "escalated": False,
            "escalation_count": 0,
        }
        return result

    routing = _resolve_adaptive_route(
        query=query,
        chat_history=chat_history,
        groq_api_key=groq_api_key,
        model_name=model_name,
        router_model=router_model,
        execution_mode=execution_mode,
        groq_factory=groq_factory,
        classify_fn=classify_fn,
    )
    routing = {**routing, "escalated": False, "escalation_count": 0}
    mode_norm = (execution_mode or "adaptive").strip().lower()
    is_manual_override = mode_norm in ("direct", "direct_rag", "fast_direct_rag", "agent", "autonomous", "autonomous_agent")
    use_agent = routing.get("route") == "AUTONOMOUS_AGENT"
    route_reason = routing.get("reason", "Adaptive routing decision.")

    if routing.get("route") == "CONVERSATIONAL":
        conv = detect_conversational_intent(query) or {
            "answer": "Hey! Ready whenever you are—what Pega topic would you like to explore?"
        }
        conv_trace = [{
            "step": 1,
            "tool": "conversational_fast_path",
            "args": {},
            "summary": "Conversational message detected; bypassed retrieval and citations.",
        }]
        exec_meta = _build_execution_meta(
            execution_strategy="Conversation",
            detected_intent="CONVERSATIONAL",
            output_format="Not applicable",
            router_model=routing.get("model_used") or routing.get("tier") or "deterministic",
            evidence_status="Not applicable",
            escalation_status="Not escalated",
            trace=conv_trace,
            sources=[],
            citation_verification="Not applicable",
            retrieval_performed=False,
        )
        _emit_routing_telemetry(query, routing)
        return {"answer": conv["answer"], "trace": conv_trace, "sources": [], "routing": routing, "execution_meta": exec_meta}

    # Direct RAG with Stage 3 evidence evaluation (avoids duplicate retrieval)
    if (not use_agent) and (not is_manual_override):
        resolved_query = contextualize_query_with_history(query, chat_history)
        chunks = _retrieve_direct_rag_chunks(
            engine=engine,
            query=query,
            chat_history=chat_history,
            top_k=top_k,
            section_filter=section_filter,
            use_reranker=use_reranker,
            dense_weight=dense_weight,
        )
        ev_eval = evaluate_evidence_sufficiency(resolved_query, chunks, intent=routing.get("intent", "TECHNICAL_EXPLANATION"))
        routing["evidence_evaluation"] = ev_eval

        if ev_eval["sufficient"]:
            answer_text = generate_answer(
                query=query,
                retrieved_chunks=chunks,
                groq_api_key=groq_api_key,
                model_name=model_name,
                temperature=temperature,
                chat_history=chat_history,
                groq_factory=groq_factory,
            )
            sources = deduplicate_sources(chunks)
            trace = [{
                "step": 1,
                "tool": "direct_grounded_retrieval",
                "args": {"model": model_name, "sources_retrieved": len(sources), "reason": route_reason},
                "summary": f"Retrieved {len(sources)} top-ranked Pega curriculum excerpts via Hybrid Search (Dense + BM25 RRF + Reranker).",
            }]
            exec_meta = _build_execution_meta(
                execution_strategy="Direct RAG",
                detected_intent=routing.get("intent", "TECHNICAL_EXPLANATION"),
                output_format=detect_output_format(query),
                router_model=routing.get("model_used") or routing.get("tier") or "deterministic",
                evidence_status="Sufficient",
                escalation_status="Not escalated",
                trace=trace,
                sources=sources,
                citation_verification={"verified": 0, "removed": 0, "mismatched_claims": 0},
                retrieval_performed=True,
            )
            _emit_routing_telemetry(query, routing)
            return {"answer": answer_text, "trace": trace, "sources": sources, "routing": routing, "execution_meta": exec_meta}

        # Evidence insufficient: check if one escalation to Autonomous Agent is permitted
        if MAX_ROUTING_ESCALATIONS >= 1 and is_tool_capable_model(model_name):
            routing["escalated"] = True
            routing["escalation_count"] = 1
            routing["escalation_reason"] = ev_eval["reason"]
            routing["final_route"] = "AUTONOMOUS_AGENT"
            result = run_agent(
                engine=engine,
                query=query,
                groq_api_key=groq_api_key,
                model_name=model_name,
                temperature=temperature,
                max_iterations=max_iterations,
                chat_history=chat_history,
                top_k=top_k,
                use_reranker=use_reranker,
                dense_weight=dense_weight,
                section_filter=section_filter,
                groq_factory=groq_factory,
                should_use_agent_fn=lambda _q: (True, f"Escalated from Direct RAG: {ev_eval['reason']}"),
            )
            post_sources = result.get("sources") or []
            post_eval = evaluate_evidence_sufficiency(resolved_query, post_sources, intent=routing.get("intent", "TECHNICAL_EXPLANATION"))
            routing["post_escalation_evidence"] = post_eval
            if not post_sources or not post_eval["sufficient"]:
                result["answer"] = _build_knowledge_gap_answer(query, post_eval, escalated=True, model_name=model_name)
            result["routing"] = routing
            _emit_routing_telemetry(query, routing)
            return result

        # Cannot escalate (non-tool model or escalation cap 0) -> honest knowledge-gap response
        routing["escalation_blocked_reason"] = (
            f"Model '{model_name}' does not support tool calling."
            if not is_tool_capable_model(model_name)
            else "Escalation cap reached."
        )
        _emit_routing_telemetry(query, routing)
        return {
            "answer": _build_knowledge_gap_answer(query, ev_eval, escalated=False, model_name=model_name),
            "trace": [{
                "step": 1,
                "tool": "direct_grounded_retrieval",
                "args": {"model": model_name, "sources_retrieved": len(chunks), "reason": ev_eval["reason"]},
                "summary": "Direct retrieval evidence was insufficient and escalation was not permitted.",
            }],
            "sources": deduplicate_sources(chunks),
            "routing": routing,
        }

    result = run_agent(
        engine=engine,
        query=query,
        groq_api_key=groq_api_key,
        model_name=model_name,
        temperature=temperature,
        max_iterations=max_iterations,
        chat_history=chat_history,
        top_k=top_k,
        use_reranker=use_reranker,
        dense_weight=dense_weight,
        section_filter=section_filter,
        groq_factory=groq_factory,
        should_use_agent_fn=lambda _q: (use_agent, route_reason),
    )
    result["routing"] = routing
    _emit_routing_telemetry(query, routing)
    return result


def run_adaptive_stream(
    engine: Any,
    query: str,
    groq_api_key: str,
    model_name: str = PRIMARY_GROQ_MODEL,
    temperature: float = 0.0,
    max_iterations: int = 4,
    chat_history: Optional[List[Dict[str, str]]] = None,
    top_k: int = DEFAULT_FINAL_EVIDENCE_K,
    use_reranker: bool = True,
    dense_weight: float = 0.5,
    section_filter: Optional[str] = None,
    router_model: str = ROUTER_MODEL,
    execution_mode: str = "adaptive",
    groq_factory: Callable[..., Any] = Groq,
    classify_fn: Optional[Callable[..., Dict[str, Any]]] = None,
):
    """
    Stage 2 + Stage 3 Adaptive RAG orchestrator (streaming).
    Connects classify_query_intent() to Direct RAG and Autonomous CLSA Agent,
    with single-step evidence-based escalation and honest knowledge-gap handling.
    """
    if not groq_api_key or not groq_api_key.strip():
        fallback_routing = {
            "intent": "SIMPLE_LOOKUP",
            "route": "DIRECT_RAG",
            "reason": "Missing Groq API key.",
            "tier": "fallback",
            "model_used": None,
            "latency_ms": 0.0,
            "tokens_used": 0,
            "escalated": False,
            "escalation_count": 0,
        }
        for ev in run_agent_stream(
            engine=engine,
            query=query,
            groq_api_key=groq_api_key,
            model_name=model_name,
            temperature=temperature,
            max_iterations=max_iterations,
            chat_history=chat_history,
            top_k=top_k,
            use_reranker=use_reranker,
            dense_weight=dense_weight,
            section_filter=section_filter,
            groq_factory=groq_factory,
        ):
            if ev.get("type") == "final":
                ev = {**ev, "routing": fallback_routing}
            yield ev
        return

    routing = _resolve_adaptive_route(
        query=query,
        chat_history=chat_history,
        groq_api_key=groq_api_key,
        model_name=model_name,
        router_model=router_model,
        execution_mode=execution_mode,
        groq_factory=groq_factory,
        classify_fn=classify_fn,
    )
    routing = {**routing, "escalated": False, "escalation_count": 0}
    mode_norm = (execution_mode or "adaptive").strip().lower()
    is_manual_override = mode_norm in ("direct", "direct_rag", "fast_direct_rag", "agent", "autonomous", "autonomous_agent")
    use_agent = routing.get("route") == "AUTONOMOUS_AGENT"
    route_reason = routing.get("reason", "Adaptive routing decision.")

    if routing.get("route") == "CONVERSATIONAL":
        conv = detect_conversational_intent(query) or {
            "answer": "Hey! Ready whenever you are—what Pega topic would you like to explore?"
        }
        conv_trace = [{
            "step": 1,
            "tool": "conversational_fast_path",
            "args": {},
            "summary": "Conversational message detected; bypassed retrieval and citations.",
        }]
        exec_meta = _build_execution_meta(
            execution_strategy="Conversation",
            detected_intent="CONVERSATIONAL",
            output_format="Not applicable",
            router_model=routing.get("model_used") or routing.get("tier") or "deterministic",
            evidence_status="Not applicable",
            escalation_status="Not escalated",
            trace=conv_trace,
            sources=[],
            citation_verification="Not applicable",
            retrieval_performed=False,
        )
        _emit_routing_telemetry(query, routing)
        yield {
            "type": "final",
            "answer": conv["answer"],
            "sources": [],
            "trace": conv_trace,
            "routing": routing,
            "execution_meta": exec_meta,
        }
        return

    yield {
        "type": "status",
        "message": f"Adaptive route: {routing.get('route')} ({routing.get('intent')}, {routing.get('tier')})",
    }

    if (not use_agent) and (not is_manual_override):
        yield {"type": "status", "message": "Retrieving Pega curriculum excerpts (Hybrid BM25 + Vector + Reranker)..."}
        resolved_query = contextualize_query_with_history(query, chat_history)
        chunks = _retrieve_direct_rag_chunks(
            engine=engine,
            query=query,
            chat_history=chat_history,
            top_k=top_k,
            section_filter=section_filter,
            use_reranker=use_reranker,
            dense_weight=dense_weight,
        )
        ev_eval = evaluate_evidence_sufficiency(resolved_query, chunks, intent=routing.get("intent", "TECHNICAL_EXPLANATION"))
        routing["evidence_evaluation"] = ev_eval

        if ev_eval["sufficient"]:
            yield {"type": "status", "message": "Generating grounded response..."}
            trace = [{
                "step": 1,
                "tool": "direct_grounded_retrieval",
                "args": {"model": model_name, "sources_retrieved": len(deduplicate_sources(chunks)), "reason": route_reason},
                "summary": "Retrieved Pega curriculum excerpts via Hybrid Search (Dense + BM25 RRF + Reranker).",
            }]
            for ev in stream_answer(
                query=query,
                retrieved_chunks=chunks,
                groq_api_key=groq_api_key,
                model_name=model_name,
                temperature=temperature,
                chat_history=chat_history,
                groq_factory=groq_factory,
            ):
                if ev.get("type") == "token":
                    yield ev
                elif ev.get("type") == "final":
                    exec_meta = _build_execution_meta(
                        execution_strategy="Direct RAG",
                        detected_intent=routing.get("intent", "TECHNICAL_EXPLANATION"),
                        output_format=ev.get("output_format") or detect_output_format(query),
                        router_model=routing.get("model_used") or routing.get("tier") or "deterministic",
                        evidence_status="Sufficient",
                        escalation_status="Not escalated",
                        trace=trace,
                        sources=ev.get("sources") or [],
                        citation_verification=ev.get("citation_stats") or {"verified": 0, "removed": 0, "mismatched_claims": 0},
                        retrieval_performed=True,
                    )
                    yield {
                        "type": "final",
                        "answer": ev["answer"],
                        "sources": ev["sources"],
                        "trace": trace,
                        "routing": routing,
                        "execution_meta": exec_meta,
                    }
            return

        if MAX_ROUTING_ESCALATIONS >= 1 and is_tool_capable_model(model_name):
            routing["escalated"] = True
            routing["escalation_count"] = 1
            routing["escalation_reason"] = ev_eval["reason"]
            routing["final_route"] = "AUTONOMOUS_AGENT"
            yield {
                "type": "status",
                "message": f"Initial evidence insufficient ({ev_eval['reason']}); escalating to Autonomous CLSA Agent...",
            }
            for ev in run_agent_stream(
                engine=engine,
                query=query,
                groq_api_key=groq_api_key,
                model_name=model_name,
                temperature=temperature,
                max_iterations=max_iterations,
                chat_history=chat_history,
                top_k=top_k,
                use_reranker=use_reranker,
                dense_weight=dense_weight,
                section_filter=section_filter,
                groq_factory=groq_factory,
                should_use_agent_fn=lambda _q: (True, f"Escalated from Direct RAG: {ev_eval['reason']}"),
            ):
                if ev.get("type") == "final":
                    post_sources = ev.get("sources") or []
                    post_eval = evaluate_evidence_sufficiency(resolved_query, post_sources, intent=routing.get("intent", "TECHNICAL_EXPLANATION"))
                    routing["post_escalation_evidence"] = post_eval
                    final_answer = ev.get("answer", "")
                    if not post_sources or not post_eval["sufficient"]:
                        final_answer = _build_knowledge_gap_answer(query, post_eval, escalated=True, model_name=model_name)
                    exec_meta = _build_execution_meta(
                        execution_strategy="Autonomous CLSA Agent",
                        detected_intent=routing.get("intent", "TECHNICAL_EXPLANATION"),
                        output_format=ev.get("output_format") or detect_output_format(query),
                        router_model=routing.get("model_used") or routing.get("tier") or "deterministic",
                        evidence_status="Sufficient" if post_eval.get("sufficient") else "Insufficient",
                        escalation_status="Escalated (1/1)",
                        trace=ev.get("trace") or [],
                        sources=post_sources,
                        citation_verification=ev.get("citation_stats") or {"verified": 0, "removed": 0, "mismatched_claims": 0},
                        retrieval_performed=True,
                    )
                    _emit_routing_telemetry(query, routing)
                    yield {**ev, "answer": final_answer, "routing": routing, "execution_meta": exec_meta}
                else:
                    yield ev
            return

        routing["escalation_blocked_reason"] = (
            f"Model '{model_name}' does not support tool calling."
            if not is_tool_capable_model(model_name)
            else "Escalation cap reached."
        )
        _emit_routing_telemetry(query, routing)
        yield {
            "type": "final",
            "answer": _build_knowledge_gap_answer(query, ev_eval, escalated=False, model_name=model_name),
            "sources": deduplicate_sources(chunks),
            "trace": [],
            "routing": routing,
        }
        return

    for ev in run_agent_stream(
        engine=engine,
        query=query,
        groq_api_key=groq_api_key,
        model_name=model_name,
        temperature=temperature,
        max_iterations=max_iterations,
        chat_history=chat_history,
        top_k=top_k,
        use_reranker=use_reranker,
        dense_weight=dense_weight,
        section_filter=section_filter,
        groq_factory=groq_factory,
        should_use_agent_fn=lambda _q: (use_agent, route_reason),
    ):
        if ev.get("type") == "final":
            _emit_routing_telemetry(query, routing)
            ev = {**ev, "routing": routing}
        yield ev
