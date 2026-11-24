"""
Stage 4: Adaptive RAG Evaluation and Observability for PegaRagAgent.

Provides lightweight, zero-extra-dependency routing telemetry (with secret redaction),
a representative Pega evaluation dataset across 9 categories, and a Markdown report generator.
"""

import re
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Optional

from agents.research_agent import (
    classify_query_intent,
    evaluate_evidence_sufficiency,
)
from config.settings import PRIMARY_GROQ_MODEL, ROUTER_MODEL

_SECRET_PATTERNS = (
    re.compile(r"gsk_[A-Za-z0-9_-]{8,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]+"),
    re.compile(r"(?i)(api[_-]?key|cookie|session[_-]?id|token)\s*[:=]\s*[^\s,;]+"),
)


def _redact_sensitive_text(text: str, max_len: int = 120) -> str:
    """Redact API keys, tokens, cookies, or session secrets before telemetry storage."""
    cleaned = (text or "").strip()
    for pat in _SECRET_PATTERNS:
        cleaned = pat.sub("[REDACTED]", cleaned)
    if len(cleaned) > max_len:
        return cleaned[: max_len - 3] + "..."
    return cleaned


def sanitize_routing_record(query: str, routing: Dict[str, Any]) -> Dict[str, Any]:
    """Build a privacy-safe observability record from a routing decision."""
    ev_eval = routing.get("evidence_evaluation") or {}
    ev_sufficient = ev_eval.get("sufficient") if isinstance(ev_eval, dict) else None
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query_preview": _redact_sensitive_text(query, max_len=120),
        "intent": str(routing.get("intent") or "UNKNOWN"),
        "selected_route": str(routing.get("route") or "DIRECT_RAG"),
        "final_route": str(routing.get("final_route") or routing.get("route") or "DIRECT_RAG"),
        "tier": str(routing.get("tier") or "unknown"),
        "model_used": routing.get("model_used"),
        "router_latency_ms": float(routing.get("latency_ms") or 0.0),
        "tokens_used": int(routing.get("tokens_used") or 0),
        "escalated": bool(routing.get("escalated", False)),
        "escalation_count": int(routing.get("escalation_count") or 0),
        "evidence_sufficient": ev_sufficient,
    }


class RoutingMetricsTracker:
    """Bounded in-memory observability tracker for adaptive routing decisions."""

    def __init__(self, max_records: int = 500):
        self._records: Deque[Dict[str, Any]] = deque(maxlen=max(1, max_records))

    def record(self, query: str, routing: Dict[str, Any]) -> Dict[str, Any]:
        entry = sanitize_routing_record(query, routing)
        self._records.append(entry)
        return entry

    def records(self) -> List[Dict[str, Any]]:
        return list(self._records)

    def clear(self) -> None:
        self._records.clear()

    def summarize(self) -> Dict[str, Any]:
        items = list(self._records)
        total = len(items)
        if total == 0:
            return {
                "total_queries": 0,
                "direct_rag_count": 0,
                "autonomous_agent_count": 0,
                "escalation_count": 0,
                "escalation_rate": 0.0,
                "avg_router_latency_ms": 0.0,
                "total_tokens_used": 0,
                "evidence_insufficient_count": 0,
                "intent_counts": {},
                "tier_counts": {},
            }

        direct_cnt = sum(1 for r in items if r["final_route"] == "DIRECT_RAG")
        agent_cnt = sum(1 for r in items if r["final_route"] == "AUTONOMOUS_AGENT")
        esc_cnt = sum(1 for r in items if r["escalated"])
        insuff_cnt = sum(1 for r in items if r["evidence_sufficient"] is False)
        avg_lat = round(sum(r["router_latency_ms"] for r in items) / total, 2)
        tot_tok = sum(r["tokens_used"] for r in items)

        intent_counts: Dict[str, int] = {}
        tier_counts: Dict[str, int] = {}
        for r in items:
            intent_counts[r["intent"]] = intent_counts.get(r["intent"], 0) + 1
            tier_counts[r["tier"]] = tier_counts.get(r["tier"], 0) + 1

        return {
            "total_queries": total,
            "direct_rag_count": direct_cnt,
            "autonomous_agent_count": agent_cnt,
            "escalation_count": esc_cnt,
            "escalation_rate": round(esc_cnt / total, 3),
            "avg_router_latency_ms": avg_lat,
            "total_tokens_used": tot_tok,
            "evidence_insufficient_count": insuff_cnt,
            "intent_counts": intent_counts,
            "tier_counts": tier_counts,
        }


ROUTING_TRACKER = RoutingMetricsTracker(max_records=500)


def record_routing_event(query: str, routing: Dict[str, Any]) -> Dict[str, Any]:
    """Record a sanitized routing telemetry event in the global tracker."""
    return ROUTING_TRACKER.record(query, routing)


def get_routing_observability_summary() -> Dict[str, Any]:
    """Return aggregated observability metrics across recorded adaptive routing calls."""
    return ROUTING_TRACKER.summarize()


PEGA_ROUTING_EVAL_DATASET: List[Dict[str, Any]] = [
    # Definitions
    {"query": "What is Lock and Roll in Pega?", "category": "definitions", "expected_route": "DIRECT_RAG"},
    {"query": "Define PyWorkPage on the Clipboard", "category": "definitions", "expected_route": "DIRECT_RAG"},
    {"query": "What is Constellation in Pega?", "category": "definitions", "expected_route": "DIRECT_RAG"},
    # Technical rules
    {"query": "Explain Rule Resolution in Pega", "category": "technical_rules", "expected_route": "DIRECT_RAG"},
    {"query": "How does Circumstancing work in Pega?", "category": "technical_rules", "expected_route": "DIRECT_RAG"},
    {"query": "What is a Data Transform in Pega?", "category": "technical_rules", "expected_route": "DIRECT_RAG"},
    # Comparisons
    {"query": "Compare Queue Processor vs Job Scheduler", "category": "comparisons", "expected_route": "DIRECT_RAG"},
    {"query": "What is the difference between RBAC and ABAC in Pega?", "category": "comparisons", "expected_route": "DIRECT_RAG"},
    {"query": "Compare Data Page vs Clipboard page", "category": "comparisons", "expected_route": "DIRECT_RAG"},
    # Data pages
    {"query": "What is a Data Page in Pega?", "category": "data_pages", "expected_route": "DIRECT_RAG"},
    {
        "query": "Explain how Data Pages and Clipboard pages store customer data and refresh strategies during a normal case lifecycle",
        "category": "data_pages",
        "expected_route": "DIRECT_RAG",
    },
    # Case management
    {"query": "Explain Case Management stages and processes", "category": "case_management", "expected_route": "DIRECT_RAG"},
    {"query": "What is an SLA goal and deadline in Pega?", "category": "case_management", "expected_route": "DIRECT_RAG"},
    {
        "query": "Explain how Case Management, SLA, and Queue Processor interact across multiple rules",
        "category": "case_management",
        "expected_route": "AUTONOMOUS_AGENT",
    },
    # Integrations
    {"query": "Explain Connect-REST and auth profile in Pega", "category": "integrations", "expected_route": "DIRECT_RAG"},
    {
        "query": "Walkthrough step-by-step how to configure Connect-REST with OAuth 2.0 and JSON transform",
        "category": "integrations",
        "expected_route": "AUTONOMOUS_AGENT",
    },
    # Architecture
    {
        "query": "Architect a multi-application enterprise design for security ABAC and DX API integration",
        "category": "architecture",
        "expected_route": "AUTONOMOUS_AGENT",
    },
    {
        "query": "Analyze the downstream impact of changing class inheritance in Rule Resolution",
        "category": "architecture",
        "expected_route": "AUTONOMOUS_AGENT",
    },
    # Troubleshooting
    {
        "query": "Troubleshoot why my Queue Processor is stuck and failing with Data Page errors",
        "category": "troubleshooting",
        "expected_route": "AUTONOMOUS_AGENT",
    },
    {
        "query": "What is the root cause of stale Clipboard data after an SLA escalation?",
        "category": "troubleshooting",
        "expected_route": "AUTONOMOUS_AGENT",
    },
    {
        "query": "Diagnose why our savable Data Page throws a case locking exception on commit",
        "category": "troubleshooting",
        "expected_route": "AUTONOMOUS_AGENT",
    },
    # Hands-on challenges
    {
        "query": "Complete the hands-on challenge for Circumstancing and Job Scheduler",
        "category": "hands_on_challenges",
        "expected_route": "AUTONOMOUS_AGENT",
    },
    {
        "query": "Walkthrough step-by-step how to build a Constellation case type with Connect-REST",
        "category": "hands_on_challenges",
        "expected_route": "AUTONOMOUS_AGENT",
    },
    {
        "query": "Read full page 14 of the CLSA guide on Rule Resolution",
        "category": "hands_on_challenges",
        "expected_route": "AUTONOMOUS_AGENT",
    },
]


def evaluate_routing_suite(
    dataset: Optional[List[Dict[str, Any]]] = None,
    groq_api_key: str = "",
    answer_model_name: str = PRIMARY_GROQ_MODEL,
    router_model: str = ROUTER_MODEL,
    classify_fn: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Run offline or mocked evaluation across representative Pega queries and compute
    routing accuracy, unnecessary Agent calls, missed escalations, latency, and token metrics.
    """
    cases = dataset if dataset is not None else PEGA_ROUTING_EVAL_DATASET
    classifier = classify_fn or classify_query_intent

    results: List[Dict[str, Any]] = []
    category_stats: Dict[str, Dict[str, Any]] = {}
    unnecessary_agent_calls = 0
    missed_escalations = 0
    correct_routes = 0
    total_tokens = 0
    latencies: List[float] = []

    for item in cases:
        query = item["query"]
        category = item.get("category", "general")
        expected = item["expected_route"]
        t0 = time.perf_counter()
        routing = classifier(
            query=query,
            chat_history=item.get("chat_history"),
            groq_api_key=groq_api_key,
            answer_model_name=answer_model_name,
            router_model=router_model,
        )
        elapsed_ms = float(routing.get("latency_ms") or round((time.perf_counter() - t0) * 1000, 2))
        latencies.append(elapsed_ms)
        total_tokens += int(routing.get("tokens_used") or 0)

        predicted = routing.get("route", "DIRECT_RAG")
        escalated = False
        if predicted == "DIRECT_RAG" and "mock_chunks" in item:
            ev_eval = evaluate_evidence_sufficiency(
                query, item["mock_chunks"], intent=routing.get("intent", "TECHNICAL_EXPLANATION")
            )
            if not ev_eval["sufficient"]:
                escalated = True
                predicted = "AUTONOMOUS_AGENT"

        is_correct = predicted == expected
        if is_correct:
            correct_routes += 1
        elif expected == "DIRECT_RAG" and predicted == "AUTONOMOUS_AGENT":
            unnecessary_agent_calls += 1
        elif expected == "AUTONOMOUS_AGENT" and predicted == "DIRECT_RAG":
            missed_escalations += 1

        cat = category_stats.setdefault(category, {"total": 0, "correct": 0, "accuracy": 0.0})
        cat["total"] += 1
        if is_correct:
            cat["correct"] += 1

        results.append({
            "query": _redact_sensitive_text(query, max_len=100),
            "category": category,
            "expected_route": expected,
            "predicted_route": predicted,
            "intent": routing.get("intent"),
            "tier": routing.get("tier"),
            "escalated": escalated,
            "correct": is_correct,
            "latency_ms": elapsed_ms,
        })

    for cat in category_stats.values():
        cat["accuracy"] = round((cat["correct"] / cat["total"]) * 100.0, 1) if cat["total"] else 0.0

    total = len(cases)
    accuracy = round((correct_routes / total) * 100.0, 1) if total else 0.0
    avg_latency = round(sum(latencies) / total, 2) if total else 0.0
    mismatches = [r for r in results if not r["correct"]]

    return {
        "total_cases": total,
        "correct_routes": correct_routes,
        "routing_accuracy": accuracy,
        "unnecessary_agent_calls": unnecessary_agent_calls,
        "missed_escalations": missed_escalations,
        "avg_router_latency_ms": avg_latency,
        "total_tokens_used": total_tokens,
        "category_breakdown": category_stats,
        "mismatches": mismatches,
        "results": results,
    }


def format_routing_eval_report(summary: Dict[str, Any]) -> str:
    """Format a Markdown evaluation report from evaluate_routing_suite() output."""
    lines = [
        "# Adaptive RAG Routing Evaluation Report",
        "",
        f"- **Total Test Cases:** {summary.get('total_cases', 0)}",
        f"- **Routing Accuracy:** {summary.get('routing_accuracy', 0.0)}% ({summary.get('correct_routes', 0)}/{summary.get('total_cases', 0)})",
        f"- **Unnecessary Agent Calls:** {summary.get('unnecessary_agent_calls', 0)}",
        f"- **Missed Escalations:** {summary.get('missed_escalations', 0)}",
        f"- **Avg Router Latency:** {summary.get('avg_router_latency_ms', 0.0)} ms",
        f"- **Total Router Tokens:** {summary.get('total_tokens_used', 0)}",
        "",
        "## Category Breakdown",
        "| Category | Correct | Total | Accuracy |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for cat, stats in sorted((summary.get("category_breakdown") or {}).items()):
        lines.append(f"| {cat} | {stats['correct']} | {stats['total']} | {stats['accuracy']}% |")

    mismatches = summary.get("mismatches") or []
    if mismatches:
        lines.extend(["", "## Mismatches"])
        for m in mismatches:
            lines.append(
                f"- [{m['category']}] `{m['query']}` -> expected `{m['expected_route']}`, got `{m['predicted_route']}`"
            )
    return "\n".join(lines)
