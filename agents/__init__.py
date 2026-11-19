"""Agents, tools, and answer generation package for PegaRagAgent."""

from importlib import import_module

_EXPORT_MAP = {
    "get_agent_tools_schema": "agents.tools",
    "execute_search_curriculum": "agents.tools",
    "execute_search_challenges": "agents.tools",
    "execute_read_full_page": "agents.tools",
    "execute_lookup_pega_terms": "agents.tools",
    "collect_tool_sources": "agents.tools",
    "GROUNDED_SYSTEM_PROMPT": "agents.answer_generator",
    "is_context_length_error": "agents.answer_generator",
    "find_best_alternative_model": "agents.answer_generator",
    "resolve_available_model": "agents.answer_generator",
    "trim_to_last_complete_sentence": "agents.answer_generator",
    "create_completion_with_completeness_guard": "agents.answer_generator",
    "stream_completion_with_completeness_guard": "agents.answer_generator",
    "execute_direct_grounded_fallback": "agents.answer_generator",
    "generate_answer": "agents.answer_generator",
    "stream_answer": "agents.answer_generator",
    "AGENT_SYSTEM_PROMPT": "agents.research_agent",
    "should_use_agent_loop": "agents.research_agent",
    "run_agent": "agents.research_agent",
    "run_agent_stream": "agents.research_agent",
}

__all__ = list(_EXPORT_MAP.keys())


def __getattr__(name: str):
    if name in _EXPORT_MAP:
        mod = import_module(_EXPORT_MAP[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'agents' has no attribute {name!r}")
