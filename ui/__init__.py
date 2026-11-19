"""Streamlit UI package for PegaRagAgent."""

from importlib import import_module

_EXPORT_MAP = {
    "LUCIDE_ICONS": "ui.styles",
    "inject_bento_styles": "ui.styles",
    "init_session_state": "ui.state",
    "queue_user_query": "ui.state",
    "format_citation_badges_html": "ui.chat_panel",
    "render_compact_evidence": "ui.chat_panel",
    "stream_assistant_turn": "ui.chat_panel",
    "render_chat_panel": "ui.chat_panel",
    "render_upload_panel": "ui.upload_panel",
    "render_explorer_panel": "ui.explorer_panel",
    "fetch_available_models": "ui.config_panel",
    "render_config_panel": "ui.config_panel",
}

__all__ = list(_EXPORT_MAP.keys())


def __getattr__(name: str):
    if name in _EXPORT_MAP:
        mod = import_module(_EXPORT_MAP[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'ui' has no attribute {name!r}")
