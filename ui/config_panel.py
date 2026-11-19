"""Extreme-right Configuration drawer panel for PegaRagAgent."""

import streamlit as st

from rag_engine import get_model_spec, is_tool_capable_model
from ui.styles import LUCIDE_ICONS


def fetch_available_models(api_key: str) -> list:
    models = []
    if api_key and api_key.strip():
        try:
            from groq import Groq
            client = Groq(api_key=api_key.strip())
            live_models = client.models.list()
            for m in live_models.data:
                m_low = m.id.lower()
                if any(x in m_low for x in ["whisper", "orpheus", "tts", "audio", "guard", "safeguard", "canopylabs"]):
                    continue
                models.append(m.id)
            models.sort(key=lambda m_id: (get_model_spec(m_id).get("priority", 4), m_id.lower()))
        except Exception:
            pass
    if not models:
        models = [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b"
        ]
    return models


def render_config_panel() -> None:
    """Renders the Configuration drawer Bento card."""
    with st.container(key="bento_config_drawer"):
        st.markdown(
            f'<div class="bento-card-header">'
            f'<span class="bento-title-group">{LUCIDE_ICONS["sliders"]} Configuration</span>'
            f'<span class="live-pill"><span class="live-dot"></span>Live</span></div>',
            unsafe_allow_html=True
        )

        st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            key="cfg_api_key"
        )

        available_models = fetch_available_models(st.session_state.cfg_api_key)
        if st.session_state.cfg_model not in available_models:
            st.session_state.cfg_model = available_models[0]

        st.selectbox(
            "LLM Reasoning Model",
            options=available_models,
            format_func=lambda m: f"{m} ({'Agent' if is_tool_capable_model(m) else 'Direct'})",
            key="cfg_model"
        )
        active_spec = get_model_spec(st.session_state.cfg_model)
        ctx_k = active_spec.get("context_window", 4096) // 1024
        mode_tag = "Tool-Calling Agent" if is_tool_capable_model(st.session_state.cfg_model) else "Direct Grounded"
        st.caption(f"Active Model: `{st.session_state.cfg_model}` · {ctx_k}K ctx · {mode_tag}")

        st.radio(
            "Intelligence Mode",
            options=[
                "Autonomous CLSA Agent (Recommended)",
                "Fast Direct RAG"
            ],
            key="cfg_mode"
        )

        st.slider(
            "Hybrid Weight (0=BM25 · 1=Vector)",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key="cfg_dense_weight"
        )

        st.toggle(
            "Cross-Encoder Reranking",
            key="cfg_use_reranker"
        )

        st.slider(
            "Top-K Evidence Chunks",
            min_value=2,
            max_value=10,
            step=1,
            key="cfg_top_k"
        )

        st.slider(
            "Temperature",
            min_value=0.0,
            max_value=0.8,
            step=0.05,
            key="cfg_temperature"
        )

        st.selectbox(
            "Curriculum Section Filter",
            options=[
                "All Sections",
                "01_LSA_Mindset",
                "02_Planning_and_Design",
                "03_Constellation_Adoption",
                "04_Infrastructure_and_Architecture",
                "05_Implementation_and_Deployment"
            ],
            key="cfg_section_filter"
        )
