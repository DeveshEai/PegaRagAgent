"""Session-state initialization and shared chat-queue helpers for PegaRagAgent."""

import os
from datetime import datetime
import streamlit as st


def init_session_state() -> None:
    """Initializes all Streamlit session_state defaults if not yet present."""
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False

    if "show_config" not in st.session_state:
        st.session_state.show_config = True

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "scroll_to_bottom" not in st.session_state:
        st.session_state.scroll_to_bottom = False

    if "pending_response" not in st.session_state:
        st.session_state.pending_response = False

    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None

    if "pending_delete_doc" not in st.session_state:
        st.session_state.pending_delete_doc = None

    if "pending_delete_all" not in st.session_state:
        st.session_state.pending_delete_all = False

    if "sync_status" not in st.session_state:
        st.session_state.sync_status = None

    if "cfg_api_key" not in st.session_state:
        st.session_state.cfg_api_key = os.environ.get("GROQ_API_KEY", "")

    if "cfg_model" not in st.session_state:
        st.session_state.cfg_model = "openai/gpt-oss-120b"

    if "cfg_mode" not in st.session_state:
        st.session_state.cfg_mode = "Autonomous CLSA Agent (Recommended)"

    if "cfg_dense_weight" not in st.session_state:
        st.session_state.cfg_dense_weight = 0.50

    if "cfg_use_reranker" not in st.session_state:
        st.session_state.cfg_use_reranker = True

    if "cfg_top_k" not in st.session_state:
        st.session_state.cfg_top_k = 4

    if "cfg_temperature" not in st.session_state:
        st.session_state.cfg_temperature = 0.0

    if "cfg_section_filter" not in st.session_state:
        st.session_state.cfg_section_filter = "All Sections"


def queue_user_query(query_text: str) -> None:
    """Immediately appends the user's question and schedules live streaming generation."""
    now_ts = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({
        "role": "user",
        "content": query_text,
        "sources": [],
        "timestamp": now_ts
    })
    st.session_state.pending_response = True
    st.session_state.pending_query = query_text
    st.session_state.scroll_to_bottom = True
