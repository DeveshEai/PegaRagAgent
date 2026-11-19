"""
Bento Modular Chat UI for Pega CLSA Knowledge Assistant
========================================================
Enterprise Pega Architecture RAG — Hybrid BM25 + Vector Search,
Cross-Encoder Reranking & Verified Citations.
"""

import logging
import os
import warnings

# Suppress Streamlit module watcher and Hugging Face warnings
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

try:
    import streamlit.watcher.local_sources_watcher as _lsw
    _orig_get_module_paths = _lsw.get_module_paths

    def _safe_get_module_paths(module):
        mod_name = getattr(module, "__name__", "")
        if mod_name.startswith("transformers") or mod_name.startswith("torch"):
            return set()
        return _orig_get_module_paths(module)

    _lsw.get_module_paths = _safe_get_module_paths
    _lsw._LOGGER.setLevel(logging.ERROR)
except Exception:
    pass

try:
    import transformers.utils.logging as _t_log
    _t_log.set_verbosity_error()
except Exception:
    pass

import streamlit as st

from rag_engine import PegaRAGEngine
from ui import (
    init_session_state,
    inject_bento_styles,
    render_chat_panel,
    render_config_panel,
    render_explorer_panel,
    render_upload_panel,
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Pega CLSA Knowledge Assistant",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# Cached Engine Initialization & Session State
# ---------------------------------------------------------

@st.cache_resource(show_spinner="Initializing Pega RAG Engine & Hybrid Index...")
def get_engine():
    return PegaRAGEngine()

engine = get_engine()
init_session_state()
inject_bento_styles(st.session_state.dark_mode)

# ---------------------------------------------------------
# 1. Centered Top Title & Top-Right Controls
# ---------------------------------------------------------

hdr_spacer, hdr_center, hdr_actions = st.columns([1.5, 5, 1.8])
with hdr_center:
    st.markdown(
        '<div class="bento-header-title">Pega CLSA Knowledge Assistant</div>',
        unsafe_allow_html=True
    )

with hdr_actions:
    b1, b2 = st.columns(2)
    with b1:
        theme_label = "Light Mode" if st.session_state.dark_mode else "Dark Mode"
        if st.button(theme_label, key="btn_theme_toggle", use_container_width=True):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
    with b2:
        cfg_label = "Close Config" if st.session_state.show_config else "Configuration"
        if st.button(cfg_label, key="btn_config_toggle", use_container_width=True):
            st.session_state.show_config = not st.session_state.show_config
            st.rerun()

# ---------------------------------------------------------
# 2. Bento Layout (+ Extreme-Right Configuration Column)
# ---------------------------------------------------------

if st.session_state.show_config:
    col_chat, col_right, col_drawer = st.columns([1.6, 1, 0.95], gap="medium")
else:
    col_chat, col_right = st.columns([1.75, 1], gap="medium")
    col_drawer = None

with col_chat:
    render_chat_panel(engine)

with col_right:
    render_upload_panel(engine)
    render_explorer_panel(engine)

if col_drawer is not None:
    with col_drawer:
        render_config_panel()
