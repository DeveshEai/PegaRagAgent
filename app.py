"""
Bento Modular Chat UI for Pega CLSA Knowledge Assistant
========================================================
Enterprise Pega Architecture RAG — Hybrid BM25 + Vector Search,
Cross-Encoder Reranking & Verified Citations.
"""

import os
import html
import time
import logging
import warnings
from datetime import datetime
from pathlib import Path

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
import streamlit.components.v1 as components
from rag_engine import (
    PegaRAGEngine,
    WORKSPACE_DIR,
    DOCS_DIR,
    compute_file_hash,
    is_tool_capable_model,
    get_model_spec,
    deduplicate_sources,
    clean_chunk_content,
    contextualize_query_with_history
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
# Crisp Inline Lucide SVG Icons
# ---------------------------------------------------------

LUCIDE_ICONS = {
    "message": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>',
    "upload": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>',
    "folder": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>',
    "sliders": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>',
    "file": '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>',
}

# ---------------------------------------------------------
# Cached Engine Initialization
# ---------------------------------------------------------

@st.cache_resource(show_spinner="Initializing Pega RAG Engine & Hybrid Index...")
def get_engine():
    return PegaRAGEngine()

engine = get_engine()

# ---------------------------------------------------------
# Session State Initialization (Live Reactive Bindings)
# ---------------------------------------------------------

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

# ---------------------------------------------------------
# Theme, Fonts & 2-Column Bento Grid CSS (Zero Glassmorphism)
# ---------------------------------------------------------

is_dark = st.session_state.dark_mode
palette = {
    "canvas": "#0B0D10" if is_dark else "#F4F5F7",
    "card": "#14171D" if is_dark else "#FFFFFF",
    "border": "#262B36" if is_dark else "#E5E7EB",
    "text_primary": "#F3F4F6" if is_dark else "#111827",
    "text_secondary": "#9CA3AF" if is_dark else "#6B7280",
    "accent": "#60A5FA" if is_dark else "#2563EB",
    "ai_bubble": "#1C2029" if is_dark else "#F8F9FB",
    "ai_text": "#F3F4F6" if is_dark else "#111827",
    "user_bubble": "#18181B",
    "user_text": "#FFFFFF",
    "chip_bg": "#1E2430" if is_dark else "#F3F4F6",
    "code_bg": "#1F2937" if is_dark else "#EFF6FF",
    "code_fg": "#93C5FD" if is_dark else "#1D4ED8",
    "input_bg": "#1C2029" if is_dark else "#FFFFFF",
    "input_fg": "#F3F4F6" if is_dark else "#111827",
    "placeholder_fg": "#9CA3AF" if is_dark else "#6B7280",
    "btn_sec_bg": "#1E293B" if is_dark else "#FFFFFF",
    "btn_sec_fg": "#F8FAFC" if is_dark else "#111827",
    "btn_sec_border": "#475569" if is_dark else "#E5E7EB",
    "btn_pri_bg": "#2563EB" if is_dark else "#18181B",
    "btn_pri_fg": "#FFFFFF",
    "dropzone_bg": "#10141B" if is_dark else "#F9FAFB",
}

st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Quantico:wght@400;700&display=swap" rel="stylesheet">
<style>
    :root {{
        --canvas-bg: {palette['canvas']};
        --card-bg: {palette['card']};
        --card-border: {palette['border']};
        --text-primary: {palette['text_primary']};
        --text-secondary: {palette['text_secondary']};
        --accent: {palette['accent']};
        --ai-bubble-bg: {palette['ai_bubble']};
        --ai-text: {palette['ai_text']};
        --user-bubble-bg: {palette['user_bubble']};
        --user-text: {palette['user_text']};
        --chip-bg: {palette['chip_bg']};
        --code-bg: {palette['code_bg']};
        --code-fg: {palette['code_fg']};
        --input-bg: {palette['input_bg']};
        --input-fg: {palette['input_fg']};
        --placeholder-fg: {palette['placeholder_fg']};
        --btn-sec-bg: {palette['btn_sec_bg']};
        --btn-sec-fg: {palette['btn_sec_fg']};
        --btn-sec-border: {palette['btn_sec_border']};
        --btn-pri-bg: {palette['btn_pri_bg']};
        --btn-pri-fg: {palette['btn_pri_fg']};
        --dropzone-bg: {palette['dropzone_bg']};
    }}

    /* Preserve Streamlit Material Icon ligatures globally */
    [data-testid="stIconMaterial"],
    .material-symbols-rounded,
    span[class*="material"] {{
        font-family: 'Material Symbols Rounded' !important;
        font-feature-settings: 'liga' 1 !important;
    }}

    /* Main UI & Chat Font: Plus Jakarta Sans (NO wildcard * or span override) */
    html, body, .stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}

    header[data-testid="stHeader"],
    section[data-testid="stSidebar"],
    div[data-testid="stToolbar"],
    #MainMenu, footer {{
        display: none !important;
    }}

    .stApp {{
        background-color: var(--canvas-bg) !important;
        color: var(--text-primary) !important;
    }}

    .block-container {{
        padding: 16px 22px !important;
        max-width: 100% !important;
    }}

    /* Centered Main Title (Zero Emojis) */
    .bento-header-title {{
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        text-align: center !important;
        font-size: 2.1rem !important;
        font-weight: 800 !important;
        line-height: 1.2 !important;
        letter-spacing: -0.02em;
        color: var(--text-primary) !important;
        margin: 0 0 14px 0 !important;
    }}

    /* Bento Cards (Solid Surfaces, 1px Border, 24px Radius, No Glassmorphism) */
    .st-key-bento_chat_card,
    .st-key-bento_upload_card,
    .st-key-bento_explorer_card {{
        background-color: var(--card-bg) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 24px !important;
        padding: 18px !important;
        box-sizing: border-box !important;
    }}

    .st-key-bento_upload_card,
    .st-key-bento_explorer_card {{
        margin-bottom: 14px !important;
    }}

    .st-key-bento_explorer_card [data-baseweb="select"] > div {{
        background-color: var(--input-bg) !important;
        border: 1px solid var(--card-border) !important;
        color: var(--input-fg) !important;
        border-radius: 12px !important;
    }}

    .st-key-bento_explorer_card [data-baseweb="select"] div,
    .st-key-bento_explorer_card [data-baseweb="select"] span {{
        color: var(--input-fg) !important;
        background-color: transparent !important;
    }}

    /* High-Contrast Button Styling (Light Mode & Dark Mode) */
    [data-testid="stBaseButton-secondary"],
    [data-testid="stFileUploaderDropzone"] button,
    .stButton > button {{
        background-color: var(--btn-sec-bg) !important;
        color: var(--btn-sec-fg) !important;
        border: 1px solid var(--btn-sec-border) !important;
        border-radius: 12px !important;
        padding: 7px 14px !important;
        line-height: 1.4 !important;
        min-height: 36px !important;
        height: auto !important;
        white-space: normal !important;
        display: inline-flex !important;
        flex-wrap: wrap !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 13px !important;
        font-weight: 600 !important;
    }}

    [data-testid="stBaseButton-secondary"] p,
    [data-testid="stBaseButton-secondary"] span,
    [data-testid="stFileUploaderDropzone"] button p,
    [data-testid="stFileUploaderDropzone"] button span,
    .stButton > button p,
    .stButton > button span {{
        color: var(--btn-sec-fg) !important;
    }}

    [data-testid="stBaseButton-secondary"]:hover,
    [data-testid="stFileUploaderDropzone"] button:hover,
    .stButton > button:hover {{
        border-color: var(--accent) !important;
        color: var(--accent) !important;
    }}

    [data-testid="stBaseButton-secondary"]:hover p,
    [data-testid="stBaseButton-secondary"]:hover span,
    .stButton > button:hover p,
    .stButton > button:hover span {{
        color: var(--accent) !important;
    }}

    [data-testid="stBaseButton-primary"],
    div[data-testid="stFormSubmitButton"] button {{
        background-color: var(--btn-pri-bg) !important;
        color: var(--btn-pri-fg) !important;
        border: 1px solid var(--btn-pri-bg) !important;
        border-radius: 12px !important;
        padding: 7px 14px !important;
        line-height: 1.4 !important;
        min-height: 38px !important;
        height: auto !important;
        font-size: 13px !important;
        font-weight: 600 !important;
    }}

    [data-testid="stBaseButton-primary"] p,
    [data-testid="stBaseButton-primary"] span,
    div[data-testid="stFormSubmitButton"] button p,
    div[data-testid="stFormSubmitButton"] button span {{
        color: var(--btn-pri-fg) !important;
    }}

    button:disabled,
    [data-testid="stBaseButton-secondary"]:disabled,
    [data-testid="stBaseButton-primary"]:disabled {{
        opacity: 0.48 !important;
        cursor: not-allowed !important;
    }}

    /* Distinct Engineering Console Styling for Configuration Drawer (Quantico + #1E2430) */
    .st-key-bento_config_drawer {{
        background-color: #1E2430 !important;
        border: 1px solid #334155 !important;
        border-radius: 24px !important;
        padding: 18px !important;
        box-sizing: border-box !important;
    }}

    .st-key-bento_config_drawer p,
    .st-key-bento_config_drawer label,
    .st-key-bento_config_drawer input,
    .st-key-bento_config_drawer button,
    .st-key-bento_config_drawer .bento-card-header,
    .st-key-bento_config_drawer div[data-baseweb="select"],
    .st-key-bento_config_drawer div[data-baseweb="radio"] {{
        font-family: 'Quantico', sans-serif !important;
        color: #F8FAFC !important;
    }}

    /* Fix white-on-white TextInput & Password Eye Toggle inside Configuration Drawer */
    .st-key-bento_config_drawer [data-testid="stTextInput"] div[data-baseweb="input"],
    .st-key-bento_config_drawer [data-testid="stTextInput"] div[data-baseweb="base-input"] {{
        background-color: #0F172A !important;
        border: 1px solid #475569 !important;
        border-radius: 10px !important;
    }}

    .st-key-bento_config_drawer [data-testid="stTextInput"] input {{
        background-color: #0F172A !important;
        color: #F8FAFC !important;
        border: none !important;
    }}

    .st-key-bento_config_drawer [data-testid="stTextInput"] button {{
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        color: #38BDF8 !important;
        z-index: 10 !important;
        padding: 0 8px !important;
        min-height: unset !important;
    }}

    .st-key-bento_config_drawer [data-testid="stTextInput"] button [data-testid="stIconMaterial"],
    .st-key-bento_config_drawer [data-testid="stTextInput"] button svg {{
        color: #38BDF8 !important;
        fill: #38BDF8 !important;
        font-family: 'Material Symbols Rounded' !important;
    }}

    /* Fix white-on-white Model Selectbox inside Configuration Drawer */
    .st-key-bento_config_drawer [data-baseweb="select"] > div {{
        background-color: #0F172A !important;
        border: 1px solid #475569 !important;
        color: #F8FAFC !important;
        border-radius: 10px !important;
    }}

    .st-key-bento_config_drawer [data-baseweb="select"] div,
    .st-key-bento_config_drawer [data-baseweb="select"] span {{
        color: #F8FAFC !important;
        background-color: transparent !important;
    }}

    .st-key-bento_config_drawer [data-baseweb="select"] svg {{
        fill: #38BDF8 !important;
        color: #38BDF8 !important;
    }}

    /* Dropdown Popover Menu Portal Styling (Dark Slate #0F172A + Light Text #F8FAFC) */
    [data-baseweb="popover"],
    [data-baseweb="popover"] [data-baseweb="menu"],
    [data-baseweb="popover"] [role="listbox"] {{
        background-color: #0F172A !important;
        border: 1px solid #475569 !important;
        border-radius: 10px !important;
        color: #F8FAFC !important;
    }}

    [data-baseweb="popover"] [role="option"] {{
        background-color: #0F172A !important;
        color: #F8FAFC !important;
        font-family: 'Quantico', sans-serif !important;
    }}

    [data-baseweb="popover"] [role="option"]:hover,
    [data-baseweb="popover"] [aria-selected="true"] {{
        background-color: #1E293B !important;
        color: #38BDF8 !important;
    }}

    .st-key-bento_config_drawer .bento-card-header {{
        color: #38BDF8 !important;
    }}

    /* Bento Card Headers (18px, weight 600) */
    .bento-card-header {{
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        margin: 0 0 12px 0 !important;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
    }}

    .bento-title-group {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }}

    /* File Uploader Dropzone */
    [data-testid="stFileUploader"] {{
        background-color: transparent !important;
    }}

    [data-testid="stFileUploaderDropzone"] {{
        background-color: var(--dropzone-bg) !important;
        border: 1px dashed var(--btn-sec-border) !important;
        border-radius: 14px !important;
        color: var(--text-primary) !important;
        padding: 12px !important;
    }}

    [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] div {{
        color: var(--text-secondary) !important;
    }}

    /* Strict Left / Right Message Bubbles */
    .msg-row-user {{
        display: flex;
        justify-content: flex-end;
        width: 100%;
        margin: 8px 0;
    }}

    .bubble-user {{
        background-color: var(--user-bubble-bg);
        color: var(--user-text);
        border-radius: 18px 18px 4px 18px;
        padding: 11px 16px;
        max-width: 78%;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 15px;
        font-weight: 400;
        line-height: 1.5;
    }}

    .msg-meta {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        opacity: 0.72;
        margin-top: 5px;
    }}

    [class*="st-key-ai_bubble_"] {{
        background-color: var(--ai-bubble-bg) !important;
        color: var(--ai-text) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 18px 18px 18px 4px !important;
        padding: 13px 17px !important;
        max-width: 86% !important;
        margin-right: auto !important;
        margin-bottom: 10px !important;
    }}

    [class*="st-key-ai_bubble_"] p,
    [class*="st-key-ai_bubble_"] li {{
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 15px !important;
        font-weight: 400 !important;
        line-height: 1.5 !important;
        color: var(--ai-text) !important;
    }}

    [class*="st-key-ai_bubble_"] code {{
        font-family: 'JetBrains Mono', monospace !important;
        background-color: var(--code-bg) !important;
        color: var(--code-fg) !important;
        padding: 2px 6px !important;
        border-radius: 6px !important;
        font-size: 12px !important;
    }}

    /* Citation Badges (#2563EB, JetBrains Mono) */
    .citation-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 8px;
    }}

    .citation-badge {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background-color: #2563EB;
        color: #FFFFFF !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        padding: 3px 10px;
        border-radius: 999px;
    }}

    /* Live Indicator Pill */
    .live-pill {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: rgba(56, 189, 248, 0.16);
        color: #38BDF8 !important;
        border: 1px solid #38BDF8;
        font-family: 'Quantico', sans-serif !important;
        font-size: 12px !important;
        font-weight: 700 !important;
        padding: 2px 10px;
        border-radius: 999px;
    }}

    .live-dot {{
        width: 7px;
        height: 7px;
        background-color: #38BDF8;
        border-radius: 50%;
        display: inline-block;
    }}

    .sync-status-pill {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        background-color: var(--chip-bg);
        color: var(--accent);
        border: 1px solid var(--card-border);
        padding: 6px 10px;
        border-radius: 10px;
        margin-top: 8px;
    }}

    .upload-hint-text {{
        font-size: 13px;
        color: var(--text-secondary);
        margin-bottom: 6px;
        line-height: 1.4;
    }}

    .empty-chat-title {{
        padding: 18px 12px 6px 12px;
        color: var(--text-primary);
        font-size: 15px;
        font-weight: 500;
        text-align: center;
    }}

    .empty-chat-sublabel {{
        padding: 4px 12px 12px 12px;
        color: var(--text-secondary);
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        text-align: center;
    }}

    /* Explorer Rows */
    .doc-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 7px 10px;
        border: 1px solid var(--card-border);
        border-radius: 12px;
        margin-bottom: 6px;
        background-color: var(--ai-bubble-bg);
        font-size: 13px;
        color: var(--text-primary);
    }}

    .chunk-badge {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        background-color: var(--chip-bg);
        color: var(--accent);
        padding: 2px 8px;
        border-radius: 999px;
    }}

    .evidence-item {{
        border: 1px solid var(--card-border);
        background-color: var(--card-bg);
        border-radius: 12px;
        padding: 9px 12px;
        margin-top: 6px;
        font-size: 12px;
        color: var(--text-secondary);
    }}

    .evidence-item summary {{
        cursor: pointer;
        color: var(--accent);
        font-weight: 600;
        font-size: 12px;
    }}

    .evidence-chunk-full {{
        margin-top: 6px;
        padding: 8px;
        border-radius: 8px;
        background-color: var(--input-bg);
        font-family: 'JetBrains Mono', monospace !important;
        white-space: pre-wrap;
        color: var(--text-primary);
        max-height: 200px;
        overflow-y: auto;
    }}

    div[data-testid="stForm"] {{
        border: none !important;
        padding: 0 !important;
        margin-top: 10px !important;
    }}

    [data-testid="stTextInput"] div[data-baseweb="input"],
    [data-testid="stTextInput"] div[data-baseweb="base-input"],
    [data-testid="stTextArea"] div[data-baseweb="textarea"],
    [data-testid="stTextArea"] div[data-baseweb="base-input"] {{
        background-color: var(--input-bg) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 14px !important;
    }}

    textarea,
    input,
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-baseweb="input"] input,
    [data-baseweb="textarea"] textarea {{
        background-color: var(--input-bg) !important;
        color: var(--input-fg) !important;
        -webkit-text-fill-color: var(--input-fg) !important;
        caret-color: var(--input-fg) !important;
        border-radius: 14px !important;
        font-size: 15px !important;
    }}

    textarea::placeholder,
    input::placeholder,
    textarea::-webkit-input-placeholder,
    input::-webkit-input-placeholder,
    [data-testid="stTextInput"] input::placeholder,
    [data-testid="stTextArea"] textarea::placeholder,
    [data-testid="stTextInput"] input::-webkit-input-placeholder,
    [data-testid="stTextArea"] textarea::-webkit-input-placeholder {{
        color: var(--placeholder-fg) !important;
        -webkit-text-fill-color: var(--placeholder-fg) !important;
        opacity: 1 !important;
    }}

    .st-key-bento_config_drawer [data-testid="stTextInput"] input {{
        color: #F8FAFC !important;
        -webkit-text-fill-color: #F8FAFC !important;
        caret-color: #F8FAFC !important;
        background-color: #0F172A !important;
    }}

    .st-key-bento_config_drawer [data-testid="stTextInput"] input::placeholder,
    .st-key-bento_config_drawer [data-testid="stTextInput"] input::-webkit-input-placeholder {{
        color: #94A3B8 !important;
        -webkit-text-fill-color: #94A3B8 !important;
        opacity: 1 !important;
    }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def format_citation_badges_html(sources: list, max_badges: int = 3) -> str:
    unique_sources = deduplicate_sources(sources)[:max_badges]
    if not unique_sources:
        return ""
    badges = []
    for src in unique_sources:
        fname = html.escape(str(src.get("source", "Document")))
        page = html.escape(str(src.get("page", "1")))
        section = html.escape(str(src.get("section", "General")))
        badges.append(
            f'<span class="citation-badge">{LUCIDE_ICONS["file"]} {fname} · p.{page} · {section}</span>'
        )
    return f'<div class="citation-row">{"".join(badges)}</div>'


def render_compact_evidence(sources: list, max_display: int = 3) -> None:
    """Renders top 2-3 deduplicated evidence previews collapsed by default."""
    unique_sources = deduplicate_sources(sources)[:max_display]
    if not unique_sources:
        return
    with st.expander(f"Sources ({len(unique_sources)})", expanded=False):
        for idx, src in enumerate(unique_sources, 1):
            fname = html.escape(str(src.get("source", "Unknown")))
            page = html.escape(str(src.get("page", "1")))
            section = html.escape(str(src.get("section", "General")))
            score = src.get("score", 0)
            full_text = clean_chunk_content(src.get("document", ""))
            preview = " ".join(full_text.split())
            if len(preview) > 190:
                preview = preview[:190].rsplit(" ", 1)[0] + "..."
            st.markdown(f"""
            <div class="evidence-item">
                <div><b>[{idx}] {fname}</b> · Page {page} · {section} · <b>{score}%</b></div>
                <div style="margin:4px 0;">{html.escape(preview)}</div>
                <details>
                    <summary>View full chunk</summary>
                    <div class="evidence-chunk-full">{html.escape(full_text)}</div>
                </details>
            </div>
            """, unsafe_allow_html=True)


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


def stream_assistant_turn(query_text: str) -> None:
    """Streams the assistant's response progressively inside the active chat bubble."""
    prior_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]

    api_key = st.session_state.cfg_api_key
    model_name = st.session_state.cfg_model
    temperature = st.session_state.cfg_temperature
    top_k = st.session_state.cfg_top_k
    use_reranker = st.session_state.cfg_use_reranker
    dense_weight = st.session_state.cfg_dense_weight
    section_filter = st.session_state.cfg_section_filter
    mode = st.session_state.cfg_mode

    with st.container(key="ai_bubble_streaming"):
        status_ph = st.empty()
        stream_ph = st.empty()
        status_ph.markdown("*Searching Pega curriculum...*")
        accumulated_text = ""
        final_answer = ""
        final_sources = []

        try:
            if mode.startswith("Autonomous"):
                event_stream = engine.run_agent_stream(
                    query=query_text,
                    groq_api_key=api_key,
                    model_name=model_name,
                    temperature=temperature,
                    max_iterations=4,
                    chat_history=prior_history,
                    top_k=top_k,
                    use_reranker=use_reranker,
                    dense_weight=dense_weight,
                    section_filter=section_filter
                )
            else:
                def _direct_stream():
                    yield {"type": "status", "message": "Retrieving Pega curriculum excerpts (Hybrid BM25 + Vector + Reranker)..."}
                    contextual_query = contextualize_query_with_history(query_text, prior_history)
                    chunks = engine.search(
                        query=contextual_query,
                        top_k=top_k,
                        section_filter=section_filter,
                        use_reranker=use_reranker,
                        dense_weight=dense_weight
                    )
                    sources = deduplicate_sources(chunks)
                    if not sources:
                        yield {
                            "type": "final",
                            "answer": "I could not find relevant sections in the indexed Pega documents for your question.",
                            "sources": []
                        }
                        return
                    yield {"type": "status", "message": "Generating grounded response..."}
                    for ev in engine.stream_answer(
                        query=query_text,
                        retrieved_chunks=sources,
                        groq_api_key=api_key,
                        model_name=model_name,
                        temperature=temperature,
                        chat_history=prior_history
                    ):
                        yield ev
                event_stream = _direct_stream()

            for ev in event_stream:
                if not isinstance(ev, dict):
                    continue
                ev_type = ev.get("type")
                if ev_type == "status":
                    status_ph.markdown(f"*{html.escape(str(ev.get('message') or 'Working...'))}*")
                elif ev_type == "token":
                    accumulated_text += str(ev.get("text") or "")
                    stream_ph.markdown(accumulated_text + " ▌")
                elif ev_type == "final":
                    status_ph.empty()
                    final_answer = ev.get("answer") or accumulated_text
                    final_sources = deduplicate_sources(ev.get("sources") or [])
                    stream_ph.markdown(final_answer)
        except Exception as exc:
            status_ph.empty()
            final_answer = f"❌ **Streaming Error:** {str(exc)}"
            stream_ph.markdown(final_answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": final_answer or accumulated_text or "No response generated.",
        "sources": final_sources,
        "timestamp": datetime.now().strftime("%H:%M")
    })
    st.session_state.pending_response = False
    st.session_state.pending_query = None
    st.session_state.scroll_to_bottom = True
    st.rerun()


# ---------------------------------------------------------
# 1. Centered Top Title & Top-Right Controls (Zero Emojis)
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
# 2-Column Bento Layout (+ Configuration Drawer at Extreme Right)
# ---------------------------------------------------------

if st.session_state.show_config:
    col_chat, col_right, col_drawer = st.columns([1.6, 1, 0.95], gap="medium")
else:
    col_chat, col_right = st.columns([1.75, 1], gap="medium")
    col_drawer = None

# ---------------------------------------------------------
# Left Bento Card: Ask Architect (Q&A)
# ---------------------------------------------------------
with col_chat:
    with st.container(key="bento_chat_card"):
        hdr_row_l, hdr_row_r = st.columns([4, 1])
        with hdr_row_l:
            st.markdown(
                f'<div class="bento-card-header"><span class="bento-title-group">{LUCIDE_ICONS["message"]} Ask Architect (Q&amp;A)</span></div>',
                unsafe_allow_html=True
            )
        with hdr_row_r:
            if st.session_state.messages:
                if st.button("Clear Chat", key="btn_clear_chat", use_container_width=True):
                    st.session_state.messages = []
                    st.rerun()

        # Fixed-Height Scrollable Message Container (never shifts input)
        chat_box = st.container(height=500, border=False)
        with chat_box:
            if not st.session_state.messages:
                st.markdown(
                    '<div class="empty-chat-title">Ask a Pega CLSA architecture question below to begin a multi-turn conversation.</div>'
                    '<div class="empty-chat-sublabel">Example questions to ask</div>',
                    unsafe_allow_html=True
                )
                demo_questions = [
                    "When should I choose Data Pages with Keyed Access over Report Definitions?",
                    "How do I design an enterprise Case Type hierarchy to avoid class group locking?",
                    "What are the best practices for asynchronous Queue Processors vs. Job Schedulers?",
                    "How does Pega Constellation decouple UI views from DX API data models?"
                ]
                d_col1, d_col2 = st.columns(2)
                for d_idx, d_query in enumerate(demo_questions):
                    target_col = d_col1 if d_idx % 2 == 0 else d_col2
                    with target_col:
                        if st.button(d_query, key=f"demo_q_{d_idx}", use_container_width=True):
                            queue_user_query(d_query)
                            st.rerun()

            for m_idx, msg in enumerate(st.session_state.messages):
                ts = html.escape(str(msg.get("timestamp", "")))
                if msg["role"] == "user":
                    safe_q = html.escape(msg["content"]).replace("\n", "<br>")
                    st.markdown(f"""
                    <div class="msg-row-user">
                        <div class="bubble-user">
                            {safe_q}
                            <div class="msg-meta">{ts}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    with st.container(key=f"ai_bubble_{m_idx}"):
                        st.markdown(msg["content"])
                        srcs = msg.get("sources", [])
                        if srcs:
                            st.markdown(format_citation_badges_html(srcs, max_badges=3), unsafe_allow_html=True)
                            render_compact_evidence(srcs, max_display=3)
                        if ts:
                            st.markdown(f'<div class="msg-meta">{ts}</div>', unsafe_allow_html=True)

            # Live Streaming Turn (executes after user message is already rendered above)
            if st.session_state.pending_response and st.session_state.pending_query:
                stream_assistant_turn(st.session_state.pending_query)

            # Anchor inside chat_box for independent container auto-scroll
            st.markdown('<div id="chat-box-end-anchor"></div>', unsafe_allow_html=True)

        # Independent Scroll Script (scrolls only chat_box container, never outer window)
        if st.session_state.scroll_to_bottom:
            components.html(
                """
                <script>
                (function() {
                    const parentDoc = window.parent.document;
                    const anchor = parentDoc.getElementById('chat-box-end-anchor');
                    if (anchor) {
                        let node = anchor.parentElement;
                        while (node && node !== parentDoc.body) {
                            const style = window.parent.getComputedStyle(node);
                            if ((style.overflowY === 'auto' || style.overflowY === 'scroll') && node.scrollHeight > node.clientHeight) {
                                node.scrollTop = node.scrollHeight;
                                break;
                            }
                            node = node.parentElement;
                        }
                    }
                })();
                </script>
                """,
                height=0
            )
            st.session_state.scroll_to_bottom = False

        # Pinned Input Bar Directly Below chat_box
        with st.form(key="bento_chat_form", clear_on_submit=True):
            in_col, send_col = st.columns([6, 1.1])
            with in_col:
                typed_query = st.text_input(
                    "Ask a question",
                    placeholder="Ask any Pega architecture, Constellation, security, or follow-up question...",
                    label_visibility="collapsed",
                    key="chat_text_input"
                )
            with send_col:
                submitted = st.form_submit_button("Send", type="primary", use_container_width=True)

        if submitted and typed_query and typed_query.strip():
            queue_user_query(typed_query.strip())
            st.rerun()

# ---------------------------------------------------------
# Right Bento Stack: Upload & Sync + Indexed Knowledge + Configuration
# ---------------------------------------------------------
with col_right:
    # Top Card: Upload & Add More Files + Sync Live Files
    with st.container(key="bento_upload_card"):
        st.markdown(
            f'<div class="bento-card-header"><span class="bento-title-group">{LUCIDE_ICONS["upload"]} Upload &amp; Add More Files</span></div>',
            unsafe_allow_html=True
        )
        up_files = st.file_uploader(
            "Browse Files",
            type=["pdf", "docx", "doc", "txt", "md"],
            accept_multiple_files=True,
            label_visibility="visible",
            key="bento_uploader"
        )
        doc_name_input = st.text_input(
            "Document Name (Optional)",
            value="",
            placeholder="Enter a descriptive name for this document",
            label_visibility="visible",
            key="bento_upload_doc_name"
        )
        if not up_files:
            st.markdown(
                '<div class="upload-hint-text">Give your document a recognizable name to make it easier to identify in the knowledge base and search results.</div>',
                unsafe_allow_html=True
            )

        act_c1, act_c2 = st.columns(2)
        with act_c1:
            upload_clicked = st.button(
                "Upload",
                type="primary",
                disabled=not bool(up_files),
                key="btn_index_uploads",
                use_container_width=True
            )
        with act_c2:
            sync_clicked = st.button(
                "Sync Live Files",
                key="btn_sync_live_files",
                use_container_width=True
            )

        if upload_clicked and up_files:
            target_dir = DOCS_DIR / "06_Custom_Materials"
            target_dir.mkdir(parents=True, exist_ok=True)
            custom_doc_name = doc_name_input.strip() if doc_name_input else ""
            total_added = 0
            for f_obj in up_files:
                dest = target_dir / f_obj.name
                with open(dest, "wb") as out_f:
                    out_f.write(f_obj.getbuffer())
                chunks_count = engine.chunk_and_index_file(dest)
                total_added += chunks_count
                manifest = engine.load_manifest()
                rel_path = str(dest.relative_to(WORKSPACE_DIR))
                manifest_entry = {
                    "hash": compute_file_hash(dest),
                    "file_name": dest.name,
                    "chunks_count": chunks_count,
                    "last_indexed": str(dest.stat().st_mtime),
                    "extractor_version": 2 if dest.suffix.lower() in {".docx", ".doc"} else 1,
                }
                if custom_doc_name:
                    manifest_entry["document_name"] = custom_doc_name
                manifest[rel_path] = manifest_entry
                engine.save_manifest(manifest)
            st.session_state.sync_status = f"Indexed {len(up_files)} file(s) ({total_added} chunks)."
            time.sleep(0.4)
            st.rerun()

        if sync_clicked:
            with st.spinner("Scanning for new or unindexed files..."):
                sync_res = engine.sync_database()
            st.session_state.sync_status = (
                f"Synced: {sync_res['new_indexed']} new ({sync_res['chunks_added']} chunks), "
                f"{sync_res['skipped']} unchanged."
            )
            st.rerun()

        if st.session_state.sync_status:
            st.markdown(
                f'<div class="sync-status-pill">{html.escape(st.session_state.sync_status)}</div>',
                unsafe_allow_html=True
            )

    # Middle Card: Indexed Knowledge & Chunk Explorer + Document Management
    with st.container(key="bento_explorer_card"):
        indexed_docs = engine.list_indexed_documents()
        total_chunks = engine.collection.count()
        st.markdown(
            f'<div class="bento-card-header">'
            f'<span class="bento-title-group">{LUCIDE_ICONS["folder"]} Indexed Knowledge &amp; Chunk Explorer</span>'
            f'<span class="chunk-badge">{len(indexed_docs)} docs · {total_chunks} chunks</span></div>',
            unsafe_allow_html=True
        )

        # Confirmation dialog for Delete All Chunks (mutually exclusive with pending_delete_doc)
        if st.session_state.pending_delete_all:
            st.warning(
                f"Confirm **Delete All Chunks**: this will remove all **{total_chunks}** indexed chunks "
                f"across **{len(indexed_docs)}** document(s) from ChromaDB and clear the manifest. "
                f"Original files on disk will NOT be deleted unless checked below."
            )
            delete_all_disk_checked = st.checkbox(
                "Also delete original source files from disk inside DOCS_DIR (permanent)",
                value=False,
                key="confirm_delete_all_disk_chk"
            )
            c_all1, c_all2 = st.columns(2)
            with c_all1:
                if st.button("Confirm Delete All", type="primary", key="btn_confirm_delete_all", use_container_width=True):
                    del_all_res = engine.delete_all_chunks(delete_disk_files=bool(delete_all_disk_checked))
                    if del_all_res.get("status") == "success":
                        disk_msg = (
                            f" Deleted {del_all_res.get('deleted_disk_files', 0)} disk file(s)."
                            if delete_all_disk_checked else ""
                        )
                        st.session_state.sync_status = (
                            f"Deleted all {del_all_res.get('deleted_chunks', 0)} chunks "
                            f"across {del_all_res.get('previous_doc_count', 0)} document(s).{disk_msg}"
                        )
                        st.session_state.pending_delete_all = False
                        st.session_state.pop("confirm_delete_all_disk_chk", None)
                        st.rerun()
                    else:
                        st.error(del_all_res.get("message", "Failed to delete all chunks."))
            with c_all2:
                if st.button("Cancel", key="btn_cancel_delete_all", use_container_width=True):
                    st.session_state.pending_delete_all = False
                    st.session_state.pop("confirm_delete_all_disk_chk", None)
                    st.rerun()

        # Confirmation dialog for pending individual deletion (prevents accidental deletion)
        if st.session_state.pending_delete_doc:
            p_doc = st.session_state.pending_delete_doc
            st.warning(
                f"Confirm index deletion for **{p_doc['file_name']}** "
                f"(`{p_doc['relative_path']}` — **{p_doc['chunks_count']}** indexed chunks). "
                f"The original file on disk will NOT be deleted."
            )
            disk_delete_checked = False
            if p_doc.get("file_exists_on_disk"):
                disk_delete_checked = st.checkbox(
                    "Also delete original file from disk (permanent)",
                    value=False,
                    key="confirm_disk_delete_chk"
                )
            c_del1, c_del2 = st.columns(2)
            with c_del1:
                if st.button("Confirm Delete", type="primary", key="btn_confirm_doc_delete", use_container_width=True):
                    del_res = engine.delete_document_chunks(
                        relative_path=p_doc["relative_path"],
                        source_name=p_doc["file_name"]
                    )
                    if disk_delete_checked:
                        try:
                            target_disk = WORKSPACE_DIR / p_doc["relative_path"]
                            if target_disk.exists() and target_disk.is_file():
                                target_disk.unlink()
                        except Exception as disk_err:
                            st.error(f"Could not delete disk file: {disk_err}")
                    st.session_state.sync_status = (
                        f"Deleted {del_res.get('deleted_chunks', 0)} chunks for {p_doc['file_name']}."
                    )
                    st.session_state.pending_delete_doc = None
                    st.rerun()
            with c_del2:
                if st.button("Cancel", key="btn_cancel_doc_delete", use_container_width=True):
                    st.session_state.pending_delete_doc = None
                    st.rerun()

        raw_search_q = st.text_input(
            "Inline Chunk Previewer",
            placeholder="Search chunks (e.g. Queue Processor)...",
            label_visibility="collapsed",
            key="explorer_search_input"
        )

        # Dedicated options to delete chunks for one selected document or all indexed chunks
        if indexed_docs:
            doc_options = {f"{d['file_name']} ({d['chunks_count']} chunks)": d for d in indexed_docs}
            sel_col, btn_col, btn_all_col = st.columns([1.9, 1, 1.1])
            with sel_col:
                selected_doc_label = st.selectbox(
                    "Select document to manage",
                    options=list(doc_options.keys()),
                    label_visibility="collapsed",
                    key="doc_mgmt_select"
                )
            with btn_col:
                if st.button("Delete Chunks", key="btn_delete_selected_doc", use_container_width=True):
                    st.session_state.pending_delete_all = False
                    st.session_state.pending_delete_doc = doc_options[selected_doc_label]
                    st.rerun()
            with btn_all_col:
                if st.button("Delete All Chunks", key="btn_delete_all_chunks", use_container_width=True):
                    st.session_state.pending_delete_doc = None
                    st.session_state.pending_delete_all = True
                    st.rerun()
        elif total_chunks > 0:
            if st.button("Delete All Chunks", key="btn_delete_all_chunks", use_container_width=True):
                st.session_state.pending_delete_doc = None
                st.session_state.pending_delete_all = True
                st.rerun()

        explorer_box = st.container(height=230, border=False)
        with explorer_box:
            if raw_search_q and raw_search_q.strip():
                hits = engine.search(
                    raw_search_q.strip(),
                    top_k=4,
                    use_reranker=st.session_state.cfg_use_reranker,
                    dense_weight=st.session_state.cfg_dense_weight
                )
                for h_idx, h in enumerate(hits, 1):
                    fname = html.escape(str(h.get("source", "Unknown")))
                    page = html.escape(str(h.get("page", "1")))
                    score = h.get("score", 0)
                    clean_doc = html.escape(clean_chunk_content(h.get("document", "")))
                    st.markdown(f"""
                    <div class="evidence-item">
                        <div><b>[{h_idx}] {fname}</b> · p.{page} · <b>{score}%</b></div>
                        <details>
                            <summary>Preview chunk</summary>
                            <div class="evidence-chunk-full">{clean_doc}</div>
                        </details>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                if indexed_docs:
                    for d_idx, doc_info in enumerate(indexed_docs):
                        doc_name = html.escape(doc_info.get("document_name") or doc_info["file_name"])
                        fname = html.escape(doc_info["file_name"])
                        rel_p = html.escape(doc_info["relative_path"])
                        cat = html.escape(doc_info["category"])
                        sec = html.escape(doc_info.get("section", "General"))
                        c_count = doc_info["chunks_count"]
                        display_title = f"{doc_name} ({fname})" if doc_name != fname else fname
                        row_c1, row_c2 = st.columns([3.4, 1])
                        with row_c1:
                            st.markdown(f"""
                            <div class="doc-row" style="flex-direction:column; align-items:flex-start; gap:3px; margin-bottom:4px;">
                                <div style="display:flex; width:100%; justify-content:space-between; align-items:center;">
                                    <span style="display:inline-flex; align-items:center; gap:5px; font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:170px;" title="{display_title}">{LUCIDE_ICONS["file"]} {display_title}</span>
                                    <span class="chunk-badge">{c_count} chunks</span>
                                </div>
                                <div style="font-size:11px; color:var(--text-secondary); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:225px;" title="{rel_p}">{cat} · {sec} · {rel_p}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with row_c2:
                            if st.button("Delete", key=f"del_doc_{d_idx}", use_container_width=True):
                                st.session_state.pending_delete_all = False
                                st.session_state.pending_delete_doc = doc_info
                                st.rerun()
                else:
                    st.info("No indexed files found in manifest.")

# ---------------------------------------------------------
# Extreme-Right Configuration Column
# ---------------------------------------------------------
if col_drawer is not None:
    with col_drawer:
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
