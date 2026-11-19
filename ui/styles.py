"""Light/Dark Bento theme CSS and inline Lucide SVG icons for PegaRagAgent."""

import streamlit as st

LUCIDE_ICONS = {
    "message": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>',
    "upload": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>',
    "folder": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>',
    "sliders": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>',
    "file": '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>',
}


def inject_bento_styles(is_dark: bool) -> None:
    """Injects Google Fonts, Light/Dark CSS variables, and Bento UI styles."""
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
