"""
Streamlit Web Interface for Pega CLSA Knowledge Assistant
==========================================================
Interactive RAG application powered by ChromaDB, SentenceTransformers, and Groq LLM.
Features:
- Chat interface with page-level citations & sources
- Dynamic upload & chunking of new documents
- Live vector database indexing & explorer
"""

import os
# Suppress Streamlit module watcher checking uninstalled vision packages in transformers
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import shutil
import time
from pathlib import Path

import streamlit as st
from rag_engine import PegaRAGEngine, WORKSPACE_DIR, DOCS_DIR

# Set page configuration
st.set_page_config(
    page_title="Pega CLSA Knowledge Assistant",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0A325C;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #556677;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F0F6FC;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #0A325C;
        margin-bottom: 12px;
    }
    .source-box {
        background-color: #F8F9FA;
        border-radius: 6px;
        padding: 10px 14px;
        border: 1px solid #E1E4E8;
        margin-bottom: 8px;
        font-size: 0.92rem;
    }
    .badge-module {
        background-color: #0A325C;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-challenge {
        background-color: #16A085;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .stButton>button {
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize RAG Engine in cache
@st.cache_resource(show_spinner="Initializing Pega RAG Engine & Embedding Model...")
def get_engine():
    return PegaRAGEngine()

engine = get_engine()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retrieved_sources" not in st.session_state:
    st.session_state.retrieved_sources = []

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/cf/Pegasystems_logo.svg/320px-Pegasystems_logo.svg.png", width=180)
    st.markdown("### ⚙️ Engine Settings")
    
    # Groq API Key Input
    env_groq_key = os.environ.get("GROQ_API_KEY", "")
    groq_api_key = st.text_input(
        "Groq API Key",
        value=env_groq_key,
        type="password",
        placeholder="gsk_...",
        help="Get your free API key at https://console.groq.com/keys"
    )
    
    if groq_api_key:
        st.success("API Key detected", icon="✅")
    else:
        st.info("💡 Paste your Groq API key above. It is free with instant setup at [console.groq.com](https://console.groq.com/keys).")

    # Dynamically fetch models from the user's specific Groq key
    active_chat_models = []
    TOOL_CAPABLE_MODELS = {
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "llama3-groq-70b-8192-tool-use-preview",
        "llama3-groq-8b-8192-tool-use-preview"
    }

    if groq_api_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_api_key.strip())
            live_models = client.models.list()
            raw_ids = [m.id for m in live_models.data]
            
            # Filter for text chat models: exclude whisper audio, tts, speech, safety guards, and canopylabs
            for m_id in raw_ids:
                m_low = m_id.lower()
                if any(x in m_low for x in ["whisper", "orpheus", "tts", "audio", "guard", "safeguard", "canopylabs"]):
                    continue
                active_chat_models.append(m_id)

            # Sort: tool-capable models first, then others
            active_chat_models.sort(key=lambda m: (0 if m in TOOL_CAPABLE_MODELS else 1, m))
        except Exception as e:
            st.sidebar.error(f"Groq Key Validation: {e}")

    if not active_chat_models:
        active_chat_models = [
            "llama-3.3-70b-versatile",
            "qwen/qwen3.8-27b",
            "openai/gpt-oss-20b",
            "llama3-8b-8192"
        ]

    def format_model_label(m_id: str) -> str:
        if m_id in TOOL_CAPABLE_MODELS:
            return f"{m_id}  (🛠️ Multi-Step Agent)"
        return f"{m_id}  (⚡ Grounded RAG)"

    selected_model = st.selectbox(
        "LLM Reasoning Model",
        options=active_chat_models,
        index=0,
        format_func=format_model_label,
        help="Models available directly on your Groq API account. 🛠️ models run multi-step agent tool calling; ⚡ models automatically use direct grounded retrieval."
    )

    if groq_api_key and active_chat_models:
        with st.sidebar.expander(f"📋 Models on Your Key ({len(active_chat_models)})"):
            for m in active_chat_models:
                st.code(m, language="text")

    # Operating Mode
    assistant_mode = st.radio(
        "🧠 Intelligence Mode",
        options=[
            "🤖 Autonomous CLSA Agent (Recommended)",
            "⚡ Fast Direct RAG"
        ],
        index=0,
        help="The Autonomous Agent plans, calls tools, performs multi-step research across modules & challenges, and reads full PDF pages to eliminate hallucinations."
    )

    # Retrieval parameters
    top_k = st.slider("Top Sources (Chunks) to Retrieve", min_value=3, max_value=12, value=5, step=1)
    temperature = st.slider(
        "Response Temperature (0.0 = Zero Hallucination)",
        min_value=0.0,
        max_value=0.8,
        value=0.0,
        step=0.05,
        help="0.0 guarantees strictly deterministic, grounded answers without creative extrapolation."
    )

    st.markdown("---")
    st.markdown("### 📚 Knowledge Base Status")
    
    # Sync & stats
    col_s1, col_s2 = st.columns(2)
    total_chunks = engine.collection.count()
    manifest = engine.load_manifest()
    total_files = len(manifest)

    with col_s1:
        st.metric("Files Indexed", total_files)
    with col_s2:
        st.metric("Total Chunks", total_chunks)

    if st.button("🔄 Sync & Scan for New Files", use_container_width=True):
        with st.spinner("Scanning directory and indexing any new or modified files..."):
            sync_res = engine.sync_database()
            st.success(f"Sync complete! Added {sync_res['new_indexed']} new files ({sync_res['chunks_added']} chunks).")
            time.sleep(1)
            st.rerun()

    st.markdown("---")
    # Section filter option
    section_filter = st.selectbox(
        "Filter by Curriculum Section",
        options=[
            "All Sections",
            "01_LSA_Mindset",
            "02_Planning_and_Design",
            "03_Constellation_Adoption",
            "04_Infrastructure_and_Architecture",
            "05_Implementation_and_Deployment"
        ],
        index=0
    )

# ---------------------------------------------------------
# Main Interface Tabs
# ---------------------------------------------------------

st.markdown('<div class="main-title">🦅 Pega CLSA Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Instant, accurate answers grounded strictly in the official Pega Lead System Architect (v7) curriculum.</div>', unsafe_allow_html=True)

tab_chat, tab_upload, tab_explorer = st.tabs([
    "💬 Ask Architect (Q&A)",
    "📤 Upload & Add More Files",
    "📑 Document & Chunk Explorer"
])

# ---------------------------------------------------------
# Tab 1: Chat & Q&A
# ---------------------------------------------------------

with tab_chat:
    # Suggested starter queries
    st.markdown("**Suggested Architectural Questions:**")
    prompt_cols = st.columns(2)
    
    suggested_prompts = [
        "What are the core differences between Constellation architecture and Traditional UI?",
        "When should an architect use Queue Processors versus Job Schedulers?",
        "How do you implement Attribute-Based Access Control (ABAC) in Pega?",
        "What are the best practices for Enterprise Class Structure (ECS) specialization?"
    ]
    
    clicked_prompt = None
    with prompt_cols[0]:
        if st.button(f"💡 {suggested_prompts[0]}", use_container_width=True):
            clicked_prompt = suggested_prompts[0]
        if st.button(f"💡 {suggested_prompts[1]}", use_container_width=True):
            clicked_prompt = suggested_prompts[1]
    with prompt_cols[1]:
        if st.button(f"💡 {suggested_prompts[2]}", use_container_width=True):
            clicked_prompt = suggested_prompts[2]
        if st.button(f"💡 {suggested_prompts[3]}", use_container_width=True):
            clicked_prompt = suggested_prompts[3]

    st.markdown("---")

    # Render chat history
    for msg_idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Display Agent Execution Trace if available
            if "trace" in msg and msg["trace"]:
                with st.expander(f"🧠 Agent Reasoning & Tool Calls ({len(msg['trace'])} actions executed)"):
                    for t_idx, step in enumerate(msg["trace"], 1):
                        st.markdown(f"**Step {t_idx}:** Called `{step['tool']}`")
                        st.json(step["args"])

            # Display citations if available
            if "sources" in msg and msg["sources"]:
                with st.expander(f"📚 Retrieved Evidence & Page Citations ({len(msg['sources'])} sources)"):
                    for s_idx, src in enumerate(msg["sources"], 1):
                        badge_cls = "badge-challenge" if src.get("category") == "Hands-on Challenge" else "badge-module"
                        st.markdown(f"""
                        <div class="source-box">
                            <b>Source [{s_idx}]:</b> <code>{src.get('source', 'Unknown')}</code> &nbsp;
                            <span class="{badge_cls}">{src.get('category', 'Module')}</span> &nbsp;
                            <b>Page:</b> {src.get('page', '1')} &nbsp;|&nbsp; 
                            <b>Section:</b> {src.get('section', 'General')} &nbsp;|&nbsp; 
                            <b>Similarity:</b> {src.get('score', 0)}%
                            <hr style="margin: 6px 0; border: none; border-top: 1px dashed #DDD;">
                            <div style="font-family: monospace; white-space: pre-wrap; color: #333; font-size: 0.85rem;">{src.get('document', '')}</div>
                        </div>
                        """, unsafe_allow_html=True)

    # Chat Input handler
    user_query = st.chat_input("Ask any architectural, design, or implementation question...")
    
    if clicked_prompt:
        user_query = clicked_prompt

    if user_query:
        # Add user query to chat
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            if assistant_mode.startswith("🤖"):
                # Autonomous Agent Execution
                with st.spinner("🤖 Autonomous CLSA Agent analyzing question, calling tools, and verifying evidence..."):
                    agent_res = engine.run_agent(
                        query=user_query,
                        groq_api_key=groq_api_key,
                        model_name=selected_model,
                        temperature=temperature,
                        max_iterations=4,
                        chat_history=st.session_state.messages[:-1]
                    )

                st.markdown(agent_res["answer"])

                # Display agent thought process trace
                if agent_res.get("trace"):
                    with st.expander(f"🧠 Agent Reasoning & Tool Calls ({len(agent_res['trace'])} actions executed)"):
                        for t_idx, step in enumerate(agent_res["trace"], 1):
                            st.markdown(f"**Step {t_idx}:** Tool `{step['tool']}`")
                            st.json(step["args"])

                # Display evidence sources
                if agent_res.get("sources"):
                    with st.expander(f"📚 Retrieved Evidence & Page Citations ({len(agent_res['sources'])} sources)"):
                        for s_idx, src in enumerate(agent_res["sources"], 1):
                            badge_cls = "badge-challenge" if src.get("category") == "Hands-on Challenge" else "badge-module"
                            st.markdown(f"""
                            <div class="source-box">
                                <b>Source [{s_idx}]:</b> <code>{src.get('source', 'Unknown')}</code> &nbsp;
                                <span class="{badge_cls}">{src.get('category', 'Module')}</span> &nbsp;
                                <b>Page:</b> {src.get('page', '1')} &nbsp;|&nbsp; 
                                <b>Section:</b> {src.get('section', 'General')} &nbsp;|&nbsp; 
                                <b>Similarity:</b> {src.get('score', 0)}%
                                <hr style="margin: 6px 0; border: none; border-top: 1px dashed #DDD;">
                                <div style="font-family: monospace; white-space: pre-wrap; color: #333; font-size: 0.85rem;">{src.get('document', '')}</div>
                            </div>
                            """, unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": agent_res["answer"],
                    "sources": agent_res.get("sources", []),
                    "trace": agent_res.get("trace", [])
                })

            else:
                # Direct Static RAG Mode
                with st.spinner("Retrieving relevant curriculum sections & page excerpts..."):
                    retrieved_chunks = engine.search(
                        query=user_query,
                        top_k=top_k,
                        section_filter=section_filter
                    )

                if not retrieved_chunks:
                    answer = "I could not find relevant sections in the indexed Pega documents for your question. Please verify that the files are indexed."
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": [], "trace": []})
                else:
                    with st.spinner("Formulating architectural answer via Groq LLM..."):
                        answer = engine.generate_answer(
                            query=user_query,
                            retrieved_chunks=retrieved_chunks,
                            groq_api_key=groq_api_key,
                            model_name=selected_model,
                            temperature=temperature
                        )
                        st.markdown(answer)
                        
                        # Display evidence accordion
                        with st.expander(f"📚 Retrieved Evidence & Page Citations ({len(retrieved_chunks)} sources)"):
                            for s_idx, src in enumerate(retrieved_chunks, 1):
                                badge_cls = "badge-challenge" if src["category"] == "Hands-on Challenge" else "badge-module"
                                st.markdown(f"""
                                <div class="source-box">
                                    <b>Source [{s_idx}]:</b> <code>{src['source']}</code> &nbsp;
                                    <span class="{badge_cls}">{src['category']}</span> &nbsp;
                                    <b>Page:</b> {src['page']} &nbsp;|&nbsp; 
                                    <b>Section:</b> {src['section']} &nbsp;|&nbsp; 
                                    <b>Similarity:</b> {src['score']}%
                                    <hr style="margin: 6px 0; border: none; border-top: 1px dashed #DDD;">
                                    <div style="font-family: monospace; white-space: pre-wrap; color: #333; font-size: 0.85rem;">{src['document']}</div>
                                </div>
                                """, unsafe_allow_html=True)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": retrieved_chunks,
                        "trace": []
                    })

# ---------------------------------------------------------
# Tab 2: Upload & Add More Files
# ---------------------------------------------------------

with tab_upload:
    st.markdown("### 📥 Add New Pega Documentation or Custom Files")
    st.markdown(
        "You can upload additional **PDFs, Word documents (.docx), or text files** at any time. "
        "The system will automatically extract text page-by-page, apply contextual chunking, generate embeddings, and update the vector database instantly."
    )

    up_col1, up_col2 = st.columns([2, 1])

    with up_col1:
        uploaded_files = st.file_uploader(
            "Select files to upload and index",
            type=["pdf", "docx", "doc", "txt", "md"],
            accept_multiple_files=True
        )

    with up_col2:
        target_section = st.selectbox(
            "Assign to Section / Folder",
            options=[
                "01_LSA_Mindset",
                "02_Planning_and_Design",
                "03_Constellation_Adoption",
                "04_Infrastructure_and_Architecture",
                "05_Implementation_and_Deployment",
                "06_Custom_Materials"
            ],
            index=5
        )

    if uploaded_files:
        if st.button("🚀 Process, Chunk & Index Uploaded Files", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()

            target_dir = DOCS_DIR / target_section
            target_dir.mkdir(parents=True, exist_ok=True)

            total_new_chunks = 0
            for idx, file_obj in enumerate(uploaded_files):
                dest_file = target_dir / file_obj.name
                status_text.text(f"Saving and chunking: {file_obj.name} ...")
                
                # Write file to disk
                with open(dest_file, "wb") as f:
                    f.write(file_obj.getbuffer())

                # Chunk and index directly
                chunks_count = engine.chunk_and_index_file(dest_file)
                total_new_chunks += chunks_count

                # Update manifest
                manifest = engine.load_manifest()
                rel_path = str(dest_file.relative_to(WORKSPACE_DIR))
                from rag_engine import compute_file_hash
                manifest[rel_path] = {
                    "hash": compute_file_hash(dest_file),
                    "file_name": dest_file.name,
                    "chunks_count": chunks_count,
                    "last_indexed": str(dest_file.stat().st_mtime)
                }
                engine.save_manifest(manifest)

                progress_bar.progress((idx + 1) / len(uploaded_files))

            status_text.empty()
            progress_bar.empty()
            st.success(f"🎉 Successfully uploaded and indexed {len(uploaded_files)} file(s) into {target_section}! Added **{total_new_chunks} searchable chunks**.")
            time.sleep(1)
            st.rerun()

# ---------------------------------------------------------
# Tab 3: Document & Chunk Explorer
# ---------------------------------------------------------

with tab_explorer:
    st.markdown("### 📑 Indexed Document Catalog & Semantic Inspection")
    
    manifest_data = engine.load_manifest()
    
    if manifest_data:
        # Build tabular summary
        table_rows = []
        for rel_p, meta in manifest_data.items():
            table_rows.append({
                "Document Name": meta.get("file_name", Path(rel_p).name),
                "Path": rel_p,
                "Chunks Generated": meta.get("chunks_count", 0)
            })

        st.dataframe(table_rows, use_container_width=True)
    else:
        st.info("No documents indexed in manifest yet. Click 'Sync & Scan' in the sidebar.")

    st.markdown("---")
    st.markdown("#### 🔬 Test Raw Retrieval (No LLM required)")
    test_query = st.text_input("Enter a test query to inspect retrieved chunks:", value="How to configure Queue Processors in Pega?")
    if st.button("Search Raw Chunks"):
        with st.spinner("Querying vector database..."):
            test_results = engine.search(test_query, top_k=3)
            for t_idx, tr in enumerate(test_results, 1):
                st.markdown(f"**Result {t_idx}** | `{tr['source']}` (Page {tr['page']}) | Similarity: **{tr['score']}%**")
                st.code(tr['document'], language="markdown")

