"""Upload & Add More Files + Sync Live Files panel for PegaRagAgent."""

import html
import re
import time
from pathlib import Path
from typing import Any

import streamlit as st

import rag_engine
from rag_engine import compute_file_hash
from ui.styles import LUCIDE_ICONS


def sanitize_uploaded_filename(raw_name: str) -> str:
    """Strips directory components and unsafe characters from an uploaded filename."""
    normalized = (raw_name or "").replace("\\", "/")
    base = Path(normalized).name.strip().lstrip(".")
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", base).strip()
    if not cleaned or cleaned in {".", ".."}:
        return "uploaded_document"
    return cleaned


def render_upload_panel(engine: Any) -> None:
    """Renders the Upload & Add More Files + Sync Live Files Bento card."""
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
            target_dir = (rag_engine.DOCS_DIR / "06_Custom_Materials").resolve()
            target_dir.mkdir(parents=True, exist_ok=True)
            custom_doc_name = doc_name_input.strip() if doc_name_input else ""
            total_added = 0
            for f_obj in up_files:
                safe_name = sanitize_uploaded_filename(f_obj.name)
                dest = (target_dir / safe_name).resolve()
                dest.relative_to(target_dir)
                with open(dest, "wb") as out_f:
                    out_f.write(f_obj.getbuffer())
                chunks_count = engine.chunk_and_index_file(dest)
                total_added += chunks_count
                manifest = engine.load_manifest()
                rel_path = str(dest.relative_to(rag_engine.WORKSPACE_DIR.resolve()))
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
