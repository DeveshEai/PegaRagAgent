"""Indexed Knowledge & Chunk Explorer + Document Management panel."""

import html
from typing import Any

import streamlit as st

import rag_engine
from rag_engine import clean_chunk_content
from ui.styles import LUCIDE_ICONS


def render_explorer_panel(engine: Any) -> None:
    """Renders the Indexed Knowledge & Chunk Explorer Bento card with deletion workflows."""
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
                            target_disk = rag_engine.WORKSPACE_DIR / p_doc["relative_path"]
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
