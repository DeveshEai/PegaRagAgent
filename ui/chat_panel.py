"""Chat history, citation badges, evidence inspector, and streaming panel."""

import html
from datetime import datetime
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from rag_engine import (
    ENABLE_ADAPTIVE_ROUTING,
    clean_chunk_content,
    contextualize_query_with_history,
    deduplicate_sources,
    detect_output_format,
)
from ui.state import queue_user_query
from ui.styles import LUCIDE_ICONS


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
    return f'<div class="citation-row" style="margin-top: 10px; margin-bottom: 6px;">{"".join(badges)}</div>'


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


def render_execution_details(exec_meta: Any, has_visible_sources: bool = False) -> None:
    """Render an expandable execution-metadata section from actual pipeline outputs."""
    if not isinstance(exec_meta, dict) or not exec_meta:
        return
    strategy = str(exec_meta.get("execution_strategy") or "Direct RAG")
    with st.expander("Execution Details", expanded=False):
        if strategy == "Conversation":
            st.caption("Execution strategy: Conversation")
            st.caption("Retrieval: Not performed")
            st.caption("Citations: Not applicable")
            return

        intent = exec_meta.get("detected_intent", "UNKNOWN")
        router_model = exec_meta.get("router_model", "deterministic")
        evidence_status = exec_meta.get("evidence_status", "Unknown")
        cite_ver = exec_meta.get("citation_verification", "Not applicable")
        if isinstance(cite_ver, dict):
            cite_str = f"verified={cite_ver.get('verified', 0)}, removed={cite_ver.get('removed', 0)}, mismatched={cite_ver.get('mismatched_claims', 0)}"
        else:
            cite_str = str(cite_ver)

        docs = exec_meta.get("documents_consulted") or []
        if has_visible_sources and docs:
            doc_str = f"{len(docs)} unique location(s) (listed in Sources above)"
        else:
            doc_str = ", ".join(f"{d.get('source')} (p.{d.get('page')})" for d in docs) if docs else "None"

        st.caption(f"Detected intent: `{intent}` · Format: `{exec_meta.get('output_format', 'PARAGRAPH')}`")
        st.caption(f"Execution strategy: `{strategy}` · Router: `{router_model}`")
        if strategy == "Autonomous CLSA Agent":
            tools = exec_meta.get("tools_used") or []
            tools_str = ", ".join(tools) if tools else "None"
            st.caption(f"Research tools used: {tools_str} · Searches: {exec_meta.get('search_count', 0)}")
            st.caption(f"Escalation: {exec_meta.get('escalation_status', 'Not escalated')} · Evidence: {evidence_status}")
            st.caption(f"Citation verification: {cite_str}")
            st.caption(f"Documents & pages consulted: {doc_str}")
        else:
            st.caption(f"Evidence sufficiency: {evidence_status}")
            st.caption(f"Citation verification: {cite_str}")
            st.caption(f"Retrieved documents & pages: {doc_str}")


def stream_assistant_turn(engine: Any, query_text: str) -> None:
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
        final_exec_meta = None

        try:
            adaptive_enabled = bool(st.session_state.get("cfg_adaptive_routing", ENABLE_ADAPTIVE_ROUTING))
            if adaptive_enabled:
                exec_mode = "adaptive" if mode.startswith("Autonomous") else "direct"
                event_stream = engine.run_adaptive_stream(
                    query=query_text,
                    groq_api_key=api_key,
                    model_name=model_name,
                    temperature=temperature,
                    max_iterations=4,
                    chat_history=prior_history,
                    top_k=top_k,
                    use_reranker=use_reranker,
                    dense_weight=dense_weight,
                    section_filter=section_filter,
                    execution_mode=exec_mode,
                )
            elif mode.startswith("Autonomous"):
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
                    conv = engine.detect_conversational_intent(query_text) if hasattr(engine, "detect_conversational_intent") else None
                    if conv is not None:
                        yield {
                            "type": "final",
                            "answer": conv["answer"],
                            "sources": [],
                            "execution_meta": {
                                "execution_strategy": "Conversation",
                                "detected_intent": "CONVERSATIONAL",
                                "output_format": "Not applicable",
                                "router_model": "deterministic",
                                "evidence_status": "Not applicable",
                                "escalation_status": "Not escalated",
                                "tools_used": [],
                                "search_count": 0,
                                "documents_consulted": [],
                                "citation_verification": "Not applicable",
                                "retrieval_performed": False,
                            },
                        }
                        return
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
                            "sources": [],
                            "execution_meta": {
                                "execution_strategy": "Direct RAG",
                                "detected_intent": "TECHNICAL_EXPLANATION",
                                "output_format": detect_output_format(query_text),
                                "router_model": "manual",
                                "evidence_status": "Insufficient",
                                "escalation_status": "Not escalated",
                                "tools_used": [],
                                "search_count": 1,
                                "documents_consulted": [],
                                "citation_verification": {"verified": 0, "removed": 0, "mismatched_claims": 0},
                                "retrieval_performed": True,
                            },
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
                        if ev.get("type") == "final":
                            ev = {
                                **ev,
                                "execution_meta": {
                                    "execution_strategy": "Direct RAG",
                                    "detected_intent": "TECHNICAL_EXPLANATION",
                                    "output_format": ev.get("output_format") or detect_output_format(query_text),
                                    "router_model": "manual",
                                    "evidence_status": "Sufficient",
                                    "escalation_status": "Not escalated",
                                    "tools_used": [],
                                    "search_count": 1,
                                    "documents_consulted": [
                                        {"source": s.get("source"), "page": s.get("page"), "section": s.get("section")}
                                        for s in (ev.get("sources") or [])
                                    ],
                                    "citation_verification": ev.get("citation_stats") or {"verified": 0, "removed": 0, "mismatched_claims": 0},
                                    "retrieval_performed": True,
                                },
                            }
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
                    final_exec_meta = ev.get("execution_meta")
                    stream_ph.markdown(final_answer)
        except Exception as exc:
            status_ph.empty()
            final_answer = f"❌ **Streaming Error:** {str(exc)}"
            stream_ph.markdown(final_answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": final_answer or accumulated_text or "No response generated.",
        "sources": final_sources,
        "execution_meta": final_exec_meta,
        "timestamp": datetime.now().strftime("%H:%M")
    })
    st.session_state.pending_response = False
    st.session_state.pending_query = None
    st.session_state.scroll_to_bottom = True
    st.rerun()


def render_chat_panel(engine: Any) -> None:
    """Renders the left Bento Ask Architect (Q&A) chat card."""
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
                        render_execution_details(msg.get("execution_meta"), has_visible_sources=bool(srcs))
                        if ts:
                            st.markdown(f'<div class="msg-meta">{ts}</div>', unsafe_allow_html=True)

            if st.session_state.pending_response and st.session_state.pending_query:
                stream_assistant_turn(engine, st.session_state.pending_query)

            st.markdown('<div id="chat-box-end-anchor"></div>', unsafe_allow_html=True)

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
