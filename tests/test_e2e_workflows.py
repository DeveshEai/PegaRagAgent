from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import streamlit as st
from streamlit.testing.v1 import AppTest

import rag_engine
from conftest import create_sample_pdf

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def test_workflow_a_knowledge_ingestion_and_explorer(isolated_env, monkeypatch):
    engine = isolated_env["engine"]
    docs_dir = isolated_env["docs_dir"]
    ws = isolated_env["workspace"]

    pdf_path = create_sample_pdf(
        docs_dir / "06_Custom_Materials" / "workflow_a.pdf",
        ["Pega Constellation decouples UI views from DX API data models."],
    )
    chunks_added = engine.chunk_and_index_file(pdf_path)
    assert chunks_added == 1

    manifest = engine.load_manifest()
    rel = str(pdf_path.relative_to(ws))
    manifest[rel] = {
        "hash": rag_engine.compute_file_hash(pdf_path),
        "file_name": pdf_path.name,
        "chunks_count": chunks_added,
        "extractor_version": 1,
        "document_name": "Constellation Overview",
    }
    engine.save_manifest(manifest)

    docs = engine.list_indexed_documents()
    assert any(d["file_name"] == "workflow_a.pdf" and d["document_name"] == "Constellation Overview" for d in docs)

    hits = engine.search("DX API data models", top_k=3, use_reranker=False)
    assert any(h["source"] == "workflow_a.pdf" for h in hits)

    st.cache_resource.clear()
    monkeypatch.setattr(rag_engine, "PegaRAGEngine", lambda db_dir=None: engine)
    at = AppTest.from_file(APP_PATH, default_timeout=20)
    at.run()
    assert not at.exception
    st.cache_resource.clear()


def test_workflow_b_knowledge_synchronization_on_modified_file(isolated_env):
    engine = isolated_env["engine"]
    docs_dir = isolated_env["docs_dir"]
    ws = isolated_env["workspace"]

    doc_path = docs_dir / "sync_guide.txt"
    doc_path.write_text("Initial version covers Pega Standard Agents.", encoding="utf-8")

    sync1 = engine.sync_database(base_dir=ws)
    assert sync1["new_indexed"] == 1
    assert any("Standard Agents" in h["document"] for h in engine.search("Standard Agents", top_k=2, use_reranker=False))

    # Modify document on disk and re-sync
    doc_path.write_text("Updated version covers Kafka Queue Processors exclusively.", encoding="utf-8")
    sync2 = engine.sync_database(base_dir=ws)
    assert sync2["new_indexed"] == 1

    hits_new = engine.search("Kafka Queue Processors", top_k=2, use_reranker=False)
    assert any("Kafka Queue Processors" in h["document"] for h in hits_new)
    assert engine._ensure_bm25_retriever().search("Standard Agents", top_k=2) == []


def test_workflow_c_document_deletion_and_delete_all_ui(isolated_env, monkeypatch):
    engine = isolated_env["engine"]
    docs_dir = isolated_env["docs_dir"]
    ws = isolated_env["workspace"]

    f1 = docs_dir / "keep_me.txt"
    f2 = docs_dir / "remove_me.txt"
    f1.write_text("Enterprise Class Structure uses Work- and Data- layers.", encoding="utf-8")
    f2.write_text("Temporary obsolete notes about legacy SOAP connectors.", encoding="utf-8")
    engine.sync_database(base_dir=ws)

    orig_col_id = engine.collection.id
    del_one = engine.delete_document_chunks(relative_path=str(f2.relative_to(ws)), source_name=f2.name)
    assert del_one["status"] == "success"
    assert any(h["source"] == "keep_me.txt" for h in engine.search("Enterprise Class Structure", top_k=2, use_reranker=False))

    st.cache_resource.clear()
    monkeypatch.setattr(rag_engine, "PegaRAGEngine", lambda db_dir=None: engine)
    at = AppTest.from_file(APP_PATH, default_timeout=20)
    at.run()
    at.button(key="btn_delete_all_chunks").click().run()
    at.button(key="btn_confirm_delete_all").click().run()
    assert not at.exception
    assert engine.collection.id == orig_col_id
    assert engine.collection.count() == 0
    assert engine.load_manifest() == {}
    st.cache_resource.clear()


def test_workflow_d_grounded_chat_context_citations_and_history(isolated_env, monkeypatch):
    engine = isolated_env["engine"]
    docs_dir = isolated_env["docs_dir"]
    ws = isolated_env["workspace"]

    pdf_path = create_sample_pdf(
        docs_dir / "05_Implementation_and_Deployment" / "qp_guide.pdf",
        ["Queue Processor rules process queued items asynchronously on BackgroundProcessing nodes."],
    )
    engine.sync_database(base_dir=ws)

    st.cache_resource.clear()
    monkeypatch.setattr(rag_engine, "PegaRAGEngine", lambda db_dir=None: engine)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_workflow_d")

    captured_messages = []

    def _fake_create(**kwargs):
        captured_messages.append(kwargs.get("messages", []))
        return iter([
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content="Queue Processor rules process queued items asynchronously [qp_guide.pdf, Page 1]."
                        ),
                        finish_reason="stop",
                    )
                ]
            )
        ])

    with patch("rag_engine.Groq") as mock_groq_cls:
        mock_client = MagicMock()
        mock_client.models.list.return_value = SimpleNamespace(data=[SimpleNamespace(id="openai/gpt-oss-120b")])
        mock_client.chat.completions.create.side_effect = _fake_create
        mock_groq_cls.return_value = mock_client

        at = AppTest.from_file(APP_PATH, default_timeout=20)
        at.run()
        at.button(key="demo_q_2").click().run()
        assert not at.exception

        assert len(captured_messages) == 1
        user_prompt_sent = captured_messages[0][-1]["content"]
        assert "--- SOURCE [1] ---" in user_prompt_sent
        assert "qp_guide.pdf" in user_prompt_sent

        assert len(at.session_state["messages"]) == 2
        assistant_msg = at.session_state["messages"][1]
        assert "[qp_guide.pdf, Page 1]" in assistant_msg["content"]
        assert len(assistant_msg["sources"]) >= 1

    st.cache_resource.clear()
