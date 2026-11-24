from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import streamlit as st
from streamlit.testing.v1 import AppTest

import rag_engine

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def test_streamlit_app_startup_theme_config_and_explorer(isolated_env, monkeypatch):
    engine = isolated_env["engine"]
    docs_dir = isolated_env["docs_dir"]
    ws = isolated_env["workspace"]

    sample_doc = docs_dir / "notes.txt"
    sample_doc.write_text("Queue Processor processes Pega background items.", encoding="utf-8")
    engine.sync_database(base_dir=ws)

    st.cache_resource.clear()
    monkeypatch.setattr(rag_engine, "PegaRAGEngine", lambda db_dir=None: engine)

    at = AppTest.from_file(APP_PATH, default_timeout=20)
    at.run()
    assert not at.exception

    # Session-state initialization & defaults
    assert at.session_state["dark_mode"] is False
    assert at.session_state["show_config"] is True
    assert at.session_state["cfg_model"] == "openai/gpt-oss-120b"
    assert at.session_state["cfg_top_k"] == 4
    assert at.session_state["cfg_temperature"] == 0.0
    assert at.session_state["cfg_section_filter"] == "All Sections"
    assert at.session_state["pending_delete_doc"] is None
    assert at.session_state["pending_delete_all"] is False

    # Light/Dark mode toggle & persistence across rerun
    at.button(key="btn_theme_toggle").click().run()
    assert not at.exception
    assert at.session_state["dark_mode"] is True

    # Configuration controls & persistence across rerun
    at.slider(key="cfg_top_k").set_value(6).run()
    at.slider(key="cfg_dense_weight").set_value(0.75).run()
    at.toggle(key="cfg_use_reranker").set_value(False).run()
    at.selectbox(key="cfg_section_filter").set_value("05_Implementation_and_Deployment").run()
    assert at.session_state["cfg_top_k"] == 6
    assert at.session_state["cfg_dense_weight"] == 0.75
    assert at.session_state["cfg_use_reranker"] is False
    assert at.session_state["cfg_section_filter"] == "05_Implementation_and_Deployment"

    # Configuration drawer toggle off and back on
    at.button(key="btn_config_toggle").click().run()
    assert at.session_state["show_config"] is False
    at.button(key="btn_config_toggle").click().run()
    assert at.session_state["show_config"] is True

    # Explorer search preview
    at.text_input(key="explorer_search_input").set_value("Queue").run()
    assert not at.exception
    at.text_input(key="explorer_search_input").set_value("").run()

    # Sync Live Files button
    at.button(key="btn_sync_live_files").click().run()
    assert not at.exception
    assert "Synced:" in (at.session_state["sync_status"] or "")

    # Delete All Chunks confirmation dialog + Cancel
    at.button(key="btn_delete_all_chunks").click().run()
    assert at.session_state["pending_delete_all"] is True
    at.button(key="btn_cancel_delete_all").click().run()
    assert at.session_state["pending_delete_all"] is False
    assert engine.collection.count() == 1

    # Individual Delete confirmation dialog + Confirm Delete (keeps disk file)
    at.button(key="del_doc_0").click().run()
    assert at.session_state["pending_delete_doc"] is not None
    at.button(key="btn_confirm_doc_delete").click().run()
    assert at.session_state["pending_delete_doc"] is None
    assert engine.collection.count() == 0
    assert sample_doc.exists()

    st.cache_resource.clear()


def test_streamlit_chat_demo_question_and_clear_chat(isolated_env, monkeypatch):
    engine = isolated_env["engine"]
    docs_dir = isolated_env["docs_dir"]
    ws = isolated_env["workspace"]

    sample_doc = docs_dir / "05_Implementation_and_Deployment" / "qp.txt"
    sample_doc.parent.mkdir(parents=True, exist_ok=True)
    sample_doc.write_text("Data Pages with Keyed Access cache lookup results efficiently.", encoding="utf-8")
    engine.sync_database(base_dir=ws)

    st.cache_resource.clear()
    monkeypatch.setattr(rag_engine, "PegaRAGEngine", lambda db_dir=None: engine)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_ui")

    ok_chunk = SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content="Use Keyed Access Data Pages [qp.txt, Page 1]."), finish_reason="stop")]
    )

    with patch("rag_engine.Groq") as mock_groq_cls:
        mock_client = MagicMock()
        mock_client.models.list.return_value = SimpleNamespace(data=[SimpleNamespace(id="openai/gpt-oss-120b")])
        mock_client.chat.completions.create.return_value = iter([ok_chunk])
        mock_groq_cls.return_value = mock_client

        at = AppTest.from_file(APP_PATH, default_timeout=20)
        at.run()
        assert not at.exception

        at.button(key="demo_q_0").click().run()
        assert not at.exception
        assert len(at.session_state["messages"]) == 2
        assert at.session_state["messages"][1]["role"] == "assistant"
        assert "Keyed Access" in at.session_state["messages"][1]["content"]
        assert at.session_state["pending_response"] is False

        # Clear Chat button clears conversation history
        at.button(key="btn_clear_chat").click().run()
        assert not at.exception
        assert at.session_state["messages"] == []

    st.cache_resource.clear()


def test_sanitize_uploaded_filename_and_batch_launchers():
    from ui.upload_panel import sanitize_uploaded_filename

    assert sanitize_uploaded_filename("report.pdf") == "report.pdf"
    assert sanitize_uploaded_filename("../../etc/passwd.pdf") == "passwd.pdf"
    assert sanitize_uploaded_filename(r"..\..\Windows\evil.docx") == "evil.docx"
    assert sanitize_uploaded_filename(r"C:\Users\hp\Desktop\spec:v1?.txt") == "spec_v1_.txt"
    assert sanitize_uploaded_filename("...") == "uploaded_document"
    assert sanitize_uploaded_filename("") == "uploaded_document"

    project_root = Path(__file__).resolve().parent.parent
    launch_bat = (project_root / "LAUNCH_AI_ASSISTANT.bat").read_text(encoding="utf-8")
    downloader_bat = (project_root / "RUN_DOWNLOADER.bat").read_text(encoding="utf-8")

    assert r"PEGA-EXP-1\Scripts\python.exe" in launch_bat
    assert "streamlit run app.py" in launch_bat

    assert r"PEGA-EXP-1\Scripts\python.exe" in downloader_bat
    assert "pega_downloader.py" in downloader_bat
    assert "download_challenges.py" in downloader_bat
    assert downloader_bat.index("pega_downloader.py") < downloader_bat.index("download_challenges.py")
