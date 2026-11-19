from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import rag_engine
from conftest import create_sample_pdf


def test_agent_tools_schema_and_local_tool_execution(isolated_env):
    engine = isolated_env["engine"]
    ws = isolated_env["workspace"]
    docs_dir = isolated_env["docs_dir"]

    schema = engine.get_agent_tools_schema()
    names = {item["function"]["name"] for item in schema}
    assert names == {"search_curriculum", "search_challenges", "read_full_page", "lookup_pega_terms"}

    create_sample_pdf(
        docs_dir / "05_Implementation_and_Deployment" / "01_bg.pdf",
        ["Page 1 Queue Processor intro.", "Page 2 pyWorkPage locking details."],
    )
    create_sample_pdf(
        docs_dir / "Challenges" / "challenge_01.pdf",
        ["Challenge step 1 configure Queue Processor."],
    )
    engine.sync_database(base_dir=ws)

    r1 = engine.tool_search_curriculum("Queue Processor")
    r2 = engine.tool_search_challenges("Queue Processor")
    r3 = engine.tool_read_full_page("01_bg.pdf", 2)
    r4 = engine.tool_lookup_pega_terms("pyWorkPage")

    assert r1["status"] == "success" and r1["results_count"] >= 1
    assert r2["status"] == "success" and r2["results_count"] >= 1
    assert r3["status"] == "success" and "pyWorkPage" in r3["full_page_text"]
    assert r4["status"] == "success" and len(r4["results"]) >= 1


def test_stream_answer_tokens_completeness_guard_and_model_fallback(isolated_env):
    engine = isolated_env["engine"]
    chunks = [{"source": "01_bg.pdf", "page": "1", "section": "05", "document": "Queue Processor runs async.", "score": 95.0}]

    chunk_1 = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="First sentence. "), finish_reason=None)])
    chunk_2 = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Cut off mid"), finish_reason="length")])

    with patch("rag_engine.Groq") as mock_groq_cls:
        mock_client = MagicMock()
        mock_client.models.list.return_value = SimpleNamespace(data=[SimpleNamespace(id="openai/gpt-oss-120b")])
        mock_client.chat.completions.create.return_value = iter([chunk_1, chunk_2])
        mock_groq_cls.return_value = mock_client

        events = list(engine.stream_answer("Explain Queue Processor", chunks, groq_api_key="gsk_test"))
        final_ev = [e for e in events if e["type"] == "final"][-1]
        assert any(e["type"] == "token" for e in events)
        assert "First sentence." in final_ev["answer"]
        assert "output length limit" in final_ev["answer"]

    # Model fallback from unavailable primary to openai/gpt-oss-20b
    ok_chunk = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Fallback ok."), finish_reason="stop")])
    with patch("rag_engine.Groq") as mock_groq_cls:
        mock_client = MagicMock()
        mock_client.models.list.return_value = SimpleNamespace(data=[SimpleNamespace(id="openai/gpt-oss-20b")])
        mock_client.chat.completions.create.return_value = iter([ok_chunk])
        mock_groq_cls.return_value = mock_client

        events = list(engine.stream_answer("Explain Queue Processor", chunks, groq_api_key="gsk_test", model_name="openai/gpt-oss-120b"))
        final_ev = [e for e in events if e["type"] == "final"][-1]
        assert final_ev["model_used"] == "openai/gpt-oss-20b"
        assert "Fallback ok." in final_ev["answer"]


def test_run_agent_stream_null_tool_args_and_non_tool_fallback(isolated_env, monkeypatch):
    engine = isolated_env["engine"]
    docs_dir = isolated_env["docs_dir"]
    f = docs_dir / "notes.txt"
    f.write_text("Queue Processor async details.", encoding="utf-8")
    engine.chunk_and_index_file(f)

    monkeypatch.setattr(rag_engine, "should_use_agent_loop", lambda q: (True, "forced for test"))

    # 1. Tool call with arguments == "null"
    null_tc = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="search_curriculum", arguments="null"),
    )
    step1 = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[null_tc], content=None), finish_reason="tool_calls")])
    step2 = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None, content="Handled null args."), finish_reason="stop")])

    with patch("rag_engine.Groq") as mock_groq_cls:
        mock_client = MagicMock()
        mock_client.models.list.return_value = SimpleNamespace(data=[SimpleNamespace(id="openai/gpt-oss-120b")])
        mock_client.chat.completions.create.side_effect = [step1, step2]
        mock_groq_cls.return_value = mock_client

        evs = list(engine.run_agent_stream("Explain Queue Processor", groq_api_key="gsk_test"))
        final_ev = [e for e in evs if e["type"] == "final"][-1]
        assert "Handled null args." in final_ev["answer"]

    # 2. Non-tool-capable model falls back to Direct Grounded Streaming
    direct_chunk = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Direct grounded answer."), finish_reason="stop")])
    with patch("rag_engine.Groq") as mock_groq_cls:
        mock_client = MagicMock()
        mock_client.models.list.return_value = SimpleNamespace(data=[SimpleNamespace(id="allam-2-7b")])
        mock_client.chat.completions.create.return_value = iter([direct_chunk])
        mock_groq_cls.return_value = mock_client

        evs = list(engine.run_agent_stream("Explain Queue Processor", groq_api_key="gsk_test", model_name="allam-2-7b"))
        final_ev = [e for e in evs if e["type"] == "final"][-1]
        assert "Direct grounded answer." in final_ev["answer"]


def test_stream_answer_context_length_retry_recovery(isolated_env):
    engine = isolated_env["engine"]
    chunks = [{"source": "01_bg.pdf", "page": "1", "section": "05", "document": "Queue Processor runs async.", "score": 95.0}]
    retry_chunk = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Recovered answer."), finish_reason="stop")])

    with patch("rag_engine.Groq") as mock_groq_cls:
        mock_client = MagicMock()
        mock_client.models.list.return_value = SimpleNamespace(data=[SimpleNamespace(id="openai/gpt-oss-120b")])
        mock_client.chat.completions.create.side_effect = [
            RuntimeError("context_length_exceeded: reduce the length of the messages"),
            iter([retry_chunk]),
        ]
        mock_groq_cls.return_value = mock_client

        evs = list(engine.stream_answer("Explain Queue Processor", chunks, groq_api_key="gsk_test"))
        final_ev = [e for e in evs if e["type"] == "final"][-1]
        assert "Recovered answer." in final_ev["answer"]
        assert mock_client.chat.completions.create.call_count == 2


def test_agents_modules_reexport_identity_and_independence():
    import subprocess
    import sys
    from agents import answer_generator, research_agent, tools

    assert answer_generator.GROUNDED_SYSTEM_PROMPT is rag_engine.GROUNDED_SYSTEM_PROMPT
    assert answer_generator.is_context_length_error is rag_engine.is_context_length_error
    assert research_agent.AGENT_SYSTEM_PROMPT is rag_engine.AGENT_SYSTEM_PROMPT
    assert research_agent.should_use_agent_loop is rag_engine.should_use_agent_loop
    assert tools.get_agent_tools_schema() == rag_engine.PegaRAGEngine.get_agent_tools_schema()

    for mod_name in ("agents.tools", "agents.answer_generator", "agents.research_agent"):
        check_code = (
            f"import sys; sys.path.insert(0, 'PegaRagAgent'); "
            f"import {mod_name}; "
            "forbidden = {'rag_engine', 'app', 'streamlit', 'chromadb'}; "
            "assert forbidden.isdisjoint(sys.modules.keys()), f'Leaked imports: {forbidden & sys.modules.keys()}'"
        )
        proc = subprocess.run([sys.executable, "-c", check_code], capture_output=True, text=True, check=True)
        assert proc.returncode == 0
