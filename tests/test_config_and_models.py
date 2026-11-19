from types import SimpleNamespace
from unittest.mock import MagicMock

import rag_engine


def test_model_registry_contains_primary_and_fallback():
    assert rag_engine.PRIMARY_GROQ_MODEL == "openai/gpt-oss-120b"
    assert rag_engine.FALLBACK_GROQ_MODEL == "openai/gpt-oss-20b"
    assert rag_engine.PRIMARY_GROQ_MODEL in rag_engine.MODEL_REGISTRY
    assert rag_engine.FALLBACK_GROQ_MODEL in rag_engine.MODEL_REGISTRY

    retired_ids = {
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "deepseek-r1-distill-llama-70b",
    }
    assert retired_ids.isdisjoint(rag_engine.MODEL_REGISTRY.keys())


def test_get_model_spec_and_tool_capability():
    primary_spec = rag_engine.get_model_spec("openai/gpt-oss-120b")
    fallback_spec = rag_engine.get_model_spec("openai/gpt-oss-20b")
    assert primary_spec["context_window"] == 131072
    assert fallback_spec["context_window"] == 131072
    assert rag_engine.is_tool_capable_model("openai/gpt-oss-120b") is True
    assert rag_engine.is_tool_capable_model("openai/gpt-oss-20b") is True

    empty_spec = rag_engine.get_model_spec("")
    assert empty_spec == primary_spec

    unknown_spec = rag_engine.get_model_spec("some-unregistered-model")
    assert unknown_spec == rag_engine.DEFAULT_UNKNOWN_SPEC
    assert rag_engine.is_tool_capable_model("some-unregistered-model") is False
    assert rag_engine.is_tool_capable_model("allam-2-7b") is False


def test_resolve_available_model_uses_mocked_groq_models(isolated_env):
    engine = isolated_env["engine"]
    mock_client = MagicMock()
    mock_client.models.list.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(id="whisper-large-v3"),
            SimpleNamespace(id="openai/gpt-oss-120b"),
            SimpleNamespace(id="openai/gpt-oss-20b"),
            SimpleNamespace(id="qwen/qwen3.8-27b"),
        ]
    )

    resolved, notice = engine.resolve_available_model(mock_client, "openai/gpt-oss-120b")
    assert resolved == "openai/gpt-oss-120b"
    assert notice is None

    # Retired model falls back to openai/gpt-oss-20b
    fb_model, fb_notice = engine.resolve_available_model(mock_client, "llama-3.3-70b-versatile")
    assert fb_model == "openai/gpt-oss-20b"
    assert fb_notice is not None and "openai/gpt-oss-20b" in fb_notice


def test_find_best_alternative_model_skips_current_and_audio(isolated_env):
    engine = isolated_env["engine"]
    mock_client = MagicMock()
    mock_client.models.list.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(id="whisper-large-v3"),
            SimpleNamespace(id="canopylabs/orpheus-v1-english"),
            SimpleNamespace(id="openai/gpt-oss-safeguard-20b"),
            SimpleNamespace(id="openai/gpt-oss-120b"),
            SimpleNamespace(id="openai/gpt-oss-20b"),
        ]
    )

    alt = engine.find_best_alternative_model(mock_client, "openai/gpt-oss-120b")
    assert alt == "openai/gpt-oss-20b"

    alt_from_20b = engine.find_best_alternative_model(mock_client, "openai/gpt-oss-20b")
    assert alt_from_20b == "openai/gpt-oss-120b"


def test_config_settings_reexport_identity_paths_and_independence():
    import subprocess
    import sys
    from config import settings

    assert settings.MODEL_REGISTRY is rag_engine.MODEL_REGISTRY
    assert settings.get_model_spec is rag_engine.get_model_spec
    assert settings.is_tool_capable_model is rag_engine.is_tool_capable_model
    assert settings.WORKSPACE_DIR == rag_engine.WORKSPACE_DIR
    assert settings.DOCS_DIR == rag_engine.DOCS_DIR
    assert settings.DB_DIR == rag_engine.DB_DIR
    assert settings.MANIFEST_PATH == rag_engine.MANIFEST_PATH

    assert settings.COLLECTION_NAME == "pega_lsa_curriculum"
    assert settings.EMBEDDING_MODEL_NAME == "all-MiniLM-L6-v2"
    assert settings.CHUNK_SIZE == 950
    assert settings.CHUNK_OVERLAP == 180
    assert settings.DOCX_EXTRACTOR_VERSION == 2
    assert settings.TABLE_ATOMIC_CHAR_LIMIT == 1800
    assert settings.RRF_K == 60
    assert settings.BM25_K1 == 1.5
    assert settings.BM25_B == 0.75

    # Verify config.settings imports cleanly in a fresh Python process without heavy runtime libs
    check_code = (
        "import sys; sys.path.insert(0, 'PegaRagAgent'); "
        "import config.settings; "
        "forbidden = {'rag_engine', 'app', 'streamlit', 'chromadb', 'groq'}; "
        "assert forbidden.isdisjoint(sys.modules.keys()), f'Leaked imports: {forbidden & sys.modules.keys()}'"
    )
    proc = subprocess.run([sys.executable, "-c", check_code], capture_output=True, text=True, check=True)
    assert proc.returncode == 0
