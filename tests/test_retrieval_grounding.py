from unittest.mock import MagicMock

import rag_engine


def test_tokenize_pega_text_preserves_identifiers_and_subtokens():
    tokens = rag_engine.tokenize_pega_text("Document: a.pdf\nContent:\npyWorkPage Rule-Obj-Class D_pxGetPage")
    assert "pyworkpage" in tokens
    assert "rule-obj-class" in tokens
    assert "rule" in tokens and "obj" in tokens and "class" in tokens
    assert "d_pxgetpage" in tokens


def test_bm25_retriever_search_and_filters():
    docs = [
        "Document: a.pdf\nContent:\npyWorkPage holds active case data.",
        "Document: b.pdf\nContent:\nQueue Processor handles async jobs.",
        "Document: c.pdf\nContent:\nJob Scheduler runs recurring tasks.",
    ]
    metas = [
        {"source": "a.pdf", "page": "1", "section": "02_Planning", "category": "Master Module"},
        {"source": "b.pdf", "page": "2", "section": "05_Impl", "category": "Master Module"},
        {"source": "c.pdf", "page": "3", "section": "05_Impl", "category": "Hands-on Challenge"},
    ]
    bm25 = rag_engine.BM25Retriever(docs, metas)

    hits = bm25.search("pyWorkPage", top_k=2)
    assert len(hits) == 1
    assert hits[0]["source"] == "a.pdf"

    filtered = bm25.search("async recurring", top_k=5, section_filter="05_Impl", category_filter="Hands-on Challenge")
    assert len(filtered) == 1
    assert filtered[0]["source"] == "c.pdf"


def test_compute_similarity_score_and_rrf():
    cos_exact = rag_engine.compute_similarity_score(0.0, "cosine")
    cos_far = rag_engine.compute_similarity_score(1.0, "cosine")
    assert cos_exact == 100.0
    assert 0.0 <= cos_far < cos_exact

    l2_close = rag_engine.compute_similarity_score(0.1, "l2")
    l2_far = rag_engine.compute_similarity_score(2.0, "l2")
    assert 0.0 <= l2_far < l2_close <= 100.0

    dense = [
        {"source": "a.pdf", "page": "1", "document": "Doc A", "score": 90.0},
        {"source": "b.pdf", "page": "1", "document": "Doc B", "score": 80.0},
    ]
    sparse = [
        {"source": "b.pdf", "page": "1", "document": "Doc B", "bm25_score": 12.0},
        {"source": "c.pdf", "page": "1", "document": "Doc C", "bm25_score": 8.0},
    ]
    fused = rag_engine.reciprocal_rank_fusion(dense, sparse, rrf_k=60, top_k=5, dense_weight=0.5)
    assert [item["source"] for item in fused] == ["b.pdf", "a.pdf", "c.pdf"]


def test_cross_encoder_reranker_fallbacks():
    candidates = [
        {"source": "a.pdf", "page": "1", "document": "First"},
        {"source": "b.pdf", "page": "2", "document": "Second"},
    ]
    disabled_reranker = rag_engine.CrossEncoderReranker(enabled=False)
    assert disabled_reranker.rerank("query", candidates, top_k=1) == candidates[:1]

    broken_reranker = rag_engine.CrossEncoderReranker(enabled=True)
    broken_reranker._load_model = lambda: True
    broken_reranker._model = MagicMock()
    broken_reranker._model.predict.side_effect = RuntimeError("scoring error")
    assert broken_reranker.rerank("query", candidates, top_k=2) == candidates


def test_empty_collection_search_returns_empty(isolated_env):
    engine = isolated_env["engine"]
    assert engine._dense_search("pyWorkPage", top_k=5) == []
    assert engine.search("pyWorkPage", top_k=5) == []


def test_deduplicate_sources_clean_chunk_and_context_prompt():
    prefix = "Document: r.docx\nSection: General | Category: Master Module | Page: Table\nContent:\n" + ("X" * 110)
    s1 = {"source": "r.docx", "page": "Table", "section": "General", "category": "Master Module", "document": prefix + " Row 1"}
    s2 = {"source": "r.docx", "page": "Table", "section": "General", "category": "Master Module", "document": prefix + " Row 2"}
    deduped = rag_engine.deduplicate_sources([s1, s2, s1])
    assert len(deduped) == 2
    assert rag_engine.clean_chunk_content(s1["document"]).startswith("X")

    ctx, retained = rag_engine.build_grounded_context_prompt([s1, s2], max_token_budget=5000)
    assert len(retained) == 2
    assert "--- SOURCE [1] ---" in ctx
    assert "--- SOURCE [2] ---" in ctx


def test_verify_citations_reference_validation_and_contextualize():
    sources = [
        {
            "source": "01_bg.pdf",
            "page": "11",
            "document": "Queue Processor processes work items asynchronously via Kafka topics.",
        }
    ]
    # Valid reference retained, fabricated reference stripped
    text = "Queue Processor runs asynchronously [01_bg.pdf, Page 11]. Unrelated claim [fake.pdf, Page 99]."
    cleaned, stats = rag_engine.verify_citations(text, sources)
    assert "[01_bg.pdf, Page 11]" in cleaned
    assert "[fake.pdf, Page 99]" not in cleaned
    assert stats["verified"] == 1
    assert stats["removed"] == 1

    # Stage 4 defect fix: preserve Markdown hard line breaks, indentation, paragraphs, and tables
    md_answer = (
        "Summary line  \n"
        "Next line after hard break.\n\n"
        "- Parent bullet [01_bg.pdf, Page 11]\n"
        "  - Indented sub-bullet\n"
        "1. Numbered step [fake.pdf, Page 99]\n\n"
        "| Col1 | Col2 |\n"
        "| --- | --- |\n"
        "| Val1 | Val2 |"
    )
    md_cleaned, md_stats = rag_engine.verify_citations(md_answer, sources)
    assert "Summary line  \nNext line" in md_cleaned
    assert "\n  - Indented sub-bullet" in md_cleaned
    assert "1. Numbered step\n\n| Col1 | Col2 |" in md_cleaned
    assert md_stats["verified"] == 1
    assert md_stats["removed"] == 1

    # Contextualize short follow-up vs empty history
    assert rag_engine.contextualize_query_with_history("What is pyWorkPage?", []) == "What is pyWorkPage?"
    rewritten = rag_engine.contextualize_query_with_history(
        "How does it handle retries?",
        [{"role": "user", "content": "Explain Queue Processors in Pega."}],
    )
    assert "Queue Processors" in rewritten

    # Stage 1 defect fix: 'its' / 'their' resolution and multi-turn subject lookup
    rewritten_its = rag_engine.contextualize_query_with_history(
        "What about its caching strategy?",
        [
            {"role": "user", "content": "Explain Data Pages in Pega."},
            {"role": "assistant", "content": "Data Pages load and cache data on demand."},
        ],
    )
    assert "Data Pages" in rewritten_its
    assert "caching strategy" in rewritten_its

    # Multi-turn follow-up skips bare pronoun follow-up to find the real subject
    rewritten_multi = rag_engine.contextualize_query_with_history(
        "What about their refresh strategy?",
        [
            {"role": "user", "content": "Explain Data Pages in Pega."},
            {"role": "assistant", "content": "Data Pages cache data across scopes."},
            {"role": "user", "content": "What about its scopes?"},
            {"role": "assistant", "content": "Thread, Requestor, and Node scopes."},
        ],
    )
    assert "Data Pages" in rewritten_multi

    # Standalone long technical query without pronouns is NOT polluted by old history
    standalone = "Explain how Queue Processors process delayed items in Pega background processing."
    assert rag_engine.contextualize_query_with_history(
        standalone,
        [{"role": "user", "content": "Explain Data Pages in Pega."}],
    ) == standalone


def test_retrieval_modules_reexport_identity_and_independence():
    import subprocess
    import sys
    from retrieval import grounding, hybrid

    for name in (
        "tokenize_pega_text",
        "BM25Retriever",
        "reciprocal_rank_fusion",
        "CrossEncoderReranker",
        "compute_similarity_score",
    ):
        assert getattr(hybrid, name) is getattr(rag_engine, name)

    for name in (
        "estimate_tokens",
        "clean_chunk_content",
        "deduplicate_sources",
        "build_grounded_context_prompt",
        "verify_citations",
        "contextualize_query_with_history",
    ):
        assert getattr(grounding, name) is getattr(rag_engine, name)

    for mod_name in ("retrieval.hybrid", "retrieval.grounding"):
        check_code = (
            f"import sys; sys.path.insert(0, 'PegaRagAgent'); "
            f"import {mod_name}; "
            "forbidden = {'rag_engine', 'app', 'streamlit', 'chromadb', 'groq'}; "
            "assert forbidden.isdisjoint(sys.modules.keys()), f'Leaked imports: {forbidden & sys.modules.keys()}'"
        )
        proc = subprocess.run([sys.executable, "-c", check_code], capture_output=True, text=True, check=True)
        assert proc.returncode == 0
