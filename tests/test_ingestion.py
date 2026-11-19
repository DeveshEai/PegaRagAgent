import rag_engine
from conftest import create_sample_docx, create_sample_pdf


def test_extract_pdf_pages_preserves_page_boundaries(tmp_path):
    pdf_path = create_sample_pdf(
        tmp_path / "sample.pdf",
        ["Page one Pega Enterprise Class Structure.", "Page two Queue Processor details."],
    )
    pages = rag_engine.extract_pdf_pages(pdf_path)
    assert len(pages) == 2
    assert pages[0]["page_number"] == 1
    assert "Enterprise Class Structure" in pages[0]["text"]
    assert pages[1]["page_number"] == 2
    assert "Queue Processor" in pages[1]["text"]


def test_extract_docx_sections_tables_headings_and_empty_table(tmp_path):
    docx_path = create_sample_docx(
        tmp_path / "report.docx",
        include_empty_table=True,
        include_large_table=False,
    )
    sections = rag_engine.extract_docx_sections(docx_path)
    # Section 1 (prose), Section 2 (heading-only + table), Section 3 (empty table -> skipped)
    assert [s["page_number"] for s in sections] == ["1. Executive Overview", "2. Regional Revenue"]
    assert sections[1]["heading_path"] == "1. Executive Overview > 2. Regional Revenue"
    assert sections[1]["blocks"][0]["type"] == "table"
    assert "| Region | Target | Actual |" in sections[1]["blocks"][0]["markdown"]
    assert "| UAE | $1,800,000 | $1,800,000 |" in sections[1]["blocks"][0]["markdown"]


def test_oversized_docx_table_splits_at_row_boundaries_with_repeated_headers(isolated_env):
    engine = isolated_env["engine"]
    docs_dir = isolated_env["docs_dir"]
    docx_path = create_sample_docx(
        docs_dir / "large_report.docx",
        include_empty_table=False,
        include_large_table=True,
    )

    total_added = engine.chunk_and_index_file(docx_path)
    assert total_added >= 3

    stored = engine.collection.get(where={"source": docx_path.name}, include=["documents", "metadatas"])
    large_table_docs = [
        (doc, meta)
        for doc, meta in zip(stored["documents"], stored["metadatas"])
        if meta.get("content_type") == "table" and meta.get("page") == "4. Large Budget Table"
    ]
    assert len(large_table_docs) >= 2
    for doc_txt, _ in large_table_docs:
        assert "| DepartmentCode | ApprovedBudget | ActualSpend |" in doc_txt


def test_hashing_incremental_sync_extractor_version_and_idempotency(isolated_env):
    engine = isolated_env["engine"]
    ws = isolated_env["workspace"]
    docs_dir = isolated_env["docs_dir"]

    pdf_path = create_sample_pdf(docs_dir / "01_LSA_Mindset" / "guide.pdf", ["LSA mindset page 1."])
    docx_path = create_sample_docx(docs_dir / "06_Custom_Materials" / "fin.docx", include_empty_table=False)

    sha = rag_engine.compute_file_hash(pdf_path)
    assert len(sha) == 64

    sync1 = engine.sync_database(base_dir=ws)
    assert sync1["new_indexed"] == 2
    assert sync1["skipped"] == 0

    # Simulate legacy MD5 manifest on PDF and outdated extractor_version=1 on DOCX with custom document_name
    manifest = engine.load_manifest()
    pdf_rel = str(pdf_path.relative_to(ws))
    docx_rel = str(docx_path.relative_to(ws))
    manifest[pdf_rel]["hash"] = rag_engine._legacy_md5_hash(pdf_path)
    manifest[docx_rel]["extractor_version"] = 1
    manifest[docx_rel]["document_name"] = "Custom Finance Title"
    engine.save_manifest(manifest)

    sync2 = engine.sync_database(base_dir=ws)
    assert sync2["skipped"] == 1  # PDF upgraded MD5->SHA256 without re-indexing
    assert sync2["reindexed"] == 1  # DOCX selectively re-indexed
    updated_manifest = engine.load_manifest()
    assert updated_manifest[pdf_rel]["hash"] == sha
    assert updated_manifest[docx_rel]["extractor_version"] == rag_engine.DOCX_EXTRACTOR_VERSION
    assert updated_manifest[docx_rel]["document_name"] == "Custom Finance Title"

    # Repeated sync is idempotent
    sync3 = engine.sync_database(base_dir=ws)
    assert sync3["new_indexed"] == 0
    assert sync3["reindexed"] == 0
    assert sync3["skipped"] == 2
    assert sync3["total_chunks_in_db"] == sync2["total_chunks_in_db"]


def test_ingestion_extractors_reexport_identity_and_independence():
    import subprocess
    import sys
    from ingestion import extractors

    assert extractors.extract_pdf_pages is rag_engine.extract_pdf_pages
    assert extractors.extract_docx_sections is rag_engine.extract_docx_sections
    assert extractors._normalize_docx_table_markdown is rag_engine._normalize_docx_table_markdown
    assert extractors._split_markdown_table_by_rows is rag_engine._split_markdown_table_by_rows
    assert extractors.compute_file_hash is rag_engine.compute_file_hash
    assert extractors._legacy_md5_hash is rag_engine._legacy_md5_hash

    check_code = (
        "import sys; sys.path.insert(0, 'PegaRagAgent'); "
        "import ingestion.extractors; "
        "forbidden = {'rag_engine', 'app', 'streamlit', 'chromadb', 'groq'}; "
        "assert forbidden.isdisjoint(sys.modules.keys()), f'Leaked imports: {forbidden & sys.modules.keys()}'"
    )
    proc = subprocess.run([sys.executable, "-c", check_code], capture_output=True, text=True, check=True)
    assert proc.returncode == 0
