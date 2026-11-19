from conftest import create_sample_docx


def test_list_and_delete_individual_document(isolated_env):
    engine = isolated_env["engine"]
    ws = isolated_env["workspace"]
    docs_dir = isolated_env["docs_dir"]

    doc_a = docs_dir / "alpha.txt"
    doc_b = docs_dir / "beta.txt"
    doc_a.write_text("Alpha Pega Queue Processor notes.", encoding="utf-8")
    doc_b.write_text("Beta Pega Job Scheduler notes.", encoding="utf-8")
    engine.sync_database(base_dir=ws)

    listed = engine.list_indexed_documents()
    assert len(listed) == 2
    assert {d["file_name"] for d in listed} == {"alpha.txt", "beta.txt"}
    assert all("section" in d and "document_name" in d for d in listed)

    rel_b = str(doc_b.relative_to(ws))
    del_res = engine.delete_document_chunks(relative_path=rel_b, source_name="beta.txt")
    assert del_res["status"] == "success"
    assert del_res["deleted_chunks"] == 1
    assert del_res["remaining_chunks_in_db"] == 1
    assert doc_b.exists()  # Disk file untouched


def test_delete_all_chunks_keeps_collection_and_clears_bm25(isolated_env):
    engine = isolated_env["engine"]
    ws = isolated_env["workspace"]
    docs_dir = isolated_env["docs_dir"]

    f1 = create_sample_docx(docs_dir / "fin.docx", include_empty_table=False)
    outside = ws / "outside_file.txt"
    outside.write_text("Outside file must remain untouched.", encoding="utf-8")
    engine.sync_database(base_dir=ws)

    orig_col_id = engine.collection.id
    assert engine.collection.count() >= 2

    # Keep disk files
    res_keep = engine.delete_all_chunks(delete_disk_files=False, docs_dir=docs_dir, workspace_dir=ws)
    assert res_keep["status"] == "success"
    assert res_keep["remaining_chunks"] == 0
    assert res_keep["deleted_disk_files"] == 0
    assert engine.collection.id == orig_col_id
    assert engine.load_manifest() == {}
    assert engine._bm25_retriever is None
    assert engine.search("Regional Revenue", top_k=3) == []
    assert f1.exists() and outside.exists()

    # Already empty collection
    res_empty = engine.delete_all_chunks(delete_disk_files=False, docs_dir=docs_dir, workspace_dir=ws)
    assert res_empty["status"] == "success"
    assert res_empty["deleted_chunks"] == 0

    # Re-index new file and verify no stale BM25 results from deleted fin.docx
    f2 = docs_dir / "new_doc.txt"
    f2.write_text("pyWorkPage locking behavior.", encoding="utf-8")
    engine.chunk_and_index_file(f2)
    stale_hits = engine.search("Regional Revenue", top_k=3, use_reranker=False)
    assert not any(h["source"] == "fin.docx" for h in stale_hits)
    assert engine._ensure_bm25_retriever().search("Regional Revenue", top_k=3) == []
    assert len(engine.search("pyWorkPage", top_k=2, use_reranker=False)) == 1

    # Delete all with disk deletion enabled: deletes inside DOCS_DIR, protects outside files
    engine.sync_database(base_dir=ws)
    res_disk = engine.delete_all_chunks(delete_disk_files=True, docs_dir=docs_dir, workspace_dir=ws)
    assert res_disk["status"] == "success"
    assert res_disk["remaining_chunks"] == 0
    assert not f1.exists() and not f2.exists()
    assert outside.exists()
