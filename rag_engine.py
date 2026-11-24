"""
Pega Architecture RAG Engine
=============================
- High-precision page-aware document parsing (PDF via PyMuPDF, DOCX via python-docx)
- Semantic chunking with rich context headers (Document name, section, category, page)
- Incremental indexing using persistent ChromaDB + SentenceTransformers (all-MiniLM-L6-v2)
- Fast generation via Groq API (openai/gpt-oss-120b / openai/gpt-oss-20b)
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

from config.settings import (
    WORKSPACE_DIR,
    DOCS_DIR,
    DB_DIR,
    MANIFEST_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DOCX_EXTRACTOR_VERSION,
    TABLE_ATOMIC_CHAR_LIMIT,
    RERANKER_MODEL_NAME,
    ENABLE_RERANKER,
    RRF_K,
    DEFAULT_CANDIDATE_K,
    DEFAULT_FINAL_EVIDENCE_K,
    MIN_RERANK_PROBABILITY,
    BM25_K1,
    BM25_B,
    PRIMARY_GROQ_MODEL,
    FALLBACK_GROQ_MODEL,
    ROUTER_MODEL,
    ENABLE_ADAPTIVE_ROUTING,
    MAX_ROUTING_ESCALATIONS,
    MODEL_REGISTRY,
    DEFAULT_UNKNOWN_SPEC,
    get_model_spec,
    is_tool_capable_model,
)


from retrieval.grounding import (
    estimate_tokens,
    clean_chunk_content,
    deduplicate_sources,
    build_grounded_context_prompt,
    _CITATION_STOPWORDS,
    _extract_meaningful_tokens,
    verify_citations,
    contextualize_query_with_history,
)
from retrieval.hybrid import (
    tokenize_pega_text,
    BM25Retriever,
    reciprocal_rank_fusion,
    CrossEncoderReranker,
    compute_similarity_score,
)


from agents.tools import (
    get_agent_tools_schema as _get_agent_tools_schema,
    execute_search_curriculum as _execute_search_curriculum,
    execute_search_challenges as _execute_search_challenges,
    execute_read_full_page as _execute_read_full_page,
    execute_lookup_pega_terms as _execute_lookup_pega_terms,
)
from agents.answer_generator import (
    GROUNDED_SYSTEM_PROMPT,
    is_context_length_error,
    find_best_alternative_model as _find_best_alternative_model,
    resolve_available_model as _resolve_available_model,
    trim_to_last_complete_sentence as _trim_sentence_impl,
    create_completion_with_completeness_guard as _create_completion_guard,
    stream_completion_with_completeness_guard as _stream_completion_guard,
    execute_direct_grounded_fallback as _execute_direct_grounded_fallback_impl,
    generate_answer as _generate_answer_impl,
    stream_answer as _stream_answer_impl,
    detect_output_format,
)
from agents.research_agent import (
    AGENT_SYSTEM_PROMPT,
    should_use_agent_loop,
    detect_conversational_intent,
    classify_query_intent as _classify_query_intent_impl,
    evaluate_evidence_sufficiency,
    run_agent as _run_agent_impl,
    run_agent_stream as _run_agent_stream_impl,
    run_adaptive as _run_adaptive_impl,
    run_adaptive_stream as _run_adaptive_stream_impl,
)
from agents.routing_eval import (
    PEGA_ROUTING_EVAL_DATASET,
    RoutingMetricsTracker,
    evaluate_routing_suite,
    format_routing_eval_report,
    get_routing_observability_summary,
    record_routing_event,
    sanitize_routing_record,
)

# ---------------------------------------------------------
# Document Extraction & Hashing (re-exported from ingestion)
# ---------------------------------------------------------

from ingestion.extractors import (
    extract_pdf_pages,
    _normalize_docx_table_markdown,
    _split_markdown_table_by_rows,
    extract_docx_sections,
    compute_file_hash,
    _legacy_md5_hash,
)


# ---------------------------------------------------------
# RAG Engine Class
# ---------------------------------------------------------

class PegaRAGEngine:
    def __init__(self, db_dir: Optional[Path] = None):
        self.db_dir = db_dir or DB_DIR
        self.db_dir.mkdir(parents=True, exist_ok=True)
        
        # Local SentenceTransformer embedding function (100% offline, free, fast)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
        
        # Persistent ChromaDB client (preserve existing collection metadata/index; new DBs use cosine)
        self.chroma_client = chromadb.PersistentClient(path=str(self.db_dir))
        existing_names = {c.name for c in self.chroma_client.list_collections()}
        if COLLECTION_NAME in existing_names:
            self.collection = self.chroma_client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=self.embedding_fn
            )
        else:
            self.collection = self.chroma_client.create_collection(
                name=COLLECTION_NAME,
                embedding_function=self.embedding_fn,
                metadata={
                    "description": "Pega Academy LSA Curriculum & Challenges",
                    "hnsw:space": "cosine"
                }
            )
        self.distance_metric = ((self.collection.metadata or {}).get("hnsw:space") or "l2").lower()
        
        # Recursive text splitter tuned for technical enterprise docs
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""]
        )

        # Lazy-initialized in-memory BM25 retriever & Cross-Encoder reranker
        self._bm25_retriever: Optional[BM25Retriever] = None
        self.reranker = CrossEncoderReranker(
            model_name=RERANKER_MODEL_NAME,
            enabled=ENABLE_RERANKER
        )

    def invalidate_bm25_cache(self) -> None:
        """Forces BM25 index refresh on the next search after new files are indexed."""
        self._bm25_retriever = None

    def _ensure_bm25_retriever(self) -> Optional[BM25Retriever]:
        """Builds or returns the cached in-memory BM25 index from existing ChromaDB chunks."""
        if self._bm25_retriever is not None:
            return self._bm25_retriever
        try:
            data = self.collection.get(include=["documents", "metadatas"])
            docs = data.get("documents") or []
            metas = data.get("metadatas") or []
            if docs:
                self._bm25_retriever = BM25Retriever(docs, metas)
            return self._bm25_retriever
        except Exception as e:
            print(f"BM25 index initialization failed, falling back to dense search: {e}")
            return None

    def load_manifest(self) -> Dict[str, Any]:
        if MANIFEST_PATH.exists():
            try:
                with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save_manifest(self, manifest: Dict[str, Any]):
        try:
            with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            print(f"Error saving manifest: {e}")

    def chunk_and_index_file(self, file_path: Path) -> int:
        """
        Parses a single file, applies structure-aware & table-aware contextual chunking,
        and upserts into ChromaDB without leaving stale chunks behind.
        """
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            raw_pages = extract_pdf_pages(file_path)
        elif suffix in [".docx", ".doc"]:
            raw_pages = extract_docx_sections(file_path)
        elif suffix in [".txt", ".md"]:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            raw_pages = [{"page_number": 1, "text": content}]
        else:
            return 0

        if not raw_pages:
            return 0

        # Infer section and category from folder structure
        parent_name = file_path.parent.name
        is_challenge = "challenge" in str(file_path).lower() or parent_name.lower() == "challenges"
        category = "Hands-on Challenge" if is_challenge else "Master Module"

        # Find section name (e.g., 02_Planning_and_Design)
        section = "General"
        for part in file_path.parts:
            if any(part.startswith(prefix) for prefix in ["01_", "02_", "03_", "04_", "05_"]):
                section = part
                break

        rel_path = str(file_path.relative_to(WORKSPACE_DIR))
        # Ensure re-indexing a modified/upgraded file removes any prior chunks for this relative_path
        try:
            existing = self.collection.get(where={"relative_path": rel_path}, include=[])
            old_ids = list(existing.get("ids") or [])
            if old_ids:
                for i in range(0, len(old_ids), 200):
                    self.collection.delete(ids=old_ids[i:i + 200])
        except Exception as cleanup_err:
            print(f"Warning: pre-index cleanup failed for {rel_path}: {cleanup_err}")

        chunks_to_add = []
        ids_to_add = []
        metadatas_to_add = []

        chunk_counter = 0
        prose_chunk_count = 0
        table_chunk_count = 0

        def _append_chunk(chunk_body: str, page_num: str, content_type: str, extra_meta: Optional[Dict[str, Any]] = None) -> None:
            nonlocal chunk_counter, prose_chunk_count, table_chunk_count
            chunk_counter += 1
            chunk_id = f"{file_path.stem}_p{page_num}_c{chunk_counter}"
            enriched_text = (
                f"Document: {file_path.name}\n"
                f"Section: {section} | Category: {category} | Page: {page_num}\n"
                f"Content:\n{chunk_body}"
            )
            meta = {
                "source": file_path.name,
                "relative_path": rel_path,
                "section": section,
                "category": category,
                "page": page_num,
                "chunk_index": chunk_counter,
                "content_type": content_type,
            }
            if extra_meta:
                meta.update(extra_meta)
            chunks_to_add.append(enriched_text)
            ids_to_add.append(chunk_id)
            metadatas_to_add.append(meta)
            if content_type == "table":
                table_chunk_count += 1
            else:
                prose_chunk_count += 1

        for page_data in raw_pages:
            page_num = str(page_data["page_number"])
            heading_path = str(page_data.get("heading_path") or page_num)
            blocks = page_data.get("blocks")

            if blocks:
                for block in blocks:
                    if block["type"] == "prose":
                        sub_chunks = self.text_splitter.split_text(block["text"])
                        for chunk_text in sub_chunks:
                            body_with_section = f"Section Context: {heading_path}\n{chunk_text}"
                            _append_chunk(body_with_section, page_num, "prose")
                    elif block["type"] == "table":
                        prefix_ctx = f"Document: {file_path.name} | Section Context: {heading_path}"
                        full_table = f"{prefix_ctx}\n{block['markdown']}"
                        extra = {
                            "table_rows": int(block.get("rows", 0)),
                            "table_cols": int(block.get("cols", 0)),
                        }
                        if len(full_table) <= TABLE_ATOMIC_CHAR_LIMIT:
                            _append_chunk(full_table, page_num, "table", extra)
                        else:
                            row_chunks = _split_markdown_table_by_rows(
                                header_lines=block["header_lines"],
                                data_lines=block["data_lines"],
                                prefix_context=prefix_ctx,
                                max_chars=TABLE_ATOMIC_CHAR_LIMIT,
                            )
                            for rc in row_chunks:
                                _append_chunk(rc, page_num, "table", extra)
            else:
                page_text = page_data["text"]
                sub_chunks = self.text_splitter.split_text(page_text)
                for chunk_text in sub_chunks:
                    _append_chunk(chunk_text, page_num, "prose")

        # Batch upsert into ChromaDB
        if chunks_to_add:
            batch_size = 200
            for i in range(0, len(chunks_to_add), batch_size):
                self.collection.upsert(
                    ids=ids_to_add[i:i+batch_size],
                    documents=chunks_to_add[i:i+batch_size],
                    metadatas=metadatas_to_add[i:i+batch_size]
                )
            self.invalidate_bm25_cache()

        print(
            f"[Chunking] {file_path.name}: prose_chunks={prose_chunk_count}, "
            f"table_chunks={table_chunk_count}, total_chunks={len(chunks_to_add)}"
        )
        return len(chunks_to_add)

    def sync_database(self, base_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Scans all supported documents in the directory.
        - Skips unchanged PDFs and up-to-date DOCX files (SHA-256 + extractor_version).
        - Selectively re-indexes DOCX files created with an older extractor_version.
        - Indexes NEW or MODIFIED files incrementally.
        """
        target_base = base_dir or WORKSPACE_DIR
        manifest = self.load_manifest()
        new_indexed = 0
        reindexed_count = 0
        skipped = 0
        total_chunks_added = 0

        # Scan for PDFs and Word documents
        supported_exts = {".pdf", ".docx", ".doc", ".txt", ".md"}
        all_files = []
        for ext in supported_exts:
            all_files.extend(list(target_base.rglob(f"*{ext}")))

        # Filter out ChromaDB directory itself
        all_files = [f for f in all_files if "pega_rag_db" not in str(f)]

        for file_path in all_files:
            rel_path = str(file_path.relative_to(target_base))
            current_hash = compute_file_hash(file_path)
            is_docx = file_path.suffix.lower() in {".docx", ".doc"}
            target_version = DOCX_EXTRACTOR_VERSION if is_docx else 1

            is_selective_reindex = False
            preserved_doc_name = None

            if rel_path in manifest:
                entry = manifest[rel_path]
                stored_hash = entry.get("hash", "")
                stored_version = int(entry.get("extractor_version", 1))
                preserved_doc_name = entry.get("document_name")

                hash_matches = (stored_hash == current_hash) or (
                    len(stored_hash) == 32 and stored_hash == _legacy_md5_hash(file_path)
                )

                if hash_matches:
                    if len(stored_hash) == 32:
                        # Upgrade legacy MD5 entry to SHA-256 without re-chunking PDFs
                        entry["hash"] = current_hash
                    if is_docx and stored_version < DOCX_EXTRACTOR_VERSION:
                        is_selective_reindex = True
                    else:
                        skipped += 1
                        continue

            # Process & index new, modified, or selectively re-indexed file
            chunks_count = self.chunk_and_index_file(file_path)
            new_entry = {
                "hash": current_hash,
                "file_name": file_path.name,
                "chunks_count": chunks_count,
                "last_indexed": str(file_path.stat().st_mtime),
                "extractor_version": target_version,
            }
            if preserved_doc_name:
                new_entry["document_name"] = preserved_doc_name
            manifest[rel_path] = new_entry

            if is_selective_reindex:
                reindexed_count += 1
            else:
                new_indexed += 1
            total_chunks_added += chunks_count

        self.save_manifest(manifest)
        total_docs = self.collection.count()
        print(
            f"[Sync] new_indexed={new_indexed}, selectively_reindexed={reindexed_count}, "
            f"skipped={skipped}, total_chunks_in_db={total_docs}"
        )

        return {
            "new_indexed": new_indexed,
            "reindexed": reindexed_count,
            "skipped": skipped,
            "total_files": len(all_files),
            "chunks_added": total_chunks_added,
            "total_chunks_in_db": total_docs
        }

    def list_indexed_documents(self) -> List[Dict[str, Any]]:
        """
        Returns a consolidated list of all indexed documents combining ChromaDB metadata
        and indexed_manifest.json, safely handling stale manifest entries and zero-chunk docs.
        """
        manifest = self.load_manifest()
        chroma_stats: Dict[str, Dict[str, Any]] = {}

        try:
            data = self.collection.get(include=["metadatas"])
            for meta in (data.get("metadatas") or []):
                if not meta:
                    continue
                rel = meta.get("relative_path") or meta.get("source") or "Unknown"
                if rel not in chroma_stats:
                    chroma_stats[rel] = {
                        "file_name": meta.get("source", Path(rel).name),
                        "relative_path": rel,
                        "category": meta.get("category", "Document"),
                        "section": meta.get("section", "General"),
                        "chunks_count": 0
                    }
                chroma_stats[rel]["chunks_count"] += 1
        except Exception as e:
            print(f"Error inspecting ChromaDB metadata: {e}")

        docs_map: Dict[str, Dict[str, Any]] = {}
        # Populate from manifest first
        for rel_p, m_info in manifest.items():
            fname = m_info.get("file_name", Path(rel_p).name)
            doc_display_name = m_info.get("document_name") or fname
            live_stat = chroma_stats.get(rel_p)
            live_count = live_stat["chunks_count"] if live_stat else 0
            cat = live_stat["category"] if live_stat else (
                "Hands-on Challenge" if "challenge" in rel_p.lower() else "Master Module"
            )
            sec = live_stat["section"] if live_stat else "General"
            disk_path = WORKSPACE_DIR / rel_p
            docs_map[rel_p] = {
                "document_name": doc_display_name,
                "file_name": fname,
                "relative_path": rel_p,
                "category": cat,
                "section": sec,
                "chunks_count": live_count,
                "manifest_chunks": m_info.get("chunks_count", 0),
                "file_exists_on_disk": disk_path.exists()
            }

        # Include any ChromaDB documents missing from manifest
        for rel_p, c_stat in chroma_stats.items():
            if rel_p not in docs_map:
                disk_path = WORKSPACE_DIR / rel_p
                docs_map[rel_p] = {
                    "document_name": c_stat["file_name"],
                    "file_name": c_stat["file_name"],
                    "relative_path": rel_p,
                    "category": c_stat["category"],
                    "section": c_stat["section"],
                    "chunks_count": c_stat["chunks_count"],
                    "manifest_chunks": 0,
                    "file_exists_on_disk": disk_path.exists()
                }

        return sorted(docs_map.values(), key=lambda d: (d["file_name"].lower(), d["relative_path"].lower()))

    def delete_document_chunks(
        self,
        relative_path: Optional[str] = None,
        source_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Deletes ONLY the selected document's chunks from ChromaDB, removes its entry from
        indexed_manifest.json, and rebuilds the in-memory BM25 corpus.
        Does NOT delete the physical file from disk.
        """
        rel_clean = (relative_path or "").strip()
        src_clean = (source_name or "").strip()

        if not rel_clean and not src_clean:
            return {
                "status": "error",
                "message": "A valid relative_path or source_name is required for targeted deletion.",
                "deleted_chunks": 0
            }

        matched_ids: List[str] = []
        try:
            if rel_clean:
                res = self.collection.get(where={"relative_path": rel_clean}, include=[])
                matched_ids = list(res.get("ids") or [])

            # Fallback to source metadata only if relative_path yielded nothing
            if not matched_ids and src_clean:
                res = self.collection.get(where={"source": src_clean}, include=["metadatas"])
                ids = res.get("ids") or []
                metas = res.get("metadatas") or []
                for cid, meta in zip(ids, metas):
                    m_rel = (meta or {}).get("relative_path", "")
                    if not rel_clean or m_rel == rel_clean or not m_rel:
                        matched_ids.append(cid)

            if matched_ids:
                # Batch delete matched IDs only (never broad collection deletion)
                batch_size = 200
                for i in range(0, len(matched_ids), batch_size):
                    self.collection.delete(ids=matched_ids[i:i + batch_size])
        except Exception as e:
            return {
                "status": "error",
                "message": f"ChromaDB targeted deletion failed: {e}",
                "deleted_chunks": 0
            }

        # Remove from indexed_manifest.json
        manifest = self.load_manifest()
        removed_manifest_keys = []
        if rel_clean and rel_clean in manifest:
            del manifest[rel_clean]
            removed_manifest_keys.append(rel_clean)
        elif src_clean:
            for k in list(manifest.keys()):
                if manifest[k].get("file_name") == src_clean or k == src_clean:
                    del manifest[k]
                    removed_manifest_keys.append(k)
                    break

        if removed_manifest_keys:
            self.save_manifest(manifest)

        # Invalidate and rebuild BM25 corpus immediately so deleted content cannot be retrieved
        self.invalidate_bm25_cache()
        self._ensure_bm25_retriever()

        return {
            "status": "success",
            "deleted_chunks": len(matched_ids),
            "manifest_removed": bool(removed_manifest_keys),
            "remaining_chunks_in_db": self.collection.count(),
            "remaining_docs_in_manifest": len(manifest)
        }

    def delete_all_chunks(
        self,
        delete_disk_files: bool = False,
        docs_dir: Optional[Path] = None,
        workspace_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Deletes every indexed record from the existing ChromaDB collection using safe batch deletion
        (without deleting or recreating the collection), clears indexed_manifest.json after
        successful deletion, invalidates BM25 cache, and optionally deletes original source files
        strictly contained inside DOCS_DIR.
        """
        target_docs_dir = (docs_dir or DOCS_DIR).resolve()
        target_workspace_dir = (workspace_dir or WORKSPACE_DIR).resolve()

        manifest = self.load_manifest()
        previous_doc_count = len(self.list_indexed_documents())

        try:
            existing = self.collection.get(include=["metadatas"] if delete_disk_files else [])
            all_ids = list(existing.get("ids") or [])
            all_metas = list(existing.get("metadatas") or []) if delete_disk_files else []
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to inspect ChromaDB collection before deletion: {e}",
                "deleted_chunks": 0,
                "previous_doc_count": previous_doc_count,
                "remaining_chunks": self.collection.count(),
                "deleted_disk_files": 0
            }

        try:
            if all_ids:
                batch_size = 200
                for i in range(0, len(all_ids), batch_size):
                    self.collection.delete(ids=all_ids[i:i + batch_size])
        except Exception as e:
            return {
                "status": "error",
                "message": f"ChromaDB batch deletion failed: {e}",
                "deleted_chunks": 0,
                "previous_doc_count": previous_doc_count,
                "remaining_chunks": self.collection.count(),
                "deleted_disk_files": 0
            }

        # Clear manifest only after ChromaDB deletion succeeded
        self.save_manifest({})

        # Invalidate BM25 cache so empty-collection searches never return stale results
        self.invalidate_bm25_cache()

        deleted_disk_files = 0
        if delete_disk_files:
            candidate_rel_paths = set(manifest.keys())
            for meta in all_metas:
                if isinstance(meta, dict) and meta.get("relative_path"):
                    candidate_rel_paths.add(str(meta["relative_path"]))

            for rel_p in candidate_rel_paths:
                try:
                    candidate_path = (target_workspace_dir / rel_p).resolve()
                    # Restrict disk deletion strictly to files inside target_docs_dir
                    candidate_path.relative_to(target_docs_dir)
                    if candidate_path != target_docs_dir and candidate_path.exists() and candidate_path.is_file():
                        candidate_path.unlink()
                        deleted_disk_files += 1
                except Exception:
                    # Skip paths outside DOCS_DIR, missing files, or missing directories safely
                    continue

        return {
            "status": "success",
            "deleted_chunks": len(all_ids),
            "previous_doc_count": previous_doc_count,
            "remaining_chunks": self.collection.count(),
            "deleted_disk_files": deleted_disk_files
        }

    def _dense_search(
        self,
        query: str,
        top_k: int = 15,
        section_filter: Optional[str] = None,
        category_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Performs dense vector search against ChromaDB with optional section/category filters."""
        total_count = self.collection.count()
        if total_count <= 0:
            return []
        effective_k = min(top_k, total_count)

        clauses = []
        if section_filter and section_filter != "All Sections":
            clauses.append({"section": section_filter})
        if category_filter:
            clauses.append({"category": category_filter})

        where_filter = None
        if len(clauses) == 1:
            where_filter = clauses[0]
        elif len(clauses) > 1:
            where_filter = {"$and": clauses}

        results = self.collection.query(
            query_texts=[query],
            n_results=effective_k,
            where=where_filter
        )

        formatted_results = []
        if results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(docs)

            for doc_text, meta, dist in zip(docs, metas, distances):
                similarity_score = compute_similarity_score(dist, self.distance_metric)
                formatted_results.append({
                    "document": doc_text,
                    "source": meta.get("source", "Unknown"),
                    "page": str(meta.get("page", "1")),
                    "section": meta.get("section", "General"),
                    "category": meta.get("category", "Document"),
                    "relative_path": meta.get("relative_path", ""),
                    "score": similarity_score
                })

        return formatted_results

    def search(
        self,
        query: str,
        top_k: int = 5,
        section_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
        use_hybrid: bool = True,
        use_reranker: bool = True,
        dense_weight: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval pipeline:
        1. Dense retrieval (ChromaDB) + Sparse lexical retrieval (BM25)
        2. Weighted Reciprocal Rank Fusion (RRF)
        3. Cross-Encoder reranking and relevance filtering
        """
        if not use_hybrid:
            return self._dense_search(
                query=query,
                top_k=top_k,
                section_filter=section_filter,
                category_filter=category_filter
            )

        candidate_k = max(DEFAULT_CANDIDATE_K, top_k * 3)
        dense_hits = self._dense_search(
            query=query,
            top_k=candidate_k,
            section_filter=section_filter,
            category_filter=category_filter
        )

        bm25 = self._ensure_bm25_retriever()
        sparse_hits = (
            bm25.search(
                query=query,
                top_k=candidate_k,
                section_filter=section_filter,
                category_filter=category_filter
            )
            if bm25 is not None
            else []
        )

        fused_candidates = reciprocal_rank_fusion(
            dense_results=dense_hits,
            sparse_results=sparse_hits,
            rrf_k=RRF_K,
            top_k=candidate_k,
            dense_weight=dense_weight
        )

        if use_reranker:
            return self.reranker.rerank(query=query, candidates=fused_candidates, top_k=top_k)

        return fused_candidates[:top_k]

    # ---------------------------------------------------------
    # Autonomous Agent Tools, Answer Generation & Streaming
    # (Thin adapters preserving engine state & monkeypatch bindings)
    # ---------------------------------------------------------

    def tool_search_curriculum(self, query: str, section: Optional[str] = None, top_k: int = 4) -> Dict[str, Any]:
        return _execute_search_curriculum(self, query=query, section=section, top_k=top_k)

    def tool_search_challenges(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        return _execute_search_challenges(self, query=query, top_k=top_k)

    def tool_read_full_page(self, document_name: str, page_number: int) -> Dict[str, Any]:
        return _execute_read_full_page(document_name=document_name, page_number=page_number, docs_dir=DOCS_DIR)

    def tool_lookup_pega_terms(self, term: str, top_k: int = 4) -> Dict[str, Any]:
        return _execute_lookup_pega_terms(self, term=term, top_k=top_k)

    @staticmethod
    def get_agent_tools_schema() -> List[Dict[str, Any]]:
        return _get_agent_tools_schema()

    def find_best_alternative_model(self, client: Groq, current_model: str) -> Optional[str]:
        return _find_best_alternative_model(client, current_model)

    def resolve_available_model(self, client: Groq, requested_model: Optional[str]) -> Tuple[str, Optional[str]]:
        return _resolve_available_model(client, requested_model)

    @staticmethod
    def _trim_to_last_complete_sentence(text: str) -> str:
        return _trim_sentence_impl(text)

    def _create_completion_with_completeness_guard(
        self,
        client: Groq,
        model_name: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int
    ) -> Tuple[str, Optional[str]]:
        return _create_completion_guard(client, model_name, messages, temperature, max_tokens)

    def _execute_direct_grounded_fallback(
        self,
        query: str,
        client: Groq,
        model_name: str,
        temperature: float = 0.0,
        chat_history: Optional[List[Dict[str, str]]] = None,
        execution_trace: Optional[List[Dict[str, Any]]] = None,
        reason: str = "",
        top_k: int = DEFAULT_FINAL_EVIDENCE_K,
        use_reranker: bool = True,
        dense_weight: float = 0.5,
        section_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        return _execute_direct_grounded_fallback_impl(
            engine=self,
            query=query,
            client=client,
            model_name=model_name,
            temperature=temperature,
            chat_history=chat_history,
            execution_trace=execution_trace,
            reason=reason,
            top_k=top_k,
            use_reranker=use_reranker,
            dense_weight=dense_weight,
            section_filter=section_filter,
        )

    def run_agent(
        self,
        query: str,
        groq_api_key: str,
        model_name: str = PRIMARY_GROQ_MODEL,
        temperature: float = 0.0,
        max_iterations: int = 4,
        chat_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = DEFAULT_FINAL_EVIDENCE_K,
        use_reranker: bool = True,
        dense_weight: float = 0.5,
        section_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        return _run_agent_impl(
            engine=self,
            query=query,
            groq_api_key=groq_api_key,
            model_name=model_name,
            temperature=temperature,
            max_iterations=max_iterations,
            chat_history=chat_history,
            top_k=top_k,
            use_reranker=use_reranker,
            dense_weight=dense_weight,
            section_filter=section_filter,
            groq_factory=Groq,
            should_use_agent_fn=should_use_agent_loop,
        )

    def detect_conversational_intent(self, query: str) -> Optional[Dict[str, Any]]:
        return detect_conversational_intent(query)

    def classify_query_intent(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        groq_api_key: str = "",
        answer_model_name: str = PRIMARY_GROQ_MODEL,
        router_model: str = ROUTER_MODEL,
    ) -> Dict[str, Any]:
        return _classify_query_intent_impl(
            query=query,
            chat_history=chat_history,
            groq_api_key=groq_api_key,
            answer_model_name=answer_model_name,
            router_model=router_model,
            groq_factory=Groq,
            should_use_agent_fn=should_use_agent_loop,
        )

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        groq_api_key: str,
        model_name: str = PRIMARY_GROQ_MODEL,
        temperature: float = 0.2,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        return _generate_answer_impl(
            query=query,
            retrieved_chunks=retrieved_chunks,
            groq_api_key=groq_api_key,
            model_name=model_name,
            temperature=temperature,
            chat_history=chat_history,
            groq_factory=Groq,
        )

    def _stream_completion_with_completeness_guard(
        self,
        client: Groq,
        model_name: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int
    ):
        return _stream_completion_guard(client, model_name, messages, temperature, max_tokens)

    def stream_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        groq_api_key: str,
        model_name: str = PRIMARY_GROQ_MODEL,
        temperature: float = 0.0,
        chat_history: Optional[List[Dict[str, str]]] = None
    ):
        return _stream_answer_impl(
            query=query,
            retrieved_chunks=retrieved_chunks,
            groq_api_key=groq_api_key,
            model_name=model_name,
            temperature=temperature,
            chat_history=chat_history,
            groq_factory=Groq,
        )

    def run_agent_stream(
        self,
        query: str,
        groq_api_key: str,
        model_name: str = PRIMARY_GROQ_MODEL,
        temperature: float = 0.0,
        max_iterations: int = 4,
        chat_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = DEFAULT_FINAL_EVIDENCE_K,
        use_reranker: bool = True,
        dense_weight: float = 0.5,
        section_filter: Optional[str] = None
    ):
        return _run_agent_stream_impl(
            engine=self,
            query=query,
            groq_api_key=groq_api_key,
            model_name=model_name,
            temperature=temperature,
            max_iterations=max_iterations,
            chat_history=chat_history,
            top_k=top_k,
            use_reranker=use_reranker,
            dense_weight=dense_weight,
            section_filter=section_filter,
            groq_factory=Groq,
            should_use_agent_fn=should_use_agent_loop,
        )

    def run_adaptive(
        self,
        query: str,
        groq_api_key: str,
        model_name: str = PRIMARY_GROQ_MODEL,
        temperature: float = 0.0,
        max_iterations: int = 4,
        chat_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = DEFAULT_FINAL_EVIDENCE_K,
        use_reranker: bool = True,
        dense_weight: float = 0.5,
        section_filter: Optional[str] = None,
        router_model: str = ROUTER_MODEL,
        execution_mode: str = "adaptive",
    ) -> Dict[str, Any]:
        return _run_adaptive_impl(
            engine=self,
            query=query,
            groq_api_key=groq_api_key,
            model_name=model_name,
            temperature=temperature,
            max_iterations=max_iterations,
            chat_history=chat_history,
            top_k=top_k,
            use_reranker=use_reranker,
            dense_weight=dense_weight,
            section_filter=section_filter,
            router_model=router_model,
            execution_mode=execution_mode,
            groq_factory=Groq,
        )

    def run_adaptive_stream(
        self,
        query: str,
        groq_api_key: str,
        model_name: str = PRIMARY_GROQ_MODEL,
        temperature: float = 0.0,
        max_iterations: int = 4,
        chat_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = DEFAULT_FINAL_EVIDENCE_K,
        use_reranker: bool = True,
        dense_weight: float = 0.5,
        section_filter: Optional[str] = None,
        router_model: str = ROUTER_MODEL,
        execution_mode: str = "adaptive",
    ):
        return _run_adaptive_stream_impl(
            engine=self,
            query=query,
            groq_api_key=groq_api_key,
            model_name=model_name,
            temperature=temperature,
            max_iterations=max_iterations,
            chat_history=chat_history,
            top_k=top_k,
            use_reranker=use_reranker,
            dense_weight=dense_weight,
            section_filter=section_filter,
            router_model=router_model,
            execution_mode=execution_mode,
            groq_factory=Groq,
        )
