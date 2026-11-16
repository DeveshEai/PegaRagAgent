"""
Pega Architecture RAG Engine
=============================
- High-precision page-aware document parsing (PDF via PyMuPDF, DOCX via python-docx)
- Semantic chunking with rich context headers (Document name, section, category, page)
- Incremental indexing using persistent ChromaDB + SentenceTransformers (all-MiniLM-L6-v2)
- Fast generation via Groq API (llama-3.3-70b-versatile / llama-3.1-8b-instant)
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv

# Load local .env if present
load_dotenv(Path(__file__).parent / ".env")

import fitz  # PyMuPDF
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

# Base Directories
WORKSPACE_DIR = Path(__file__).parent.resolve()
DOCS_DIR = WORKSPACE_DIR / "Lead System Architect"
DB_DIR = WORKSPACE_DIR / "pega_rag_db"
MANIFEST_PATH = DB_DIR / "indexed_manifest.json"

COLLECTION_NAME = "pega_lsa_curriculum"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ---------------------------------------------------------
# Document Extraction with Rich Page-Level Metadata
# ---------------------------------------------------------

def extract_pdf_pages(file_path: Path) -> List[Dict[str, Any]]:
    """Extracts text page-by-page from a PDF, preserving exact page numbers."""
    pages_data = []
    try:
        doc = fitz.open(file_path)
        for page_idx, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                pages_data.append({
                    "page_number": page_idx + 1,
                    "text": text
                })
        doc.close()
    except Exception as e:
        print(f"Error reading PDF {file_path.name}: {e}")
    return pages_data


def extract_docx_sections(file_path: Path) -> List[Dict[str, Any]]:
    """Extracts text from DOCX documents section by section."""
    sections_data = []
    try:
        doc = docx.Document(file_path)
        current_heading = "General Overview"
        current_paragraphs = []

        for p in doc.paragraphs:
            text = p.text.strip()
            if not text:
                continue
            if p.style.name.startswith("Heading"):
                if current_paragraphs:
                    sections_data.append({
                        "page_number": current_heading,
                        "text": "\n".join(current_paragraphs)
                    })
                    current_paragraphs = []
                current_heading = text
            else:
                current_paragraphs.append(text)

        if current_paragraphs:
            sections_data.append({
                "page_number": current_heading,
                "text": "\n".join(current_paragraphs)
            })
    except Exception as e:
        print(f"Error reading DOCX {file_path.name}: {e}")
    return sections_data


def compute_file_hash(file_path: Path) -> str:
    """Computes quick SHA256 of file modification time and size."""
    stat = file_path.stat()
    hash_str = f"{file_path.name}_{stat.st_size}_{stat.st_mtime}"
    return hashlib.md5(hash_str.encode()).hexdigest()

# ---------------------------------------------------------
# RAG Engine Class
# ---------------------------------------------------------

class PegaRAGEngine:
    def __init__(self, db_dir: Path = DB_DIR):
        self.db_dir = db_dir
        self.db_dir.mkdir(parents=True, exist_ok=True)
        
        # Local SentenceTransformer embedding function (100% offline, free, fast)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
        
        # Persistent ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=str(self.db_dir))
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"description": "Pega Academy LSA Curriculum & Challenges"}
        )
        
        # Recursive text splitter tuned for technical enterprise docs
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=950,
            chunk_overlap=180,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""]
        )

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
        """Parses a single file, applies contextual chunking, and upserts into ChromaDB."""
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

        chunks_to_add = []
        ids_to_add = []
        metadatas_to_add = []

        chunk_counter = 0
        for page_data in raw_pages:
            page_num = str(page_data["page_number"])
            page_text = page_data["text"]

            # Split text while keeping within boundary
            sub_chunks = self.text_splitter.split_text(page_text)
            for sub_idx, chunk_text in enumerate(sub_chunks):
                chunk_counter += 1
                chunk_id = f"{file_path.stem}_p{page_num}_c{chunk_counter}"
                
                # Context-enriched chunk text (helps embedding similarity match specific architectural topics)
                enriched_text = (
                    f"Document: {file_path.name}\n"
                    f"Section: {section} | Category: {category} | Page: {page_num}\n"
                    f"Content:\n{chunk_text}"
                )

                chunks_to_add.append(enriched_text)
                ids_to_add.append(chunk_id)
                metadatas_to_add.append({
                    "source": file_path.name,
                    "relative_path": str(file_path.relative_to(WORKSPACE_DIR)),
                    "section": section,
                    "category": category,
                    "page": page_num,
                    "chunk_index": chunk_counter
                })

        # Batch upsert into ChromaDB
        if chunks_to_add:
            # ChromaDB handles up to ~500 items per batch comfortably
            batch_size = 200
            for i in range(0, len(chunks_to_add), batch_size):
                self.collection.upsert(
                    ids=ids_to_add[i:i+batch_size],
                    documents=chunks_to_add[i:i+batch_size],
                    metadatas=metadatas_to_add[i:i+batch_size]
                )

        return len(chunks_to_add)

    def sync_database(self, base_dir: Path = WORKSPACE_DIR) -> Dict[str, Any]:
        """
        Scans all supported documents in the directory.
        Only indexes NEW or MODIFIED files, making future file additions instantaneous.
        """
        manifest = self.load_manifest()
        new_indexed = 0
        skipped = 0
        total_chunks_added = 0

        # Scan for PDFs and Word documents
        supported_exts = {".pdf", ".docx", ".doc", ".txt", ".md"}
        all_files = []
        for ext in supported_exts:
            all_files.extend(list(base_dir.rglob(f"*{ext}")))

        # Filter out ChromaDB directory itself
        all_files = [f for f in all_files if "pega_rag_db" not in str(f)]

        for file_path in all_files:
            rel_path = str(file_path.relative_to(base_dir))
            current_hash = compute_file_hash(file_path)

            # Check if file has changed
            if rel_path in manifest and manifest[rel_path].get("hash") == current_hash:
                skipped += 1
                continue

            # Process & index new/modified file
            chunks_count = self.chunk_and_index_file(file_path)
            manifest[rel_path] = {
                "hash": current_hash,
                "file_name": file_path.name,
                "chunks_count": chunks_count,
                "last_indexed": str(file_path.stat().st_mtime)
            }
            new_indexed += 1
            total_chunks_added += chunks_count

        self.save_manifest(manifest)
        total_docs = self.collection.count()

        return {
            "new_indexed": new_indexed,
            "skipped": skipped,
            "total_files": len(all_files),
            "chunks_added": total_chunks_added,
            "total_chunks_in_db": total_docs
        }

    def search(self, query: str, top_k: int = 5, section_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves top-k most relevant chunks with similarity metrics and metadata."""
        where_filter = None
        if section_filter and section_filter != "All Sections":
            where_filter = {"section": section_filter}

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter
        )

        formatted_results = []
        if results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(docs)

            for doc_text, meta, dist in zip(docs, metas, distances):
                # Cosine distance to similarity percentage approximation
                similarity_score = max(0.0, min(100.0, (1.0 - (dist / 2.0)) * 100))
                formatted_results.append({
                    "document": doc_text,
                    "source": meta.get("source", "Unknown"),
                    "page": meta.get("page", "1"),
                    "section": meta.get("section", "General"),
                    "category": meta.get("category", "Document"),
                    "relative_path": meta.get("relative_path", ""),
                    "score": round(similarity_score, 1)
                })

        return formatted_results

    # ---------------------------------------------------------
    # Autonomous Agent Tools Implementation
    # ---------------------------------------------------------

    def tool_search_curriculum(self, query: str, section: Optional[str] = None, top_k: int = 4) -> Dict[str, Any]:
        """Tool: Search theoretical concepts & design patterns in the 28 Master Modules."""
        where_filter = {"category": "Master Module"}
        if section and section != "All Sections":
            where_filter = {"$and": [{"category": "Master Module"}, {"section": section}]}

        results = self.search(query=query, top_k=top_k)
        # Filter for master modules if metadata is present
        filtered = [r for r in results if r.get("category") == "Master Module"]
        if not filtered:
            filtered = results[:top_k]

        formatted = []
        for r in filtered:
            formatted.append({
                "document": r["source"],
                "section": r["section"],
                "page": r["page"],
                "relevance_score": f"{r['score']}%",
                "excerpt": r["document"]
            })
        return {"status": "success", "results_count": len(formatted), "results": formatted}

    def tool_search_challenges(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        """Tool: Search hands-on implementation steps and solutions in the 40 Challenges."""
        results = self.search(query=query, top_k=top_k * 2)
        # Filter for hands-on challenges
        filtered = [r for r in results if r.get("category") == "Hands-on Challenge"]
        if not filtered:
            filtered = results[:top_k]
        else:
            filtered = filtered[:top_k]

        formatted = []
        for r in filtered:
            formatted.append({
                "document": r["source"],
                "section": r["section"],
                "page": r["page"],
                "relevance_score": f"{r['score']}%",
                "excerpt": r["document"]
            })
        return {"status": "success", "results_count": len(formatted), "results": formatted}

    def tool_read_full_page(self, document_name: str, page_number: int) -> Dict[str, Any]:
        """Tool: Read the complete, untruncated content of a specific page in any Pega PDF."""
        # Find document on disk
        matches = list(DOCS_DIR.rglob(document_name))
        if not matches:
            matches = [f for f in DOCS_DIR.rglob("*.pdf") if document_name.lower() in f.name.lower()]
        
        if not matches:
            return {"status": "error", "message": f"Document '{document_name}' not found in curriculum."}

        target_path = matches[0]
        try:
            doc = fitz.open(target_path)
            total_pages = len(doc)
            if 1 <= page_number <= total_pages:
                page_text = doc[page_number - 1].get_text("text").strip()
                doc.close()
                return {
                    "status": "success",
                    "document": target_path.name,
                    "page": page_number,
                    "total_pages": total_pages,
                    "full_page_text": page_text
                }
            else:
                doc.close()
                return {
                    "status": "error",
                    "message": f"Requested page {page_number} is out of bounds (Document has {total_pages} pages)."
                }
        except Exception as e:
            return {"status": "error", "message": f"Failed reading page: {str(e)}"}

    def tool_lookup_pega_terms(self, term: str, top_k: int = 4) -> Dict[str, Any]:
        """Tool: Specialized lookup for exact Pega rules, classes, APIs, or architectural terms."""
        results = self.search(query=term, top_k=top_k)
        formatted = []
        for r in results:
            formatted.append({
                "document": r["source"],
                "section": r["section"],
                "page": r["page"],
                "relevance_score": f"{r['score']}%",
                "excerpt": r["document"]
            })
        return {"status": "success", "term": term, "results": formatted}

    # ---------------------------------------------------------
    # Agent Tool Schemas for Groq Function Calling
    # ---------------------------------------------------------

    @staticmethod
    def get_agent_tools_schema() -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_curriculum",
                    "description": "Searches theoretical architectural concepts, principles, rulesets, and design patterns in the 28 Pega Master Modules.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Specific search query focused on architectural principles, design patterns, or system governance."
                            },
                            "section": {
                                "type": "string",
                                "enum": ["All Sections", "01_LSA_Mindset", "02_Planning_and_Design", "03_Constellation_Adoption", "04_Infrastructure_and_Architecture", "05_Implementation_and_Deployment"],
                                "description": "Optional curriculum section to filter search."
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_challenges",
                    "description": "Searches hands-on implementation steps, configuration walkthroughs, and solutions in the 40 Pega Challenges.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Practical configuration or exercise query (e.g. 'how to configure SAML in App Studio', 'creating field groups in Constellation')."
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_full_page",
                    "description": "Reads the entire, complete text of a specific page from any Pega PDF document. Use this when a search chunk is truncated or when you need full step-by-step instructions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "document_name": {
                                "type": "string",
                                "description": "The exact name or partial name of the PDF file (e.g. '01_fundamental-concepts-constellation.pdf')."
                            },
                            "page_number": {
                                "type": "integer",
                                "description": "The exact 1-indexed page number to read."
                            }
                        },
                        "required": ["document_name", "page_number"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "lookup_pega_terms",
                    "description": "Looks up exact Pega technical rules, classes (e.g. 'Queue Processor', 'Job Scheduler', 'ABAC', 'BIX', 'Rule-Obj-Class', 'DX API').",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "term": {
                                "type": "string",
                                "description": "The exact technical Pega rule or class name to search."
                            }
                        },
                        "required": ["term"]
                    }
                }
            }
        ]

    def find_best_alternative_model(self, client: Groq, current_model: str) -> Optional[str]:
        """Dynamically finds an active text chat model from the user's specific Groq account."""
        try:
            live_models = client.models.list()
            raw_ids = [m.id for m in live_models.data]
            valid_ids = []
            for m_id in raw_ids:
                m_low = m_id.lower()
                if any(x in m_low for x in ["whisper", "orpheus", "tts", "audio", "guard", "safeguard", "canopylabs"]):
                    continue
                if m_id != current_model:
                    valid_ids.append(m_id)

            priority = [
                "llama-3.3-70b-versatile",
                "llama-3.1-70b-versatile",
                "llama-3.1-8b-instant",
                "llama3-70b-8192",
                "mixtral-8x7b-32768",
                "qwen/qwen3.8-27b",
                "openai/gpt-oss-20b"
            ]
            for p in priority:
                if p in valid_ids:
                    return p
            return valid_ids[0] if valid_ids else None
        except Exception:
            return None

    def _execute_direct_grounded_fallback(
        self,
        query: str,
        client: Groq,
        model_name: str,
        temperature: float = 0.0,
        chat_history: Optional[List[Dict[str, str]]] = None,
        execution_trace: Optional[List[Dict[str, Any]]] = None,
        reason: str = ""
    ) -> Dict[str, Any]:
        """
        Direct Grounded Retrieval Fallback when a model does not support Groq function/tool calling.
        Retrieves verified excerpts from ChromaDB and sends a strict zero-hallucination prompt
        WITHOUT the 'tools' parameter.
        """
        if execution_trace is None:
            execution_trace = []

        # 1. Retrieve top chunks from curriculum
        retrieved_chunks = self.search(query=query, top_k=6)

        # Also retrieve challenges if query appears practical/implementation oriented
        query_lower = query.lower()
        if any(w in query_lower for w in ["how to", "configure", "implement", "step", "challenge", "exercise", "create", "setup"]):
            try:
                challenge_res = self.tool_search_challenges(query=query, top_k=3)
                if "results" in challenge_res:
                    for r in challenge_res["results"]:
                        if not any(c.get("document") == r.get("excerpt") for c in retrieved_chunks):
                            retrieved_chunks.append({
                                "source": r.get("document", "Challenge"),
                                "page": str(r.get("page", "1")),
                                "section": r.get("section", "Hands-on Challenges"),
                                "category": "Hands-on Challenge",
                                "score": float(r.get("relevance_score", "0%").replace("%", "")),
                                "document": r.get("excerpt", "")
                            })
            except Exception:
                pass

        collected_sources = []
        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            source_name = chunk.get("source", "Unknown")
            page_num = str(chunk.get("page", "1"))
            section_name = chunk.get("section", "General")
            category_name = chunk.get("category", "Master Module")
            score = chunk.get("score", 0.0)
            doc_text = chunk.get("document", "")

            collected_sources.append({
                "source": source_name,
                "page": page_num,
                "section": section_name,
                "category": category_name,
                "score": score,
                "document": doc_text
            })

            context_blocks.append(
                f"--- SOURCE [{i}] ---\n"
                f"File: {source_name} (Page {page_num})\n"
                f"Section: {section_name} ({category_name})\n"
                f"Content:\n{doc_text}\n"
            )

        context_str = "\n".join(context_blocks)

        execution_trace.append({
            "step": 1,
            "tool": "direct_grounded_retrieval",
            "args": {
                "model": model_name,
                "sources_retrieved": len(collected_sources),
                "reason": reason or f"Model '{model_name}' operates with direct grounded retrieval."
            },
            "summary": f"Switched to Direct Grounded Retrieval with {len(collected_sources)} verified Pega source excerpts."
        })

        system_prompt = (
            "You are the Senior Pega Certified Lead System Architect (CLSA) AI Assistant.\n"
            "You possess authoritative knowledge of Pega Platform architecture, Enterprise Class Structure (ECS), "
            "Constellation UI, background processing, security, and deployment.\n\n"
            "CRITICAL ZERO-HALLUCINATION RULES:\n"
            "1. Answer strictly and exclusively using the facts, rules, and architectural patterns provided in the excerpts below.\n"
            "2. For EVERY architectural claim, rule name, or configuration step, cite the exact source file and page: e.g. `[01_fundamental-concepts-constellation.pdf, Page 5]`.\n"
            "3. If the provided context does not contain enough information to answer the question, clearly state: 'The provided Pega curriculum documentation does not specify [X].' NEVER invent rules, classes, or patterns.\n"
            "4. Structure your response clearly with headings, bullet points, and code/rule names in backticks (`Rule-Obj-Class`, `Queue Processor`, etc.)."
        )

        user_content = (
            f"USER QUESTION:\n{query}\n\n"
            f"OFFICIAL PEGA ACADEMY CONTEXT EXCERPTS:\n"
            f"{context_str}\n\n"
            f"Please provide an authoritative, zero-hallucination architectural answer with exact page citations based strictly on the context above:"
        )

        messages = [{"role": "system", "content": system_prompt}]
        if chat_history:
            for h in chat_history[-6:]:
                messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": user_content})

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=2500
            )
            return {
                "answer": response.choices[0].message.content or "No response generated.",
                "trace": execution_trace,
                "sources": collected_sources
            }
        except Exception as e:
            err_str = str(e).lower()
            is_model_issue = any(x in err_str for x in [
                "model_not_found", "model_terms_required", "model_decommissioned",
                "decommissioned", "deprecated", "does not exist", "not have access"
            ])
            if is_model_issue:
                alt_model = self.find_best_alternative_model(client, model_name)
                if alt_model and alt_model != model_name:
                    try:
                        response = client.chat.completions.create(
                            model=alt_model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=2500
                        )
                        return {
                            "answer": f"> *Note: Requested model `{model_name}` was unavailable. Auto-switched to `{alt_model}`.*\n\n" + (response.choices[0].message.content or ""),
                            "trace": execution_trace,
                            "sources": collected_sources
                        }
                    except Exception as retry_err:
                        return {
                            "answer": f"❌ **Groq API Error:** {str(retry_err)}",
                            "trace": execution_trace,
                            "sources": collected_sources
                        }
            return {
                "answer": f"❌ **Groq API Error:** {str(e)}",
                "trace": execution_trace,
                "sources": collected_sources
            }

    # ---------------------------------------------------------
    # Autonomous Agent Execution Loop
    # ---------------------------------------------------------

    def run_agent(
        self,
        query: str,
        groq_api_key: str,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.0,
        max_iterations: int = 4,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Runs an autonomous agent loop powered by Groq Function Calling:
        1. Agent plans which tools to call (curriculum search, challenge search, or full-page read).
        2. Executes tools locally against ChromaDB & PDFs.
        3. Feeds verified evidence back to the model.
        4. If a model does not support tool calling (or encounters a Groq 400), automatically
           falls back to Direct Grounded Retrieval without failing.
        """
        if not groq_api_key or not groq_api_key.strip():
            return {
                "answer": "⚠️ **Groq API Key Required:** Please provide your Groq API key in the sidebar to activate the Agent. You can get one free at [console.groq.com/keys](https://console.groq.com/keys).",
                "trace": [],
                "sources": []
            }

        if not model_name:
            model_name = "llama-3.3-70b-versatile"

        client = Groq(api_key=groq_api_key.strip())
        execution_trace = []
        collected_sources = []

        # Known non-tool models or models without function calling support
        model_low = model_name.lower()
        known_no_tools = any(x in model_low for x in [
            "deepseek", "qwen", "gemma", "distill", "guard", "audio", "orpheus"
        ])
        if known_no_tools:
            return self._execute_direct_grounded_fallback(
                query=query,
                client=client,
                model_name=model_name,
                temperature=temperature,
                chat_history=chat_history,
                execution_trace=execution_trace,
                reason=f"Model '{model_name}' operates with Direct Grounded Retrieval (tool calling not supported by model architecture)."
            )

        tools_schema = self.get_agent_tools_schema()

        system_prompt = (
            "You are the Senior Pega Certified Lead System Architect (CLSA) Autonomous AI Agent.\n"
            "You possess authoritative knowledge of Pega Platform architecture, Enterprise Class Structure (ECS), "
            "Constellation UI, background processing, security, and deployment.\n\n"
            "CRITICAL OPERATIONAL RULES (ZERO-HALLUCINATION POLICY):\n"
            "1. MANDATORY TOOL USE: You do NOT rely on ungrounded memory. You MUST call your tools (`search_curriculum`, `search_challenges`, `read_full_page`, `lookup_pega_terms`) to retrieve verified Pega documentation excerpts before answering.\n"
            "2. MULTI-STEP REASONING: If a user query requires both architectural theory and practical steps, query both `search_curriculum` and `search_challenges`.\n"
            "3. INCOMPLETE CHUNKS: If a retrieved chunk looks incomplete or references a crucial diagram/table, call `read_full_page` to inspect the full page.\n"
            "4. CITATIONS: Every architectural claim, rule name, and step MUST be accompanied by a verifiable citation, e.g., `[01_fundamental-concepts-constellation.pdf, Page 5]`.\n"
            "5. ADMIT UNKNOWN: If the official curriculum does not contain information on the query after searching, explicitly state: 'The official Pega Lead System Architect curriculum does not specify [X].' NEVER invent or extrapolate Pega rules."
        )

        messages = [{"role": "system", "content": system_prompt}]

        # Append recent conversation history if provided
        if chat_history:
            for h in chat_history[-6:]:
                messages.append({"role": h["role"], "content": h["content"]})

        messages.append({"role": "user", "content": query})

        current_iteration = 0

        while current_iteration < max_iterations:
            current_iteration += 1

            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=tools_schema,
                    tool_choice="auto",
                    temperature=temperature,
                    max_tokens=2500
                )
            except Exception as e:
                err_str = str(e).lower()

                # Check 1: Tool calling is not supported with this model
                is_tool_err = any(x in err_str for x in [
                    "tool calling is not supported",
                    "tool calling",
                    "tools is not supported",
                    "does not support tools",
                    "param': 'tool calling'",
                    "param': 'tools'"
                ])
                if is_tool_err:
                    return self._execute_direct_grounded_fallback(
                        query=query,
                        client=client,
                        model_name=model_name,
                        temperature=temperature,
                        chat_history=chat_history,
                        execution_trace=execution_trace,
                        reason=f"Groq API reported tool calling unsupported for model '{model_name}'. Auto-switched to Direct Grounded Retrieval."
                    )

                # Check 2: Model decommissioned, not found, or requires terms
                is_model_issue = any(x in err_str for x in [
                    "model_not_found", "model_terms_required", "model_decommissioned",
                    "decommissioned", "deprecated", "does not exist", "not have access"
                ])
                if is_model_issue:
                    alt_model = self.find_best_alternative_model(client, model_name)
                    if alt_model and alt_model != model_name:
                        model_name = alt_model
                        # If alternative model is known non-tool, fallback immediately
                        if any(x in alt_model.lower() for x in ["deepseek", "qwen", "gemma", "distill"]):
                            return self._execute_direct_grounded_fallback(
                                query=query,
                                client=client,
                                model_name=model_name,
                                temperature=temperature,
                                chat_history=chat_history,
                                execution_trace=execution_trace,
                                reason=f"Auto-switched from unavailable model to '{model_name}' (Direct Grounded Retrieval)."
                            )
                        try:
                            response = client.chat.completions.create(
                                model=model_name,
                                messages=messages,
                                tools=tools_schema,
                                tool_choice="auto",
                                temperature=temperature,
                                max_tokens=2500
                            )
                        except Exception as retry_err:
                            retry_err_str = str(retry_err).lower()
                            if any(x in retry_err_str for x in ["tool calling", "tools"]):
                                return self._execute_direct_grounded_fallback(
                                    query=query,
                                    client=client,
                                    model_name=model_name,
                                    temperature=temperature,
                                    chat_history=chat_history,
                                    execution_trace=execution_trace,
                                    reason=f"Model '{model_name}' does not support tools. Switched to Direct Grounded Retrieval."
                                )
                            return {
                                "answer": f"❌ **Groq API Error:** {str(retry_err)}",
                                "trace": execution_trace,
                                "sources": collected_sources
                            }
                    else:
                        return {
                            "answer": f"❌ **Groq API Error:** {str(e)}",
                            "trace": execution_trace,
                            "sources": collected_sources
                        }
                else:
                    return {
                        "answer": f"❌ **Groq API Error:** {str(e)}",
                        "trace": execution_trace,
                        "sources": collected_sources
                    }

            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            # If the model produced a final answer without calling tools
            if not tool_calls:
                return {
                    "answer": response_message.content or "No response generated.",
                    "trace": execution_trace,
                    "sources": collected_sources
                }

            # Convert response message to dict and append to message history
            # The tool_calls must be preserved in the assistant message
            messages.append(response_message)

            # Execute each tool call
            for tool_call in tool_calls:
                func_name = tool_call.function.name
                func_args_str = tool_call.function.arguments or "{}"
                
                try:
                    func_args = json.loads(func_args_str)
                except Exception:
                    func_args = {}

                # Execute appropriate tool
                tool_output_dict = {}
                if func_name == "search_curriculum":
                    tool_output_dict = self.tool_search_curriculum(
                        query=func_args.get("query", query),
                        section=func_args.get("section")
                    )
                elif func_name == "search_challenges":
                    tool_output_dict = self.tool_search_challenges(
                        query=func_args.get("query", query)
                    )
                elif func_name == "read_full_page":
                    tool_output_dict = self.tool_read_full_page(
                        document_name=func_args.get("document_name", ""),
                        page_number=int(func_args.get("page_number", 1))
                    )
                elif func_name == "lookup_pega_terms":
                    tool_output_dict = self.tool_lookup_pega_terms(
                        term=func_args.get("term", query)
                    )
                else:
                    tool_output_dict = {"status": "error", "message": f"Unknown tool '{func_name}'"}

                # Collect sources for UI display
                if "results" in tool_output_dict:
                    for r in tool_output_dict["results"]:
                        collected_sources.append({
                            "source": r.get("document", "Unknown"),
                            "page": str(r.get("page", "1")),
                            "section": r.get("section", "General"),
                            "category": "Hands-on Challenge" if "challenge" in r.get("document", "").lower() else "Master Module",
                            "score": float(r.get("relevance_score", "0%").replace("%", "")),
                            "document": r.get("excerpt", "")
                        })
                elif "full_page_text" in tool_output_dict:
                    collected_sources.append({
                        "source": tool_output_dict.get("document", "Unknown"),
                        "page": str(tool_output_dict.get("page", "1")),
                        "section": "Direct Page Read",
                        "category": "Full Page Read",
                        "score": 100.0,
                        "document": tool_output_dict.get("full_page_text", "")[:600] + "..."
                    })

                # Log trace step
                execution_trace.append({
                    "step": current_iteration,
                    "tool": func_name,
                    "args": func_args,
                    "summary": f"Executed `{func_name}` with {func_args}"
                })

                # Append tool response message
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": json.dumps(tool_output_dict, ensure_ascii=False)
                })

        # Final synthesis if loop exhausted
        try:
            final_res = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=2500
            )
            return {
                "answer": final_res.choices[0].message.content or "",
                "trace": execution_trace,
                "sources": collected_sources
            }
        except Exception as e:
            return {
                "answer": f"Completed research iterations. Trace: {len(execution_trace)} actions. Error on final pass: {e}",
                "trace": execution_trace,
                "sources": collected_sources
            }

    # ---------------------------------------------------------
    # Legacy Direct Generator (Backwards-compatible)
    # ---------------------------------------------------------


    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        groq_api_key: str,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.2
    ) -> str:
        """Sends context and question to Groq LLM for a grounded, accurate response with citations."""
        if not groq_api_key or not groq_api_key.strip():
            return (
                "⚠️ **Groq API Key Required:** Please provide your Groq API key in the sidebar to generate answers. "
                "You can get a free key instantly at [https://console.groq.com/keys](https://console.groq.com/keys)."
            )

        # Build context prompt
        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            block = (
                f"--- SOURCE [{i}] ---\n"
                f"File: {chunk['source']} (Page {chunk['page']})\n"
                f"Section: {chunk['section']} ({chunk['category']})\n"
                f"Content:\n{chunk['document']}\n"
            )
            context_blocks.append(block)

        context_str = "\n".join(context_blocks)

        system_prompt = (
            "You are an expert Pega Certified Lead System Architect (CLSA) AI assistant.\n"
            "Your task is to provide accurate, authoritative, and practical architectural advice and answers based EXCLUSIVELY on the provided Pega Academy documentation excerpts.\n\n"
            "CRITICAL GUIDELINES:\n"
            "1. Grounding: Answer strictly using the facts and architectural principles described in the context. Do not invent rules, classes, or patterns not supported by the context.\n"
            "2. Specific Citations: Whenever you state an architectural rule, pattern, best practice, or solution step, explicitly cite the source file and page number, e.g. `[01_fundamental-concepts-constellation.pdf, Page 4]`.\n"
            "3. Structure: Use clear formatting, bullet points, and code/rule names in backticks (`Rule-Obj-Class`, `Queue Processor`, etc.).\n"
            "4. Missing Context: If the provided context does not contain enough information to fully answer the question, state what is known from the context and clearly mention what specific details are absent."
        )

        user_prompt = (
            f"USER QUESTION:\n{query}\n\n"
            f"RELEVANT PEGA ACADEMY CONTEXT EXCERPTS:\n"
            f"{context_str}\n\n"
            f"Please provide a thorough, accurate architectural answer with direct citations:"
        )

        try:
            if not model_name:
                model_name = "llama-3.3-70b-versatile"

            client = Groq(api_key=groq_api_key.strip())
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    max_tokens=2048
                )
                return response.choices[0].message.content
            except Exception as model_err:
                err_lower = str(model_err).lower()
                is_model_err = any(x in err_lower for x in [
                    "model_not_found", "model_terms_required", "model_decommissioned",
                    "decommissioned", "deprecated", "does not exist", "not have access"
                ])
                if is_model_err:
                    alt_model = self.find_best_alternative_model(client, model_name)
                    if alt_model and alt_model != model_name:
                        response = client.chat.completions.create(
                            model=alt_model,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            temperature=temperature,
                            max_tokens=2048
                        )
                        return f"> *Note: Requested model `{model_name}` was unavailable. Auto-switched to `{alt_model}`.*\n\n" + response.choices[0].message.content
                raise model_err
        except Exception as e:
            return f"❌ **Error generating response from Groq:** {str(e)}"
