"""
Agent research tool schemas, tool execution helpers, and source normalization
for PegaRagAgent.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF

from config.settings import DOCS_DIR


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
                            "enum": [
                                "All Sections",
                                "01_LSA_Mindset",
                                "02_Planning_and_Design",
                                "03_Constellation_Adoption",
                                "04_Infrastructure_and_Architecture",
                                "05_Implementation_and_Deployment",
                            ],
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


def execute_search_curriculum(
    engine: Any,
    query: str,
    section: Optional[str] = None,
    top_k: int = 4
) -> Dict[str, Any]:
    """Tool: Search theoretical concepts & design patterns in the 28 Master Modules."""
    results = engine.search(
        query=query,
        top_k=top_k,
        section_filter=section,
        category_filter="Master Module"
    )
    if not results:
        results = engine.search(query=query, top_k=top_k, section_filter=section)

    formatted = []
    for r in results[:top_k]:
        formatted.append({
            "document": r["source"],
            "section": r["section"],
            "page": r["page"],
            "relevance_score": f"{r['score']}%",
            "excerpt": r["document"]
        })
    return {"status": "success", "results_count": len(formatted), "results": formatted}


def execute_search_challenges(engine: Any, query: str, top_k: int = 4) -> Dict[str, Any]:
    """Tool: Search hands-on implementation steps and solutions in the 40 Challenges."""
    results = engine.search(
        query=query,
        top_k=top_k,
        category_filter="Hands-on Challenge"
    )
    if not results:
        results = engine.search(query=query, top_k=top_k)

    formatted = []
    for r in results[:top_k]:
        formatted.append({
            "document": r["source"],
            "section": r["section"],
            "page": r["page"],
            "relevance_score": f"{r['score']}%",
            "excerpt": r["document"]
        })
    return {"status": "success", "results_count": len(formatted), "results": formatted}


def execute_read_full_page(
    document_name: str,
    page_number: int,
    docs_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Tool: Read the complete, untruncated content of a specific page in any Pega PDF."""
    target_docs_dir = docs_dir or DOCS_DIR
    matches = list(target_docs_dir.rglob(document_name))
    if not matches:
        matches = [f for f in target_docs_dir.rglob("*.pdf") if document_name.lower() in f.name.lower()]

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
        doc.close()
        return {
            "status": "error",
            "message": f"Requested page {page_number} is out of bounds (Document has {total_pages} pages)."
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed reading page: {str(e)}"}


def execute_lookup_pega_terms(engine: Any, term: str, top_k: int = 4) -> Dict[str, Any]:
    """Tool: Specialized lookup for exact Pega rules, classes, APIs, or architectural terms."""
    results = engine.search(query=term, top_k=top_k)
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


def collect_tool_sources(
    tool_output_dict: Any,
    collected_sources: List[Dict[str, Any]],
    include_preview: bool = False
) -> None:
    """Safely extracts normalized source dicts from a tool output payload."""
    if not isinstance(tool_output_dict, dict):
        return

    if isinstance(tool_output_dict.get("results"), list):
        for r in tool_output_dict["results"]:
            if not isinstance(r, dict):
                continue
            doc_label = str(r.get("document") or "Unknown")
            rel_score = str(r.get("relevance_score") or "0%").replace("%", "")
            try:
                score_val = float(rel_score)
            except ValueError:
                score_val = 0.0
            collected_sources.append({
                "source": doc_label,
                "page": str(r.get("page") or "1"),
                "section": r.get("section") or "General",
                "category": "Hands-on Challenge" if "challenge" in doc_label.lower() else "Master Module",
                "score": score_val,
                "document": r.get("excerpt") or ""
            })
    elif "full_page_text" in tool_output_dict:
        full_text = tool_output_dict.get("full_page_text") or ""
        entry: Dict[str, Any] = {
            "source": tool_output_dict.get("document") or "Unknown",
            "page": str(tool_output_dict.get("page") or "1"),
            "section": "Direct Page Read",
            "category": "Full Page Read",
            "score": 100.0,
            "document": full_text,
        }
        if include_preview:
            entry["preview"] = full_text[:600] + ("..." if len(full_text) > 600 else "")
        collected_sources.append(entry)
