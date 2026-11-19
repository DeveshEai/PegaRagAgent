"""
Document extraction (PDF via PyMuPDF, structured XML-order DOCX via python-docx)
and content hashing utilities for PegaRagAgent.
"""

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import docx
import fitz  # PyMuPDF

from config.settings import TABLE_ATOMIC_CHAR_LIMIT


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


def _normalize_docx_table_markdown(table: Any) -> Dict[str, Any]:
    """
    Converts a python-docx Table into a clean Markdown table representation,
    normalizing whitespace and multiline/merged cells.
    Returns dict with keys: markdown, header_lines, data_lines, rows, cols, is_empty.
    """
    raw_rows: List[List[str]] = []
    for row in table.rows:
        cells_text: List[str] = []
        for cell in row.cells:
            # Collapse internal newlines/tabs/multiple spaces in merged or multiline cells
            cleaned = re.sub(r"\s+", " ", cell.text or "").strip()
            # Escape literal pipes so Markdown column alignment stays valid
            cleaned = cleaned.replace("|", "\\|")
            cells_text.append(cleaned)
        if any(c for c in cells_text):
            raw_rows.append(cells_text)

    if not raw_rows:
        return {
            "markdown": "",
            "header_lines": [],
            "data_lines": [],
            "rows": 0,
            "cols": len(table.columns) if hasattr(table, "columns") else 0,
            "is_empty": True,
        }

    num_cols = max(len(r) for r in raw_rows)
    padded_rows = [r + [""] * (num_cols - len(r)) for r in raw_rows]

    header = padded_rows[0]
    header_line = "| " + " | ".join(header) + " |"
    sep_line = "| " + " | ".join(["---"] * num_cols) + " |"
    data_lines = ["| " + " | ".join(r) + " |" for r in padded_rows[1:]]

    all_lines = [header_line, sep_line] + data_lines
    return {
        "markdown": "\n".join(all_lines),
        "header_lines": [header_line, sep_line],
        "data_lines": data_lines,
        "rows": len(padded_rows),
        "cols": num_cols,
        "is_empty": False,
    }


def _split_markdown_table_by_rows(
    header_lines: List[str],
    data_lines: List[str],
    prefix_context: str,
    max_chars: int = TABLE_ATOMIC_CHAR_LIMIT
) -> List[str]:
    """
    Splits an oversized Markdown table at row boundaries while repeating
    the header + separator rows and context prefix in every chunk.
    """
    header_block = "\n".join(header_lines)
    if not data_lines:
        return [f"{prefix_context}\n{header_block}".strip()]

    chunks: List[str] = []
    current_rows: List[str] = []
    base_len = len(prefix_context) + 1 + len(header_block) + 1

    for row_line in data_lines:
        projected_len = base_len + sum(len(r) + 1 for r in current_rows) + len(row_line)
        if current_rows and projected_len > max_chars:
            chunk_body = f"{prefix_context}\n{header_block}\n" + "\n".join(current_rows)
            chunks.append(chunk_body.strip())
            current_rows = [row_line]
        else:
            current_rows.append(row_line)

    if current_rows:
        chunk_body = f"{prefix_context}\n{header_block}\n" + "\n".join(current_rows)
        chunks.append(chunk_body.strip())

    return chunks


def extract_docx_sections(file_path: Path) -> List[Dict[str, Any]]:
    """
    Extracts structured sections from DOCX documents in true XML document order,
    preserving Heading 1/2/3 hierarchy, paragraphs, and Markdown-formatted tables.
    """
    sections_data: List[Dict[str, Any]] = []
    para_count = 0
    heading_count = 0
    table_count = 0
    empty_table_count = 0
    table_dims: List[str] = []

    try:
        from docx.oxml.text.paragraph import CT_P
        from docx.oxml.table import CT_Tbl
        from docx.text.paragraph import Paragraph
        from docx.table import Table

        doc = docx.Document(file_path)
        heading_stack: Dict[int, Optional[str]] = {1: None, 2: None, 3: None}
        current_heading = "General Overview"
        current_heading_path = "General Overview"
        current_blocks: List[Dict[str, Any]] = []
        prose_buffer: List[str] = []

        def _flush_prose() -> None:
            if prose_buffer:
                current_blocks.append({
                    "type": "prose",
                    "text": "\n".join(prose_buffer)
                })
                prose_buffer.clear()

        def _flush_section() -> None:
            _flush_prose()
            if current_blocks:
                flat_parts = [
                    b["text"] if b["type"] == "prose" else b["markdown"]
                    for b in current_blocks
                ]
                sections_data.append({
                    "page_number": current_heading,
                    "heading_path": current_heading_path,
                    "text": "\n\n".join(flat_parts),
                    "blocks": list(current_blocks),
                })
                current_blocks.clear()

        def _build_heading_path() -> str:
            parts = [heading_stack[lvl] for lvl in (1, 2, 3) if heading_stack[lvl]]
            return " > ".join(parts) if parts else current_heading

        for child in doc.element.body.iterchildren():
            if isinstance(child, CT_P):
                p = Paragraph(child, doc)
                text = re.sub(r"\s+", " ", p.text or "").strip()
                if not text:
                    continue
                style_name = (p.style.name if p.style and p.style.name else "").strip()
                if style_name.startswith("Heading") or style_name == "Title":
                    _flush_section()
                    heading_count += 1
                    lvl_match = re.search(r"Heading\s+(\d+)", style_name)
                    if lvl_match:
                        lvl = min(3, max(1, int(lvl_match.group(1))))
                        heading_stack[lvl] = text
                        for sub_lvl in range(lvl + 1, 4):
                            heading_stack[sub_lvl] = None
                    else:
                        heading_stack[1] = text
                        heading_stack[2] = None
                        heading_stack[3] = None
                    current_heading = text
                    current_heading_path = _build_heading_path()
                else:
                    para_count += 1
                    prose_buffer.append(text)

            elif isinstance(child, CT_Tbl):
                _flush_prose()
                tbl = Table(child, doc)
                tbl_info = _normalize_docx_table_markdown(tbl)
                if tbl_info["is_empty"]:
                    empty_table_count += 1
                    print(f"[DOCX Extract WARNING] {file_path.name}: skipped empty table under '{current_heading_path}'")
                    continue
                table_count += 1
                table_dims.append(f"{tbl_info['rows']}x{tbl_info['cols']}")
                current_blocks.append({
                    "type": "table",
                    "markdown": tbl_info["markdown"],
                    "header_lines": tbl_info["header_lines"],
                    "data_lines": tbl_info["data_lines"],
                    "rows": tbl_info["rows"],
                    "cols": tbl_info["cols"],
                })

        _flush_section()
        print(
            f"[DOCX Extract] {file_path.name}: paragraphs={para_count}, "
            f"headings={heading_count}, tables={table_count} ({', '.join(table_dims) if table_dims else 'none'}), "
            f"empty_tables={empty_table_count}"
        )
    except Exception as e:
        print(f"Error reading DOCX {file_path.name}: {e}")
    return sections_data


def compute_file_hash(file_path: Path) -> str:
    """Computes SHA-256 content hash of a file's bytes."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)
    return sha256.hexdigest()


def _legacy_md5_hash(file_path: Path) -> str:
    """Legacy filename/size/mtime MD5 hash used only for backward-compatible manifest migration."""
    stat = file_path.stat()
    hash_str = f"{file_path.name}_{stat.st_size}_{stat.st_mtime}"
    return hashlib.md5(hash_str.encode()).hexdigest()
