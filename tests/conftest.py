import hashlib
import sys
from pathlib import Path
from typing import List

import docx
import fitz
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import rag_engine

PROD_WORKSPACE_DIR = rag_engine.WORKSPACE_DIR.resolve()
PROD_DOCS_DIR = rag_engine.DOCS_DIR.resolve()
PROD_DB_DIR = rag_engine.DB_DIR.resolve()
PROD_MANIFEST_PATH = rag_engine.MANIFEST_PATH.resolve()


class DummyEmbeddingFunction:
    """Deterministic offline embedding function for tests (no model downloads)."""

    def __init__(self, *args, **kwargs):
        self._dim = 32

    @staticmethod
    def name() -> str:
        return "dummy-offline-test-embeddings"

    @staticmethod
    def is_legacy() -> bool:
        return False

    def _embed_one(self, text: str) -> List[float]:
        vec = [0.0] * self._dim
        for tok in (text or "").lower().split():
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            idx = digest[0] % self._dim
            vec[idx] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        else:
            vec[0] = 1.0
        return vec

    def __call__(self, input: List[str]) -> List[List[float]]:
        return [self._embed_one(doc) for doc in input]

    def embed_query(self, input: List[str]) -> List[List[float]]:
        return self(input)


@pytest.fixture
def isolated_env(tmp_path: Path, monkeypatch):
    """
    Provides an isolated temporary workspace, DOCS_DIR, ChromaDB directory, and manifest.
    Prevents any access to production pega_rag_db or course documents and blocks model downloads.
    """
    tmp_ws = tmp_path / "workspace"
    tmp_docs = tmp_ws / "Lead System Architect"
    tmp_db = tmp_ws / "pega_rag_db"
    tmp_docs.mkdir(parents=True, exist_ok=True)
    tmp_db.mkdir(parents=True, exist_ok=True)
    tmp_manifest = tmp_db / "indexed_manifest.json"

    assert tmp_ws.resolve() != PROD_WORKSPACE_DIR
    assert tmp_docs.resolve() != PROD_DOCS_DIR
    assert tmp_db.resolve() != PROD_DB_DIR
    assert tmp_manifest.resolve() != PROD_MANIFEST_PATH

    monkeypatch.setattr(rag_engine, "WORKSPACE_DIR", tmp_ws)
    monkeypatch.setattr(rag_engine, "DOCS_DIR", tmp_docs)
    monkeypatch.setattr(rag_engine, "DB_DIR", tmp_db)
    monkeypatch.setattr(rag_engine, "MANIFEST_PATH", tmp_manifest)
    monkeypatch.setattr(
        rag_engine.embedding_functions,
        "SentenceTransformerEmbeddingFunction",
        DummyEmbeddingFunction,
    )
    monkeypatch.setattr(rag_engine.CrossEncoderReranker, "_load_model", lambda self: False)

    engine = rag_engine.PegaRAGEngine(db_dir=tmp_db)
    assert engine.db_dir.resolve() == tmp_db.resolve()
    assert engine.db_dir.resolve() != PROD_DB_DIR

    return {
        "engine": engine,
        "workspace": tmp_ws,
        "docs_dir": tmp_docs,
        "db_dir": tmp_db,
        "manifest_path": tmp_manifest,
    }


def create_sample_pdf(path: Path, pages: List[str]) -> Path:
    """Creates a controlled multi-page PDF fixture using PyMuPDF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()
    return path


def create_sample_docx(
    path: Path,
    include_empty_table: bool = True,
    include_large_table: bool = False,
) -> Path:
    """Creates a controlled DOCX fixture with Heading 1/2/3, paragraphs, and tables."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = docx.Document()
    doc.add_heading("1. Executive Overview", level=1)
    doc.add_paragraph("Queue Processor handles asynchronous background work in Pega.")

    doc.add_heading("2. Regional Revenue", level=2)
    tbl = doc.add_table(rows=3, cols=3)
    headers = ["Region", "Target", "Actual"]
    rows = [["UAE", "$1,800,000", "$1,800,000"], ["India", "$800,000", "$750,000"]]
    for c_idx, h in enumerate(headers):
        tbl.rows[0].cells[c_idx].text = h
    for r_idx, row_vals in enumerate(rows, start=1):
        for c_idx, val in enumerate(row_vals):
            tbl.rows[r_idx].cells[c_idx].text = val

    if include_empty_table:
        doc.add_heading("3. Empty Table Section", level=3)
        doc.add_table(rows=2, cols=2)

    if include_large_table:
        doc.add_heading("4. Large Budget Table", level=2)
        large_tbl = doc.add_table(rows=45, cols=3)
        large_tbl.rows[0].cells[0].text = "DepartmentCode"
        large_tbl.rows[0].cells[1].text = "ApprovedBudget"
        large_tbl.rows[0].cells[2].text = "ActualSpend"
        for r in range(1, 45):
            large_tbl.rows[r].cells[0].text = f"DEPT-UNIT-{r:03d}-REGION-EMEA"
            large_tbl.rows[r].cells[1].text = f"${100000 + r * 1000:,}"
            large_tbl.rows[r].cells[2].text = f"${105000 + r * 1000:,}"

    doc.save(path)
    return path
