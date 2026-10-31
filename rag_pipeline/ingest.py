"""
Phase 1 — Member B: unstructured pipeline, ingestion + chunking.

Reads PDFs from data/pdfs/, extracts text per page, and splits into
overlapping chunks. Chunking is "semantic-ish": we split on paragraph
boundaries first and only fall back to a hard character split for
oversized paragraphs, which keeps chunks topically coherent for retrieval.
"""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from pypdf import PdfReader

PDF_DIR = Path(__file__).resolve().parent.parent / "data" / "pdfs"

CHUNK_SIZE = 800       # target characters per chunk
CHUNK_OVERLAP = 120    # characters of overlap between consecutive chunks


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    page: int
    text: str


def extract_pages(pdf_path: Path) -> Iterator[tuple]:
    reader = PdfReader(str(pdf_path))
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        yield i + 1, text


def _split_paragraphs(text: str) -> list:
    paras = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paras if p.strip()]


def chunk_text(paragraphs: list, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list:
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= chunk_size:
            current = f"{current}\n\n{para}".strip()
            continue

        if current:
            chunks.append(current)
            # start next chunk with overlap tail of the previous one
            current = current[-overlap:] + "\n\n" + para
        else:
            # single paragraph longer than chunk_size: hard split
            for i in range(0, len(para), chunk_size - overlap):
                chunks.append(para[i:i + chunk_size])
            current = ""

    if current:
        chunks.append(current)
    return chunks


def ingest_single_pdf(pdf_path: Path, doc_id: str = None) -> list:
    """Ingest and chunk exactly one PDF file. Returns its Chunk list.

    doc_id defaults to the file's stem (matches ingest_pdfs()'s existing
    behavior) but can be overridden — used by the live-upload feature to
    assign a collision-safe ID instead of trusting the user's filename.
    """
    if doc_id is None:
        doc_id = pdf_path.stem
    chunks = []
    for page_num, page_text in extract_pages(pdf_path):
        paras = _split_paragraphs(page_text)
        if not paras:
            continue
        for j, chunk in enumerate(chunk_text(paras)):
            chunks.append(
                Chunk(doc_id=doc_id, chunk_id=f"{doc_id}-p{page_num}-c{j}",
                      page=page_num, text=chunk)
            )
    return chunks


def ingest_pdfs(pdf_dir: Path = PDF_DIR) -> list:
    """Returns a flat list of Chunk objects across every PDF in pdf_dir."""
    all_chunks = []
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {pdf_dir}. Add source documents there.")
        return all_chunks

    for pdf_path in pdf_files:
        all_chunks.extend(ingest_single_pdf(pdf_path))
    print(f"Ingested {len(pdf_files)} PDFs into {len(all_chunks)} chunks.")
    return all_chunks


if __name__ == "__main__":
    ingest_pdfs()