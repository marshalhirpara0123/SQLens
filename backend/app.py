# """
# Phase 2 — Member C: unified /chat endpoint wired to the LangGraph router.

# /sql/ask and /rag/ask (Phase 1) are kept as-is — they're useful for
# isolated debugging and for the Phase 3 ablation study, which needs to
# call the SQL agent directly without going through routing. /chat is the
# new, primary, user-facing endpoint.
# """
# import re
# import sys
# import uuid
# from pathlib import Path

# sys.path.append(str(Path(__file__).resolve().parent.parent))

# from fastapi import FastAPI, HTTPException, UploadFile, File
# from fastapi.responses import HTMLResponse
# from pydantic import BaseModel

# from sql_agent.agent import ask as sql_ask
# from rag_pipeline.retrieve import answer as rag_answer, add_pdf_to_index, list_indexed_documents, delete_document
# from backend.llm_client import benchmark, LLMError
# from backend.graph import run_query

# app = FastAPI(title="Omni-Agent Analytics")

# # --- Live PDF upload (Phase 4) ---
# # Uploads are saved under their own subfolder, separate from the original
# # demo PDFs in data/pdfs/, and are gitignored — they're live/local additions,
# # not part of the committed demo baseline. Permanent/shared: once uploaded,
# # a document is indexed for every future question, by every user, until the
# # server's ChromaDB collection is reset.
# UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "pdfs" / "uploads"
# MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
# PDF_MAGIC = b"%PDF-"


# def _safe_doc_id_fragment(filename: str) -> str:
#     """Build a readable doc_id fragment from the user's filename. This is
#     NEVER used as an actual filesystem path (the file itself is always
#     saved under a random uuid name) — it's purely for making citations in
#     the UI look like the original filename instead of a random hex string."""
#     stem = Path(filename or "document").stem
#     safe = re.sub(r"[^A-Za-z0-9_-]", "-", stem)[:40]
#     return safe or "document"


# class Question(BaseModel):
#     question: str


# @app.get("/health")
# def health():
#     return {"status": "ok", "phase": 2}


# @app.post("/chat")
# def chat_endpoint(q: Question):
#     """Unified endpoint: routes the question through the LangGraph graph
#     (router -> sql_tool and/or rag_tool -> synthesizer) and returns one
#     grounded, cited answer regardless of which source(s) it came from."""
#     question = q.question.strip()
#     if not question:
#         raise HTTPException(status_code=400, detail="Question cannot be empty.")
#     try:
#         result = run_query(question)
#     except LLMError as e:
#         raise HTTPException(status_code=503, detail=str(e))
#     except Exception as e:
#         # Anything unexpected inside the graph (bad DB state, chroma error,
#         # etc.) — surface it as a clean 500 instead of an unhandled crash,
#         # so the UI can show a friendly message instead of hanging.
#         raise HTTPException(status_code=500, detail=f"Unexpected error while answering: {e}")
#     return result


# @app.post("/documents/upload")
# async def upload_document(file: UploadFile = File(...)):
#     """Upload a PDF (max 10MB) to add live to the shared document index.
#     Indexed permanently — every future question from any user can draw on
#     it until the collection is reset. Never trusts the filename for disk
#     paths or the declared content-type; validates actual file content."""
#     if not (file.filename or "").lower().endswith(".pdf"):
#         raise HTTPException(status_code=400, detail="Only .pdf files are accepted.")

#     # Read in chunks with a hard cap — Content-Length can be absent or
#     # spoofed, so we enforce the limit on actual bytes read, not the header.
#     data = bytearray()
#     while True:
#         piece = await file.read(1024 * 1024)
#         if not piece:
#             break
#         data.extend(piece)
#         if len(data) > MAX_UPLOAD_BYTES:
#             raise HTTPException(status_code=413, detail="File exceeds the 10MB limit.")

#     if not data:
#         raise HTTPException(status_code=400, detail="Uploaded file is empty.")
#     if bytes(data[:5]) != PDF_MAGIC:
#         raise HTTPException(
#             status_code=400,
#             detail="File does not look like a valid PDF (missing %PDF- header).",
#         )

#     UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
#     unique_id = uuid.uuid4().hex
#     doc_id = f"{_safe_doc_id_fragment(file.filename)}-{unique_id[:8]}"
#     # Saved under a random name on disk — the user's filename is never used
#     # as a filesystem path, avoiding path-traversal/overwrite risks entirely.
#     disk_path = UPLOAD_DIR / f"{unique_id}.pdf"
#     disk_path.write_bytes(bytes(data))

#     try:
#         n_chunks = add_pdf_to_index(disk_path, doc_id=doc_id)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to index the uploaded PDF: {e}")

#     if n_chunks == 0:
#         raise HTTPException(
#             status_code=422,
#             detail="No extractable text was found in this PDF (it may be scanned/image-only).",
#         )

#     return {
#         "doc_id": doc_id,
#         "original_filename": file.filename,
#         "chunks_indexed": n_chunks,
#     }


# @app.get("/documents")
# def documents_endpoint():
#     """List every document currently searchable by the RAG pipeline —
#     the original demo PDFs plus anything uploaded live."""
#     return {"documents": list_indexed_documents()}


# @app.delete("/documents/{doc_id}")
# def remove_document(doc_id: str):
#     """Remove a document (and all its chunks) from the shared index —
#     used to clean up a live upload, including test uploads made during
#     development."""
#     n_removed = delete_document(doc_id)
#     if n_removed == 0:
#         raise HTTPException(status_code=404, detail=f"No document found with doc_id '{doc_id}'.")
#     return {"doc_id": doc_id, "chunks_removed": n_removed}


# @app.post("/sql/ask")
# def sql_endpoint(q: Question):
#     try:
#         result = sql_ask(q.question)
#     except LLMError as e:
#         raise HTTPException(status_code=503, detail=str(e))
#     return {
#         "question": result.question,
#         "sql": result.final_sql,
#         "columns": result.columns,
#         "rows": result.rows,
#         "succeeded": result.succeeded,
#         "attempts": result.attempts,
#         "used_self_correction": result.used_self_correction,
#         "error": result.error,
#     }


# @app.post("/rag/ask")
# def rag_endpoint(q: Question):
#     try:
#         result = rag_answer(q.question)
#     except LLMError as e:
#         raise HTTPException(status_code=503, detail=str(e))
#     return result


# @app.get("/llm/benchmark")
# def llm_benchmark():
#     try:
#         return benchmark()
#     except LLMError as e:
#         raise HTTPException(status_code=503, detail=str(e))


# @app.get("/", response_class=HTMLResponse)
# def chat_ui():
#     return Path(__file__).with_name("static_index.html").read_text(encoding="utf-8")

"""
Phase 2 — Member C: unified /chat endpoint wired to the LangGraph router.

/sql/ask and /rag/ask (Phase 1) are kept as-is — they're useful for
isolated debugging and for the Phase 3 ablation study, which needs to
call the SQL agent directly without going through routing. /chat is the
new, primary, user-facing endpoint.
"""
import asyncio
import re
import sqlite3
import sys
import uuid
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from sql_agent import db_setup
from sql_agent.agent import ask as sql_ask
from sql_agent.guardrails import UnsafeScriptError
from rag_pipeline.retrieve import answer as rag_answer, add_pdf_to_index, list_indexed_documents, delete_document
from backend.llm_client import benchmark, LLMError
from backend.graph import run_query

app = FastAPI(title="Omni-Agent Analytics")

# --- Live PDF upload (Phase 4) ---
# Uploads are saved under their own subfolder, separate from the original
# demo PDFs in data/pdfs/, and are gitignored — they're live/local additions,
# not part of the committed demo baseline. Permanent/shared: once uploaded,
# a document is indexed for every future question, by every user, until the
# server's ChromaDB collection is reset.
UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "pdfs" / "uploads"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
PDF_MAGIC = b"%PDF-"


def _safe_doc_id_fragment(filename: str) -> str:
    """Build a readable doc_id fragment from the user's filename. This is
    NEVER used as an actual filesystem path (the file itself is always
    saved under a random uuid name) — it's purely for making citations in
    the UI look like the original filename instead of a random hex string."""
    stem = Path(filename or "document").stem
    safe = re.sub(r"[^A-Za-z0-9_-]", "-", stem)[:40]
    return safe or "document"


class Question(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "phase": 2}


@app.post("/chat")
def chat_endpoint(q: Question):
    """Unified endpoint: routes the question through the LangGraph graph
    (router -> sql_tool and/or rag_tool -> synthesizer) and returns one
    grounded, cited answer regardless of which source(s) it came from."""
    question = q.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        result = run_query(question)
    except LLMError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        # Anything unexpected inside the graph (bad DB state, chroma error,
        # etc.) — surface it as a clean 500 instead of an unhandled crash,
        # so the UI can show a friendly message instead of hanging.
        raise HTTPException(status_code=500, detail=f"Unexpected error while answering: {e}")
    return result


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF (max 10MB) to add live to the shared document index.
    Indexed permanently — every future question from any user can draw on
    it until the collection is reset. Never trusts the filename for disk
    paths or the declared content-type; validates actual file content."""
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted.")

    # Read in chunks with a hard cap — Content-Length can be absent or
    # spoofed, so we enforce the limit on actual bytes read, not the header.
    data = bytearray()
    while True:
        piece = await file.read(1024 * 1024)
        if not piece:
            break
        data.extend(piece)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="File exceeds the 10MB limit.")

    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if bytes(data[:5]) != PDF_MAGIC:
        raise HTTPException(
            status_code=400,
            detail="File does not look like a valid PDF (missing %PDF- header).",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    unique_id = uuid.uuid4().hex
    doc_id = f"{_safe_doc_id_fragment(file.filename)}-{unique_id[:8]}"
    # Saved under a random name on disk — the user's filename is never used
    # as a filesystem path, avoiding path-traversal/overwrite risks entirely.
    disk_path = UPLOAD_DIR / f"{unique_id}.pdf"
    disk_path.write_bytes(bytes(data))

    try:
        n_chunks = add_pdf_to_index(disk_path, doc_id=doc_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to index the uploaded PDF: {e}")

    if n_chunks == 0:
        raise HTTPException(
            status_code=422,
            detail="No extractable text was found in this PDF (it may be scanned/image-only).",
        )

    return {
        "doc_id": doc_id,
        "original_filename": file.filename,
        "chunks_indexed": n_chunks,
    }


@app.get("/documents")
def documents_endpoint():
    """List every document currently searchable by the RAG pipeline —
    the original demo PDFs plus anything uploaded live."""
    return {"documents": list_indexed_documents()}


@app.delete("/documents/{doc_id}")
def remove_document(doc_id: str):
    """Remove a document (and all its chunks) from the shared index —
    used to clean up a live upload, including test uploads made during
    development."""
    n_removed = delete_document(doc_id)
    if n_removed == 0:
        raise HTTPException(status_code=404, detail=f"No document found with doc_id '{doc_id}'.")
    return {"doc_id": doc_id, "chunks_removed": n_removed}


# ---------------------------------------------------------------------------
# Live database upload (Phase 4): replace the active database with one built
# from a user-supplied .sql script. One active database at a time; Chinook is
# the default and can always be restored. The script is validated by its own
# narrow guardrail (sql_agent/guardrails.py::validate_schema_script) and
# executed against a brand-new isolated file — never the live database.
# ---------------------------------------------------------------------------

@app.get("/database/status")
def database_status():
    return {
        "active": db_setup.get_active_db_label(),
        "is_default": db_setup.is_using_default_db(),
        "tables": db_setup.list_tables(),
    }


@app.post("/database/upload")
async def upload_database(file: UploadFile = File(...)):
    filename = file.filename or ""
    if not filename.lower().endswith(".sql"):
        raise HTTPException(status_code=400, detail="Only .sql files are accepted.")

    # Same pattern as the PDF upload: enforce the cap on bytes actually read.
    data = bytearray()
    while True:
        piece = await file.read(1024 * 1024)
        if not piece:
            break
        data.extend(piece)
        if len(data) > db_setup.MAX_DB_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="File exceeds the 10MB limit.")

    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if b"\x00" in data:
        raise HTTPException(status_code=400, detail="This doesn't look like a text .sql script (binary content found).")
    try:
        script = bytes(data).decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="The .sql file must be UTF-8 encoded text.")

    try:
        # Building can take seconds for a big script; run it off the event
        # loop so the rest of the app (chat, /health) stays responsive.
        new_path = await asyncio.to_thread(db_setup.build_database_from_script, script)
    except UnsafeScriptError as e:
        raise HTTPException(status_code=422, detail=f"Script rejected by the safety guardrail: {e}")
    except sqlite3.Error as e:
        raise HTTPException(
            status_code=422,
            detail=(
                f"The script could not be executed as SQLite SQL: {e}. "
                "Only SQLite-dialect SQL is supported — dumps from MySQL or PostgreSQL "
                "often use syntax SQLite rejects."
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build the database: {e}")

    db_setup.set_active_db(new_path, label=Path(filename).name[:80])
    return {
        "active": db_setup.get_active_db_label(),
        "is_default": False,
        "tables": db_setup.list_tables(),
    }


@app.post("/database/reset")
def reset_database():
    """Revert to the bundled Chinook demo database."""
    db_setup.reset_to_default_db()
    return {
        "active": db_setup.get_active_db_label(),
        "is_default": True,
        "tables": db_setup.list_tables(),
    }


@app.post("/sql/ask")
def sql_endpoint(q: Question):
    try:
        result = sql_ask(q.question)
    except LLMError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {
        "question": result.question,
        "sql": result.final_sql,
        "columns": result.columns,
        "rows": result.rows,
        "succeeded": result.succeeded,
        "attempts": result.attempts,
        "used_self_correction": result.used_self_correction,
        "error": result.error,
    }


@app.post("/rag/ask")
def rag_endpoint(q: Question):
    try:
        result = rag_answer(q.question)
    except LLMError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return result


@app.get("/llm/benchmark")
def llm_benchmark():
    try:
        return benchmark()
    except LLMError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/", response_class=HTMLResponse)
def chat_ui():
    return Path(__file__).with_name("static_index.html").read_text(encoding="utf-8")