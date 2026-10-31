"""
Container entrypoint: make sure the vector index exists, then start the API.

Why this exists: data/chroma is (rightly) not committed to git, so a fresh
clone, or a brand-new Docker volume, starts with an EMPTY document index and
every document question would answer "No indexed documents". On startup we
check the index; if it is empty we build it from the PDFs in data/pdfs/.

Never destructive: an index that already has content (including documents
uploaded through the UI) is left exactly as it is.
"""
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))


def ensure_index(get_collection=None, build_index=None, log=print) -> str:
    """Returns 'existing', 'built' or 'empty'. The two callables are
    injectable only so this logic can be unit-tested without ChromaDB or the
    embedding model."""
    if get_collection is None or build_index is None:
        from rag_pipeline.retrieve import build_index as _build, get_collection as _get
        get_collection = get_collection or _get
        build_index = build_index or _build

    existing = get_collection().count()
    if existing > 0:
        log(f"[startup] Document index already has {existing} chunks - leaving it untouched.")
        return "existing"

    log("[startup] Document index is empty - building it from data/pdfs/ ...")
    added = build_index(reset=False)
    if added == 0:
        log("[startup] WARNING: no readable PDFs found in data/pdfs/. Document questions "
            "will have nothing to search until a PDF is uploaded in the UI.")
        return "empty"
    log(f"[startup] Indexed {added} chunks.")
    return "built"


def main() -> None:
    try:
        ensure_index()
    except Exception as e:  # the SQL side still works without the index
        print(f"[startup] WARNING: could not prepare the document index: {e}", flush=True)

    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()