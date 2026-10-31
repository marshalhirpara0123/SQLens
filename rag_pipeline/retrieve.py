"""
Phase 1 — Member B: local embeddings, ChromaDB indexing, retrieval + answer.

Uses sentence-transformers (all-MiniLM-L6-v2, runs on CPU, no API key)
for embeddings, persisted in a local ChromaDB collection.

Phase 2 fix: answer() now filters out chunks that are too far away in
embedding space to be genuinely relevant, instead of always using the
top-K nearest chunks regardless of distance. Without this, a query with
no good match in the index (e.g. "how does the FAQ describe genre
categorization?" when the FAQ never discusses that) still returned its
4 nearest chunks as "sources" even though none of them were relevant —
misleading in the UI and in the hybrid-answer synthesizer.

Phase 4 addition: add_pdf_to_index() and list_indexed_documents() support
the live PDF-upload feature. add_pdf_to_index() upserts ONE new file into
the existing collection without touching anything already indexed —
unlike build_index(reset=True), which wipes and rebuilds everything from
data/pdfs/ and is only meant for the original one-time setup.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import chromadb
from sentence_transformers import SentenceTransformer

from backend.llm_client import generate
from rag_pipeline.ingest import ingest_pdfs, ingest_single_pdf

CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma"
COLLECTION_NAME = "omni_agent_docs"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 4

# ChromaDB's default distance space is L2 (squared-then-rooted euclidean).
# all-MiniLM-L6-v2 embeddings are normalized (unit length), so for this
# model L2 distance ranges roughly 0 (identical) to 2 (opposite meaning).
# For normalized vectors, L2 = sqrt(2 * (1 - cosine_similarity)), so an
# L2 distance of ~1.0 corresponds to cosine similarity ~0.5 — a common
# "no longer relevant" cutoff. This is a reasonable starting default, NOT
# a guaranteed-correct number for your specific documents — run this file
# directly (see __main__ below) to see real distances on your data and
# adjust DISTANCE_THRESHOLD if it's rejecting good matches or accepting
# bad ones.
DISTANCE_THRESHOLD = 1.0

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(COLLECTION_NAME)


def build_index(reset: bool = False) -> int:
    """Ingest PDFs, embed chunks, and upsert into the ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    collection = client.get_or_create_collection(COLLECTION_NAME)

    chunks = ingest_pdfs()
    if not chunks:
        return 0

    embedder = get_embedder()
    texts = [c.text for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"doc_id": c.doc_id, "page": c.page} for c in chunks],
    )
    return len(chunks)


def add_pdf_to_index(pdf_path: Path, doc_id: str = None) -> int:
    """Ingest and index ONE additional PDF into the EXISTING collection,
    without touching anything already indexed. Returns the number of
    chunks added (0 if the PDF had no extractable text).

    Used by the live PDF-upload feature. Deliberately does not call
    build_index(reset=True) — that wipes the whole collection, which would
    delete every previously uploaded/indexed document just to add one more.
    """
    chunks = ingest_single_pdf(pdf_path, doc_id=doc_id)
    if not chunks:
        return 0

    collection = get_collection()
    embedder = get_embedder()
    texts = [c.text for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"doc_id": c.doc_id, "page": c.page} for c in chunks],
    )
    return len(chunks)


def list_indexed_documents() -> list:
    """Returns a sorted list of distinct doc_ids currently in the
    collection — the original 3 PDFs plus anything added via
    add_pdf_to_index(). Used by the UI to show what's searchable."""
    collection = get_collection()
    data = collection.get(include=["metadatas"])
    doc_ids = {m.get("doc_id") for m in data.get("metadatas", []) if m.get("doc_id")}
    return sorted(doc_ids)


def delete_document(doc_id: str) -> int:
    """Remove every chunk belonging to doc_id from the collection. Returns
    the number of chunks removed (0 if doc_id wasn't found). Used to clean
    up test uploads and to let the UI remove a live-uploaded document."""
    collection = get_collection()
    matches = collection.get(where={"doc_id": doc_id})
    ids = matches.get("ids", [])
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def retrieve(query: str, top_k: int = TOP_K) -> list:
    """Raw retrieval — returns the top_k nearest chunks with their distances,
    unfiltered. Relevance filtering happens in answer(), not here, so this
    function stays useful for debugging/calibration (see __main__)."""
    collection = get_collection()
    embedder = get_embedder()
    query_emb = embedder.encode([query]).tolist()

    results = collection.query(query_embeddings=query_emb, n_results=top_k)
    hits = []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    for i in range(len(ids)):
        hits.append({
            "chunk_id": ids[i],
            "text": docs[i],
            "doc_id": metas[i].get("doc_id"),
            "page": metas[i].get("page"),
            "distance": dists[i],
        })
    return hits


ANSWER_SYSTEM_PROMPT = """Answer the user's question using ONLY the provided context.
If the context does not contain the answer, say you don't know — do not guess.
Cite sources inline like [doc_id p.page] after any claim drawn from that source."""

NO_RELEVANT_DOCS_MESSAGE = (
    "I don't have any indexed documents that are relevant to this question."
)


def answer(query: str, top_k: int = TOP_K, model: str = None,
           distance_threshold: float = DISTANCE_THRESHOLD) -> dict:
    hits = retrieve(query, top_k=top_k)
    if not hits:
        return {"answer": "No indexed documents to search. Run build_index() first.",
                "sources": []}

    relevant_hits = [h for h in hits if h["distance"] <= distance_threshold]
    if not relevant_hits:
        # Nearest chunks exist but are too far away to be genuinely
        # relevant — report honestly instead of citing noise.
        return {"answer": NO_RELEVANT_DOCS_MESSAGE, "sources": []}

    context = "\n\n".join(
        f"[{h['doc_id']} p.{h['page']}]\n{h['text']}" for h in relevant_hits
    )
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = generate(prompt, model=model, system=ANSWER_SYSTEM_PROMPT)
    return {
        "answer": response,
        "sources": [{"doc_id": h["doc_id"], "page": h["page"], "chunk_id": h["chunk_id"]}
                    for h in relevant_hits],
    }


if __name__ == "__main__":
    n = build_index(reset=True)
    print(f"Indexed {n} chunks.")
    if n and len(sys.argv) > 1:
        query = sys.argv[1]

        # Calibration aid: show raw distances for every nearest chunk,
        # before the relevance filter, so you can judge whether
        # DISTANCE_THRESHOLD is actually cutting at the right place for
        # your documents.
        print(f"\nRaw nearest chunks for {query!r} (threshold = {DISTANCE_THRESHOLD}):")
        for h in retrieve(query):
            flag = "KEEP" if h["distance"] <= DISTANCE_THRESHOLD else "drop"
            print(f"  [{flag}] distance={h['distance']:.4f}  {h['doc_id']} p.{h['page']}")

        result = answer(query)
        print("\nAnswer:", result["answer"])
        print("Sources:", result["sources"])