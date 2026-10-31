"""
One-command Phase 1 setup: builds chinook.db and the ChromaDB vector
index in one go, so you don't have to run sql_agent.db_setup and
rag_pipeline.retrieve separately.

Usage:
    python scripts/setup.py

What it does, in order:
    1. Builds data/db/chinook.db from the bundled SQL script (fast,
       stdlib sqlite3 only — no network needed).
    2. Checks data/pdfs/ for source PDFs. If none are found, it warns
       and skips the vector-index step rather than failing, since an
       empty index isn't an error, just an incomplete one.
    3. Builds data/chroma/ from whatever PDFs are present. This step
       downloads the all-MiniLM-L6-v2 embedding model from Hugging Face
       on first run (~80MB, cached afterwards) — needs real internet
       access and the chromadb + sentence-transformers packages
       installed (`pip install -r requirements.txt`).

Safe to re-run: step 1 skips rebuilding if chinook.db already exists
(pass --force-db to rebuild anyway); step 3 always rebuilds the index
from whatever is currently in data/pdfs/ so newly-added PDFs get picked
up on a re-run.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-db", action="store_true",
                         help="Rebuild chinook.db even if it already exists")
    parser.add_argument("--skip-index", action="store_true",
                         help="Skip the ChromaDB vector-index build step")
    args = parser.parse_args()

    # --- Step 1: SQLite database ---------------------------------------
    print("=" * 60)
    print("Step 1/3: Building chinook.db")
    print("=" * 60)
    from sql_agent.db_setup import build_database, get_readonly_connection

    db_path = build_database(force=args.force_db)
    conn = get_readonly_connection()
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    conn.close()
    print(f"OK  {db_path}")
    print(f"OK  {len(tables)} tables: {', '.join(t['name'] for t in tables)}\n")

    # --- Step 2: check for source PDFs ----------------------------------
    print("=" * 60)
    print("Step 2/3: Checking data/pdfs/ for source documents")
    print("=" * 60)
    pdf_dir = Path(__file__).resolve().parent.parent / "data" / "pdfs"
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"WARN  No PDFs found in {pdf_dir}")
        print("WARN  Add source PDFs there, then re-run this script "
              "(or run `python rag_pipeline/retrieve.py` directly).")
        print("\nSetup finished with the SQL side ready; the RAG index was skipped.")
        return
    print(f"OK  Found {len(pdf_files)} PDF(s): "
          f"{', '.join(p.name for p in pdf_files)}\n")

    if args.skip_index:
        print("Skipping vector-index build (--skip-index passed).")
        return

    # --- Step 3: ChromaDB vector index -----------------------------------
    print("=" * 60)
    print("Step 3/3: Building the ChromaDB vector index")
    print("=" * 60)
    print("This downloads the embedding model on first run (~80MB) — "
          "needs internet access. Please wait...")
    try:
        from rag_pipeline.retrieve import build_index
    except ImportError as e:
        print(f"ERROR  Missing dependency: {e}")
        print("ERROR  Run `pip install -r requirements.txt` first.")
        sys.exit(1)

    try:
        n_chunks = build_index(reset=True)
    except Exception as e:
        print(f"ERROR  Failed to build the vector index: {e}")
        print("ERROR  Check your internet connection (needed to download "
              "the embedding model on first run) and try again.")
        sys.exit(1)

    if n_chunks == 0:
        print("WARN  Indexed 0 chunks — PDFs were found but produced no "
              "extractable text (are they scanned images without OCR?).")
    else:
        print(f"OK  Indexed {n_chunks} chunks into data/chroma/")

    print("\nSetup complete. Both the SQL database and the RAG vector "
          "index are ready.")
    print("Next: `uvicorn backend.app:app --reload` and open http://localhost:8000")


if __name__ == "__main__":
    main()
