"""
Phase 1 — Member A: Structured pipeline foundations.

Builds the local sample database (Chinook) from the bundled SQL script and
exposes a read-only connection helper. SQLite has no user/role system, so
"read-only" is enforced at the connection layer (uri=... mode=ro) and again
in the agent's guardrail layer (sql_agent/guardrails.py) — defense in depth,
matching the roadmap's "reject before execution" principle.

Phase 4 addition: the active database is now swappable at runtime via
set_active_db() / reset_to_default_db(), for the live .sql database-upload
feature. get_readonly_connection() and get_schema_summary() both read the
CURRENT active path indirectly, so sql_agent/agent.py and backend/graph.py
need no changes at all to pick up a swapped-in database — confirmed when
this feature was scoped, since neither of those modules references DB_PATH
directly, only these two functions.
"""
import re
import sqlite3
import time
import uuid
from pathlib import Path

from sql_agent.guardrails import validate_schema_script

DB_DIR = Path(__file__).resolve().parent.parent / "data" / "db"
SQL_SCRIPT = DB_DIR / "Chinook_Sqlite.sql"
DEFAULT_DB_PATH = DB_DIR / "chinook.db"
DB_PATH = DEFAULT_DB_PATH  # kept for backward compatibility with any existing references

# Uploaded databases (built from a .sql script) are written here, kept
# separate from the bundled demo database.
UPLOADED_DB_DIR = DB_DIR / "uploads"
MAX_DB_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB — same cap as the PDF upload feature
SCRIPT_TIMEOUT_SECONDS = 20  # hard ceiling on executing an uploaded script

_active_db_path = DEFAULT_DB_PATH
_active_db_label = "Chinook (default demo database)"


def build_database(force: bool = False) -> Path:
    """Create chinook.db from the bundled SQL script if it doesn't exist yet."""
    if DB_PATH.exists() and not force:
        return DB_PATH

    if DB_PATH.exists():
        DB_PATH.unlink()

    if not SQL_SCRIPT.exists():
        raise FileNotFoundError(
            f"Missing {SQL_SCRIPT}. Download it from "
            "https://github.com/lerocha/chinook-database (DataSources/Chinook_Sqlite.sql)"
        )

    conn = sqlite3.connect(DB_PATH)
    try:
        script = SQL_SCRIPT.read_text(encoding="utf-8", errors="ignore")
        conn.executescript(script)
        conn.commit()
    finally:
        conn.close()
    return DB_PATH


def get_readonly_connection() -> sqlite3.Connection:
    """Open the ACTIVE database in SQLite's immutable read-only URI mode.

    Any INSERT/UPDATE/DELETE/DDL attempted through this connection fails at
    the driver level, independent of the guardrail regex checks upstream.
    Reads whatever the active path currently is — the default Chinook DB,
    or a live-uploaded one swapped in via set_active_db().
    """
    path = _active_db_path
    if path == DEFAULT_DB_PATH and not path.exists():
        build_database()
    if not path.exists():
        raise FileNotFoundError(
            f"Active database file is missing: {path}. It may have been "
            "deleted externally — try resetting to the default database."
        )
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def get_active_db_path() -> Path:
    return _active_db_path


def get_active_db_label() -> str:
    return _active_db_label


def is_using_default_db() -> bool:
    return _active_db_path == DEFAULT_DB_PATH


def list_tables() -> list:
    """Names of the user tables in the ACTIVE database (internal sqlite_*
    tables excluded). Used by the upload endpoint/UI to confirm what was
    just loaded."""
    conn = get_readonly_connection()
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        return [r["name"] for r in rows]
    finally:
        conn.close()


def _split_identifier(name: str) -> set:
    """'PlaylistTrack' / 'order_items' -> {'playlisttrack', 'playlist', 'track'} etc."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    parts = {p.lower() for p in re.split(r"[^A-Za-z0-9]+", spaced) if p}
    parts.add(re.sub(r"[^a-z0-9]", "", name.lower()))
    return {p for p in parts if p}


def get_schema_terms() -> set:
    """Lowercase vocabulary of the ACTIVE database's entities: its table
    names (also split into their camelCase / snake_case parts) plus any
    *id column names* (e.g. 'invoiceid'). Used by backend/router_rules.py so
    that words which are really data in the connected database (a table
    called 'policies', say) aren't mistaken for references to documents."""
    conn = get_readonly_connection()
    try:
        terms = set()
        tables = [
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        ]
        for table in tables:
            terms |= _split_identifier(table)
            for col in conn.execute("SELECT name FROM pragma_table_info(?)", (table,)).fetchall():
                col_name = col[0].lower()
                if col_name.endswith("id") and len(col_name) > 2:
                    terms.add(col_name)
        return terms
    finally:
        conn.close()


def set_active_db(path: Path, label: str = None) -> None:
    """Swap the active database. Every future get_readonly_connection() /
    get_schema_summary() call — and therefore every sql_agent.agent.ask()
    and backend.graph.run_query() call — immediately uses the new path.

    If the database being REPLACED was itself a previous upload (not the
    default), its file is deleted, since only one active DB is kept at a
    time and uploads shouldn't accumulate on disk across a session.
    """
    global _active_db_path, _active_db_label
    old_path = _active_db_path
    _active_db_path = path
    _active_db_label = label or path.name
    if old_path != DEFAULT_DB_PATH and old_path.exists() and old_path != path:
        try:
            old_path.unlink()
        except Exception:
            pass


def reset_to_default_db() -> Path:
    """Revert to the bundled Chinook demo database, deleting the current
    uploaded database file (if any)."""
    global _active_db_path, _active_db_label
    old_path = _active_db_path
    _active_db_path = DEFAULT_DB_PATH
    _active_db_label = "Chinook (default demo database)"
    if old_path != DEFAULT_DB_PATH and old_path.exists():
        try:
            old_path.unlink()
        except Exception:
            pass
    return _active_db_path


def build_database_from_script(sql_script: str) -> Path:
    """Validate and execute an uploaded .sql script against a BRAND-NEW,
    isolated database file — never the live active database, so a bad or
    malicious script can't touch anything currently in use.

    Raises guardrails.UnsafeScriptError if the script contains anything
    outside the narrow schema-import allowlist (see guardrails.py).
    Raises sqlite3.Error if a validated statement still fails to execute
    (e.g. a genuine syntax error, or non-SQLite-dialect SQL such as a
    MySQL/PostgreSQL dump).

    Returns the new file's path on success. Does NOT activate it — the
    caller decides whether to call set_active_db() with the result, so a
    script that executes but fails integrity_check never becomes active.
    """
    statements = validate_schema_script(sql_script)  # raises UnsafeScriptError on violation

    UPLOADED_DB_DIR.mkdir(parents=True, exist_ok=True)
    new_path = UPLOADED_DB_DIR / f"{uuid.uuid4().hex}.db"

    conn = sqlite3.connect(new_path)
    # Two layers, because neither alone is enough:
    #  - the progress handler interrupts ONE slow statement (e.g. a
    #    CREATE INDEX over a big table) — but SQLite counts its steps per
    #    statement, so thousands of tiny INSERTs never trip it;
    #  - the per-statement deadline check in the loop below catches a long
    #    script made of many fast statements.
    # Either one raises sqlite3.OperationalError, which the cleanup below
    # turns into a deleted half-built file and a clean error.
    deadline = time.monotonic() + SCRIPT_TIMEOUT_SECONDS
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 1000)
    try:
        # Statements may arrive in any order relative to foreign-key
        # dependencies between tables (e.g. a child table's INSERTs before
        # its parent table's, or vice versa) — off during import, exactly
        # as a bulk data-load tool like Chinook_Sqlite.sql's own loader
        # would do, since we're not relying on FK enforcement for safety
        # here (the guardrail already restricts what can run).
        conn.execute("PRAGMA foreign_keys = OFF;")
        for stmt in statements:
            if time.monotonic() > deadline:
                raise sqlite3.OperationalError(
                    f"Script execution exceeded the {SCRIPT_TIMEOUT_SECONDS}s time limit."
                )
            conn.execute(stmt)
        conn.commit()
    except sqlite3.Error:
        conn.close()
        new_path.unlink(missing_ok=True)
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass

    # Integrity check on the freshly built file BEFORE it's ever trusted,
    # independent of whether every individual statement "succeeded" above.
    check_conn = sqlite3.connect(f"file:{new_path}?mode=ro", uri=True)
    try:
        result = check_conn.execute("PRAGMA integrity_check;").fetchone()
        if not result or result[0] != "ok":
            check_conn.close()
            new_path.unlink(missing_ok=True)
            raise sqlite3.DatabaseError(f"Integrity check failed on the built database: {result}")
    finally:
        try:
            check_conn.close()
        except Exception:
            pass

    return new_path


def get_schema_summary() -> str:
    """Return a compact schema description used as LLM context for text-to-SQL."""
    conn = get_readonly_connection()
    try:
        cur = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        lines = []
        for row in cur.fetchall():
            lines.append(row["sql"])
        return "\n\n".join(lines)
    finally:
        conn.close()


if __name__ == "__main__":
    path = build_database()
    print(f"Database ready at {path}")
    conn = get_readonly_connection()
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    print(f"{len(tables)} tables:", ", ".join(t["name"] for t in tables))
    conn.close()