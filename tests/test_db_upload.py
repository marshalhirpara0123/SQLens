import sqlite3
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest

from sql_agent import db_setup
from sql_agent.guardrails import (
    ADVERSARIAL_SCHEMA_EXAMPLES,
    UnsafeScriptError,
    validate_schema_script,
)

VALID_SCRIPT = """
-- a tiny test database
CREATE TABLE pets (id INTEGER PRIMARY KEY, name TEXT, species TEXT);
INSERT INTO pets (id, name, species) VALUES (1, 'Rex', 'Dog');
INSERT INTO pets (id, name, species) VALUES (2, 'Milo', 'Cat');
CREATE INDEX idx_species ON pets(species);
"""


def _upload_files():
    if not db_setup.UPLOADED_DB_DIR.exists():
        return set()
    return set(db_setup.UPLOADED_DB_DIR.glob("*.db"))


@pytest.fixture(autouse=True)
def restore_default_db():
    """Every test starts and ends on the default database, and never leaves
    an uploaded .db file behind (including databases a test builds but
    never activates, which reset_to_default_db() wouldn't know about)."""
    db_setup.reset_to_default_db()
    before = _upload_files()
    yield
    db_setup.reset_to_default_db()
    for leftover in _upload_files() - before:
        leftover.unlink(missing_ok=True)


# ---------------------------------------------------------------- guardrail

def test_valid_script_passes_validation():
    statements = validate_schema_script(VALID_SCRIPT)
    assert len(statements) == 4
    assert statements[0].upper().startswith("CREATE TABLE")


@pytest.mark.parametrize("bad_sql", ADVERSARIAL_SCHEMA_EXAMPLES)
def test_every_documented_adversarial_example_is_rejected(bad_sql):
    with pytest.raises(UnsafeScriptError):
        validate_schema_script(bad_sql)


def test_ctas_with_cte_is_rejected():
    # The bypass a plain "AS SELECT" check would miss: AS WITH ... SELECT
    sql = "CREATE TABLE evil AS WITH x AS (SELECT 1) SELECT * FROM pragma_database_list;"
    with pytest.raises(UnsafeScriptError):
        validate_schema_script(sql)


def test_empty_or_comment_only_script_is_rejected():
    with pytest.raises(UnsafeScriptError):
        validate_schema_script("-- nothing but a comment\n/* and another */")


def test_statement_count_cap():
    script = "CREATE TABLE a (x INT); CREATE TABLE b (x INT); CREATE TABLE c (x INT);"
    with pytest.raises(UnsafeScriptError):
        validate_schema_script(script, max_statements=2)


def test_semicolon_inside_string_does_not_split_statement():
    script = """
    CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT);
    INSERT INTO notes (id, body) VALUES (1, 'first; second; third');
    """
    assert len(validate_schema_script(script)) == 2


def test_dangerous_looking_words_inside_data_are_allowed_and_preserved():
    """Real data contains words like select/update/drop/view and even '--'
    or '/*'. Those are text, not SQL, and must neither be rejected nor
    corrupted by comment stripping."""
    tricky = "Please select a view, then DROP by to UPDATE -- and /* delete */ stuff; ok"
    script = (
        "CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT);\n"
        f"INSERT INTO notes (id, body) VALUES (1, '{tricky}');\n"
    )
    path = db_setup.build_database_from_script(script)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        stored = conn.execute("SELECT body FROM notes WHERE id = 1").fetchone()[0]
    finally:
        conn.close()
    assert stored == tricky


def test_escaped_quote_in_data_is_handled():
    script = (
        "CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT);\n"
        "INSERT INTO t (id, name) VALUES (1, 'Guns N'' Roses');\n"
    )
    path = db_setup.build_database_from_script(script)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        assert conn.execute("SELECT name FROM t").fetchone()[0] == "Guns N' Roses"
    finally:
        conn.close()


# ----------------------------------------------------------- build / swap

def test_build_swap_query_and_reset_lifecycle():
    assert db_setup.is_using_default_db()

    new_path = db_setup.build_database_from_script(VALID_SCRIPT)
    assert new_path.exists()
    assert db_setup.is_using_default_db()  # building alone must NOT activate it

    db_setup.set_active_db(new_path, label="pets.sql")
    assert not db_setup.is_using_default_db()
    assert db_setup.get_active_db_label() == "pets.sql"

    # The agent reads schema + connection ONLY through these two functions,
    # so this proves agent.py / graph.py need no changes.
    assert "pets" in db_setup.get_schema_summary()
    conn = db_setup.get_readonly_connection()
    try:
        names = [r["name"] for r in conn.execute("SELECT name FROM pets ORDER BY id")]
    finally:
        conn.close()
    assert names == ["Rex", "Milo"]

    db_setup.reset_to_default_db()
    assert db_setup.is_using_default_db()
    assert not new_path.exists()  # uploaded file cleaned up on reset


def test_active_connection_is_really_read_only():
    new_path = db_setup.build_database_from_script(VALID_SCRIPT)
    db_setup.set_active_db(new_path, label="pets.sql")
    conn = db_setup.get_readonly_connection()
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM pets")
    finally:
        conn.close()


def test_replacing_an_upload_deletes_the_previous_one():
    first = db_setup.build_database_from_script(VALID_SCRIPT)
    db_setup.set_active_db(first, label="first.sql")
    second = db_setup.build_database_from_script(
        "CREATE TABLE other (id INTEGER PRIMARY KEY);"
    )
    db_setup.set_active_db(second, label="second.sql")
    assert not first.exists()
    assert second.exists()


def test_failed_build_leaves_no_file_and_no_state_change():
    before = _upload_files()
    broken = "CREATE TABLE broken (id INTEGER PRIMARY KEY,"  # passes guardrail, fails in SQLite
    with pytest.raises(sqlite3.Error):
        db_setup.build_database_from_script(broken)
    assert _upload_files() == before
    assert db_setup.is_using_default_db()


def test_rejected_script_creates_no_file():
    before = _upload_files()
    with pytest.raises(UnsafeScriptError):
        db_setup.build_database_from_script("DROP TABLE Customer;")
    assert _upload_files() == before


# ------------------------------------------------- timeout + table listing

def test_script_exceeding_the_time_limit_is_aborted_cleanly(monkeypatch):
    # A deadline that has already passed: the progress handler must abort
    # the script, delete the half-built file, and leave state untouched.
    monkeypatch.setattr(db_setup, "SCRIPT_TIMEOUT_SECONDS", -1)
    inserts = "\n".join(f"INSERT INTO big (id, v) VALUES ({i}, 'row {i}');" for i in range(2000))
    script = "CREATE TABLE big (id INTEGER PRIMARY KEY, v TEXT);\n" + inserts
    before = _upload_files()
    with pytest.raises(sqlite3.OperationalError):
        db_setup.build_database_from_script(script)
    assert _upload_files() == before
    assert db_setup.is_using_default_db()


def test_list_tables_reports_user_tables_of_the_active_db():
    new_path = db_setup.build_database_from_script(
        "CREATE TABLE b_table (id INTEGER PRIMARY KEY);\n"
        "CREATE TABLE a_table (id INTEGER PRIMARY KEY);\n"
    )
    db_setup.set_active_db(new_path, label="two_tables.sql")
    assert db_setup.list_tables() == ["a_table", "b_table"]


# ------------------------------------------------------------ sample file

def test_bundled_bookstore_sample_loads_and_is_queryable():
    """The demo script shipped in data/samples/ must always pass the guardrail
    and build a usable database — it's what gets dragged in during a demo."""
    sample = Path(__file__).resolve().parent.parent / "data" / "samples" / "bookstore_demo.sql"
    script = sample.read_text(encoding="utf-8")
    new_path = db_setup.build_database_from_script(script)
    db_setup.set_active_db(new_path, label="bookstore_demo.sql")
    assert db_setup.list_tables() == ["authors", "books", "customers", "order_items", "orders"]
    conn = db_setup.get_readonly_connection()
    try:
        assert conn.execute("SELECT COUNT(*) FROM books").fetchone()[0] == 16
        # "Which author has written the most books?" must have ONE clear
        # answer for the demo — a tie makes the generated answer arbitrary.
        top_two = conn.execute(
            "SELECT a.name, COUNT(*) AS n FROM books b JOIN authors a ON a.author_id = b.author_id "
            "GROUP BY a.author_id ORDER BY n DESC, a.name LIMIT 2"
        ).fetchall()
        assert top_two[0]["name"] == "Haruki Murakami" and top_two[0]["n"] > top_two[1]["n"]
        joined = conn.execute(
            "SELECT b.title FROM books b JOIN authors a ON a.author_id = b.author_id "
            "WHERE a.name = 'George Orwell' ORDER BY b.title"
        ).fetchall()
        assert [r["title"] for r in joined] == ["Animal Farm", "Nineteen Eighty-Four"]
    finally:
        conn.close()