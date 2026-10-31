"""
SQL safety guardrails.

Per the roadmap's safety note: guardrails must reject unsafe statements
BEFORE execution, not after. The read-only DB connection is a backstop,
not the primary control — this module is the primary control.
"""
import re

FORBIDDEN_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE",
    "REPLACE", "ATTACH", "DETACH", "PRAGMA", "VACUUM", "REINDEX",
    "GRANT", "REVOKE",
]

MAX_ROW_LIMIT = 200
QUERY_TIMEOUT_SECONDS = 5


class UnsafeQueryError(Exception):
    pass


def _strip_comments(sql: str) -> str:
    sql = re.sub(r"--.*?$", " ", sql, flags=re.MULTILINE)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    return sql


def validate_select_only(sql: str) -> str:
    """Raise UnsafeQueryError unless `sql` is a single, safe SELECT statement.

    Returns the (possibly row-limited) query on success.
    """
    cleaned = _strip_comments(sql).strip().rstrip(";")

    if not cleaned:
        raise UnsafeQueryError("Empty query.")

    # Reject stacked statements (e.g. "SELECT ...; DROP TABLE ...")
    if ";" in cleaned:
        raise UnsafeQueryError("Multiple statements are not allowed.")

    if not re.match(r"^\s*(WITH\b.*?\bSELECT|SELECT)\b", cleaned, re.IGNORECASE | re.DOTALL):
        raise UnsafeQueryError("Only SELECT (or WITH ... SELECT) statements are allowed.")

    upper = cleaned.upper()
    for kw in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{kw}\b", upper):
            raise UnsafeQueryError(f"Forbidden keyword detected: {kw}")

    # Enforce a row limit — append one if the model didn't include it.
    if not re.search(r"\bLIMIT\s+\d+\b", upper):
        cleaned = f"{cleaned}\nLIMIT {MAX_ROW_LIMIT}"
    else:
        # Cap an existing LIMIT to MAX_ROW_LIMIT
        def _cap(match):
            n = int(match.group(1))
            return f"LIMIT {min(n, MAX_ROW_LIMIT)}"
        cleaned = re.sub(r"LIMIT\s+(\d+)", _cap, cleaned, flags=re.IGNORECASE)

    return cleaned


# Adversarial examples used by the golden set (see eval/golden_set_template.json)
ADVERSARIAL_EXAMPLES = [
    "DROP TABLE Customer;",
    "SELECT * FROM Customer; DELETE FROM Customer;",
    "UPDATE Employee SET Salary = 0",
    "SELECT * FROM NonExistentTable",
    "SELECT * FROM Customer -- ; DROP TABLE Customer",
]


# ---------------------------------------------------------------------------
# Phase 4 — schema-import guardrail for the live .sql database-upload feature.
#
# This is a SEPARATE, DELIBERATELY NARROWER guardrail from
# validate_select_only() above. validate_select_only() governs read-only
# question-answering (only SELECT is ever allowed). This one governs
# building a brand-new database from an uploaded .sql script, which needs
# to allow schema/data statements that validate_select_only() correctly
# forbids (CREATE TABLE, INSERT) — so it cannot reuse that function's
# keyword list, and instead defines its own, narrower allowlist.
# ---------------------------------------------------------------------------

class UnsafeScriptError(Exception):
    """Raised when an uploaded .sql script contains anything outside the
    narrow schema-import allowlist. Kept as a distinct exception type from
    UnsafeQueryError so it's unambiguous in code and logs which guardrail
    (live querying vs. database import) rejected what."""
    pass


SCHEMA_ALLOWED_PATTERNS = [
    re.compile(r"^\s*CREATE\s+TABLE\b", re.IGNORECASE),
    re.compile(r"^\s*CREATE\s+(UNIQUE\s+)?INDEX\b", re.IGNORECASE),
    re.compile(r"^\s*INSERT\s+(OR\s+(REPLACE|IGNORE|ABORT|FAIL|ROLLBACK)\s+)?INTO\b", re.IGNORECASE),
]

# Note: this list deliberately does NOT include CREATE, INSERT, or REPLACE
# (unlike FORBIDDEN_KEYWORDS above) since those are the statement types this
# guardrail exists to allow. It covers everything else that would let an
# uploaded script modify data outside the new database, touch the
# filesystem, or alter schema after creation.
SCHEMA_FORBIDDEN_KEYWORDS = [
    "ATTACH", "DETACH", "PRAGMA", "DROP", "ALTER", "DELETE", "UPDATE",
    "TRUNCATE", "VACUUM", "REINDEX", "GRANT", "REVOKE", "TRIGGER", "VIEW",
]

MAX_SCRIPT_STATEMENTS = 5000


_QUOTE_CLOSERS = {"'": "'", '"': '"', "`": "`", "[": "]"}


def _split_statements(sql: str) -> list:
    """Split a SQL script into individual statements on top-level
    semicolons, and strip comments — both done in ONE quote-aware pass.

    Semicolons, '--' and '/*' sequences that appear INSIDE a quoted
    section ('...', "...", `...`, [...]) are data, not syntax, so they
    neither split a statement nor start a comment. (Stripping comments
    with a plain regex first would corrupt text values like 'a -- b'.)
    """
    statements = []
    current = []
    closer = None  # the character that closes the quoted section we're in, if any
    i = 0
    n = len(sql)
    while i < n:
        ch = sql[i]

        if closer is not None:
            current.append(ch)
            if ch == closer:
                if closer != "]" and i + 1 < n and sql[i + 1] == closer:
                    current.append(sql[i + 1])  # doubled quote = escaped quote, still inside
                    i += 2
                    continue
                closer = None
            i += 1
            continue

        if ch == "-" and i + 1 < n and sql[i + 1] == "-":  # line comment
            j = sql.find("\n", i)
            i = n if j == -1 else j
            current.append(" ")
            continue
        if ch == "/" and i + 1 < n and sql[i + 1] == "*":  # block comment
            j = sql.find("*/", i + 2)
            i = n if j == -1 else j + 2
            current.append(" ")
            continue

        if ch in _QUOTE_CLOSERS:
            closer = _QUOTE_CLOSERS[ch]
            current.append(ch)
        elif ch == ";":
            stmt = "".join(current).strip()
            if stmt:
                statements.append(stmt)
            current = []
        else:
            current.append(ch)
        i += 1

    tail = "".join(current).strip()
    if tail:
        statements.append(tail)
    return statements


def _mask_literals(stmt: str) -> str:
    """Return stmt with the CONTENTS of every quoted section replaced by
    spaces (the quote characters themselves are kept). Keyword and pattern
    checks run on this masked copy, so they only ever look at actual SQL
    code — never at data inside string literals, where words like
    'select', 'update' or 'drop' are perfectly ordinary text."""
    out = []
    closer = None
    i = 0
    n = len(stmt)
    while i < n:
        ch = stmt[i]
        if closer is not None:
            if ch == closer:
                if closer != "]" and i + 1 < n and stmt[i + 1] == closer:
                    out.append("  ")
                    i += 2
                    continue
                closer = None
                out.append(ch)
            else:
                out.append(" ")
            i += 1
            continue
        if ch in _QUOTE_CLOSERS:
            closer = _QUOTE_CLOSERS[ch]
        out.append(ch)
        i += 1
    return "".join(out)


def validate_schema_script(sql_script: str, max_statements: int = MAX_SCRIPT_STATEMENTS) -> list:
    """Validate an uploaded .sql script for the live database-upload
    feature. Allows only: CREATE TABLE, CREATE INDEX, and INSERT INTO.

    No SELECT or WITH may appear anywhere in the code of an allowed
    statement (text inside string literals is exempt — it's just data).
    That rules out CREATE TABLE ... AS SELECT/WITH, INSERT ... SELECT, and
    subqueries generally. It matters because SQLite's pragma_*() table-
    valued functions (e.g. pragma_database_list) can be invoked from
    inside a SELECT, which would reach PRAGMA-level behavior without the
    literal word PRAGMA appearing as its own token. Blocking SELECT/WITH
    outright closes that off instead of chasing individual function names.

    Raises UnsafeScriptError on the first violation found. Returns the
    list of validated statements, in order, ready to execute.
    """
    statements = _split_statements(sql_script)

    if not statements:
        raise UnsafeScriptError("No SQL statements were found in the uploaded file.")
    if len(statements) > max_statements:
        raise UnsafeScriptError(
            f"Script contains {len(statements)} statements, exceeding the limit of {max_statements}."
        )

    for stmt in statements:
        code = _mask_literals(stmt)  # string/identifier contents blanked out
        upper = code.upper()

        if not any(p.match(code) for p in SCHEMA_ALLOWED_PATTERNS):
            raise UnsafeScriptError(
                "Only CREATE TABLE, CREATE INDEX, and INSERT INTO statements are allowed "
                f"in an uploaded database script. Rejected statement (truncated): {stmt[:80]!r}"
            )

        for kw in SCHEMA_FORBIDDEN_KEYWORDS:
            if re.search(rf"\b{kw}\b", upper):
                raise UnsafeScriptError(
                    f"Forbidden keyword detected: {kw} (truncated statement: {stmt[:80]!r})"
                )

        if re.search(r"\b(SELECT|WITH)\b", upper):
            raise UnsafeScriptError(
                "SELECT / WITH (subqueries, CREATE TABLE ... AS SELECT, INSERT ... SELECT) are not "
                f"allowed in an uploaded database script. Statement (truncated): {stmt[:80]!r}"
            )

    return statements


# Adversarial examples for the schema-import guardrail (mirrors
# ADVERSARIAL_EXAMPLES above, for the database-upload feature specifically)
ADVERSARIAL_SCHEMA_EXAMPLES = [
    "ATTACH DATABASE '/etc/passwd' AS pwn;",
    "CREATE TABLE evil AS SELECT * FROM pragma_database_list;",
    "INSERT INTO t SELECT * FROM pragma_table_info('sqlite_master');",
    "DROP TABLE Customer;",
    "PRAGMA writable_schema=1;",
    "CREATE TRIGGER evil AFTER INSERT ON t BEGIN DELETE FROM t; END;",
]