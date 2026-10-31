"""
Phase 1 — Member A: text-to-SQL chain with a self-correction loop.

Flow: NL question -> LLM generates SQL -> guardrail validation ->
execution -> on error, feed the error back to the LLM and retry
(up to max_retries) -> return result + the query actually run.

This is the piece the Month-3 ablation study measures: first-attempt
success rate vs. success rate after self-correction retries.

Phase 3 change: MAX_RETRIES is still the production default (3), but
ask() now also accepts an explicit max_retries override. This lets
eval/run_ablation.py force max_retries=1 (no self-correction) for the
"before" condition without touching production behavior, which still
calls ask(question) with no override and gets the same 3-attempt
self-correction loop as before.
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.llm_client import generate, LLMError
from sql_agent.db_setup import get_readonly_connection, get_schema_summary
from sql_agent.guardrails import validate_select_only, UnsafeQueryError

MAX_RETRIES = 3

SYSTEM_PROMPT = """You are a careful SQLite query writer.
Rules:
- Output ONLY a single valid SQLite SELECT statement. No prose, no markdown fences.
- Never write INSERT, UPDATE, DELETE, DROP, ALTER, or any statement other than SELECT.
- Use only the tables/columns given in the schema below.
- Always include a LIMIT clause (200 or fewer rows) unless the question asks for a single aggregate value.

Schema:
{schema}
"""


@dataclass
class SqlAgentResult:
    question: str
    final_sql: str = ""
    rows: list = field(default_factory=list)
    columns: list = field(default_factory=list)
    attempts: int = 0
    succeeded: bool = False
    used_self_correction: bool = False
    error: str = ""
    attempt_log: list = field(default_factory=list)  # for the ablation study


def _extract_sql(raw: str) -> str:
    """Strip markdown fences / stray prose the LLM might add despite instructions."""
    text = raw.strip()
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            p = p.strip()
            if p.lower().startswith("sql"):
                p = p[3:].strip()
            if p.upper().lstrip().startswith(("SELECT", "WITH")):
                return p
    return text


def ask(question: str, model: str = None, max_retries: int = MAX_RETRIES) -> SqlAgentResult:
    """
    max_retries: maximum attempts allowed, including the first. Defaults to
    MAX_RETRIES (3), matching production behavior. Pass max_retries=1 to
    disable self-correction entirely (used by the Phase 3 ablation study).
    """
    schema = get_schema_summary()
    system = SYSTEM_PROMPT.format(schema=schema)
    result = SqlAgentResult(question=question)

    prompt = f"Question: {question}\nSQL:"
    error_context = ""

    for attempt in range(1, max_retries + 1):
        result.attempts = attempt
        full_prompt = prompt if not error_context else f"{prompt}\n\n{error_context}"

        try:
            raw = generate(full_prompt, model=model, system=system)
        except LLMError as e:
            result.error = str(e)
            result.attempt_log.append({"attempt": attempt, "stage": "llm_call", "error": str(e)})
            break

        sql = _extract_sql(raw)

        try:
            safe_sql = validate_select_only(sql)
        except UnsafeQueryError as e:
            result.attempt_log.append({"attempt": attempt, "sql": sql, "stage": "guardrail", "error": str(e)})
            error_context = (
                f"Your previous query was rejected by the safety guardrail: {e}\n"
                f"Previous query:\n{sql}\nWrite a corrected, SELECT-only query."
            )
            if attempt > 1:
                result.used_self_correction = True
            continue

        conn = get_readonly_connection()
        try:
            cur = conn.execute(safe_sql)
            rows = [dict(r) for r in cur.fetchall()]
            columns = [d[0] for d in cur.description] if cur.description else []
            result.final_sql = safe_sql
            result.rows = rows
            result.columns = columns
            result.succeeded = True
            result.attempt_log.append({"attempt": attempt, "sql": safe_sql, "stage": "success"})
            if attempt > 1:
                result.used_self_correction = True
            return result
        except Exception as e:  # sqlite3.OperationalError, etc.
            result.attempt_log.append({"attempt": attempt, "sql": safe_sql, "stage": "execution", "error": str(e)})
            error_context = (
                f"Your previous query failed with this database error:\n{e}\n"
                f"Previous query:\n{safe_sql}\nWrite a corrected query that fixes this error."
            )
            if attempt > 1:
                result.used_self_correction = True
        finally:
            conn.close()

    result.error = result.error or "Exceeded max retries without a successful query."
    return result


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "How many customers are there in the database?"
    res = ask(q)
    print(f"Question: {res.question}")
    print(f"Succeeded: {res.succeeded} (attempts: {res.attempts}, self-correction used: {res.used_self_correction})")
    print(f"Final SQL: {res.final_sql}")
    if res.succeeded:
        print(f"Columns: {res.columns}")
        for row in res.rows[:10]:
            print(row)
    else:
        print(f"Error: {res.error}")