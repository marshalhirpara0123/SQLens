import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest
from unittest.mock import patch

from backend.graph import run_query
from backend.llm_client import LLMError
from sql_agent.guardrails import validate_select_only, UnsafeQueryError, ADVERSARIAL_EXAMPLES


@patch("backend.graph.route_question", side_effect=LLMError("Could not reach Ollama"))
def test_run_query_handles_missing_ollama_gracefully(_mock_route):
    result = run_query("How many customers do we have in Canada?")
    assert result["route"] in {"unknown", "both"}
    assert "couldn't" in result["answer"].lower() or "ollama" in result["answer"].lower()


def test_safe_select_passes():
    sql = validate_select_only("SELECT * FROM Customer WHERE Country = 'Canada'")
    assert sql.upper().startswith("SELECT")
    assert "LIMIT" in sql.upper()


def test_adversarial_examples_that_should_block():
    blocking_cases = [
        "DROP TABLE Customer;",
        "SELECT * FROM Customer; DELETE FROM Customer;",
        "UPDATE Employee SET Salary = 0",
    ]
    for sql in blocking_cases:
        with pytest.raises(UnsafeQueryError):
            validate_select_only(sql)


def test_existing_limit_is_capped():
    sql = validate_select_only("SELECT * FROM Track LIMIT 100000")
    assert "LIMIT 200" in sql.upper()


def test_comment_stripping_does_not_bypass_guardrail():
    # Comments are stripped before validation, so a hidden second statement
    # inside a comment is still just a single safe SELECT once cleaned.
    sql = validate_select_only("SELECT * FROM Customer -- ; DROP TABLE Customer")
    assert "DROP" not in sql.upper()
