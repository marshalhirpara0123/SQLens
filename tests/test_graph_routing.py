"""
Runs the REAL LangGraph graph (backend/graph.py) with the LLM classifier and
both tools replaced by fakes, to verify that the rule-based safety net
changes which tools actually execute — not just which label comes back.
No Ollama, embeddings, or network needed.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest

from backend import graph
from sql_agent import db_setup


class _FakeSqlResult:
    succeeded = True
    final_sql = "SELECT 1 AS x"
    rows = [{"x": 1}]
    columns = ["x"]
    attempts = 1
    used_self_correction = False
    error = ""
    attempt_log = []


@pytest.fixture
def tools(monkeypatch):
    db_setup.reset_to_default_db()
    calls = {"sql": 0, "rag": 0}

    def fake_sql_ask(question, model=None):
        calls["sql"] += 1
        return _FakeSqlResult()

    def fake_rag_answer(question, model=None):
        calls["rag"] += 1
        return {
            "answer": "Refunds are credited within 5 to 7 business days.",
            "sources": [{"doc_id": "returns_policy", "page": 3, "chunk_id": "returns_policy-p3-c0"}],
        }

    monkeypatch.setattr(graph, "sql_ask", fake_sql_ask)
    monkeypatch.setattr(graph, "rag_answer", fake_rag_answer)
    monkeypatch.setattr(graph, "generate", lambda *a, **k: "synthesized answer")
    monkeypatch.setattr(
        graph, "list_indexed_documents",
        lambda: ["catalog_faq", "employee_handbook", "returns_policy"],
    )
    return calls


def _with_llm_route(monkeypatch, route):
    monkeypatch.setattr(graph, "route_question", lambda question, model=None: route)


def test_reported_bug_policy_question_misrouted_to_sql_now_reaches_the_documents(tools, monkeypatch):
    _with_llm_route(monkeypatch, "sql")  # the LLM gets it wrong, as in the bug report
    result = graph.run_query("In how many days refunds credited based on return policy?")

    assert tools["rag"] == 1, "the RAG tool must run"
    assert tools["sql"] == 0, "no pointless SQL attempt for a pure policy question"
    assert result["route"] == "rag"
    assert result["route_llm"] == "sql"
    assert "policy" in result["route_note"]
    assert any(s["type"] == "document" for s in result["sources"])


def test_genuine_data_question_is_left_exactly_as_the_llm_routed_it(tools, monkeypatch):
    _with_llm_route(monkeypatch, "sql")
    result = graph.run_query("How many customers do we have in Canada?")

    assert (tools["sql"], tools["rag"]) == (1, 0)
    assert result["route"] == "sql"
    assert result["route_note"] is None


def test_hybrid_question_misrouted_to_sql_runs_both_tools(tools, monkeypatch):
    _with_llm_route(monkeypatch, "sql")
    result = graph.run_query(
        "How many playlists exist in total, and per the FAQ, can a customer purchase a playlist directly?"
    )
    assert (tools["sql"], tools["rag"]) == (1, 1)
    assert result["route"] == "both"


def test_llm_both_is_never_downgraded(tools, monkeypatch):
    _with_llm_route(monkeypatch, "both")
    result = graph.run_query("What is the returns policy?")
    assert (tools["sql"], tools["rag"]) == (1, 1)
    assert result["route"] == "both"
    assert result["route_note"] is None


def test_safety_net_failure_falls_back_to_the_llm_route_instead_of_breaking(tools, monkeypatch):
    _with_llm_route(monkeypatch, "sql")

    def broken():
        raise RuntimeError("chroma is unavailable")

    monkeypatch.setattr(graph, "list_indexed_documents", broken)
    result = graph.run_query("What is the returns policy?")

    assert result["route"] == "sql"  # the LLM's own route, untouched
    assert result["route_note"] is None
    assert tools["sql"] == 1