import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest

from backend.router_rules import analyze, refine_route, stem
from sql_agent import db_setup

DOCS = ["catalog_faq", "employee_handbook", "returns_policy"]
GOLDEN = Path(__file__).resolve().parent.parent / "eval" / "golden_set_template.json"


@pytest.fixture(scope="module")
def chinook_terms():
    db_setup.reset_to_default_db()
    return db_setup.get_schema_terms()


def _covers(route: str) -> set:
    """Which tools a route actually runs (graph.py treats 'unknown' as both)."""
    return {"sql": {"sql"}, "rag": {"rag"}}.get(route, {"sql", "rag"})


# ------------------------------------------------------------ the reported bug

def test_reported_failure_policy_question_is_no_longer_sql_only(chinook_terms):
    q = "In how many days refunds credited based on return policy?"
    route, note = refine_route("sql", q, DOCS, chinook_terms)
    assert route == "rag"
    assert "policy" in note


def test_how_many_days_is_a_duration_not_a_row_count():
    assert analyze("In how many days are refunds credited?", DOCS, set())["data_strong"] is False
    assert analyze("How many customers are there?", DOCS, set())["data_strong"] is True


# --------------------------------------------------------------- core rules

def test_both_is_never_changed(chinook_terms):
    assert refine_route("both", "What is the returns policy?", DOCS, chinook_terms) == ("both", None)


def test_no_document_cue_means_llm_route_is_untouched(chinook_terms):
    q = "How many customers do we have in Canada?"
    for llm_route in ("sql", "rag", "both", "unknown"):
        assert refine_route(llm_route, q, DOCS, chinook_terms) == (llm_route, None)


def test_rag_plus_strong_data_request_becomes_both(chinook_terms):
    q = "How many tracks are in the Rock genre, and what does the FAQ say about genres?"
    route, note = refine_route("rag", q, DOCS, chinook_terms)
    assert route == "both" and note


def test_already_correct_route_produces_no_note(chinook_terms):
    assert refine_route("rag", "What is the returns policy?", DOCS, chinook_terms) == ("rag", None)


def test_unknown_with_document_cue_becomes_rag():
    route, _ = refine_route("unknown", "Summarise the employee handbook", DOCS, set())
    assert route == "rag"


@pytest.mark.parametrize("question", [
    "What is the returns policy?",
    "How many customers do we have in Canada?",
    "Can an employee process a refund without an InvoiceId?",
    "Which employee has the most customers assigned to them?",
    "Per the handbook, who do I contact about a billing question?",
    "Tell me about our customers.",
    "What's our policy?",
    "In how many days refunds credited based on return policy?",
    "How many tracks are in the Rock genre, and what does the FAQ say?",
])
@pytest.mark.parametrize("llm_route", ["sql", "rag", "both", "unknown"])
def test_net_is_additive_except_for_the_one_documented_exception(chinook_terms, question, llm_route):
    """Coverage can only grow, EXCEPT: sql/unknown -> rag when a document is
    named explicitly and the question has no database vocabulary at all."""
    final, _ = refine_route(llm_route, question, DOCS, chinook_terms)
    cues = analyze(question, DOCS, chinook_terms)
    is_the_exception = (
        llm_route in ("sql", "unknown")
        and final == "rag"
        and cues["doc_cue_explicit"]
        and not cues["data_strong"]
        and not cues["data_weak"]
    )
    assert _covers(llm_route) <= _covers(final) or is_the_exception


def test_exception_explicit_document_and_no_data_words_drops_sql():
    assert refine_route("sql", "What does the returns policy say?", DOCS, {"customer"})[0] == "rag"


def test_loose_policy_phrasing_alone_never_drops_sql():
    # "what happens if" is a phrase cue, not an explicit document reference
    assert refine_route("sql", "What happens if I am late?", DOCS, {"customer"})[0] == "both"


def test_explicit_document_plus_database_vocabulary_keeps_sql():
    assert refine_route("sql", "What does the policy say about each customer?", DOCS, {"customer"})[0] == "both"


# ----------------------------------------------- false-positive protections

def test_policy_word_is_data_when_the_active_database_has_a_policies_table():
    terms = {"policies", "policyid"}
    route, note = refine_route("sql", "How many policies were sold in 2024?", DOCS, terms)
    assert (route, note) == ("sql", None)


def test_document_named_after_a_table_cannot_hijack_data_questions(chinook_terms):
    docs = ["customers-1a2b3c4d"]
    assert refine_route("sql", "How many customers are in Canada?", docs, chinook_terms) == ("sql", None)


def test_short_single_word_document_names_are_ignored():
    assert analyze("what is the plan for tomorrow", ["faq-1a2b3c4d"[:3] + "-1a2b3c4d"], set())["doc_cue"] is None


def test_uploaded_document_name_is_recognised_without_its_uuid_suffix():
    cues = analyze("Tell me the refund rules", ["refund_rules-a453105b"], set())
    assert cues["doc_cue"] == "refund rule"


def test_plural_and_singular_forms_match():
    assert stem("policies") == stem("policy")
    assert stem("genres") == stem("genre")
    assert stem("invoices") == stem("invoice")


# ------------------------------------------------ golden set, worst-case LLM

def _golden(categories):
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))["questions"]
    return [q for q in data if q["category"] in categories]


def test_golden_sql_questions_are_never_pulled_toward_documents(chinook_terms):
    """No false positives: every SQL-only golden question keeps its route even
    when the LLM is given the chance to be wrong in either direction."""
    for q in _golden({"sql_only"}):
        for llm_route in ("sql", "both"):
            route, note = refine_route(llm_route, q["question"], DOCS, chinook_terms)
            assert (route, note) == (llm_route, None), q["id"]


def test_golden_document_questions_recover_documents_even_if_llm_says_sql(chinook_terms):
    """Worst case: the LLM sends EVERYTHING to sql. The net must put documents
    back for the document-needing questions it can see a cue for."""
    needing_docs = _golden({"doc_only", "hybrid"})
    recovered = [
        q["id"] for q in needing_docs
        if "rag" in _covers(refine_route("sql", q["question"], DOCS, chinook_terms)[0])
    ]
    missed = {q["id"] for q in needing_docs} - set(recovered)
    # doc-005 has no explicit document cue ("How does the store distinguish a
    # physical CD track from a digital download...") — a known residual the
    # LLM alone must handle. If this set changes, the net's coverage changed.
    assert missed == {"doc-005"}


def test_hybrid_questions_end_up_covering_both_sources(chinook_terms):
    for q in _golden({"hybrid"}):
        route, _ = refine_route("sql", q["question"], DOCS, chinook_terms)
        assert _covers(route) == {"sql", "rag"}, q["id"]