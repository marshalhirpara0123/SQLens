"""
Phase 1 deliverable: sketch of the LangGraph router logic.

This is intentionally NOT wired up yet — per the roadmap, full graph
integration (router node -> structured/unstructured tool nodes ->
synthesizer node, with the low-confidence fallback that queries both
tools and reconciles) is Phase 2 work. This file documents the intended
shape so Phase 2 starts from an agreed design instead of a blank page.

Real implementation (Phase 2) will use `langgraph.graph.StateGraph` with:

    class GraphState(TypedDict):
        question: str
        route: Literal["sql", "rag", "both", "unknown"]
        sql_result: dict | None
        rag_result: dict | None
        final_answer: str
        sources: list

    graph = StateGraph(GraphState)
    graph.add_node("router", route_question)
    graph.add_node("sql_tool", run_sql_agent)
    graph.add_node("rag_tool", run_rag_pipeline)
    graph.add_node("synthesizer", synthesize_answer)
    graph.add_conditional_edges("router", pick_branch, {
        "sql": "sql_tool", "rag": "rag_tool", "both": ["sql_tool", "rag_tool"],
    })
    graph.add_edge("sql_tool", "synthesizer")
    graph.add_edge("rag_tool", "synthesizer")
    graph.set_entry_point("router")
    graph.set_finish_point("synthesizer")
"""
from backend.llm_client import generate, LLMError

ROUTER_SYSTEM_PROMPT = """Classify the user's question into exactly one label:
- "sql": the question is about structured/tabular data (counts, sums, filters, joins,
  specific records) that lives in a relational database.
- "rag": the question is about narrative/unstructured content that lives in documents
  (explanations, policies, definitions, "why"/"how" questions).
- "both": the question plausibly needs both, or you're not confident.
Respond with exactly one word: sql, rag, or both."""


def route_question(question: str, model: str = None) -> str:
    """Rough Phase-1 stand-in for the Phase-2 router node.

    This is a single LLM classification call — not yet a graph node, and
    the "both" branch does not yet reconcile two tool outputs (that's the
    Phase-2/3 low-confidence fallback + synthesizer work).
    """
    try:
        raw = generate(question, model=model, system=ROUTER_SYSTEM_PROMPT).strip().lower()
    except LLMError:
        return "unknown"

    has_sql = "sql" in raw
    has_rag = "rag" in raw
    if "both" in raw or (has_sql and has_rag):
        # Ambiguous or explicit "both" — per the roadmap's low-confidence
        # fallback rule, unclear classification should query both tools
        # and reconcile, not silently fall through to "unknown".
        return "both"
    if has_sql:
        return "sql"
    if has_rag:
        return "rag"
    return "unknown"


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "How many customers do we have in Canada?"
    print(f"Route for {q!r}: {route_question(q)}")
