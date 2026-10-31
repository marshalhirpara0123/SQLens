# """
# Phase 2 — the real LangGraph StateGraph.
# Phase 3 update: graceful degradation when Ollama is unreachable mid-graph.

# Wires backend/router_sketch.py's classifier and sql_agent/agent.py and
# rag_pipeline/retrieve.py (both already built and tested in Phase 1) into
# one graph:

#     router -> sql_tool            -> synthesizer
#            -> rag_tool             -> synthesizer
#            -> sql_tool -> rag_tool -> synthesizer   (route == "both"/"unknown")
#     router -> router_unavailable   (if the router's own LLM call fails)

# Low-confidence fallback: route_question() already collapses ambiguous
# classifications into "both" (see router_sketch.py). "unknown" (a totally
# unparseable classification) is also treated as "both" here, on the same
# "don't guess — query everything and reconcile" principle the roadmap
# states explicitly.

# NOTE on "both": this runs sql_tool then rag_tool *sequentially*, not in
# parallel. The original router_sketch.py docstring sketches a parallel
# fan-out; sequential was chosen instead for reliability across LangGraph
# versions. It costs a little latency on hybrid questions only.

# PHASE 3 CHANGE — graceful LLM-outage handling:
# sql_ask() already catches LLMError internally (see sql_agent/agent.py),
# so sql_node needed no change. But router_node and rag_node's underlying
# calls (route_question, rag_answer) did NOT catch LLMError, and
# synthesizer_node's own generate() call didn't either — any of these
# could crash run_query() uncaught if Ollama is down/crashed mid-request
# (exactly the CUDA-crash failure mode hit during Phase 2 testing). All
# three are now wrapped:
#   - router_node: on LLMError, sets route="unknown" and short-circuits
#     straight to a router_unavailable node with a plain user-facing
#     message — no point calling sql/rag tools if the LLM is unreachable.
#   - rag_node: on LLMError, returns a degraded rag_result with an
#     explanatory answer and empty sources (see "Known limitation" note
#     on _rag_ask_safe below).
#   - synthesizer_node: on LLMError, falls back to returning the raw
#     gathered evidence directly instead of an LLM-synthesized summary,
#     rather than crashing after tool calls already succeeded.

# This makes run_query() ALWAYS return a dict — it should never raise
# LLMError to its caller anymore. backend/app.py's /chat endpoint will
# now return HTTP 200 with a degraded-but-honest answer in this case,
# rather than HTTP 503. That's a deliberate behavior change from Phase 2 —
# flag if you'd rather /chat still surface 503 for this case instead.

# Usage (standalone, before app.py is wired to this):
#     python backend/graph.py "How many customers do we have in Canada?"
#     python backend/graph.py "What is the returns policy?"
#     python backend/graph.py "Which product had the most returns, and does our policy explain why?"
# """
# import sys
# import time
# from pathlib import Path
# from typing import Optional, TypedDict

# sys.path.append(str(Path(__file__).resolve().parent.parent))

# from langgraph.graph import StateGraph, END

# from backend.llm_client import generate, LLMError
# from backend.router_sketch import route_question
# from sql_agent.agent import ask as sql_ask
# from rag_pipeline.retrieve import answer as rag_answer

# MAX_SQL_ROWS_IN_CONTEXT = 20

# SYNTH_SYSTEM_PROMPT = """You are the final-answer synthesizer for a local data-analyst assistant.
# You are given evidence from up to two sources: a structured database query result and/or a document search result.
# Rules:
# - Answer the user's question using ONLY the evidence given below. Do not invent facts not present in the evidence.
# - If both a database result and a document result are given, combine them into one coherent answer, and explicitly note if they agree, disagree, or are complementary.
# - If one source clearly and directly answers the question, and the other source is empty, irrelevant, returned no rows, or failed, GIVE the answer from the source that worked. Do not hedge, say the answer is "unclear," or refuse to answer just because the other source didn't help — a working, relevant source is not undermined by an unrelated one failing. Only say the answer is unclear if NEITHER source actually answers the question.
# - If a source failed, errored, or found nothing relevant, say so plainly rather than ignoring it or pretending it succeeded.
# - If evidence shows a failure, state ONLY the reason given in that evidence. NEVER invent or guess a different cause (for example, do not claim a service was "unavailable" or "offline" unless the evidence explicitly says that) — if no specific reason is given, say the cause is unclear rather than making one up.
# - Be concise and direct. Do not repeat the raw evidence verbatim; synthesize it into a normal-language answer.
# """


# class GraphState(TypedDict):
#     question: str
#     model: Optional[str]
#     route: str                      # "sql" | "rag" | "both" | "unknown"
#     route_error: Optional[str]      # set only if the router's own LLM call failed
#     sql_result: Optional[dict]
#     rag_result: Optional[dict]
#     final_answer: str
#     sources: list
#     contexts: list                  # raw evidence text blocks fed to the synthesizer LLM (Phase 3: needed by the Ragas harness for faithfulness scoring)
#     timings: dict                   # Phase 3: per-node wall-clock seconds, for the latency breakdown study


# # ---------------------------------------------------------------------------
# # Nodes
# # ---------------------------------------------------------------------------

# def router_node(state: GraphState) -> GraphState:
#     state.setdefault("timings", {})
#     start = time.perf_counter()
#     try:
#         route = route_question(state["question"], model=state.get("model"))
#     except LLMError as e:
#         state["timings"]["router"] = time.perf_counter() - start
#         # Can't classify at all without the LLM — no point calling any tool.
#         # Short-circuit via _route_after_router -> "error" -> router_unavailable_node.
#         state["route"] = "unknown"
#         state["route_error"] = str(e)
#         return state
#     state["timings"]["router"] = time.perf_counter() - start
#     state["route"] = route
#     state["route_error"] = None
#     return state


# def router_unavailable_node(state: GraphState) -> GraphState:
#     err = state.get("route_error") or "the local model service is unavailable"
#     state["final_answer"] = (
#         "I couldn't process this question because the local Ollama service is "
#         f"unavailable right now. Please make sure `ollama serve` is running and "
#         f"try again. (Detail: {err})"
#     )
#     state["sources"] = []
#     state["contexts"] = []
#     return state


# def sql_node(state: GraphState) -> GraphState:
#     # sql_ask() already catches LLMError internally and returns a result with
#     # succeeded=False + a populated .error, so no try/except is needed here.
#     state.setdefault("timings", {})
#     start = time.perf_counter()
#     result = sql_ask(state["question"], model=state.get("model"))
#     state["timings"]["sql_tool"] = time.perf_counter() - start
#     state["sql_result"] = {
#         "succeeded": result.succeeded,
#         "final_sql": result.final_sql,
#         "rows": result.rows,
#         "columns": result.columns,
#         "attempts": result.attempts,
#         "used_self_correction": result.used_self_correction,
#         "error": result.error,
#         "attempt_log": result.attempt_log,
#     }
#     return state


# def rag_node(state: GraphState) -> GraphState:
#     state.setdefault("timings", {})
#     start = time.perf_counter()
#     try:
#         result = rag_answer(state["question"], model=state.get("model"))
#     except LLMError as e:
#         state["timings"]["rag_tool"] = time.perf_counter() - start
#         # Known limitation: rag_pipeline/retrieve.py's answer() does its own
#         # ChromaDB retrieval (no LLM needed) THEN calls generate() for the
#         # final synthesized text, as one function. If generate() fails here,
#         # the chunks it already retrieved are lost with it — we can't recover
#         # them from this side. Sources are deliberately left empty rather
#         # than guessed, since citing chunks without a verified grounded
#         # answer would be misleading. If this turns out to matter in
#         # practice (LLM outages that still leave retrieval usable), the
#         # real fix is in retrieve.py itself: catch LLMError inside answer()
#         # and return the raw retrieved chunks with a plain note instead of
#         # raising — that's Member B's file, so flagging rather than
#         # changing it here.
#         result = {
#             "answer": (
#                 "Document search could not produce an answer because the local "
#                 f"model service is unavailable right now: {e}"
#             ),
#             "sources": [],
#         }
#     else:
#         state["timings"]["rag_tool"] = time.perf_counter() - start
#     state["rag_result"] = result
#     return state


# def _last_attempt_error(sql_result: dict) -> str:
#     """Pull the most specific error from attempt_log (guardrail rejection or
#     execution error) rather than the generic 'exceeded max retries' fallback,
#     so the synthesizer has a real reason instead of having to guess one."""
#     for entry in reversed(sql_result.get("attempt_log", [])):
#         if entry.get("error"):
#             return f"{entry.get('stage', 'unknown stage')}: {entry['error']}"
#     return sql_result.get("error") or "no further detail was recorded"


# def _format_sql_context(sql_result: dict) -> str:
#     if not sql_result.get("succeeded"):
#         return (
#             f"Query attempted: {sql_result.get('final_sql') or '(no valid SELECT query could be produced)'}\n"
#             f"The database lookup failed after {sql_result.get('attempts')} attempt(s). "
#             f"Specific reason: {_last_attempt_error(sql_result)}"
#         )
#     rows = sql_result.get("rows", [])
#     columns = sql_result.get("columns", [])
#     if not rows:
#         return f"Query: {sql_result.get('final_sql')}\n(No rows returned.)"
#     lines = [f"Query: {sql_result.get('final_sql')}", "Columns: " + ", ".join(columns)]
#     for row in rows[:MAX_SQL_ROWS_IN_CONTEXT]:
#         lines.append(str(row))
#     if len(rows) > MAX_SQL_ROWS_IN_CONTEXT:
#         lines.append(f"... ({len(rows) - MAX_SQL_ROWS_IN_CONTEXT} more rows omitted)")
#     return "\n".join(lines)


# def synthesizer_node(state: GraphState) -> GraphState:
#     sql_result = state.get("sql_result")
#     rag_result = state.get("rag_result")
#     question = state["question"]

#     context_blocks = []
#     sources = []

#     if sql_result is not None:
#         context_blocks.append(f"[Database result]\n{_format_sql_context(sql_result)}")
#         if sql_result.get("succeeded"):
#             sources.append({
#                 "type": "sql",
#                 "label": "Database query",
#                 "sql": sql_result.get("final_sql"),
#                 "row_count": len(sql_result.get("rows", [])),
#             })

#     if rag_result is not None:
#         rag_sources = rag_result.get("sources", [])
#         if rag_sources:
#             context_blocks.append(f"[Document search result]\n{rag_result.get('answer')}")
#             for s in rag_sources:
#                 sources.append({
#                     "type": "document",
#                     "label": f"{s.get('doc_id')} p.{s.get('page')}",
#                     "doc_id": s.get("doc_id"),
#                     "page": s.get("page"),
#                     "chunk_id": s.get("chunk_id"),
#                 })
#         else:
#             context_blocks.append(f"[Document search result]\n{rag_result.get('answer', 'No relevant documents were found.')}")

#     if not context_blocks:
#         state["final_answer"] = "I couldn't determine how to answer this question — no tool produced a result."
#         state["sources"] = []
#         state["contexts"] = []
#         return state

#     state["contexts"] = list(context_blocks)
#     combined_context = "\n\n".join(context_blocks)
#     prompt = f"Question: {question}\n\nAvailable evidence:\n{combined_context}\n\nAnswer:"

#     state.setdefault("timings", {})
#     start = time.perf_counter()
#     try:
#         final = generate(prompt, model=state.get("model"), system=SYNTH_SYSTEM_PROMPT)
#     except LLMError as e:
#         state["timings"]["synthesizer"] = time.perf_counter() - start
#         # Tool calls already succeeded (we have real evidence) — don't throw
#         # it away just because the final synthesis step failed. Hand back
#         # the raw evidence directly instead of an LLM-written summary.
#         final = (
#             "I gathered results but couldn't generate a natural-language summary "
#             f"because the local model service is unavailable right now ({e}). "
#             f"Here is the raw evidence that was found:\n\n{combined_context}"
#         )
#     else:
#         state["timings"]["synthesizer"] = time.perf_counter() - start

#     state["final_answer"] = final
#     state["sources"] = sources
#     return state


# # ---------------------------------------------------------------------------
# # Edges
# # ---------------------------------------------------------------------------

# def _route_after_router(state: GraphState) -> str:
#     if state.get("route_error"):
#         return "error"
#     route = state.get("route", "unknown")
#     if route == "sql":
#         return "sql"
#     if route == "rag":
#         return "rag"
#     # "both" and "unknown" both fall through to the sql-first hybrid path,
#     # per the roadmap's low-confidence fallback ("query both, reconcile"
#     # rather than guessing).
#     return "both"


# def _route_after_sql(state: GraphState) -> str:
#     route = state.get("route", "unknown")
#     if route in ("both", "unknown"):
#         return "continue_to_rag"
#     return "done"


# def build_graph():
#     graph = StateGraph(GraphState)

#     graph.add_node("router", router_node)
#     graph.add_node("router_unavailable", router_unavailable_node)
#     graph.add_node("sql_tool", sql_node)
#     graph.add_node("rag_tool", rag_node)
#     graph.add_node("synthesizer", synthesizer_node)

#     graph.set_entry_point("router")

#     graph.add_conditional_edges(
#         "router",
#         _route_after_router,
#         {"sql": "sql_tool", "rag": "rag_tool", "both": "sql_tool", "error": "router_unavailable"},
#     )
#     graph.add_conditional_edges(
#         "sql_tool",
#         _route_after_sql,
#         {"continue_to_rag": "rag_tool", "done": "synthesizer"},
#     )
#     graph.add_edge("rag_tool", "synthesizer")
#     graph.add_edge("synthesizer", END)
#     graph.add_edge("router_unavailable", END)

#     return graph.compile()


# _compiled_graph = None


# def get_graph():
#     global _compiled_graph
#     if _compiled_graph is None:
#         _compiled_graph = build_graph()
#     return _compiled_graph


# def run_query(question: str, model: str = None) -> dict:
#     """Entry point backend/app.py calls for the unified /chat endpoint.

#     Always returns a dict — does not raise LLMError even if Ollama is
#     completely unreachable (see module docstring for the Phase 3 change)."""
#     initial_state: GraphState = {
#         "question": question,
#         "model": model,
#         "route": "",
#         "route_error": None,
#         "sql_result": None,
#         "rag_result": None,
#         "final_answer": "",
#         "sources": [],
#         "contexts": [],
#         "timings": {},
#     }
#     total_start = time.perf_counter()
#     final_state = get_graph().invoke(initial_state)
#     total_elapsed = time.perf_counter() - total_start

#     timings = dict(final_state.get("timings", {}))
#     timings["total"] = total_elapsed

#     return {
#         "question": question,
#         "route": final_state.get("route"),
#         "answer": final_state.get("final_answer"),
#         "sources": final_state.get("sources"),
#         "contexts": final_state.get("contexts", []),
#         "timings": timings,
#         "sql_debug": final_state.get("sql_result"),
#         "rag_debug": final_state.get("rag_result"),
#     }


# if __name__ == "__main__":
#     q = sys.argv[1] if len(sys.argv) > 1 else "How many customers do we have in Canada?"
#     result = run_query(q)
#     print(f"Question: {result['question']}")
#     print(f"Route chosen: {result['route']}")
#     print("-" * 60)
#     print(result["answer"])
#     print("-" * 60)
#     print("Sources:")
#     for s in result["sources"]:
#         print(" -", s)
#     print("-" * 60)
#     print("Timings (seconds):")
#     for k, v in result["timings"].items():
#         print(f"  {k}: {v:.2f}")
"""
Phase 2 — the real LangGraph StateGraph.
Phase 3 update: graceful degradation when Ollama is unreachable mid-graph.

Wires backend/router_sketch.py's classifier and sql_agent/agent.py and
rag_pipeline/retrieve.py (both already built and tested in Phase 1) into
one graph:

    router -> sql_tool            -> synthesizer
           -> rag_tool             -> synthesizer
           -> sql_tool -> rag_tool -> synthesizer   (route == "both"/"unknown")
    router -> router_unavailable   (if the router's own LLM call fails)

Low-confidence fallback: route_question() already collapses ambiguous
classifications into "both" (see router_sketch.py). "unknown" (a totally
unparseable classification) is also treated as "both" here, on the same
"don't guess — query everything and reconcile" principle the roadmap
states explicitly.

NOTE on "both": this runs sql_tool then rag_tool *sequentially*, not in
parallel. The original router_sketch.py docstring sketches a parallel
fan-out; sequential was chosen instead for reliability across LangGraph
versions. It costs a little latency on hybrid questions only.

PHASE 3 CHANGE — graceful LLM-outage handling:
sql_ask() already catches LLMError internally (see sql_agent/agent.py),
so sql_node needed no change. But router_node and rag_node's underlying
calls (route_question, rag_answer) did NOT catch LLMError, and
synthesizer_node's own generate() call didn't either — any of these
could crash run_query() uncaught if Ollama is down/crashed mid-request
(exactly the CUDA-crash failure mode hit during Phase 2 testing). All
three are now wrapped:
  - router_node: on LLMError, sets route="unknown" and short-circuits
    straight to a router_unavailable node with a plain user-facing
    message — no point calling sql/rag tools if the LLM is unreachable.
  - rag_node: on LLMError, returns a degraded rag_result with an
    explanatory answer and empty sources (see "Known limitation" note
    on _rag_ask_safe below).
  - synthesizer_node: on LLMError, falls back to returning the raw
    gathered evidence directly instead of an LLM-synthesized summary,
    rather than crashing after tool calls already succeeded.

This makes run_query() ALWAYS return a dict — it should never raise
LLMError to its caller anymore. backend/app.py's /chat endpoint will
now return HTTP 200 with a degraded-but-honest answer in this case,
rather than HTTP 503. That's a deliberate behavior change from Phase 2 —
flag if you'd rather /chat still surface 503 for this case instead.

Usage (standalone, before app.py is wired to this):
    python backend/graph.py "How many customers do we have in Canada?"
    python backend/graph.py "What is the returns policy?"
    python backend/graph.py "Which product had the most returns, and does our policy explain why?"
"""
import sys
import time
from pathlib import Path
from typing import Optional, TypedDict

sys.path.append(str(Path(__file__).resolve().parent.parent))

from langgraph.graph import StateGraph, END

from backend.llm_client import generate, LLMError
from backend.router_rules import refine_route
from backend.router_sketch import route_question
from sql_agent import db_setup
from sql_agent.agent import ask as sql_ask
from rag_pipeline.retrieve import answer as rag_answer, list_indexed_documents

MAX_SQL_ROWS_IN_CONTEXT = 20

SYNTH_SYSTEM_PROMPT = """You are the final-answer synthesizer for a local data-analyst assistant.
You are given evidence from up to two sources: a structured database query result and/or a document search result.
Rules:
- Answer the user's question using ONLY the evidence given below. Do not invent facts not present in the evidence.
- If both a database result and a document result are given, combine them into one coherent answer, and explicitly note if they agree, disagree, or are complementary.
- If one source clearly and directly answers the question, and the other source is empty, irrelevant, returned no rows, or failed, GIVE the answer from the source that worked. Do not hedge, say the answer is "unclear," or refuse to answer just because the other source didn't help — a working, relevant source is not undermined by an unrelated one failing. Only say the answer is unclear if NEITHER source actually answers the question.
- If a source failed, errored, or found nothing relevant, say so plainly rather than ignoring it or pretending it succeeded.
- If evidence shows a failure, state ONLY the reason given in that evidence. NEVER invent or guess a different cause (for example, do not claim a service was "unavailable" or "offline" unless the evidence explicitly says that) — if no specific reason is given, say the cause is unclear rather than making one up.
- Be concise and direct. Do not repeat the raw evidence verbatim; synthesize it into a normal-language answer.
"""


class GraphState(TypedDict):
    question: str
    model: Optional[str]
    route: str                      # "sql" | "rag" | "both" | "unknown"
    route_error: Optional[str]      # set only if the router's own LLM call failed
    route_llm: Optional[str]        # what the LLM classifier said, before the rule-based safety net
    route_note: Optional[str]       # why the safety net changed the route (None if it didn't)
    sql_result: Optional[dict]
    rag_result: Optional[dict]
    final_answer: str
    sources: list
    contexts: list                  # raw evidence text blocks fed to the synthesizer LLM (Phase 3: needed by the Ragas harness for faithfulness scoring)
    timings: dict                   # Phase 3: per-node wall-clock seconds, for the latency breakdown study


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def router_node(state: GraphState) -> GraphState:
    state.setdefault("timings", {})
    start = time.perf_counter()
    try:
        route = route_question(state["question"], model=state.get("model"))
    except LLMError as e:
        state["timings"]["router"] = time.perf_counter() - start
        # Can't classify at all without the LLM — no point calling any tool.
        # Short-circuit via _route_after_router -> "error" -> router_unavailable_node.
        state["route"] = "unknown"
        state["route_error"] = str(e)
        return state
    state["timings"]["router"] = time.perf_counter() - start
    state["route_llm"] = route
    state["route_note"] = None
    # Deterministic safety net (backend/router_rules.py): if the question
    # explicitly refers to documents but the LLM's route would skip them,
    # add them. It must never be able to break routing, so any failure while
    # gathering its inputs just falls back to the LLM's own route.
    try:
        route, note = refine_route(
            route,
            state["question"],
            list_indexed_documents(),
            db_setup.get_schema_terms(),
        )
        state["route_note"] = note
    except Exception:
        pass
    state["route"] = route
    state["route_error"] = None
    return state


def router_unavailable_node(state: GraphState) -> GraphState:
    err = state.get("route_error") or "the local model service is unavailable"
    state["final_answer"] = (
        "I couldn't process this question because the local Ollama service is "
        f"unavailable right now. Please make sure `ollama serve` is running and "
        f"try again. (Detail: {err})"
    )
    state["sources"] = []
    state["contexts"] = []
    return state


def sql_node(state: GraphState) -> GraphState:
    # sql_ask() already catches LLMError internally and returns a result with
    # succeeded=False + a populated .error, so no try/except is needed here.
    state.setdefault("timings", {})
    start = time.perf_counter()
    result = sql_ask(state["question"], model=state.get("model"))
    state["timings"]["sql_tool"] = time.perf_counter() - start
    state["sql_result"] = {
        "succeeded": result.succeeded,
        "final_sql": result.final_sql,
        "rows": result.rows,
        "columns": result.columns,
        "attempts": result.attempts,
        "used_self_correction": result.used_self_correction,
        "error": result.error,
        "attempt_log": result.attempt_log,
    }
    return state


def rag_node(state: GraphState) -> GraphState:
    state.setdefault("timings", {})
    start = time.perf_counter()
    try:
        result = rag_answer(state["question"], model=state.get("model"))
    except LLMError as e:
        state["timings"]["rag_tool"] = time.perf_counter() - start
        # Known limitation: rag_pipeline/retrieve.py's answer() does its own
        # ChromaDB retrieval (no LLM needed) THEN calls generate() for the
        # final synthesized text, as one function. If generate() fails here,
        # the chunks it already retrieved are lost with it — we can't recover
        # them from this side. Sources are deliberately left empty rather
        # than guessed, since citing chunks without a verified grounded
        # answer would be misleading. If this turns out to matter in
        # practice (LLM outages that still leave retrieval usable), the
        # real fix is in retrieve.py itself: catch LLMError inside answer()
        # and return the raw retrieved chunks with a plain note instead of
        # raising — that's Member B's file, so flagging rather than
        # changing it here.
        result = {
            "answer": (
                "Document search could not produce an answer because the local "
                f"model service is unavailable right now: {e}"
            ),
            "sources": [],
        }
    else:
        state["timings"]["rag_tool"] = time.perf_counter() - start
    state["rag_result"] = result
    return state


def _last_attempt_error(sql_result: dict) -> str:
    """Pull the most specific error from attempt_log (guardrail rejection or
    execution error) rather than the generic 'exceeded max retries' fallback,
    so the synthesizer has a real reason instead of having to guess one."""
    for entry in reversed(sql_result.get("attempt_log", [])):
        if entry.get("error"):
            return f"{entry.get('stage', 'unknown stage')}: {entry['error']}"
    return sql_result.get("error") or "no further detail was recorded"


def _format_sql_context(sql_result: dict) -> str:
    if not sql_result.get("succeeded"):
        return (
            f"Query attempted: {sql_result.get('final_sql') or '(no valid SELECT query could be produced)'}\n"
            f"The database lookup failed after {sql_result.get('attempts')} attempt(s). "
            f"Specific reason: {_last_attempt_error(sql_result)}"
        )
    rows = sql_result.get("rows", [])
    columns = sql_result.get("columns", [])
    if not rows:
        return f"Query: {sql_result.get('final_sql')}\n(No rows returned.)"
    lines = [f"Query: {sql_result.get('final_sql')}", "Columns: " + ", ".join(columns)]
    for row in rows[:MAX_SQL_ROWS_IN_CONTEXT]:
        lines.append(str(row))
    if len(rows) > MAX_SQL_ROWS_IN_CONTEXT:
        lines.append(f"... ({len(rows) - MAX_SQL_ROWS_IN_CONTEXT} more rows omitted)")
    return "\n".join(lines)


def synthesizer_node(state: GraphState) -> GraphState:
    sql_result = state.get("sql_result")
    rag_result = state.get("rag_result")
    question = state["question"]

    context_blocks = []
    sources = []

    if sql_result is not None:
        context_blocks.append(f"[Database result]\n{_format_sql_context(sql_result)}")
        if sql_result.get("succeeded"):
            sources.append({
                "type": "sql",
                "label": "Database query",
                "sql": sql_result.get("final_sql"),
                "row_count": len(sql_result.get("rows", [])),
            })

    if rag_result is not None:
        rag_sources = rag_result.get("sources", [])
        if rag_sources:
            context_blocks.append(f"[Document search result]\n{rag_result.get('answer')}")
            for s in rag_sources:
                sources.append({
                    "type": "document",
                    "label": f"{s.get('doc_id')} p.{s.get('page')}",
                    "doc_id": s.get("doc_id"),
                    "page": s.get("page"),
                    "chunk_id": s.get("chunk_id"),
                })
        else:
            context_blocks.append(f"[Document search result]\n{rag_result.get('answer', 'No relevant documents were found.')}")

    if not context_blocks:
        state["final_answer"] = "I couldn't determine how to answer this question — no tool produced a result."
        state["sources"] = []
        state["contexts"] = []
        return state

    state["contexts"] = list(context_blocks)
    combined_context = "\n\n".join(context_blocks)
    prompt = f"Question: {question}\n\nAvailable evidence:\n{combined_context}\n\nAnswer:"

    state.setdefault("timings", {})
    start = time.perf_counter()
    try:
        final = generate(prompt, model=state.get("model"), system=SYNTH_SYSTEM_PROMPT)
    except LLMError as e:
        state["timings"]["synthesizer"] = time.perf_counter() - start
        # Tool calls already succeeded (we have real evidence) — don't throw
        # it away just because the final synthesis step failed. Hand back
        # the raw evidence directly instead of an LLM-written summary.
        final = (
            "I gathered results but couldn't generate a natural-language summary "
            f"because the local model service is unavailable right now ({e}). "
            f"Here is the raw evidence that was found:\n\n{combined_context}"
        )
    else:
        state["timings"]["synthesizer"] = time.perf_counter() - start

    state["final_answer"] = final
    state["sources"] = sources
    return state


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------

def _route_after_router(state: GraphState) -> str:
    if state.get("route_error"):
        return "error"
    route = state.get("route", "unknown")
    if route == "sql":
        return "sql"
    if route == "rag":
        return "rag"
    # "both" and "unknown" both fall through to the sql-first hybrid path,
    # per the roadmap's low-confidence fallback ("query both, reconcile"
    # rather than guessing).
    return "both"


def _route_after_sql(state: GraphState) -> str:
    route = state.get("route", "unknown")
    if route in ("both", "unknown"):
        return "continue_to_rag"
    return "done"


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("router", router_node)
    graph.add_node("router_unavailable", router_unavailable_node)
    graph.add_node("sql_tool", sql_node)
    graph.add_node("rag_tool", rag_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        _route_after_router,
        {"sql": "sql_tool", "rag": "rag_tool", "both": "sql_tool", "error": "router_unavailable"},
    )
    graph.add_conditional_edges(
        "sql_tool",
        _route_after_sql,
        {"continue_to_rag": "rag_tool", "done": "synthesizer"},
    )
    graph.add_edge("rag_tool", "synthesizer")
    graph.add_edge("synthesizer", END)
    graph.add_edge("router_unavailable", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_query(question: str, model: str = None) -> dict:
    """Entry point backend/app.py calls for the unified /chat endpoint.

    Always returns a dict — does not raise LLMError even if Ollama is
    completely unreachable (see module docstring for the Phase 3 change)."""
    initial_state: GraphState = {
        "question": question,
        "model": model,
        "route": "",
        "route_error": None,
        "route_llm": None,
        "route_note": None,
        "sql_result": None,
        "rag_result": None,
        "final_answer": "",
        "sources": [],
        "contexts": [],
        "timings": {},
    }
    total_start = time.perf_counter()
    final_state = get_graph().invoke(initial_state)
    total_elapsed = time.perf_counter() - total_start

    timings = dict(final_state.get("timings", {}))
    timings["total"] = total_elapsed

    return {
        "question": question,
        "route": final_state.get("route"),
        "route_llm": final_state.get("route_llm"),
        "route_note": final_state.get("route_note"),
        "answer": final_state.get("final_answer"),
        "sources": final_state.get("sources"),
        "contexts": final_state.get("contexts", []),
        "timings": timings,
        "sql_debug": final_state.get("sql_result"),
        "rag_debug": final_state.get("rag_result"),
    }


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "How many customers do we have in Canada?"
    result = run_query(q)
    print(f"Question: {result['question']}")
    print(f"Route chosen: {result['route']}")
    print("-" * 60)
    print(result["answer"])
    print("-" * 60)
    print("Sources:")
    for s in result["sources"]:
        print(" -", s)
    print("-" * 60)
    print("Timings (seconds):")
    for k, v in result["timings"].items():
        print(f"  {k}: {v:.2f}")