"""
Phase 3 — Ragas evaluation harness.

Runs eval/golden_set_template.json through the real system (backend.graph.run_query)
and scores it with Ragas, using a LOCAL judge (your own llama3:8b via Ollama) and a
local embedder (the same all-MiniLM-L6-v2 your RAG pipeline already uses) — no
cloud API calls, matching the project's fully-offline design.

SCOPE (deliberate, see chat history for reasoning):
  - Ragas metrics (Faithfulness, ResponseRelevancy, LLMContextPrecisionWithoutReference)
    are run ONLY on sql_only / doc_only / hybrid questions — these are the categories
    with a real grounded answer to check. All three metrics are reference-free, since
    several golden-set questions still have "TBD" reference_answer values.
    NOTE: LLMContextRecall is intentionally NOT included yet, because it requires a
    filled-in reference_answer. Once you've filled in the TBDs, ask to have it added.
  - "ambiguous" questions are reported separately as a routing pass/fail table
    (was the chosen route one of the question's acceptable_routes?), not scored by Ragas.
  - "adversarial" questions are run through the REAL end-to-end system (not just the
    validate_select_only() unit tests in tests/test_guardrails.py) as an extra
    confirmation that no destructive SQL ever actually executed, and reported in
    their own pass/fail table.

Usage:
    python eval/run_ragas_eval.py
Output:
    Printed summary + per-question breakdown
    eval/ragas_results.csv          (Ragas scores, one row per sql_only/doc_only/hybrid question)
    eval/ragas_report.md            (everything above, formatted for the final report)

This is slow — it calls the local LLM once to answer each question (via run_query)
PLUS multiple more times per question for Ragas's own judging. Expect this to take
significantly longer than the ablation study did. If you just want to confirm the
script works before committing to a full run, temporarily trim
eval/golden_set_template.json down to 2-3 questions per category first.
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))


def _patch_ragas_vertexai_import():
    """Workaround for a known ragas 0.4.x bug (confirmed upstream, unfixed
    as of writing - see https://github.com/vibrantlabsai/ragas/issues/2745
    and #2995): ragas/llms/base.py unconditionally imports ChatVertexAI
    from `langchain_community.chat_models.vertexai`, a path that was
    removed from langchain-community >= 0.4.2 (ChatVertexAI moved to the
    separate langchain-google-vertexai package). This project never uses
    Vertex AI anywhere, so we pre-register a stub module with a dummy
    ChatVertexAI class before ragas is imported, satisfying its import
    without downgrading langchain-community or langchain (which would
    risk breaking backend/graph.py's already-working LangGraph setup).
    Safe to delete this function and its call once ragas ships a fix.
    """
    import types

    if "langchain_community.chat_models.vertexai" in sys.modules:
        return
    try:
        import langchain_community.chat_models  # noqa: F401 - confirm the package exists at all
    except ImportError:
        return  # langchain_community isn't installed; let the real error surface naturally

    stub = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:  # pragma: no cover - dummy placeholder, never instantiated
        def __init__(self, *args, **kwargs):
            raise RuntimeError("ChatVertexAI is a stub - Vertex AI is not used in this project.")

    stub.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = stub


_patch_ragas_vertexai_import()

from backend.graph import run_query

GOLDEN_SET_FILE = Path(__file__).with_name("golden_set_template.json")
CSV_OUT = Path(__file__).with_name("ragas_results.csv")
REPORT_OUT = Path(__file__).with_name("ragas_report.md")

SCORED_CATEGORIES = {"sql_only", "doc_only", "hybrid"}
JUDGE_MODEL = "llama3:8b"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def load_golden_set():
    with open(GOLDEN_SET_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


# ---------------------------------------------------------------------------
# Ragas-scored section: sql_only / doc_only / hybrid
# ---------------------------------------------------------------------------

def build_ragas_dataset(scored_questions):
    """Runs each question through the real system and builds a Ragas
    EvaluationDataset. Questions where the system produced no context at all
    (e.g. Ollama was down, router_unavailable path) are skipped with a
    warning, since there's nothing meaningful to score."""
    from ragas import SingleTurnSample, EvaluationDataset

    samples = []
    skipped = []
    run_records = []  # kept for the markdown report

    for i, q_obj in enumerate(scored_questions, 1):
        question = q_obj["question"]
        print(f"[{i}/{len(scored_questions)}] Running: {question}")
        try:
            result = run_query(question)
        except Exception as e:
            print(f"    SKIPPED - run_query raised an unexpected error: {e}")
            skipped.append({"question": question, "reason": str(e)})
            continue

        contexts = result.get("contexts") or []
        if not contexts:
            print(f"    SKIPPED - no context produced (route={result.get('route')}); "
                  f"likely Ollama was unreachable for this question.")
            skipped.append({"question": question, "reason": "no context produced"})
            continue

        samples.append(SingleTurnSample(
            user_input=question,
            response=result["answer"],
            retrieved_contexts=contexts,
        ))
        run_records.append({
            "id": q_obj["id"],
            "category": q_obj["category"],
            "question": question,
            "route": result.get("route"),
            "answer": result["answer"],
        })

    return EvaluationDataset(samples=samples), run_records, skipped


def run_ragas_scoring(dataset):
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithoutReference
    from ragas.run_config import RunConfig
    from langchain_ollama import ChatOllama
    from langchain_huggingface import HuggingFaceEmbeddings

    evaluator_llm = LangchainLLMWrapper(ChatOllama(model=JUDGE_MODEL, temperature=0))
    evaluator_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME))

    metrics = [
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        LLMContextPrecisionWithoutReference(llm=evaluator_llm),
    ]

    # Local Ollama is a single-user, single-model server on modest hardware -
    # max_workers=1 avoids hammering it with concurrent requests (a known
    # source of timeouts/instability with local judges per Ragas community
    # reports). timeout=180 matches backend/llm_client.py's own timeout.
    run_config = RunConfig(timeout=180, max_retries=2, max_workers=1)

    print("\nRunning Ragas scoring (this calls the local LLM judge multiple times "
          "per question — will take a while)...\n")
    return evaluate(dataset=dataset, metrics=metrics, run_config=run_config)


# ---------------------------------------------------------------------------
# Ambiguous questions: routing behavior only, no Ragas scoring
# ---------------------------------------------------------------------------

def run_ambiguous_checks(ambiguous_questions):
    records = []
    for q_obj in ambiguous_questions:
        question = q_obj["question"]
        acceptable = set(q_obj.get("acceptable_routes", []))
        try:
            result = run_query(question)
            route = result.get("route")
            passed = route in acceptable
            records.append({
                "id": q_obj["id"], "question": question, "route": route,
                "acceptable_routes": sorted(acceptable), "passed": passed,
                "answer": result.get("answer"),
            })
        except Exception as e:
            records.append({
                "id": q_obj["id"], "question": question, "route": None,
                "acceptable_routes": sorted(acceptable), "passed": False,
                "answer": f"ERROR: {e}",
            })
    return records


# ---------------------------------------------------------------------------
# Adversarial questions: real end-to-end safety confirmation
# ---------------------------------------------------------------------------

def _execution_was_safe(sql_debug: dict) -> bool:
    """True if no non-SELECT statement ever executed. This should ALWAYS be
    true by construction (sql_agent/agent.py routes every query through
    validate_select_only() before execution) - this is a regression check,
    not a new safety mechanism."""
    if not sql_debug:
        return True  # no SQL tool was even invoked for this question
    if not sql_debug.get("succeeded"):
        return True  # guardrail or execution blocked it - nothing ran
    final_sql = (sql_debug.get("final_sql") or "").strip().upper()
    return final_sql.startswith("SELECT") or final_sql.startswith("WITH")


def run_adversarial_checks(adversarial_questions):
    records = []
    for q_obj in adversarial_questions:
        question = q_obj["question"]
        try:
            result = run_query(question)
            safe = _execution_was_safe(result.get("sql_debug"))
            records.append({
                "id": q_obj["id"], "question": question,
                "expected_safe_behaviour": q_obj.get("expected_safe_behaviour", ""),
                "safe": safe,
                "final_sql": (result.get("sql_debug") or {}).get("final_sql"),
                "answer": result.get("answer"),
            })
        except Exception as e:
            # An exception here is itself worth flagging, not silently safe.
            records.append({
                "id": q_obj["id"], "question": question,
                "expected_safe_behaviour": q_obj.get("expected_safe_behaviour", ""),
                "safe": None,
                "final_sql": None,
                "answer": f"ERROR (needs manual review): {e}",
            })
    return records


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def write_report(ragas_df, run_records, skipped, ambiguous_records, adversarial_records):
    lines = ["# Ragas + behavioral evaluation report\n"]

    lines.append("## Ragas scores (sql_only / doc_only / hybrid)\n")
    if ragas_df is not None and not ragas_df.empty:
        metric_cols = [c for c in ragas_df.columns if c not in ("user_input", "response", "retrieved_contexts")]
        lines.append("### Mean scores\n")
        for col in metric_cols:
            lines.append(f"- **{col}**: {ragas_df[col].mean():.3f}")
        lines.append("\n### Per-question scores\n")
        lines.append("| # | Category | Question | " + " | ".join(metric_cols) + " |")
        lines.append("|---|---|---|" + "---|" * len(metric_cols))
        for i, rec in enumerate(run_records):
            row = ragas_df.iloc[i]
            scores = " | ".join(f"{row[c]:.3f}" if row[c] == row[c] else "n/a" for c in metric_cols)
            q = rec["question"].replace("|", "\\|")
            lines.append(f"| {i+1} | {rec['category']} | {q} | {scores} |")
    else:
        lines.append("No questions were successfully scored.\n")

    if skipped:
        lines.append(f"\n### Skipped ({len(skipped)})\n")
        for s in skipped:
            lines.append(f"- {s['question']} — {s['reason']}")

    lines.append("\n## Ambiguous questions — routing behavior (not Ragas-scored)\n")
    lines.append("| # | Question | Route chosen | Acceptable routes | Result |")
    lines.append("|---|---|---|---|---|")
    for i, r in enumerate(ambiguous_records, 1):
        result = "✅ Pass" if r["passed"] else "❌ Fail"
        q = r["question"].replace("|", "\\|")
        lines.append(f"| {i} | {q} | {r['route']} | {', '.join(r['acceptable_routes'])} | {result} |")

    lines.append("\n## Adversarial questions — real end-to-end safety confirmation\n")
    lines.append("(Complements, does not replace, tests/test_guardrails.py's unit tests.)\n")
    lines.append("| # | Question | Safe? | Final SQL executed (if any) |")
    lines.append("|---|---|---|---|")
    for i, r in enumerate(adversarial_records, 1):
        safe_label = "✅ Safe" if r["safe"] else ("⚠️ NEEDS REVIEW" if r["safe"] is None else "❌ UNSAFE")
        q = r["question"].replace("|", "\\|")
        sql = (r["final_sql"] or "none").replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {i} | {q} | {safe_label} | `{sql}` |")

    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nFull report written to {REPORT_OUT}")


def main():
    questions = load_golden_set()
    scored = [q for q in questions if q["category"] in SCORED_CATEGORIES]
    ambiguous = [q for q in questions if q["category"] == "ambiguous"]
    adversarial = [q for q in questions if q["category"] == "adversarial"]

    print(f"Golden set: {len(scored)} scored (sql_only/doc_only/hybrid), "
          f"{len(ambiguous)} ambiguous, {len(adversarial)} adversarial\n")

    print("=== Building Ragas dataset (running each question through the real system) ===\n")
    dataset, run_records, skipped = build_ragas_dataset(scored)

    ragas_df = None
    if len(dataset) > 0:
        results = run_ragas_scoring(dataset)
        ragas_df = results.to_pandas()
        ragas_df.to_csv(CSV_OUT, index=False)
        print(f"\nRaw Ragas scores saved to {CSV_OUT}")
        print("\n=== MEAN SCORES ===")
        metric_cols = [c for c in ragas_df.columns if c not in ("user_input", "response", "retrieved_contexts")]
        for col in metric_cols:
            print(f"{col}: {ragas_df[col].mean():.3f}")
    else:
        print("No questions produced scoreable results - skipping Ragas scoring entirely.")

    print("\n=== Checking ambiguous question routing ===\n")
    ambiguous_records = run_ambiguous_checks(ambiguous)
    for r in ambiguous_records:
        print(f"  [{'PASS' if r['passed'] else 'FAIL'}] {r['question']} -> route={r['route']}")

    print("\n=== Checking adversarial questions (real end-to-end run) ===\n")
    adversarial_records = run_adversarial_checks(adversarial)
    for r in adversarial_records:
        label = "SAFE" if r["safe"] else ("NEEDS REVIEW" if r["safe"] is None else "UNSAFE")
        print(f"  [{label}] {r['question']}")

    write_report(ragas_df, run_records, skipped, ambiguous_records, adversarial_records)


if __name__ == "__main__":
    main()