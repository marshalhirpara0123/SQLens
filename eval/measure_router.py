"""
Router measurement — real LLM classifier, before vs. after the rule-based
safety net (backend/router_rules.py).

For every question it asks the real llama3:8b router for a route (the
"before"), applies the safety net (the "after"), and compares both against
what the question needs:

    sql_only -> needs the SQL tool        doc_only -> needs the RAG tool
    hybrid   -> needs both

Two numbers matter:
  * SOURCE COVERAGE  - % of questions where every needed tool actually runs.
    This decides whether the user gets a correct answer. ("both"/"unknown"
    run both tools.)
  * EXACT ROUTE      - % where the route equals the ideal one. "both" for a
    doc-only question still answers correctly, just slower; that gap is the
    latency cost of being safe, reported as OVER-ROUTED.

Every rate is shown with a 95% Wilson confidence interval, because with ~20
questions a bare percentage overstates how precisely it is known.

Two question sets:
  (default)     eval/golden_set_template.json. The safety net's cue words were
                chosen after seeing failures on THIS set, so results here are
                IN-SAMPLE: they show the known failures are fixed, not that it
                generalises.
  --holdout     eval/router_holdout.json. Questions written without looking
                at the rules. Results here are the honest generalisation
                estimate. The file is strictly validated (see
                eval/router_eval_utils.py).

Only the router is called (no SQL, RAG, or answer generation), so a run takes
about a minute or two.

Usage:
    python eval/measure_router.py                # golden set (in-sample)
    python eval/measure_router.py --holdout      # held-out set
Output:
    printed tables + eval/router_report[_holdout].md and .json
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent))
sys.path.insert(0, str(HERE))

from backend.llm_client import LLMError
from backend.router_rules import refine_route
from backend.router_sketch import route_question
from rag_pipeline.retrieve import list_indexed_documents
from router_eval_utils import HoldoutError, load_holdout, wilson_interval
from sql_agent import db_setup

GOLDEN = HERE / "golden_set_template.json"
HOLDOUT = HERE / "router_holdout.json"

NEEDS = {"sql_only": {"sql"}, "doc_only": {"rag"}, "hybrid": {"sql", "rag"}}
IDEAL = {"sql_only": "sql", "doc_only": "rag", "hybrid": "both"}


def covers(route: str) -> set:
    """Tools a route runs. backend/graph.py treats 'unknown' as 'both'."""
    return {"sql": {"sql"}, "rag": {"rag"}}.get(route, {"sql", "rag"})


def rate(k: int, n: int) -> str:
    if n == 0:
        return "n/a"
    lo, hi = wilson_interval(k, n)
    return f"{100 * k / n:.1f}% ({k}/{n}; 95% CI {100 * lo:.0f}-{100 * hi:.0f}%)"


def load_questions(holdout: bool) -> list:
    if holdout:
        return load_holdout(HOLDOUT, GOLDEN)
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))["questions"]
    return [q for q in data if q["category"] in NEEDS]


def main():
    parser = argparse.ArgumentParser(description="Measure the router before/after the safety net.")
    parser.add_argument("--holdout", action="store_true",
                        help="use eval/router_holdout.json (held-out) instead of the golden set (in-sample)")
    args = parser.parse_args()
    label = "HELD-OUT" if args.holdout else "golden set (in-sample)"
    suffix = "_holdout" if args.holdout else ""
    md_out = HERE / f"router_report{suffix}.md"
    json_out = HERE / f"router_results{suffix}.json"

    try:
        questions = load_questions(args.holdout)
    except HoldoutError as e:
        print(f"Cannot run the held-out test: {e}")
        sys.exit(2)

    db_setup.reset_to_default_db()  # both sets assume the Chinook database
    docs = list_indexed_documents()
    terms = db_setup.get_schema_terms()

    print(f"Question set: {label} - {len(questions)} questions")
    print(f"Indexed documents the router can see: {docs}\n")

    rows = []
    for i, q in enumerate(questions, 1):
        try:
            before = route_question(q["question"])
        except LLMError as e:
            print(f"Ollama is unreachable ({e}). Start it with `ollama serve` and re-run.")
            sys.exit(1)
        after, note = refine_route(before, q["question"], docs, terms)
        rows.append({"id": q["id"], "category": q["category"], "question": q["question"],
                     "author": q.get("author"), "before": before, "after": after, "note": note})
        flag = "  <- adjusted" if note else ""
        print(f"[{i:2}/{len(questions)}] {q['id']:<11} llm={before:<7} final={after:<7}{flag}")

    def stats(key):
        out = {}
        for cat in NEEDS:
            sub = [r for r in rows if r["category"] == cat]
            out[cat] = {
                "n": len(sub),
                "covered": sum(NEEDS[cat] <= covers(r[key]) for r in sub),
                "exact": sum(r[key] == IDEAL[cat] for r in sub),
                "over": sum(covers(r[key]) > NEEDS[cat] for r in sub),
            }
        tot = {k: sum(v[k] for v in out.values()) for k in ("n", "covered", "exact", "over")}
        return out, tot

    b_cat, b_tot = stats("before")
    a_cat, a_tot = stats("after")

    L = [f"# Router measurement - {label} (real llama3:8b)\n"]
    if args.holdout:
        authors = Counter(r["author"] for r in rows)
        L.append("**Held-out set:** written without reference to the routing rules. "
                 f"Authors: {', '.join(f'{a} ({n})' for a, n in authors.items())}. "
                 "If the rules are changed after seeing these results, this set is no longer "
                 "held out and any new numbers must be reported as in-sample.\n")
    else:
        L.append("**In-sample:** the safety net's cue words were chosen after seeing failures "
                 "on this set, so these numbers show the known failures are fixed, "
                 "not that the net generalises. See the held-out report for that.\n")
    L.append(f"Questions: {b_tot['n']}. Documents visible: {', '.join(docs)}\n")
    L += ["## Source coverage - every needed tool actually runs\n",
          "| Category | LLM only | + safety net |", "|---|---|---|"]
    for cat in NEEDS:
        L.append(f"| {cat} | {rate(b_cat[cat]['covered'], b_cat[cat]['n'])} | {rate(a_cat[cat]['covered'], a_cat[cat]['n'])} |")
    L.append(f"| **all** | **{rate(b_tot['covered'], b_tot['n'])}** | **{rate(a_tot['covered'], a_tot['n'])}** |")
    L += ["\n## Exact ideal route, and over-routing (an extra tool runs = extra latency)\n",
          "| | LLM only | + safety net |", "|---|---|---|",
          f"| Exact route | {rate(b_tot['exact'], b_tot['n'])} | {rate(a_tot['exact'], a_tot['n'])} |",
          f"| Over-routed | {rate(b_tot['over'], b_tot['n'])} | {rate(a_tot['over'], a_tot['n'])} |"]

    changed = [r for r in rows if r["note"]]
    L.append(f"\n## Routes changed by the safety net ({len(changed)})\n")
    L += [f"- **{r['id']}** `{r['before']}` -> `{r['after']}` ({r['category']}) - {r['question']}" for r in changed] or ["None."]
    missing = [r for r in rows if not NEEDS[r["category"]] <= covers(r["after"])]
    L.append(f"\n## Still missing a needed tool after the safety net ({len(missing)})\n")
    L += [f"- **{r['id']}** (`{r['after']}`, needs {sorted(NEEDS[r['category']])}) - {r['question']}" for r in missing] or ["None."]
    over = [r for r in rows if covers(r["after"]) > NEEDS[r["category"]]]
    L.append(f"\n## Over-routed after the safety net ({len(over)}) - correct but slower\n")
    L += [f"- **{r['id']}** `{r['after']}` (ideal `{IDEAL[r['category']]}`) - {r['question']}" for r in over] or ["None."]

    md_out.write_text("\n".join(L) + "\n", encoding="utf-8")
    json_out.write_text(json.dumps({"set": label, "rows": rows, "before": b_tot, "after": a_tot}, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"[{label}]")
    print(f"SOURCE COVERAGE  LLM only: {rate(b_tot['covered'], b_tot['n'])}")
    print(f"                 +net:     {rate(a_tot['covered'], a_tot['n'])}")
    print(f"OVER-ROUTED      LLM only: {rate(b_tot['over'], b_tot['n'])}")
    print(f"                 +net:     {rate(a_tot['over'], a_tot['n'])}")
    print(f"\nReport written to {md_out}")


if __name__ == "__main__":
    main()