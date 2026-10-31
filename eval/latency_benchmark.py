"""
Phase 3 — latency breakdown study.

Runs eval/golden_set_template.json through backend.graph.run_query() and
reports latency broken down by stage (router / sql_tool / rag_tool /
synthesizer), both overall and per route type (sql / rag / both /
unknown) — the roadmap's Month-3 milestone: "measure latency broken down
by router / retrieval / generation time to find the real bottleneck."

Mapping to the roadmap's language: "router" = the classification call;
"retrieval" = sql_tool + rag_tool (each also includes their own internal
LLM generation call - text-to-SQL writing, RAG answer writing - not just
raw data fetching, since that's how the existing code is structured);
"generation" = the synthesizer's final-answer call. This is noted
explicitly here rather than glossed over, since it's a real scoping
choice for the report's methodology section.

RUNS_PER_QUESTION controls how many times each question is run (default
1, for speed). Bump to 3 for more statistically stable numbers if you
have time before the deadline - this multiplies total run time
accordingly.

Usage:
    python eval/run_latency_benchmark.py
Output:
    Printed summary (overall + per-route breakdown)
    eval/latency_results.csv   (one row per run, every stage's timing)
    eval/latency_report.md     (formatted for the final report)
"""
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.graph import run_query

GOLDEN_SET_FILE = Path(__file__).with_name("golden_set_template.json")
CSV_OUT = Path(__file__).with_name("latency_results.csv")
REPORT_OUT = Path(__file__).with_name("latency_report.md")

RUNS_PER_QUESTION = 1
STAGE_KEYS = ["router", "sql_tool", "rag_tool", "synthesizer", "total"]


def load_questions():
    with open(GOLDEN_SET_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


def run_all(questions):
    records = []
    total_runs = len(questions) * RUNS_PER_QUESTION
    run_num = 0

    for q_obj in questions:
        question = q_obj["question"]
        for run_i in range(1, RUNS_PER_QUESTION + 1):
            run_num += 1
            print(f"[{run_num}/{total_runs}] ({q_obj['category']}, run {run_i}) {question}")
            try:
                result = run_query(question)
            except Exception as e:
                print(f"    ERROR - {e}")
                continue

            timings = result.get("timings", {})
            record = {
                "id": q_obj["id"],
                "category": q_obj["category"],
                "question": question,
                "run": run_i,
                "route": result.get("route"),
            }
            for k in STAGE_KEYS:
                record[k] = timings.get(k)
            records.append(record)

            print(f"    route={record['route']} " +
                  ", ".join(f"{k}={record[k]:.2f}s" for k in STAGE_KEYS if record[k] is not None))

    return records


def _mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def summarize(records, group_key=None):
    """Returns {group_value: {stage: mean_seconds}}. group_key=None means one overall group."""
    groups = {}
    for r in records:
        key = r[group_key] if group_key else "overall"
        groups.setdefault(key, {stage: [] for stage in STAGE_KEYS})
        for stage in STAGE_KEYS:
            if r[stage] is not None:
                groups[key][stage].append(r[stage])
    return {
        key: {stage: (statistics.mean(vals) if vals else None) for stage, vals in stages.items()}
        for key, stages in groups.items()
    }


def write_csv(records):
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "category", "question", "run", "route"] + STAGE_KEYS)
        writer.writeheader()
        for r in records:
            writer.writerow(r)


def write_report(records, overall, by_route):
    lines = ["# Latency breakdown report\n"]
    lines.append(f"Runs per question: {RUNS_PER_QUESTION}. Total runs: {len(records)}.\n")

    lines.append("## Overall mean latency by stage (seconds)\n")
    lines.append("| Stage | Mean seconds |")
    lines.append("|---|---|")
    for stage in STAGE_KEYS:
        val = overall["overall"][stage]
        lines.append(f"| {stage} | {val:.2f} |" if val is not None else f"| {stage} | n/a |")

    total_mean = overall["overall"]["total"]
    if total_mean:
        lines.append("\n### Share of total time per stage\n")
        for stage in STAGE_KEYS:
            if stage == "total":
                continue
            val = overall["overall"][stage]
            if val is not None:
                lines.append(f"- **{stage}**: {val:.2f}s ({val / total_mean * 100:.1f}% of total)")

    lines.append("\n## Mean latency by route type (seconds)\n")
    lines.append("| Route | Router | SQL tool | RAG tool | Synthesizer | Total |")
    lines.append("|---|---|---|---|---|---|")
    for route, stages in by_route.items():
        row = [route] + [
            f"{stages[s]:.2f}" if stages[s] is not None else "n/a"
            for s in ["router", "sql_tool", "rag_tool", "synthesizer", "total"]
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines.append("\n## Notes on the sequential \"both\" route\n")
    both_total = by_route.get("both", {}).get("total")
    sql_total = by_route.get("sql", {}).get("total")
    rag_total = by_route.get("rag", {}).get("total")
    if both_total and sql_total and rag_total:
        lines.append(
            f"The 'both' route (sql_tool then rag_tool sequentially, per the Phase 2 design "
            f"decision) averages {both_total:.2f}s total, versus {sql_total:.2f}s for sql-only "
            f"and {rag_total:.2f}s for rag-only. This is the concrete cost of the sequential-not-"
            f"parallel choice made in backend/graph.py — worth citing directly in the report's "
            f"discussion of that design trade-off."
        )

    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nFull report written to {REPORT_OUT}")


def main():
    questions = load_questions()
    print(f"Running latency benchmark over {len(questions)} questions "
          f"({RUNS_PER_QUESTION} run(s) each)...\n")

    records = run_all(questions)
    write_csv(records)
    print(f"\nRaw per-run timings saved to {CSV_OUT}")

    overall = summarize(records, group_key=None)
    by_route = summarize(records, group_key="route")

    print("\n=== OVERALL MEAN LATENCY BY STAGE ===")
    for stage in STAGE_KEYS:
        val = overall["overall"][stage]
        print(f"  {stage}: {val:.2f}s" if val is not None else f"  {stage}: n/a")

    print("\n=== MEAN LATENCY BY ROUTE ===")
    for route, stages in by_route.items():
        print(f"  {route}: total={stages['total']:.2f}s" if stages["total"] is not None else f"  {route}: n/a")

    write_report(records, overall, by_route)


if __name__ == "__main__":
    main()