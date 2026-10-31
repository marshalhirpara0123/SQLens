"""
Phase 3 — formats eval/ablation_results.json into a markdown table and
highlights the specific question(s) that failed without self-correction
and succeeded with it, for the final report's ablation section.

Usage:
    python eval/print_ablation_report.py
"""
import json
from pathlib import Path

RESULTS_FILE = Path(__file__).with_name("ablation_results.json")


def _specific_error(attempt_result: dict) -> str:
    """Pull the specific error from attempt_log rather than the generic
    'exceeded max retries' fallback, matching the fix already applied in
    backend/graph.py's synthesizer. Falls back to the generic message for
    older results.json files that don't have attempt_log saved."""
    for entry in reversed(attempt_result.get("attempt_log", [])):
        if entry.get("error"):
            return f"{entry.get('stage', 'unknown stage')}: {entry['error']}"
    return attempt_result.get("error") or "no further detail was recorded"


def main():
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["per_question_results"]

    print("## Ablation results\n")
    print(f"- Total questions: {data['total_questions']}")
    print(f"- Success WITHOUT self-correction: {sum(1 for r in results if r['without_self_correction']['succeeded'])}/{data['total_questions']} "
          f"({data['without_self_correction_pass_rate']:.1f}%)")
    print(f"- Success WITH self-correction: {sum(1 for r in results if r['with_self_correction']['succeeded'])}/{data['total_questions']} "
          f"({data['with_self_correction_pass_rate']:.1f}%)")
    print(f"- Delta: +{data['delta_percentage_points']:.1f} percentage points\n")

    print("| # | Difficulty | Question | Without SC | With SC | Attempts (with SC) |")
    print("|---|---|---|---|---|---|")
    for i, r in enumerate(results, 1):
        wo = "✅ Pass" if r["without_self_correction"]["succeeded"] else "❌ Fail"
        wi = "✅ Pass" if r["with_self_correction"]["succeeded"] else "❌ Fail"
        attempts = r["with_self_correction"]["attempts"]
        q = r["question"].replace("|", "\\|")
        print(f"| {i} | {r['difficulty']} | {q} | {wo} | {wi} | {attempts} |")

    print("\n## Questions where self-correction made the difference\n")
    rescued = [r for r in results
               if not r["without_self_correction"]["succeeded"] and r["with_self_correction"]["succeeded"]]
    if not rescued:
        print("None in this run — every question either passed on the first attempt "
              "in both conditions, or failed in both.")
    for r in rescued:
        print(f"### {r['question']}\n")
        print(f"**First-attempt failure:** {_specific_error(r['without_self_correction'])}\n")
        print(f"**Recovered after {r['with_self_correction']['attempts']} attempt(s), "
              f"final query:**\n```sql\n{r['with_self_correction']['final_sql']}\n```\n")

    still_failing = [r for r in results if not r["with_self_correction"]["succeeded"]]
    if still_failing:
        print("## Questions that failed even WITH self-correction\n")
        print("(These are candidates for your report's \"documented failure cases\" section.)\n")
        for r in still_failing:
            print(f"- **{r['question']}** — {_specific_error(r['with_self_correction'])}")


if __name__ == "__main__":
    main()