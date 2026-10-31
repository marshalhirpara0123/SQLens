"""
Phase 3 — SQL self-correction ablation study.

Runs every question in eval/ablation_questions.json twice through
sql_agent.agent.ask(): once with max_retries=1 (self-correction
disabled) and once with max_retries=3 (production default, self-
correction enabled). Reports the success-rate delta — the headline
number the roadmap's Month-3 milestone asks for.

This calls the real LLM ~2x per question (more if retries actually
fire), so it is slow — expect several minutes for the full 16-question
starter set, more if Ollama is running on modest hardware. If you just
want a quick sanity check that the script itself works before
committing to a full run, temporarily trim eval/ablation_questions.json
down to 2-3 questions, run it, then restore the full list.

Usage:
    python eval/run_ablation.py
Output:
    Printed per-question progress + summary table
    eval/ablation_results.json (raw results, for the final report)
"""
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from sql_agent.agent import ask

QUESTIONS_FILE = Path(__file__).with_name("ablation_questions.json")
RESULTS_FILE = Path(__file__).with_name("ablation_results.json")

NO_SELF_CORRECTION_RETRIES = 1
WITH_SELF_CORRECTION_RETRIES = 3  # matches sql_agent/agent.py's MAX_RETRIES default


def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def run_one(question: str, max_retries: int) -> dict:
    result = ask(question, max_retries=max_retries)
    return {
        "succeeded": result.succeeded,
        "attempts": result.attempts,
        "used_self_correction": result.used_self_correction,
        "final_sql": result.final_sql,
        "error": result.error,
        "attempt_log": result.attempt_log,
    }


def main():
    questions = load_questions()
    results = []

    print(f"Running ablation over {len(questions)} questions "
          f"(this calls the local LLM {len(questions) * 2}+ times — may take a while)...\n")

    start = time.time()
    for i, q_obj in enumerate(questions, 1):
        question = q_obj["question"]
        difficulty = q_obj.get("difficulty", "unspecified")
        print(f"[{i}/{len(questions)}] ({difficulty}) {question}")

        no_sc = run_one(question, NO_SELF_CORRECTION_RETRIES)
        print(f"    Without self-correction: {'PASS' if no_sc['succeeded'] else 'FAIL'} "
              f"(attempts={no_sc['attempts']})")

        with_sc = run_one(question, WITH_SELF_CORRECTION_RETRIES)
        print(f"    With self-correction:    {'PASS' if with_sc['succeeded'] else 'FAIL'} "
              f"(attempts={with_sc['attempts']}, used_self_correction={with_sc['used_self_correction']})")

        results.append({
            "question": question,
            "difficulty": difficulty,
            "without_self_correction": no_sc,
            "with_self_correction": with_sc,
        })

    elapsed = time.time() - start
    total = len(results)
    no_sc_passes = sum(1 for r in results if r["without_self_correction"]["succeeded"])
    with_sc_passes = sum(1 for r in results if r["with_self_correction"]["succeeded"])

    no_sc_rate = (no_sc_passes / total * 100) if total else 0.0
    with_sc_rate = (with_sc_passes / total * 100) if total else 0.0

    print("\n" + "=" * 60)
    print("ABLATION SUMMARY")
    print("=" * 60)
    print(f"Total questions:                  {total}")
    print(f"Success WITHOUT self-correction:  {no_sc_passes}/{total} ({no_sc_rate:.1f}%)")
    print(f"Success WITH self-correction:     {with_sc_passes}/{total} ({with_sc_rate:.1f}%)")
    print(f"Delta (self-correction lift):     +{with_sc_rate - no_sc_rate:.1f} percentage points")
    print(f"Total run time:                   {elapsed:.1f}s")

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "total_questions": total,
            "without_self_correction_pass_rate": no_sc_rate,
            "with_self_correction_pass_rate": with_sc_rate,
            "delta_percentage_points": with_sc_rate - no_sc_rate,
            "per_question_results": results,
        }, f, indent=2)

    print(f"\nRaw results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()