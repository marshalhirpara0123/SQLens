"""
Helpers for the held-out router evaluation (eval/measure_router.py --holdout).

Kept free of any heavy imports (no Ollama, ChromaDB, embeddings) so the
validation and statistics are unit-tested in tests/test_router_eval_utils.py.

The validation is deliberately strict. A held-out set only means something
if it really is held out, so a file that is too small, unbalanced, still
contains template placeholders, repeats itself, or re-uses a golden-set
question is rejected with a message saying exactly what to fix.
"""
import json
import math
import re
from pathlib import Path

VALID_CATEGORIES = ("sql_only", "doc_only", "hybrid")
MIN_QUESTIONS = 12
MIN_PER_CATEGORY = 3


class HoldoutError(Exception):
    """The held-out question file is missing, empty, or not usable."""


def wilson_interval(successes: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a proportion, as (low, high) fractions.

    Preferred over the textbook normal approximation for small samples and
    for proportions near 0% or 100% — exactly the situation with ~15
    questions and a coverage rate near 95%."""
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def load_holdout(path: Path, golden_path: Path) -> list:
    """Load and validate the held-out file. Returns a list of question dicts
    (id, category, question, author). Raises HoldoutError with a specific,
    actionable message on any problem."""
    path = Path(path)
    if not path.exists():
        raise HoldoutError(f"Held-out file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise HoldoutError(f"{path.name} is not valid JSON: {e}")

    questions = data.get("questions") if isinstance(data, dict) else None
    if not isinstance(questions, list) or not questions:
        raise HoldoutError(
            f"{path.name} has no questions yet. Add at least {MIN_QUESTIONS} "
            f"(see the _instructions at the top of that file)."
        )

    golden_texts = set()
    if Path(golden_path).exists():
        golden = json.loads(Path(golden_path).read_text(encoding="utf-8"))
        golden_texts = {_normalise(q["question"]) for q in golden.get("questions", [])}

    cleaned, seen_ids, seen_texts = [], set(), set()
    for i, q in enumerate(questions, 1):
        if not isinstance(q, dict):
            raise HoldoutError(f"Entry #{i} is not an object with id/category/question.")
        text = str(q.get("question", "")).strip()
        category = q.get("category")
        qid = str(q.get("id") or f"h-{i:02d}")

        if not text or "REPLACE" in text.upper():
            raise HoldoutError(f"Entry {qid}: the question text is empty or still a placeholder.")
        if category not in VALID_CATEGORIES:
            raise HoldoutError(
                f"Entry {qid}: category must be one of {list(VALID_CATEGORIES)}, got {category!r}."
            )
        if qid in seen_ids:
            raise HoldoutError(f"Duplicate id {qid!r}.")
        norm = _normalise(text)
        if norm in seen_texts:
            raise HoldoutError(f"Entry {qid}: this question appears twice in the held-out file.")
        if norm in golden_texts:
            raise HoldoutError(
                f"Entry {qid}: this question is already in the golden set, so it isn't held out. "
                "Write a new one."
            )
        seen_ids.add(qid)
        seen_texts.add(norm)
        cleaned.append({
            "id": qid,
            "category": category,
            "question": text,
            "author": str(q.get("author", "unspecified")).strip() or "unspecified",
        })

    if len(cleaned) < MIN_QUESTIONS:
        raise HoldoutError(
            f"Only {len(cleaned)} questions; need at least {MIN_QUESTIONS} for the "
            "result to mean anything."
        )
    counts = {c: sum(q["category"] == c for q in cleaned) for c in VALID_CATEGORIES}
    thin = [c for c, n in counts.items() if n < MIN_PER_CATEGORY]
    if thin:
        raise HoldoutError(
            f"Need at least {MIN_PER_CATEGORY} questions in every category; "
            f"too few in: {', '.join(thin)} (counts: {counts})."
        )
    return cleaned