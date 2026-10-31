import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "eval"))

import pytest

from eval.router_eval_utils import HoldoutError, load_holdout, wilson_interval

GOLDEN = ROOT / "eval" / "golden_set_template.json"


def _make(tmp_path, questions):
    p = tmp_path / "holdout.json"
    p.write_text(json.dumps({"questions": questions}), encoding="utf-8")
    return p


def _valid_questions():
    cats = ["sql_only", "doc_only", "hybrid"]
    return [
        {"id": f"h-{i:02d}", "category": cats[i % 3], "question": f"A brand new held out question number {i}?", "author": "teammate"}
        for i in range(12)
    ]


# ------------------------------------------------------------------ Wilson

def test_wilson_matches_known_reference_values():
    lo, hi = wilson_interval(8, 10)
    assert (round(lo, 3), round(hi, 3)) == (0.490, 0.943)  # standard published example
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and round(hi, 3) == 0.278


def test_wilson_stays_inside_zero_and_one_at_the_extremes():
    lo, hi = wilson_interval(19, 19)
    assert hi == 1.0 and 0.8 < lo < 1.0
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_wilson_interval_narrows_as_sample_grows():
    small = wilson_interval(9, 10)
    large = wilson_interval(90, 100)
    assert (large[1] - large[0]) < (small[1] - small[0])


# -------------------------------------------------------------- validation

def test_valid_file_loads_and_defaults_are_filled(tmp_path):
    qs = _valid_questions()
    del qs[0]["author"]
    loaded = load_holdout(_make(tmp_path, qs), GOLDEN)
    assert len(loaded) == 12
    assert loaded[0]["author"] == "unspecified"


def test_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(HoldoutError, match="not found"):
        load_holdout(tmp_path / "nope.json", GOLDEN)


def test_empty_template_tells_you_what_to_do(tmp_path):
    with pytest.raises(HoldoutError, match="no questions yet"):
        load_holdout(_make(tmp_path, []), GOLDEN)


def test_too_few_questions_rejected(tmp_path):
    with pytest.raises(HoldoutError, match="at least 12"):
        load_holdout(_make(tmp_path, _valid_questions()[:11]), GOLDEN)


def test_each_category_needs_a_minimum(tmp_path):
    qs = _valid_questions()
    for q in qs:
        if q["category"] == "hybrid":
            q["category"] = "sql_only"
    with pytest.raises(HoldoutError, match="hybrid"):
        load_holdout(_make(tmp_path, qs), GOLDEN)


def test_invalid_category_rejected(tmp_path):
    qs = _valid_questions()
    qs[3]["category"] = "adversarial"
    with pytest.raises(HoldoutError, match="category must be one of"):
        load_holdout(_make(tmp_path, qs), GOLDEN)


def test_placeholder_text_rejected(tmp_path):
    qs = _valid_questions()
    qs[5]["question"] = "REPLACE_WITH_YOUR_QUESTION"
    with pytest.raises(HoldoutError, match="placeholder"):
        load_holdout(_make(tmp_path, qs), GOLDEN)


def test_duplicate_question_within_file_rejected(tmp_path):
    qs = _valid_questions()
    qs[7]["question"] = qs[2]["question"].upper() + "!!"  # same text, different case/punctuation
    with pytest.raises(HoldoutError, match="appears twice"):
        load_holdout(_make(tmp_path, qs), GOLDEN)


def test_duplicate_id_rejected(tmp_path):
    qs = _valid_questions()
    qs[1]["id"] = qs[0]["id"]
    with pytest.raises(HoldoutError, match="Duplicate id"):
        load_holdout(_make(tmp_path, qs), GOLDEN)


def test_reusing_a_golden_set_question_is_rejected_as_not_held_out(tmp_path):
    golden_q = json.loads(GOLDEN.read_text(encoding="utf-8"))["questions"][0]["question"]
    qs = _valid_questions()
    qs[4]["question"] = golden_q.lower()
    with pytest.raises(HoldoutError, match="already in the golden set"):
        load_holdout(_make(tmp_path, qs), GOLDEN)


def test_invalid_json_is_a_clear_error(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{ not json", encoding="utf-8")
    with pytest.raises(HoldoutError, match="not valid JSON"):
        load_holdout(p, GOLDEN)