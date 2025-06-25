import os
from unittest.mock import patch

from .healthbench_eval import RubricItem, calculate_score, build_grader_sampler


def test_calculate_score():
    rubric_items = [
        RubricItem(criterion="test", points=7, tags=[]),
        RubricItem(criterion="test", points=5, tags=[]),
        RubricItem(criterion="test", points=10, tags=[]),
        RubricItem(criterion="test", points=-6, tags=[]),
    ]
    grading_response_list = [
        {"criteria_met": True},
        {"criteria_met": False},
        {"criteria_met": True},
        {"criteria_met": True},
    ]
    total_possible = 7 + 5 + 10
    achieved = 7 + 0 + 10 - 6
    assert (
        calculate_score(rubric_items, grading_response_list)
        == achieved / total_possible
    )


def test_build_grader_sampler_gemini():
    dummy_resp = {
        "candidates": [{"content": {"parts": [{"text": "ok"}]}}]
    }
    with patch("simple_evals.sampler.gemini_sampler.requests.post") as post:
        post.return_value.json.return_value = dummy_resp
        post.return_value.raise_for_status.return_value = None
        with patch.dict(os.environ, {"GEMINI_API_KEY": "dummy"}):
            sampler = build_grader_sampler("gemini-test")
            resp = sampler([{"role": "user", "content": "hi"}])
    assert resp.response_text == "ok"


if __name__ == "__main__":
    test_calculate_score()
