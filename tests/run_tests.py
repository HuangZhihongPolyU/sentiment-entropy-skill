from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sentiment_entropy_agent.core import compute_entropy, prepare_rag_context, rank_items, score_item


def assert_close(actual: float, expected: float, tolerance: float = 1e-3) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{actual} != {expected}")


def test_entropy_formula() -> None:
    assert_close(compute_entropy({"positive": 1, "neutral": 1, "negative": 1}).entropy, math.log2(3))
    assert_close(compute_entropy({"positive": 1, "neutral": 0, "negative": 1}).entropy, 1.0)
    assert_close(compute_entropy({"positive": 0, "neutral": 0, "negative": 2}).entropy, 0.0)


def test_score_item() -> None:
    scored = score_item(
        {
            "id": "x",
            "aspect_sentiments": [
                {"aspect": "food", "sentiment": "positive"},
                {"aspect": "service", "sentiment": "negative"},
                {"aspect": "staff", "sentiment": "neutral"},
            ],
        }
    )
    assert scored["diagnosticity"] == "high"
    assert scored["routes"]["human_review"] is True


def test_rank_and_rag() -> None:
    payload = json.loads((ROOT / "examples" / "rank_request.json").read_text())
    ranked = rank_items(payload)
    assert ranked["ranked_items"][0]["id"] == "review_1"

    rag = prepare_rag_context(json.loads((ROOT / "examples" / "rag_context_request.json").read_text()))
    assert len(rag["context_items"]) == 3
    assert "diagnostic_summary" in rag["context_items"][0]


def main() -> None:
    test_entropy_formula()
    test_score_item()
    test_rank_and_rag()
    print("All tests passed.")


if __name__ == "__main__":
    main()

