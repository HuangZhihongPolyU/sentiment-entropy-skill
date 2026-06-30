"""
Sentiment Entropy Agent Skill — End-to-End Demo

Demonstrates the full pipeline: score → rank → RAG context selection,
using real Yelp reviews from Wong & Lo (2026).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sentiment_entropy_agent.core import (
    MAX_THREE_CATEGORY_ENTROPY,
    prepare_rag_context,
    rank_items,
    score_item,
)


def divider(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def main() -> None:
    rank_payload = json.loads(
        (ROOT / "examples" / "rank_request.json").read_text()
    )
    items = rank_payload["items"]

    ############################################################################
    # 1. Score individual items
    ############################################################################
    divider("STEP 1 — Score Each Review Individually")

    scored = [score_item(item) for item in items]
    for s in scored:
        text = s.get("text", "")
        print(f"\n[{s['id']}]")
        print(f"  Text       : {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"  Entropy    : {s['entropy']:.4f}  (normalized: {s['normalized_entropy']:.4f},  max: {MAX_THREE_CATEGORY_ENTROPY:.4f})")
        print(f"  Diagnosticity: {s['diagnosticity']}")
        print(f"  Counts     : pos={s['counts']['positive']},  neu={s['counts']['neutral']},  neg={s['counts']['negative']}")
        print(f"  Route      : {s['routes']['primary_route']}")
        if s["routes"]["human_review"]:
            reasons = ", ".join(s["routes"]["human_review_reasons"])
            print(f"  Human Review: REQUIRED — {reasons}")
        else:
            print(f"  Human Review: not required")
        if s.get("warnings"):
            print(f"  Warnings   : {s['warnings']}")

    ############################################################################
    # 2. Rank all items
    ############################################################################
    divider("STEP 2 — Rank All Reviews by Normalized Entropy (Descending)")

    ranking = rank_items(rank_payload)
    print(f"\n  Total items: {ranking['total_items']}  |  Top-K: {ranking['top_k']}")
    print(f"\n  {'Rank':<5} {'ID':<18} {'Entropy':>9} {'Norm.E':>7} {'Diagnosticity':>15} {'Primary Route':>25}")
    print(f"  {'-'*5} {'-'*18} {'-'*9} {'-'*7} {'-'*15} {'-'*25}")
    for i, item in enumerate(ranking["ranked_items"], 1):
        print(
            f"  {i:<5} {item.get('id','?'):<18} {item['entropy']:>9.4f} "
            f"{item['normalized_entropy']:>7.4f} "
            f"{item['diagnosticity']:>15} "
            f"{item['routes']['primary_route']:>25}"
        )

    ############################################################################
    # 3. Select RAG context
    ############################################################################
    divider("STEP 3 — Select Top-K Reviews as RAG / Agent Context")

    rag = prepare_rag_context(
        json.loads((ROOT / "examples" / "rag_context_request.json").read_text())
    )
    print(f"\n  Selection policy: {rag['selection_policy']}")
    print(f"  Total items: {rag['total_items']}  |  Top-K: {rag['top_k']}")
    for i, ctx in enumerate(rag["context_items"], 1):
        print(f"\n  [{i}] {ctx['id']}")
        print(f"      Entropy: {ctx['entropy']:.4f}  |  Diagnosticity: {ctx['diagnosticity']}")
        print(f"      Primary route: {ctx['primary_route']}")
        print(f"      Human review:  {ctx['human_review']}")
        if "diagnostic_summary" in ctx:
            print(f"      Summary: {ctx['diagnostic_summary']}")

    ############################################################################
    # Summary
    ############################################################################
    divider("DEMO COMPLETE")
    print(f"""
  Pipeline executed:
    {len(items)} Yelp reviews -> scored -> ranked -> top-{rag['top_k']} selected as RAG context

  Key insight:
    - High entropy (approx {MAX_THREE_CATEGORY_ENTROPY:.2f}) = balanced pos/neu/neg = high diagnostic value
    - Low entropy (approx 0.0) = all sentiments concentrated = low diagnostic value
    - High-entropy + cross-functional = human-review flag triggered
""")


if __name__ == "__main__":
    main()
