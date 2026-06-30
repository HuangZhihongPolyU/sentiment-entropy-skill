"""Core sentiment entropy logic.

The preferred input is labelled aspect-level sentiment, because that is the
construct used in the paper. Text-only extraction is supported as a demo
fallback through ``text_heuristic.extract_demo_aspect_sentiments``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .routing import recommend_routes
from .text_heuristic import extract_demo_aspect_sentiments

SENTIMENTS = ("positive", "neutral", "negative")
MAX_THREE_CATEGORY_ENTROPY = math.log2(3)

SENTIMENT_ALIASES = {
    "pos": "positive",
    "+": "positive",
    "positive": "positive",
    "good": "positive",
    "neu": "neutral",
    "neutral": "neutral",
    "mixed": "neutral",
    "0": "neutral",
    "neg": "negative",
    "-": "negative",
    "negative": "negative",
    "bad": "negative",
}


class SentimentEntropyError(ValueError):
    """Raised when a request cannot be scored."""


@dataclass(frozen=True)
class EntropyResult:
    counts: dict[str, int]
    proportions: dict[str, float]
    entropy: float
    normalized_entropy: float


def normalize_sentiment(value: Any) -> str:
    """Normalize a sentiment label to positive, neutral, or negative."""

    key = str(value).strip().lower()
    if key not in SENTIMENT_ALIASES:
        raise SentimentEntropyError(
            f"Unknown sentiment {value!r}; expected positive, neutral, or negative."
        )
    return SENTIMENT_ALIASES[key]


def counts_from_aspect_sentiments(
    aspect_sentiments: list[dict[str, Any]] | list[str],
) -> dict[str, int]:
    """Convert aspect sentiment labels into sentiment counts."""

    counts = {sentiment: 0 for sentiment in SENTIMENTS}
    for item in aspect_sentiments:
        if isinstance(item, str):
            sentiment = normalize_sentiment(item)
        elif isinstance(item, dict):
            if "sentiment" not in item:
                raise SentimentEntropyError(
                    "Each aspect sentiment object must include a 'sentiment' field."
                )
            sentiment = normalize_sentiment(item["sentiment"])
        else:
            raise SentimentEntropyError(
                "Aspect sentiments must be strings or objects with a sentiment field."
            )
        counts[sentiment] += 1
    return counts


def normalize_counts(raw_counts: dict[str, Any]) -> dict[str, int]:
    """Normalize count keys and validate nonnegative integer counts."""

    counts = {sentiment: 0 for sentiment in SENTIMENTS}
    for key, value in raw_counts.items():
        sentiment = normalize_sentiment(key)
        try:
            number = int(value)
        except (ValueError, TypeError) as exc:
            raise SentimentEntropyError(f"Invalid count for {key!r}: {value!r}") from exc
        if number < 0:
            raise SentimentEntropyError(f"Counts must be nonnegative: {key!r}={value!r}")
        counts[sentiment] += number
    return counts


def compute_entropy(raw_counts: dict[str, Any]) -> EntropyResult:
    """Compute Shannon entropy over positive, neutral, and negative counts."""

    counts = normalize_counts(raw_counts)
    total = sum(counts.values())
    if total == 0:
        proportions = {sentiment: 0.0 for sentiment in SENTIMENTS}
        return EntropyResult(
            counts=counts,
            proportions=proportions,
            entropy=0.0,
            normalized_entropy=0.0,
        )

    proportions = {sentiment: counts[sentiment] / total for sentiment in SENTIMENTS}
    entropy = -sum(p * math.log2(p) for p in proportions.values() if p > 0)
    if abs(entropy) < 1e-12:
        entropy = 0.0
    normalized_entropy = entropy / MAX_THREE_CATEGORY_ENTROPY
    if abs(normalized_entropy) < 1e-12:
        normalized_entropy = 0.0
    return EntropyResult(
        counts=counts,
        proportions=proportions,
        entropy=entropy,
        normalized_entropy=normalized_entropy,
    )


def diagnostic_label(normalized_entropy: float) -> str:
    """Map normalized entropy to a human-readable diagnosticity label."""

    if normalized_entropy >= 0.67:
        return "high"
    if normalized_entropy >= 0.34:
        return "moderate"
    return "low"


def _aspect_sentiments_from_payload(payload: dict[str, Any]) -> tuple[list[dict[str, str]], str, list[str]]:
    warnings: list[str] = []
    if "aspect_sentiments" in payload and payload["aspect_sentiments"] is not None:
        raw = payload["aspect_sentiments"]
        if not isinstance(raw, list):
            raise SentimentEntropyError("'aspect_sentiments' must be a list.")
        normalized: list[dict[str, str]] = []
        for item in raw:
            if isinstance(item, str):
                normalized.append({"aspect": "unspecified", "sentiment": normalize_sentiment(item)})
            elif isinstance(item, dict):
                normalized.append(
                    {
                        "aspect": str(item.get("aspect", "unspecified")),
                        "sentiment": normalize_sentiment(item.get("sentiment")),
                    }
                )
            else:
                raise SentimentEntropyError(
                    "Aspect sentiment items must be strings or objects."
                )
        return normalized, "labelled_aspect_sentiments", warnings

    if "counts" in payload and payload["counts"] is not None:
        counts = normalize_counts(payload["counts"])
        expanded = []
        for sentiment, count in counts.items():
            for _ in range(count):
                expanded.append({"aspect": "count_only", "sentiment": sentiment})
        return expanded, "sentiment_counts", warnings

    text = str(payload.get("text", "")).strip()
    if text:
        warnings.append(
            "Text-only mode uses a lightweight demo heuristic; replace with ABSA or LLM extraction for research use."
        )
        return extract_demo_aspect_sentiments(text), "demo_text_heuristic", warnings

    raise SentimentEntropyError(
        "Provide one of: aspect_sentiments, counts, or text."
    )


def score_item(payload: dict[str, Any]) -> dict[str, Any]:
    """Score one UGC item and return entropy, routing, and diagnostic signals."""

    if not isinstance(payload, dict):
        raise SentimentEntropyError("Payload must be a JSON object.")

    aspects, extraction_mode, warnings = _aspect_sentiments_from_payload(payload)
    counts = counts_from_aspect_sentiments(aspects)
    entropy_result = compute_entropy(counts)
    routes = recommend_routes(
        aspect_sentiments=aspects,
        text=str(payload.get("text", "")),
        normalized_entropy=entropy_result.normalized_entropy,
    )

    result = {
        "id": payload.get("id"),
        "text": payload.get("text"),
        "extraction_mode": extraction_mode,
        "aspect_sentiments": aspects,
        "counts": entropy_result.counts,
        "proportions": {
            key: round(value, 6) for key, value in entropy_result.proportions.items()
        },
        "entropy": round(entropy_result.entropy, 6),
        "max_entropy": round(MAX_THREE_CATEGORY_ENTROPY, 6),
        "normalized_entropy": round(entropy_result.normalized_entropy, 6),
        "diagnosticity": diagnostic_label(entropy_result.normalized_entropy),
        "routes": routes,
        "warnings": warnings,
    }
    return result


def rank_items(payload: dict[str, Any]) -> dict[str, Any]:
    """Score and rank a batch of UGC items by normalized sentiment entropy."""

    items = payload.get("items")
    if not isinstance(items, list):
        raise SentimentEntropyError("'items' must be a list.")
    top_k = int(payload.get("top_k", len(items)))
    scored = [score_item(item) for item in items]
    scored.sort(
        key=lambda item: (
            item["normalized_entropy"],
            len(item.get("aspect_sentiments", [])),
        ),
        reverse=True,
    )
    return {
        "ranked_items": scored[:top_k],
        "total_items": len(scored),
        "top_k": top_k,
    }


def make_diagnostic_summary(scored_item: dict[str, Any]) -> str:
    """Create a compact summary that an agent can use as RAG context."""

    counts = scored_item["counts"]
    aspects = scored_item.get("aspect_sentiments", [])
    by_sentiment: dict[str, list[str]] = {sentiment: [] for sentiment in SENTIMENTS}
    for item in aspects:
        aspect = item.get("aspect", "unspecified")
        sentiment = item.get("sentiment", "neutral")
        if aspect not in by_sentiment[sentiment]:
            by_sentiment[sentiment].append(aspect)

    parts = [
        f"entropy={scored_item['entropy']:.3f}",
        f"diagnosticity={scored_item['diagnosticity']}",
        f"counts=positive:{counts['positive']}, neutral:{counts['neutral']}, negative:{counts['negative']}",
    ]
    for sentiment in SENTIMENTS:
        if by_sentiment[sentiment]:
            parts.append(f"{sentiment}_aspects={', '.join(by_sentiment[sentiment][:6])}")
    return "; ".join(parts)


def prepare_rag_context(payload: dict[str, Any]) -> dict[str, Any]:
    """Select high-diagnostic UGC items as RAG-ready context."""

    ranked = rank_items(payload)
    include_summaries = bool(payload.get("include_summaries", True))
    context_items = []
    for item in ranked["ranked_items"]:
        entry = {
            "id": item.get("id"),
            "text": item.get("text"),
            "entropy": item["entropy"],
            "normalized_entropy": item["normalized_entropy"],
            "diagnosticity": item["diagnosticity"],
            "primary_route": item["routes"]["primary_route"],
            "human_review": item["routes"]["human_review"],
        }
        if include_summaries:
            entry["diagnostic_summary"] = make_diagnostic_summary(item)
        context_items.append(entry)
    return {
        "context_items": context_items,
        "selection_policy": "rank_by_normalized_sentiment_entropy_desc",
        "total_items": ranked["total_items"],
        "top_k": ranked["top_k"],
    }
