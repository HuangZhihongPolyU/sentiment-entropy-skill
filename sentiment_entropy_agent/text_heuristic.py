"""Demo-only aspect sentiment extraction.

This is not ABSA. It exists so the HTTP service can run without external
models. Replace it with PyABSA, LLM extraction, or labelled data for research
and production use.
"""

from __future__ import annotations

import re

ASPECT_TERMS = {
    "food",
    "taste",
    "service",
    "staff",
    "support",
    "price",
    "value",
    "queue",
    "wait",
    "delivery",
    "shipping",
    "product",
    "quality",
    "room",
    "hotel",
    "flight",
    "train",
    "cruise",
    "seat",
    "ambience",
    "cleanliness",
    "app",
    "website",
}

POSITIVE_WORDS = {
    "good",
    "great",
    "excellent",
    "amazing",
    "fast",
    "friendly",
    "clean",
    "helpful",
    "reliable",
    "smooth",
    "love",
    "liked",
    "recommend",
    "best",
    "strong",
    "praised",
}

NEGATIVE_WORDS = {
    "bad",
    "terrible",
    "awful",
    "slow",
    "rude",
    "poor",
    "dirty",
    "late",
    "delay",
    "delayed",
    "broken",
    "unacceptable",
    "worst",
    "hate",
    "expensive",
    "problem",
    "complaint",
    "criticized",
}

NEUTRAL_WORDS = {
    "okay",
    "average",
    "fine",
    "normal",
    "standard",
    "acceptable",
    "mixed",
}


def extract_demo_aspect_sentiments(text: str) -> list[dict[str, str]]:
    """Extract coarse aspect-sentiment pairs from text using transparent rules."""

    clauses = [
        clause.strip()
        for clause in re.split(
            r"[,.;!?]|\bbut\b|\bhowever\b|\balthough\b|\bwhile\b|\band\b",
            text,
            flags=re.I,
        )
        if clause.strip()
    ]
    extracted: list[dict[str, str]] = []
    for clause in clauses:
        tokens = _tokens(clause)
        aspects = sorted(tokens & ASPECT_TERMS)
        sentiment = _clause_sentiment(tokens)

        if aspects:
            for aspect in aspects:
                extracted.append({"aspect": aspect, "sentiment": sentiment})
        elif sentiment != "neutral":
            extracted.append({"aspect": "general", "sentiment": sentiment})

    if not extracted:
        extracted.append({"aspect": "general", "sentiment": "neutral"})
    return extracted


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z]+", text.lower()))


def _clause_sentiment(tokens: set[str]) -> str:
    pos = len(tokens & POSITIVE_WORDS)
    neg = len(tokens & NEGATIVE_WORDS)
    neu = len(tokens & NEUTRAL_WORDS)
    if pos > neg and pos >= neu:
        return "positive"
    if neg > pos and neg >= neu:
        return "negative"
    return "neutral"
