"""Routing policy for agentic workflows.

This is intentionally simple and transparent. It is meant to demonstrate how
sentiment entropy can feed cross-functional routing and human-review policies.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

ROUTE_KEYWORDS = {
    "service_recovery": {
        "service",
        "staff",
        "support",
        "agent",
        "host",
        "hostess",
        "waiter",
        "employee",
        "complaint",
        "rude",
        "refund",
    },
    "operations": {
        "wait",
        "queue",
        "delay",
        "delivery",
        "shipping",
        "logistics",
        "inventory",
        "cleanliness",
        "speed",
        "availability",
        "fulfillment",
    },
    "product_quality": {
        "food",
        "product",
        "quality",
        "defect",
        "feature",
        "durability",
        "taste",
        "room",
        "seat",
        "train",
        "flight",
        "hotel",
    },
    "marketing_insight": {
        "price",
        "value",
        "brand",
        "promotion",
        "recommend",
        "ambience",
        "experience",
        "design",
        "creator",
        "sponsorship",
    },
}

HIGH_RISK_TERMS = {
    "allergy",
    "allergic",
    "injury",
    "unsafe",
    "fraud",
    "privacy",
    "legal",
    "lawsuit",
    "harassment",
    "discrimination",
    "safety",
    "refund",
    "chargeback",
}


def _tokens(text: str) -> set[str]:
    cleaned = []
    for char in text.lower():
        cleaned.append(char if char.isalnum() else " ")
    return set("".join(cleaned).split())


def recommend_routes(
    aspect_sentiments: list[dict[str, Any]],
    text: str,
    normalized_entropy: float,
) -> dict[str, Any]:
    """Recommend business routes and governance flags."""

    route_scores: Counter[str] = Counter()
    all_tokens = _tokens(text)
    for aspect_item in aspect_sentiments:
        aspect_tokens = _tokens(str(aspect_item.get("aspect", "")))
        sentiment = str(aspect_item.get("sentiment", "neutral")).lower()
        weight = 2 if sentiment == "negative" else 1
        for route, keywords in ROUTE_KEYWORDS.items():
            if aspect_tokens & keywords:
                route_scores[route] += weight

    for route, keywords in ROUTE_KEYWORDS.items():
        if all_tokens & keywords:
            route_scores[route] += 1

    if not route_scores:
        route_scores["general_customer_intelligence"] = 1

    sorted_routes = [
        {"route": route, "score": score}
        for route, score in route_scores.most_common()
    ]
    active_routes = [item["route"] for item in sorted_routes if item["score"] > 0]
    high_risk_terms = sorted(all_tokens & HIGH_RISK_TERMS)

    human_review_reasons = []
    if normalized_entropy >= 0.67 and len(active_routes) >= 2:
        human_review_reasons.append("high_entropy_cross_functional_feedback")
    if high_risk_terms:
        human_review_reasons.append("high_risk_terms_detected")
    return {
        "primary_route": sorted_routes[0]["route"],
        "candidate_routes": sorted_routes,
        "human_review": bool(human_review_reasons),
        "human_review_reasons": human_review_reasons,
        "high_risk_terms": high_risk_terms,
        "agent_action_guidance": _agent_action_guidance(
            normalized_entropy=normalized_entropy,
            human_review=bool(human_review_reasons),
            primary_route=sorted_routes[0]["route"],
        ),
    }


def _agent_action_guidance(
    normalized_entropy: float,
    human_review: bool,
    primary_route: str,
) -> str:
    if human_review:
        return (
            "Use as diagnostic context, draft a routing recommendation, and keep a human in the loop before action."
        )
    if normalized_entropy >= 0.67:
        return (
            f"Prioritize for agent summary and route to {primary_route}; content likely contains differentiated operational signals."
        )
    if normalized_entropy >= 0.34:
        return (
            f"Use as supporting context for {primary_route}; combine with additional evidence before autonomous action."
        )
    return (
        "Down-rank for RAG/context selection unless needed for coverage; content appears evaluatively concentrated."
    )
