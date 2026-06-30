"""Optional MCP adapter for the sentiment entropy agent skill.

The HTTP service is dependency-free. This module requires the Python MCP SDK.
It intentionally wraps the same core functions so agent platforms can choose
either HTTP or MCP without changing business logic.
"""

from __future__ import annotations

from typing import Any

from .core import prepare_rag_context, rank_items, score_item


def build_mcp_server() -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError(
            "Python MCP SDK is not installed. Install it in your agent platform "
            "environment, or use the dependency-free HTTP service: "
            "python -m sentiment_entropy_agent.http_server --port 8765"
        ) from exc

    mcp = FastMCP("sentiment-entropy-agent-skill")

    @mcp.tool()
    def compute_sentiment_entropy(payload: dict[str, Any]) -> dict[str, Any]:
        """Compute sentiment entropy and diagnostic routing for one UGC item."""

        return score_item(payload)

    @mcp.tool()
    def rank_ugc_by_diagnosticity(payload: dict[str, Any]) -> dict[str, Any]:
        """Rank UGC items by normalized sentiment entropy."""

        return rank_items(payload)

    @mcp.tool()
    def route_customer_feedback(payload: dict[str, Any]) -> dict[str, Any]:
        """Return routing guidance and human-review flags for one UGC item."""

        scored = score_item(payload)
        return {
            "id": scored.get("id"),
            "entropy": scored["entropy"],
            "normalized_entropy": scored["normalized_entropy"],
            "diagnosticity": scored["diagnosticity"],
            "routes": scored["routes"],
            "warnings": scored["warnings"],
        }

    @mcp.tool()
    def prepare_rag_context_tool(payload: dict[str, Any]) -> dict[str, Any]:
        """Select high-diagnostic UGC items as RAG-ready context."""

        return prepare_rag_context(payload)

    return mcp


def main() -> None:
    try:
        mcp = build_mcp_server()
    except RuntimeError as exc:
        raise SystemExit(str(exc))
    mcp.run()


if __name__ == "__main__":
    main()

