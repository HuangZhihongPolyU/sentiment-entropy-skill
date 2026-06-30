"""Command-line interface for local demos and agent tool execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .core import prepare_rag_context, rank_items, score_item


def _json_arg(value: str) -> Any:
    return json.loads(value)


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sentiment entropy agent skill CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score", help="Score one item.")
    score.add_argument("--json", help="Full JSON payload.")
    score.add_argument("--file", help="Path to JSON payload.")
    score.add_argument("--text", help="Text for demo heuristic extraction.")
    score.add_argument("--counts", help="JSON counts object.")

    rank = sub.add_parser("rank", help="Rank a batch of items.")
    rank.add_argument("--file", required=True, help="Path to rank request JSON.")

    rag = sub.add_parser("rag-context", help="Prepare RAG context.")
    rag.add_argument("--file", required=True, help="Path to RAG request JSON.")

    args = parser.parse_args(argv)
    if args.command == "score":
        payload: dict[str, Any]
        if args.file:
            payload = _load_json(args.file)
        elif args.json:
            payload = _json_arg(args.json)
        elif args.counts:
            payload = {"counts": _json_arg(args.counts)}
        elif args.text:
            payload = {"text": args.text}
        else:
            parser.error("score requires --file, --json, --counts, or --text")
        result = score_item(payload)
    elif args.command == "rank":
        result = rank_items(_load_json(args.file))
    elif args.command == "rag-context":
        result = prepare_rag_context(_load_json(args.file))
    else:
        parser.error(f"Unknown command: {args.command}")
        return

    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    print()


if __name__ == "__main__":
    main()

