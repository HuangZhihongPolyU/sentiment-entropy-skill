"""Dependency-free JSON HTTP service for the sentiment entropy skill."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .core import SentimentEntropyError, prepare_rag_context, rank_items, score_item


def schema() -> dict[str, Any]:
    return {
        "service": "sentiment-entropy-agent-skill",
        "version": "0.1.0",
        "endpoints": {
            "GET /health": "service health",
            "GET /schema": "endpoint and payload guide",
            "POST /score": "score one UGC item",
            "POST /rank": "rank UGC items by normalized entropy",
            "POST /route": "score one item and return routing guidance",
            "POST /rag_context": "select high-diagnostic RAG context items",
        },
        "input_modes": [
            {"counts": {"positive": 1, "neutral": 1, "negative": 1}},
            {
                "aspect_sentiments": [
                    {"aspect": "food", "sentiment": "positive"},
                    {"aspect": "service", "sentiment": "negative"},
                ]
            },
            {"text": "Demo-only heuristic extraction from free text."},
        ],
    }


class SentimentEntropyHandler(BaseHTTPRequestHandler):
    server_version = "SentimentEntropyHTTP/0.1"

    def do_OPTIONS(self) -> None:
        self._send_json({"ok": True})

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json({"status": "ok", "service": "sentiment-entropy-agent-skill"})
            return
        if self.path == "/schema":
            self._send_json(schema())
            return
        self._send_json({"error": f"Unknown endpoint: {self.path}"}, status=404)

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            if self.path == "/score":
                self._send_json(score_item(payload))
            elif self.path == "/route":
                scored = score_item(payload)
                self._send_json(
                    {
                        "id": scored.get("id"),
                        "entropy": scored["entropy"],
                        "normalized_entropy": scored["normalized_entropy"],
                        "diagnosticity": scored["diagnosticity"],
                        "routes": scored["routes"],
                        "warnings": scored["warnings"],
                    }
                )
            elif self.path == "/rank":
                self._send_json(rank_items(payload))
            elif self.path == "/rag_context":
                self._send_json(prepare_rag_context(payload))
            else:
                self._send_json({"error": f"Unknown endpoint: {self.path}"}, status=404)
        except SentimentEntropyError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except json.JSONDecodeError as exc:
            self._send_json({"error": f"Invalid JSON: {exc}"}, status=400)
        except Exception as exc:
            self._send_json({"error": f"Server error: {exc}"}, status=500)

    def log_message(self, fmt: str, *args: Any) -> None:
        if getattr(self.server, "quiet", False):
            return
        super().log_message(fmt, *args)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        data = json.loads(body or "{}")
        if not isinstance(data, dict):
            raise SentimentEntropyError("Request body must be a JSON object.")
        return data

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the sentiment entropy HTTP service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    httpd = ThreadingHTTPServer((args.host, args.port), SentimentEntropyHandler)
    httpd.quiet = args.quiet  # type: ignore[attr-defined]
    print(f"Sentiment entropy HTTP service listening on http://{args.host}:{args.port}")
    print("Endpoints: GET /health, GET /schema, POST /score, POST /rank, POST /route, POST /rag_context")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()

