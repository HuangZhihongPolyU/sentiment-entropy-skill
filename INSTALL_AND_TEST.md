# Installation and Testing Procedure

This guide is for testing the sentiment entropy agent skill in a local agentic
AI platform.

## 1. Unzip

```bash
unzip sentiment-entropy-agent-skill.zip
cd sentiment-entropy-agent-skill
```

## 2. Check Python

Use Python 3.10 or newer.

```bash
python --version
```

If `python` points to Python 2 or is unavailable, try:

```bash
python3 --version
```

## 3. Optional Virtual Environment

The HTTP version has no required third-party dependencies, but a virtual
environment keeps the test clean.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

If using `python3`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```

## 4. Run Unit Tests

From the `sentiment-entropy-agent-skill` folder:

```bash
python tests/run_tests.py
```

Expected output:

```text
All tests passed.
```

## 5. Test CLI Mode

```bash
python -m sentiment_entropy_agent.cli score \
  --counts '{"positive": 1, "neutral": 1, "negative": 1}'
```

Expected result:

- `entropy` should be about `1.584963`
- `normalized_entropy` should be `1.0`
- `diagnosticity` should be `high`

## 6. Start HTTP Service

```bash
python -m sentiment_entropy_agent.http_server --host 127.0.0.1 --port 8765
```

Keep this terminal open.

## 7. Test HTTP Service

Open a second terminal and run:

```bash
curl http://127.0.0.1:8765/health
```

Expected output:

```json
{
  "service": "sentiment-entropy-agent-skill",
  "status": "ok"
}
```

Test RAG context selection:

```bash
curl -s http://127.0.0.1:8765/rag_context \
  -H "Content-Type: application/json" \
  -d @examples/rag_context_request.json
```

Expected behavior:

- high-entropy items are selected first;
- each selected item includes `diagnostic_summary`;
- routing fields such as `primary_route` and `human_review` are returned.

## 8. Test Score Endpoint

```bash
curl -s http://127.0.0.1:8765/score \
  -H "Content-Type: application/json" \
  -d '{
    "id": "demo-1",
    "text": "Food was good but service was slow.",
    "aspect_sentiments": [
      {"aspect": "food", "sentiment": "positive"},
      {"aspect": "service", "sentiment": "negative"}
    ]
  }'
```

Expected result:

- `entropy` should be `1.0`;
- `counts` should show one positive and one negative aspect;
- routing should likely include service or product/service-related guidance.

## 9. Agent Platform Integration

For most agentic AI platforms, the simplest integration is HTTP:

```text
POST http://127.0.0.1:8765/score
POST http://127.0.0.1:8765/rank
POST http://127.0.0.1:8765/route
POST http://127.0.0.1:8765/rag_context
```

Recommended workflow:

```text
UGC input
-> /score or /rank
-> select high-entropy items
-> use /rag_context for RAG or summarization
-> use /route for service/product/operations/marketing routing
-> keep human review if human_review=true
```

## 10. Optional MCP Mode

If the agent platform supports MCP and has the Python MCP SDK installed:

```bash
python -m sentiment_entropy_agent.mcp_server
```

If MCP is not installed, use the HTTP service instead. HTTP is the most portable
option and has no extra dependency.

## 11. Important Research Note

Use `counts` or `aspect_sentiments` for tests that should match the paper's
formula. The text-only mode is a lightweight demo heuristic. For research or
production validation, replace it with ABSA, LLM-based extraction, or labelled
aspect-sentiment data.

