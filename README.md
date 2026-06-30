# Sentiment Entropy Agent Skill

> **Paper:** "Sentiment Entropy as a Diagnostic Signal: Evidence from Multimodal User-Generated Content"  
> **Authors:** Zhihong Huang, Kwan Yu Chris Lo (corresponding author)  
> **Affiliation:** Department of Logistics and Maritime Studies, The Hong Kong Polytechnic University

A portable, zero-dependency skill that computes sentiment entropy as an
ex-ante input-quality layer for GenAI and agentic AI workflows.

---

## What It Solves

Modern AI systems ingest massive volumes of user-generated content (UGC) --
reviews, comments, transcripts -- but lack a principled way to filter *which*
content is worth the cost of processing. Two reviews that both read
"negative" can carry vastly different diagnostic value:

| Review | Internal structure | Diagnostic value |
|--------|-------------------|------------------|
| *"Food was terrible, service was slow, ambiance was bad"* | All negative | Low -- points to a broad failure, but offers no differentiated signals |
| *"The sushi was excellent, but the 40-minute wait was unacceptable"* | Positive + negative | High -- provides separate signals for kitchen quality vs. queue management |

**Sentiment entropy** quantifies this distinction. It measures how evenly
positive, neutral, and negative evaluations are distributed across aspects
*within* a single review. Higher entropy means more balanced, more diagnostic
content -- the kind agents should prioritize.

---

## Pipeline

```text
UGC item
-> aspect sentiment distribution
-> H = -sum(p_i log2(p_i))
-> diagnosticity ranking
-> RAG context selection
-> cross-functional routing
-> human-review flag
```

In practice, the skill turns one review into four decisions:

1. Is this item diagnostically rich enough to prioritize?
2. Should it be selected as RAG context?
3. Which function should receive it: service, operations, product, or marketing?
4. Does the agent need to keep a human in the loop?

The theoretical maximum entropy across three sentiment categories is
`log₂(3) ≈ 1.585`.

---

## Quick Start

```bash
# 1. Run unit tests
python tests/run_tests.py

# 2. Run the demo with real Yelp reviews
python demo.py

# 3. Start the HTTP service
python -m sentiment_entropy_agent.http_server --port 8765
```

### Demo output (excerpt)

```
======================================================================
  STEP 1 — Score Each Review Individually
======================================================================

[review_1]
  Entropy    : 1.5683  (normalized: 0.9895)     <— near-max entropy
  Diagnosticity: high
  Counts     : pos=5,  neu=7,  neg=7            <— balanced across all 3
  Route      : product_quality
  Human Review: REQUIRED — high_entropy_cross_functional_feedback

[review_5]
  Entropy    : 0.0000  (normalized: 0.0000)     <— zero entropy
  Diagnosticity: low
  Counts     : pos=11,  neu=0,  neg=0            <— all positive
  Human Review: not required

======================================================================
  STEP 2 — Rank All Reviews by Normalized Entropy (Descending)
======================================================================
  Rank  ID                   Entropy   Diagnosticity  Primary Route
  ----- ------------------ --------- --------------- ---------------
  1     review_1              1.5683  high            product_quality
  2     review_2              1.5395  high            service_recovery
  3     review_4              0.9911  moderate        service_recovery
  4     review_3              0.9819  moderate        service_recovery
  5     review_5              0.0000  low             product_quality
  6     review_6              0.0000  low             product_quality
```

---

## Agentic AI Use Cases

| Use case | Endpoint | Description |
|----------|----------|-------------|
| **Input filtering** | `POST /rank` | Rank UGC by diagnostic value before feeding to an LLM; deprioritize low-entropy content to save tokens |
| **RAG context selection** | `POST /rag_context` | Pick the most diagnostic reviews as retrieval context for summarization or Q&A |
| **Cross-functional routing** | `POST /route` | Recommend which department (service, operations, product, marketing) should handle the feedback |
| **Human-in-the-loop guard** | (embedded in all outputs) | Flag items where `human_review=true` -- the agent should draft a recommendation but not act autonomously |

---

## HTTP API

Zero dependencies. Start with:

```bash
python -m sentiment_entropy_agent.http_server --port 8765
```

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Service health check |
| `GET` | `/schema` | Endpoint guide |
| `POST` | `/score` | Compute entropy for one item |
| `POST` | `/rank` | Rank a batch by entropy |
| `POST` | `/route` | Score + routing guidance |
| `POST` | `/rag_context` | Select top-K diagnostic items for RAG |

---

## Input Modes

The skill accepts three input formats:

**1. Labelled aspect sentiments (recommended)**
```json
{
  "id": "r1",
  "text": "Food was good but service was slow.",
  "aspect_sentiments": [
    {"aspect": "food", "sentiment": "positive"},
    {"aspect": "service", "sentiment": "negative"}
  ]
}
```

**2. Sentiment counts**
```json
{
  "id": "r2",
  "counts": {"positive": 1, "neutral": 1, "negative": 1}
}
```

**3. Text only (demo heuristic -- not ABSA)**
```json
{
  "id": "r3",
  "text": "The queue was long, but the food was excellent."
}
```

For research or production, provide labelled aspect sentiments from your own
ABSA or LLM-based extraction pipeline.

---

## MCP Adapter (optional)

If your agent platform supports MCP:

```bash
python -m sentiment_entropy_agent.mcp_server
```

Exposes four tools: `compute_sentiment_entropy`, `rank_ugc_by_diagnosticity`,
`route_customer_feedback`, `prepare_rag_context`.

Without the MCP SDK, use the HTTP service -- it has no extra dependencies.

---

## Example Data

The `examples/` directory contains real Yelp reviews with pre-computed
aspect-level sentiments:

| File | Description |
|------|-------------|
| `labelled_examples.csv` | 6 reviews spanning high, moderate, and low entropy |
| `rank_request.json` | Batch ranking input with full `aspect_sentiments` |
| `rag_context_request.json` | RAG context selection input |

---

## Repository Layout

```
sentiment_entropy_agent/
  core.py           # entropy formula, scoring, ranking, RAG context
  routing.py        # cross-functional routing + human-review policy
  text_heuristic.py # demo-only text extraction (not ABSA)
  http_server.py    # zero-dependency HTTP service
  mcp_server.py     # optional MCP adapter
  cli.py            # command-line interface
demo.py             # end-to-end pipeline demo
examples/           # real Yelp review examples
tests/              # unit tests
```

---

## Citation

```bibtex
@article{huang2026sentiment,
  title   = {Sentiment Entropy as a Diagnostic Signal: Evidence from
             Multimodal User-Generated Content},
  author  = {Huang, Zhihong and Lo, Kwan Yu Chris},
  journal = {Working Paper},
  year    = {2026},
  affiliation = {Department of Logistics and Maritime Studies,
                 The Hong Kong Polytechnic University}
}
```
