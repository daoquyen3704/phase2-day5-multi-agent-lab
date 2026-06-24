# Multi-Agent Research Lab Design

## Problem

The system is a research assistant for complex technical questions. Given a user
query, it should collect relevant sources, extract useful evidence, analyze
trade-offs, and produce a final answer with references.

The lab compares two approaches:

1. A single-agent baseline that answers the query directly in one LLM call.
2. A multi-agent workflow where a Supervisor coordinates a Researcher, Analyst,
   and Writer.

The main experiment asks whether role decomposition improves output quality,
traceability, and citation coverage enough to justify extra latency and cost.

## Why Multi-Agent?

A single-agent answer is simple and cheap, but it tends to mix source collection,
analysis, and writing into one step. That makes it harder to debug which part
failed: missing sources, weak analysis, or poor final synthesis.

The multi-agent design separates responsibilities:

- Researcher focuses on evidence collection and source notes.
- Analyst turns evidence into claims, trade-offs, and risks.
- Writer produces a readable final answer with citations.
- Supervisor enforces routing, stop conditions, and iteration limits.

This separation makes the workflow easier to inspect and benchmark. The expected
trade-off is higher latency and more moving parts.

## Agent Roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Decide next route and stop when final answer exists | Shared state: query, notes, final answer, iteration | Route: researcher, analyst, writer, or done | Wrong route, repeated loop, early stop |
| Researcher | Search or retrieve sources and produce concise research notes | Query, max sources, audience | `sources`, `research_notes`, researcher result | Search provider fails, irrelevant sources, no citations |
| Analyst | Extract claims, compare viewpoints, and identify weak evidence | Query, research notes, sources | `analysis_notes`, analyst result | Overconfident conclusions, missed trade-offs |
| Writer | Synthesize final answer with references | Query, research notes, analysis notes, sources | `final_answer`, writer result | Unsupported claims, missing references, vague answer |

The optional Critic agent performs a lightweight citation and completeness check.
It is not part of the core routing path.

## Shared State

The workflow uses `ResearchState` as the single source of truth:

| Field | Purpose |
|---|---|
| `request` | Stores query, max source count, and target audience |
| `iteration` | Counts supervisor routing steps |
| `route_history` | Explains the path taken through the workflow |
| `sources` | Stores retrieved or fallback source documents |
| `research_notes` | Handoff from Researcher to Analyst |
| `analysis_notes` | Handoff from Analyst to Writer |
| `final_answer` | Final response returned to the user |
| `agent_results` | Per-agent output and token/cost metadata |
| `trace` | Step-level events with timing and attributes |
| `errors` | Recoverable failures and fallback notes |

This state is intentionally explicit so reviewers can inspect handoffs and trace
which agent produced each part of the final answer.

## Routing Policy

The current routing policy is deterministic:

```text
start
  |
  v
Supervisor
  |-- no research_notes --> Researcher --> Supervisor
  |-- no analysis_notes --> Analyst ----> Supervisor
  |-- no final_answer ----> Writer -----> Supervisor
  |-- final_answer exists -> done
```

If `MAX_ITERATIONS` is reached before completion, the Supervisor routes to the
Writer fallback and records an error. This prevents infinite loops.

## Guardrails

- Max iterations: `MAX_ITERATIONS`, default `6`, configured in `.env`.
- Timeout: `TIMEOUT_SECONDS`, default `60`, configured in `.env`.
- Retry: LLM provider calls retry up to three times before falling back.
- Fallback:
  - Missing or unsupported OpenAI key uses deterministic local LLM fallback.
  - Missing Tavily key uses deterministic local search fallback.
  - Search provider failure records an error and continues with available notes.
- Validation:
  - `ResearchQuery` enforces minimum query length and source limits.
  - `BenchmarkMetrics` constrains quality score and citation coverage ranges.
  - Tests cover routing, fallback services, workflow execution, and benchmark
    scoring.

## Benchmark Plan

Benchmark queries are defined in `configs/lab_default.yaml`:

1. Research GraphRAG state-of-the-art and write a 500-word summary.
2. Compare single-agent and multi-agent workflows for customer support.
3. Summarize production guardrails for LLM agents.

Measured metrics:

| Metric | How it is measured |
|---|---|
| Latency | Wall-clock runtime for each runner |
| Cost | Sum of per-agent estimated LLM cost metadata |
| Quality | Heuristic score from answer presence, notes, sources, citations, and errors |
| Citation coverage | Number of cited source indexes divided by source count |
| Error count | Number of state errors recorded during the run |

The report is generated at `reports/benchmark_report.md`.

## Methodological Limitations

The current benchmark is useful for a lab demo, but it is not a rigorous final
research claim. The multi-agent workflow can retrieve sources while the baseline
answers directly from the query. This gives the multi-agent condition an
advantage that may reflect source access rather than decomposition.

A stronger experiment should add a controlled baseline with the same source
pack, same model, same answer length, and comparable token budget. That would
help separate gains from decomposition from gains caused by more context or more
inference time.
