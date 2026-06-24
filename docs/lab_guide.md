# Lab Guide: Multi-Agent Research System

## Scenario

Build a research assistant that can receive a complex question, collect
supporting information, analyze trade-offs, and write a final answer. The lab
compares two approaches:

1. Single-agent baseline: one agent handles the whole task.
2. Multi-agent workflow: Supervisor coordinates Researcher, Analyst, and Writer.

## Important Rules

- Do not add agents without a clear reason.
- Each agent must have a distinct responsibility.
- Shared state must be explicit enough for debugging.
- Each workflow step must produce trace or log information.
- The submission must include benchmark evidence, not only demo output.

## Milestone 1: Baseline

Suggested files:

- `src/multi_agent_research_lab/cli.py`
- `src/multi_agent_research_lab/services/llm_client.py`

Implemented:

- The baseline calls the shared `LLMClient`.
- The client supports OpenAI when configured and deterministic local fallback
  when no valid provider key is available.

## Milestone 2: Supervisor

Suggested files:

- `src/multi_agent_research_lab/agents/supervisor.py`
- `src/multi_agent_research_lab/graph/workflow.py`

Implemented:

- Supervisor inspects shared state and routes to `researcher`, `analyst`,
  `writer`, or `done`.
- `MAX_ITERATIONS` prevents unbounded routing.
- Workflow uses LangGraph when available and a sequential fallback otherwise.

Design questions answered:

- Call Researcher when `research_notes` is missing.
- Call Analyst when research exists but `analysis_notes` is missing.
- Call Writer when analysis exists but `final_answer` is missing.
- Stop when `final_answer` exists.
- Use fallback behavior for provider failures and bounded iteration for loops.

## Milestone 3: Worker Agents

Suggested files:

- `agents/researcher.py`
- `agents/analyst.py`
- `agents/writer.py`

Implemented:

- Researcher collects sources and writes `research_notes`.
- Analyst writes `analysis_notes`.
- Writer writes `final_answer` and appends source references.
- Each worker records `AgentResult` metadata and trace events.

## Milestone 4: Trace And Benchmark

Suggested files:

- `observability/tracing.py`
- `evaluation/benchmark.py`
- `evaluation/report.py`

Implemented:

- `trace_span` records per-agent timing and attributes.
- `multi-agent --output reports/trace_latest.json` writes a trace artifact.
- `benchmark` writes `reports/benchmark_report.md`.

Minimum benchmark metrics:

| Metric | Measurement |
|---|---|
| Latency | Wall-clock runtime |
| Cost | Estimated token/provider usage from agent metadata |
| Quality | Heuristic 0-10 score for lab comparison |
| Citation coverage | Cited source indexes divided by retrieved source count |
| Failure rate | Error count per run |

## Exit Ticket

When multi-agent is appropriate:

- Use it when the task benefits from separation of concerns, evidence
  collection, analysis, synthesis, and traceability.

When multi-agent is not appropriate:

- Avoid it for short, simple, low-risk tasks where latency and cost matter more
  than decomposition.
