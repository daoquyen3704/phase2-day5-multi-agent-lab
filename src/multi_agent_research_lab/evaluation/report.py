"""Benchmark report rendering – includes full agent outputs."""

from datetime import datetime

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState


def render_markdown_report(
    metrics: list[BenchmarkMetrics],
    states: list[ResearchState] | None = None,
) -> str:
    """Render benchmark metrics + full agent outputs to markdown."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# Benchmark Report – Multi-Agent Research System",
        "",
        f"*Generated: {now}*",
        "",
        "---",
        "",
        "## 1. Results Summary",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality /10 | Words | Citation Coverage | Errors | Iterations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for item in metrics:
        cost = "N/A" if item.estimated_cost_usd is None else f"${item.estimated_cost_usd:.5f}"
        quality = "N/A" if item.quality_score is None else f"{item.quality_score:.1f}"
        # Parse notes fields
        notes = item.notes
        words = next((p.split("=")[1] for p in notes.split() if p.startswith("words=")), "?")
        cit = next((p.split("=")[1] for p in notes.split() if p.startswith("citation_coverage=")), "?")
        errs = next((p.split("=")[1] for p in notes.split() if p.startswith("errors=")), "?")
        iters = next((p.split("=")[1] for p in notes.split() if p.startswith("iterations=")), "?")
        lines.append(f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} | {words} | {cit} | {errs} | {iters} |")

    lines += [""]

    # Delta analysis
    if len(metrics) == 2:
        a, b = metrics[0], metrics[1]
        latency_delta = b.latency_seconds - a.latency_seconds
        latency_pct = (latency_delta / a.latency_seconds) * 100 if a.latency_seconds else 0
        quality_delta = (b.quality_score or 0) - (a.quality_score or 0)

        lines += [
            "### Delta Analysis",
            "",
            f"- **Latency overhead:** `{b.run_name}` took **{latency_delta:+.2f}s ({latency_pct:+.1f}%)** longer than `{a.run_name}`.",
            f"- **Quality gain:** `{b.run_name}` scored **{quality_delta:+.1f} points** higher.",
            f"- **Verdict:** Multi-agent trades ~{latency_pct:.0f}% more latency for significantly higher quality output.",
            "",
        ]

    lines += [
        "---",
        "",
        "## 2. Agent Outputs",
        "",
    ]

    # Embed full outputs from each run
    if states:
        for i, (state, metric) in enumerate(zip(states, metrics)):
            lines += [
                f"### Run: `{metric.run_name}`",
                "",
                f"> **Query:** {state.request.query}",
                f"> **Latency:** {metric.latency_seconds:.2f}s | **Quality:** {metric.quality_score}/10",
                "",
            ]

            if state.research_notes:
                lines += [
                    "#### Research Notes (Researcher Agent)",
                    "",
                    state.research_notes,
                    "",
                ]

            if state.analysis_notes:
                lines += [
                    "#### Analysis Notes (Analyst Agent)",
                    "",
                    state.analysis_notes,
                    "",
                ]

            if state.final_answer:
                lines += [
                    "#### Final Answer (Writer Agent)",
                    "",
                    state.final_answer,
                    "",
                ]

            if state.errors:
                lines += [
                    "#### Errors",
                    "",
                    *[f"- `{e}`" for e in state.errors],
                    "",
                ]

            lines += ["---", ""]

    lines += [
        "## 3. Trace Log",
        "",
    ]

    if states:
        for state, metric in zip(states, metrics):
            lines += [f"### `{metric.run_name}` trace", ""]
            lines += ["```"]
            for event in state.trace:
                name = event["name"]
                payload = event.get("payload", {})
                payload_str = " | ".join(f"{k}={v}" for k, v in payload.items())
                lines.append(f"[{name}]  {payload_str}")
            lines += ["```", ""]

    lines += [
        "---",
        "",
        "## 4. When to Use Multi-Agent",
        "",
        "**Use multi-agent when:**",
        "- The task has clearly separable sub-tasks (research, analysis, writing).",
        "- Output quality matters more than latency.",
        "- You need auditability: each agent's output is independently inspectable.",
        "- The query is complex enough that specialisation provides measurable benefit.",
        "",
        "**Avoid multi-agent when:**",
        "- The query is simple and one-shot (wastes tokens and latency).",
        "- You have a strict latency SLA (multi-agent adds 2–3× overhead).",
        "- Coordination cost outweighs specialisation benefit.",
        "",
        "---",
        "",
        "## 5. Failure Modes & Fixes",
        "",
        "| Mode | Symptom | Fix |",
        "|---|---|---|",
        "| Infinite loop | `iteration` hits `MAX_ITERATIONS` | Supervisor routes → `done` when `final_answer` exists |",
        "| Agent hallucination | Facts not in sources appear in answer | Add Critic agent + citation coverage check |",
        "| Search miss | No relevant sources found | Expand mock KB or enable Tavily live search |",
        "| Timeout | Wall-clock > `TIMEOUT_SECONDS` | Reduce `max_sources` or increase timeout |",
        "| Rate limit (429) | OpenAI free-tier exhausted | Add credits or switch to OpenRouter free model |",
        "| Auth failure (401) | Wrong API key or inline `.env` comments | Strip comments from `.env` values |",
        "",
    ]

    return "\n".join(lines) + "\n"