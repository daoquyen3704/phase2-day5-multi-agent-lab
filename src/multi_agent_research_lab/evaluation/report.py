"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to markdown."""

    lines = [
        "# Benchmark Report",
        "",
        "## Summary",
        "",
        _summary(metrics),
        "",
        "## Metrics",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Citation Coverage | Errors | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        coverage = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        lines.append(
            "| "
            f"{_cell(item.run_name)} | "
            f"{item.latency_seconds:.2f} | "
            f"{cost} | "
            f"{quality} | "
            f"{coverage} | "
            f"{item.error_count} | "
            f"{_cell(item.notes)} |"
        )
    lines.extend(
        [
            "",
            "## Methodological Notes",
            "",
            "- The current demo baseline answers directly from the query, while the "
            "multi-agent workflow may retrieve sources. Treat quality differences as "
            "a lab signal, not a definitive architecture claim.",
            "- A stricter experiment should give both systems the same cached source "
            "pack, model, answer length, and total token budget.",
            "- To separate decomposition gains from extra inference time, add a "
            "controlled multi-call baseline with the same number of calls and token "
            "budget as the multi-agent workflow.",
            "- Local fallback responses are useful for reproducible tests, but real "
            "provider runs should report provider mode and actual token/cost usage.",
        ]
    )
    return "\n".join(lines) + "\n"


def _summary(metrics: list[BenchmarkMetrics]) -> str:
    if not metrics:
        return "No benchmark runs were recorded."

    successful = [item for item in metrics if item.error_count == 0]
    best_quality = max(
        metrics,
        key=lambda item: item.quality_score if item.quality_score is not None else -1,
    )
    fastest = min(metrics, key=lambda item: item.latency_seconds)
    return (
        f"Recorded {len(metrics)} runs; {len(successful)} completed without errors. "
        f"Best quality: `{best_quality.run_name}`. Fastest: `{fastest.run_name}`."
    )


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")
