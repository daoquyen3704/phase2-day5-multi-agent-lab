"""Command-line entrypoint for the lab starter."""

import json
from pathlib import Path
from typing import Annotated

import typer
import yaml  # type: ignore[import-untyped]
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import (
    AgentName,
    AgentResult,
    BenchmarkMetrics,
    ResearchQuery,
)
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a minimal single-agent baseline."""

    _init()
    state = run_baseline(query)
    console.print(Panel.fit(state.final_answer or "", title="Single-Agent Baseline"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Optional JSON output path for trace artifact"),
    ] = None,
) -> None:
    """Run the multi-agent workflow skeleton."""

    _init()
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    try:
        result = workflow.run(state)
    except AgentExecutionError as exc:
        console.print(Panel.fit(str(exc), title="Workflow Error", style="red"))
        raise typer.Exit(code=2) from exc
    output_json = json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(output_json + "\n", encoding="utf-8")
    console.print(output_json)


@app.command()
def benchmark(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Benchmark config file"),
    ] = Path("configs/lab_default.yaml"),
) -> None:
    """Run baseline and multi-agent benchmarks and write a markdown report."""

    _init()
    queries = _load_benchmark_queries(config)
    metrics: list[BenchmarkMetrics] = []
    store = LocalArtifactStore()

    for query in queries:
        _, baseline_metrics = run_benchmark(
            run_name=f"baseline: {query[:36]}",
            query=query,
            runner=run_baseline,
        )
        _, multi_metrics = run_benchmark(
            run_name=f"multi-agent: {query[:32]}",
            query=query,
            runner=lambda item: MultiAgentWorkflow().run(
                ResearchState(request=ResearchQuery(query=item))
            ),
        )
        metrics.extend([baseline_metrics, multi_metrics])

    report = render_markdown_report(metrics)
    path = store.write_text("benchmark_report.md", report)
    console.print(Panel.fit(f"Wrote {path}", title="Benchmark"))


def run_baseline(query: str) -> ResearchState:
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    response = LLMClient().complete(
        system_prompt=(
            "You are a single-agent research baseline. Answer directly, mention likely "
            "trade-offs, and be clear when sources were not independently searched."
        ),
        user_prompt=query,
    )
    state.final_answer = response.content
    state.agent_results.append(
        AgentResult(
            agent=AgentName.WRITER,
            content=response.content,
            metadata={
                "mode": "single-agent-baseline",
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            },
        )
    )
    state.add_trace_event("baseline.run", {"model": get_settings().openai_model})
    return state


def _load_benchmark_queries(config: Path) -> list[str]:
    default_queries = ["Research GraphRAG state-of-the-art and write a 500-word summary"]
    if not config.exists():
        return default_queries

    data = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return default_queries
    benchmark = data.get("benchmark", {})
    if not isinstance(benchmark, dict):
        return default_queries
    queries = benchmark.get("queries", [])
    if not isinstance(queries, list):
        return default_queries
    return [str(query) for query in queries if str(query).strip()]


if __name__ == "__main__":
    app()
