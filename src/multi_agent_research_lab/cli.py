"""Command-line entrypoint for the lab starter."""

import json
import os
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _single_agent_runner(query: str) -> ResearchState:
    """Single-agent baseline: one LLM call, no specialisation."""
    from multi_agent_research_lab.services.llm_client import LLMClient
    from multi_agent_research_lab.services.search_client import SearchClient

    llm = LLMClient()
    search = SearchClient()
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)

    # Search for sources
    sources = search.search(query, max_results=request.max_sources)
    state.sources = sources

    sources_text = "\n\n".join(
        f"[{i+1}] {s.title}: {s.snippet}" for i, s in enumerate(sources)
    )

    system = (
        "You are a research assistant. Given a query and sources, write a comprehensive "
        "500-word summary with key findings and takeaways. Cite sources by title."
    )
    user = f"Query: {query}\n\nSources:\n{sources_text}\n\nWrite the research summary."

    response = llm.complete(system_prompt=system, user_prompt=user)
    state.final_answer = response.content
    state.iteration = 1
    return state


def _multi_agent_runner(query: str) -> ResearchState:
    """Multi-agent runner for benchmark."""
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    return workflow.run(state)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the single-agent baseline with real LLM call."""
    _init()
    console.print("[bold blue]Running single-agent baseline...[/bold blue]")
    try:
        state = _single_agent_runner(query)
        console.print(Panel.fit(state.final_answer or "(no answer)", title="Single-Agent Baseline"))
    except Exception as exc:
        console.print(Panel.fit(str(exc), title="Error", style="red"))
        raise typer.Exit(code=1) from exc


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    critic: Annotated[bool, typer.Option("--critic/--no-critic", help="Enable critic agent")] = False,
) -> None:
    """Run the full multi-agent workflow (Supervisor → Researcher → Analyst → Writer)."""
    _init()
    console.print("[bold green]Running multi-agent workflow...[/bold green]")

    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow(enable_critic=critic)
    try:
        result = workflow.run(state)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
    except Exception as exc:
        console.print(Panel.fit(str(exc), title="Error", style="red"))
        raise typer.Exit(code=1) from exc

    console.print(Panel.fit(result.final_answer or "(no answer)", title="Multi-Agent Answer"))
    console.print(f"\n[dim]Route history: {' → '.join(result.route_history)}[/dim]")
    console.print(f"[dim]Iterations: {result.iteration} | Errors: {len(result.errors)}[/dim]")


@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")] = (
        "Research GraphRAG state-of-the-art and write a 500-word summary"
    ),
    output: Annotated[str, typer.Option("--output", "-o", help="Output report path")] = (
        "reports/benchmark_report.md"
    ),
) -> None:
    """Benchmark single-agent vs multi-agent and write a report."""
    _init()
    console.print("[bold]Running benchmark: single-agent vs multi-agent[/bold]\n")

    console.print("  [1/2] Running single-agent baseline...")
    baseline_state, baseline_metrics = run_benchmark("single-agent", query, _single_agent_runner)

    console.print("  [2/2] Running multi-agent workflow...")
    multi_state, multi_metrics = run_benchmark("multi-agent", query, _multi_agent_runner)

    metrics = [baseline_metrics, multi_metrics]
    states = [baseline_state, multi_state]

    # Rich table
    table = Table(title="Benchmark Results")
    table.add_column("Run", style="cyan")
    table.add_column("Latency (s)", justify="right")
    table.add_column("Cost (USD)", justify="right")
    table.add_column("Quality /10", justify="right")
    table.add_column("Notes")
    for m in metrics:
        table.add_row(
            m.run_name,
            f"{m.latency_seconds:.2f}",
            f"${m.estimated_cost_usd:.5f}" if m.estimated_cost_usd else "N/A",
            str(m.quality_score or "N/A"),
            m.notes,
        )
    console.print(table)

    # Write markdown report
    report_md = render_markdown_report(metrics, states)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write(report_md)
    console.print(f"\n[green]Report saved → {output}[/green]")


if __name__ == "__main__":
    app()