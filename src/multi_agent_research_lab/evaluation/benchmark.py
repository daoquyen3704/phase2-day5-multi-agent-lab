"""Benchmark skeleton for single-agent vs multi-agent."""

from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, estimate cost, and score minimal output quality."""

    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started
    citation_coverage = _citation_coverage(state)
    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=_estimated_cost(state),
        quality_score=_quality_score(state, citation_coverage),
        citation_coverage=citation_coverage,
        error_count=len(state.errors),
        notes=_notes(state),
    )
    return state, metrics


def _estimated_cost(state: ResearchState) -> float | None:
    costs: list[float] = []
    for result in state.agent_results:
        value = result.metadata.get("cost_usd")
        if isinstance(value, int | float):
            costs.append(float(value))
    if not costs:
        return None
    return sum(costs)


def _citation_coverage(state: ResearchState) -> float | None:
    if not state.sources:
        return None
    answer = state.final_answer or ""
    cited = sum(1 for index in range(1, len(state.sources) + 1) if f"[{index}]" in answer)
    return cited / len(state.sources)


def _quality_score(state: ResearchState, citation_coverage: float | None) -> float:
    score = 0.0
    if state.final_answer:
        score += 2.0
    if state.research_notes:
        score += 2.0
    if state.analysis_notes:
        score += 2.0
    if state.sources:
        score += 1.5
    if citation_coverage is not None:
        score += citation_coverage * 1.5
    if not state.errors:
        score += 1.0
    return min(10.0, score)


def _notes(state: ResearchState) -> str:
    if state.errors:
        return "; ".join(state.errors)
    if state.sources:
        return f"{len(state.sources)} sources; routes: {', '.join(state.route_history)}"
    return f"routes: {', '.join(state.route_history) or 'none'}"
