"""Benchmark: single-agent vs multi-agent comparison."""

import logging
import re
from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]


def _estimate_cost(state: ResearchState) -> float | None:
    """Sum cost_usd from all agent results that track it."""
    costs = [
        r.metadata.get("cost_usd")
        for r in state.agent_results
        if r.metadata.get("cost_usd") is not None
    ]
    return sum(costs) if costs else None


def _quality_score(state: ResearchState) -> float | None:
    """Heuristic quality score 0-10 based on answer characteristics.

    Rubric:
      +2  Has a final answer
      +2  Answer is 200+ words
      +2  Has section headers (##)
      +1  Has key takeaways / bullet points
      +1  Cites at least one source
      +1  Has research notes (shows multi-step process)
      +1  Has analysis notes
      -2  Has errors in state
    """
    if not state.final_answer:
        return 0.0

    score = 0.0
    answer = state.final_answer

    score += 2.0  # has final answer
    word_count = len(answer.split())
    if word_count >= 200:
        score += 2.0
    elif word_count >= 100:
        score += 1.0

    if re.search(r"^##", answer, re.MULTILINE):
        score += 2.0
    if re.search(r"^[-*]", answer, re.MULTILINE) or "takeaway" in answer.lower():
        score += 1.0
    if state.sources and any(s.title.lower() in answer.lower() for s in state.sources):
        score += 1.0
    if state.research_notes:
        score += 1.0
    if state.analysis_notes:
        score += 1.0

    # Penalise errors
    score -= min(len(state.errors) * 0.5, 2.0)

    return round(min(max(score, 0.0), 10.0), 2)


def _citation_coverage(state: ResearchState) -> float:
    """Fraction of sources whose title appears in the final answer."""
    if not state.sources or not state.final_answer:
        return 0.0
    cited = sum(
        1 for s in state.sources
        if s.title.lower() in state.final_answer.lower()
    )
    return round(cited / len(state.sources), 2)


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, cost, quality, and citation coverage."""
    logger.info("Benchmark start: %s | query=%s", run_name, query[:60])
    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    cost = _estimate_cost(state)
    quality = _quality_score(state)
    citation_cov = _citation_coverage(state)
    error_count = len(state.errors)

    notes = (
        f"words={len((state.final_answer or '').split())} "
        f"citation_coverage={citation_cov:.0%} "
        f"errors={error_count} "
        f"iterations={state.iteration}"
    )

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=cost,
        quality_score=quality,
        notes=notes,
    )
    logger.info(
        "Benchmark done: %s | latency=%.2fs quality=%.1f cost=%s",
        run_name, latency, quality or 0, f"${cost:.5f}" if cost else "N/A",
    )
    return state, metrics