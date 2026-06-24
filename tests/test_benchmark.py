from multi_agent_research_lab.core.schemas import (
    AgentName,
    AgentResult,
    ResearchQuery,
    SourceDocument,
)
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark


def test_benchmark_scores_state() -> None:
    def runner(query: str) -> ResearchState:
        state = ResearchState(request=ResearchQuery(query=query))
        state.sources.append(
            SourceDocument(title="Source", url="https://example.com", snippet="Evidence")
        )
        state.research_notes = "Research notes"
        state.analysis_notes = "Analysis notes"
        state.final_answer = "Answer with citation [1]"
        state.agent_results.append(
            AgentResult(agent=AgentName.WRITER, content="Answer", metadata={"cost_usd": 0.01})
        )
        return state

    _, metrics = run_benchmark("multi-agent", "Explain multi-agent systems", runner)

    assert metrics.estimated_cost_usd == 0.01
    assert metrics.citation_coverage == 1.0
    assert metrics.quality_score == 10.0
