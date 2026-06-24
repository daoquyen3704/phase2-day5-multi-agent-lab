from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def test_supervisor_routes_to_researcher_first() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    result = SupervisorAgent().run(state)
    assert result.route_history == ["researcher"]
    assert result.agent_results[-1].content == "Next route: researcher"


def test_supervisor_routes_to_done_after_final_answer() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.final_answer = "Done"
    result = SupervisorAgent().run(state)
    assert result.route_history == ["done"]
