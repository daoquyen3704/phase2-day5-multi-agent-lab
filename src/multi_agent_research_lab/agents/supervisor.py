"""Supervisor / router skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route.

        Routes through the required research -> analysis -> writing pipeline and
        enforces a bounded iteration guard.
        """

        if state.final_answer:
            route = "done"
        elif state.iteration >= self.settings.max_iterations:
            route = "writer"
            state.errors.append("Max iterations reached; routing to writer fallback.")
        elif not state.research_notes:
            route = "researcher"
        elif not state.analysis_notes:
            route = "analyst"
        else:
            route = "writer"

        state.record_route(route)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.SUPERVISOR,
                content=f"Next route: {route}",
                metadata={"iteration": state.iteration},
            )
        )
        with trace_span("supervisor.route", {"route": route, "iteration": state.iteration}) as span:
            state.add_trace_event("supervisor.route", span)
        return state
