"""Supervisor / router agent.

Routing policy:
  1. No research_notes  -> researcher
  2. No analysis_notes  -> analyst
  3. No final_answer    -> writer
  4. Has final_answer   -> done
  5. Max iterations hit -> force writer or done
"""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)

ROUTE_RESEARCHER = "researcher"
ROUTE_ANALYST = "analyst"
ROUTE_WRITER = "writer"
ROUTE_DONE = "done"


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        """Update state.route_history with the next route."""
        settings = get_settings()

        with trace_span("supervisor.route", {"iteration": state.iteration}) as span:
            next_route = self._decide(state, settings.max_iterations)
            span["attributes"]["next_route"] = next_route

        state.record_route(next_route)
        state.add_trace_event("supervisor.route", {
            "iteration": state.iteration,
            "next": next_route,
            "has_research": state.research_notes is not None,
            "has_analysis": state.analysis_notes is not None,
            "has_answer": state.final_answer is not None,
        })
        logger.info("Supervisor → %s (iter %d)", next_route, state.iteration)
        return state

    def _decide(self, state: ResearchState, max_iterations: int) -> str:
        """Core routing logic."""
        # Hard stop if too many iterations
        if state.iteration >= max_iterations:
            logger.warning("Max iterations (%d) reached, forcing done", max_iterations)
            if state.final_answer is None and state.research_notes:
                return ROUTE_WRITER  # at least produce something
            return ROUTE_DONE

        # Error accumulation guard
        if len(state.errors) >= 3:
            logger.error("Too many errors (%d), stopping", len(state.errors))
            return ROUTE_DONE

        # Normal sequential routing
        if state.research_notes is None:
            return ROUTE_RESEARCHER
        if state.analysis_notes is None:
            return ROUTE_ANALYST
        if state.final_answer is None:
            return ROUTE_WRITER
        return ROUTE_DONE