"""Multi-agent workflow – Supervisor-driven loop.

Architecture:
    User Query → Supervisor → [Researcher | Analyst | Writer] → Supervisor → ... → done

This implementation uses a pure-Python loop rather than LangGraph to avoid
heavyweight setup dependencies while preserving the same logical graph structure.
A LangGraph migration path is documented below for students who want to extend it.
"""

import logging
import time

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import (
    ROUTE_ANALYST,
    ROUTE_DONE,
    ROUTE_RESEARCHER,
    ROUTE_WRITER,
    SupervisorAgent,
)
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)


class MultiAgentWorkflow:
    """Builds and runs the multi-agent research graph.

    The loop topology is:
        supervisor → researcher → supervisor
                   → analyst   → supervisor
                   → writer    → supervisor
                   → done      (exit)

    An optional CriticAgent runs after the writer before returning.

    LangGraph Migration Note:
        To convert to LangGraph, replace the while-loop below with:
            graph = StateGraph(ResearchState)
            graph.add_node("supervisor", supervisor.run)
            graph.add_node("researcher", researcher.run)
            ... etc.
            graph.add_conditional_edges("supervisor", route_fn, {...})
            graph.set_entry_point("supervisor")
            compiled = graph.compile()
            result = compiled.invoke(state)
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        search: SearchClient | None = None,
        enable_critic: bool = False,
    ) -> None:
        _llm = llm or LLMClient()
        _search = search or SearchClient()

        self._supervisor = SupervisorAgent()
        self._researcher = ResearcherAgent(llm=_llm, search=_search)
        self._analyst = AnalystAgent(llm=_llm)
        self._writer = WriterAgent(llm=_llm)
        self._critic = CriticAgent(llm=_llm) if enable_critic else None
        self._settings = get_settings()

    def build(self) -> "MultiAgentWorkflow":
        """Return self (compatibility shim for LangGraph-style API)."""
        return self

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the supervisor→worker loop and return final state."""
        start = time.perf_counter()

        with trace_span("workflow.run", {"query": state.request.query}) as workflow_span:
            agent_map = {
                ROUTE_RESEARCHER: self._researcher,
                ROUTE_ANALYST: self._analyst,
                ROUTE_WRITER: self._writer,
            }

            while True:
                # Check wall-clock timeout
                elapsed = time.perf_counter() - start
                if elapsed > self._settings.timeout_seconds:
                    logger.warning("Workflow timeout after %.1fs", elapsed)
                    state.errors.append(f"Timeout after {elapsed:.1f}s")
                    break

                # Supervisor decides next route
                try:
                    state = self._supervisor.run(state)
                except Exception as exc:
                    logger.error("Supervisor error: %s", exc)
                    state.errors.append(f"Supervisor: {exc}")
                    break

                next_route = state.route_history[-1] if state.route_history else ROUTE_DONE

                if next_route == ROUTE_DONE:
                    logger.info("Workflow complete after %d iterations", state.iteration)
                    break

                worker = agent_map.get(next_route)
                if worker is None:
                    logger.error("Unknown route: %s", next_route)
                    state.errors.append(f"Unknown route: {next_route}")
                    break

                # Execute the worker
                try:
                    with trace_span(f"{next_route}.execute"):
                        state = worker.run(state)
                except AgentExecutionError as exc:
                    err_msg = f"{next_route} failed: {exc}"
                    logger.error(err_msg)
                    state.errors.append(err_msg)
                    # Don't break – let supervisor decide next action (may fallback to done)
                except Exception as exc:
                    err_msg = f"{next_route} unexpected error: {exc}"
                    logger.exception(err_msg)
                    state.errors.append(err_msg)
                    break

            # Optional critic pass
            if self._critic and state.final_answer:
                try:
                    state = self._critic.run(state)
                except Exception as exc:
                    logger.warning("Critic skipped: %s", exc)

            total_time = time.perf_counter() - start
            workflow_span["attributes"]["total_seconds"] = total_time
            workflow_span["attributes"]["iterations"] = state.iteration
            workflow_span["attributes"]["errors"] = len(state.errors)

        state.add_trace_event("workflow.complete", {
            "total_seconds": total_time,
            "route_history": state.route_history,
        })
        return state