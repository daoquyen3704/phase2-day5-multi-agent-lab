"""LangGraph workflow orchestration."""

from importlib import import_module
from time import perf_counter
from typing import Any, Protocol, cast

from multi_agent_research_lab.agents import (
    AnalystAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.state import ResearchState


class InvokableGraph(Protocol):
    def invoke(self, state: ResearchState) -> object:
        """Invoke a compiled graph."""


class SequentialWorkflow:
    """Fallback workflow used when LangGraph is not installed."""

    def __init__(self, workflow: "MultiAgentWorkflow") -> None:
        self.workflow = workflow

    def invoke(self, state: ResearchState) -> ResearchState:
        return self.workflow._run_sequential(state)


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(
        self,
        supervisor: SupervisorAgent | None = None,
        researcher: ResearcherAgent | None = None,
        analyst: AnalystAgent | None = None,
        writer: WriterAgent | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.supervisor = supervisor or SupervisorAgent(self.settings)
        self.agents: dict[str, BaseAgent] = {
            "researcher": researcher or ResearcherAgent(),
            "analyst": analyst or AnalystAgent(),
            "writer": writer or WriterAgent(),
        }
        self._compiled_graph: object | None = None

    def build(self) -> object:
        """Create a LangGraph graph.

        Falls back to a sequential implementation if LangGraph is unavailable.
        """

        if self._compiled_graph is not None:
            return self._compiled_graph

        try:
            self._compiled_graph = self._build_langgraph()
        except (ImportError, ModuleNotFoundError):
            self._compiled_graph = SequentialWorkflow(self)
        return self._compiled_graph

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state.

        Compiles the graph on first use and converts the result back to
        `ResearchState` for callers.
        """

        graph = cast(InvokableGraph, self.build())
        result = graph.invoke(state)
        if isinstance(result, ResearchState):
            return result
        if isinstance(result, dict):
            return ResearchState.model_validate(result)
        raise AgentExecutionError(f"Workflow returned unsupported result: {type(result).__name__}")

    def _build_langgraph(self) -> object:
        langgraph = import_module("langgraph.graph")
        state_graph_class = langgraph.__dict__["StateGraph"]
        end = langgraph.__dict__["END"]

        graph: Any = state_graph_class(ResearchState)
        graph.add_node("supervisor", self.supervisor.run)
        for route, agent in self.agents.items():
            graph.add_node(route, agent.run)

        graph.set_entry_point("supervisor")
        graph.add_conditional_edges(
            "supervisor",
            self._next_route,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "done": end,
            },
        )
        for route in self.agents:
            graph.add_edge(route, "supervisor")
        return graph.compile()

    @staticmethod
    def _next_route(state: ResearchState) -> str:
        if not state.route_history:
            return "done"
        return state.route_history[-1]

    def _run_sequential(self, state: ResearchState) -> ResearchState:
        started = perf_counter()
        max_steps = self.settings.max_iterations + len(self.agents) + 1

        for _ in range(max_steps):
            if perf_counter() - started > self.settings.timeout_seconds:
                state.errors.append("Workflow timeout reached.")
                break

            state = self.supervisor.run(state)
            route = self._next_route(state)
            if route == "done":
                return state

            agent = self.agents.get(route)
            if agent is None:
                state.errors.append(f"Unknown route: {route}")
                break

            try:
                state = agent.run(state)
            except AgentExecutionError as exc:
                state.errors.append(str(exc))
                if route != "writer":
                    state = self.agents["writer"].run(state)
                break

        if not state.final_answer:
            state = self.agents["writer"].run(state)
        return state
