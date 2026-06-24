"""Researcher agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(
        self,
        search_client: SearchClient | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.search_client = search_client or SearchClient()
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`.

        Searches for relevant documents and asks the LLM layer to turn them into
        compact handoff notes for the analyst.
        """

        with trace_span("researcher.run", {"query": state.request.query}) as span:
            try:
                sources = self.search_client.search(state.request.query, state.request.max_sources)
            except AgentExecutionError as exc:
                state.errors.append(str(exc))
                sources = []

            state.sources = self._dedupe_sources([*state.sources, *sources])
            source_block = self._format_sources(state.sources)
            response = self.llm_client.complete(
                system_prompt=(
                    "You are the Researcher agent. Produce concise research notes, "
                    "preserve source names, and avoid unsupported claims."
                ),
                user_prompt=(
                    f"Query: {state.request.query}\n"
                    f"Audience: {state.request.audience}\n"
                    f"Sources:\n{source_block}"
                ),
            )
            state.research_notes = response.content
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.RESEARCHER,
                    content=response.content,
                    metadata={
                        "source_count": len(state.sources),
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )
            span["source_count"] = len(state.sources)
            state.add_trace_event("researcher.run", span)
        return state

    @staticmethod
    def _dedupe_sources(sources: list[SourceDocument]) -> list[SourceDocument]:
        seen: set[tuple[str, str | None]] = set()
        deduped: list[SourceDocument] = []
        for source in sources:
            key = (source.title, source.url)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(source)
        return deduped

    @staticmethod
    def _format_sources(sources: list[SourceDocument]) -> str:
        if not sources:
            return "No external sources available; rely on clearly marked general knowledge."
        lines = []
        for index, source in enumerate(sources, start=1):
            url = f" ({source.url})" if source.url else ""
            lines.append(f"[{index}] {source.title}{url}: {source.snippet}")
        return "\n".join(lines)
