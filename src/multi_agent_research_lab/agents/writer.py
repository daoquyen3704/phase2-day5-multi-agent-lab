"""Writer agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`.

        Synthesizes final answer with source references where available.
        """

        with trace_span("writer.run", {"source_count": len(state.sources)}) as span:
            response = self.llm_client.complete(
                system_prompt=(
                    "You are the Writer agent. Produce a clear final answer for the "
                    "audience. Cite sources using [1], [2] style references when possible."
                ),
                user_prompt=(
                    f"Query: {state.request.query}\n"
                    f"Audience: {state.request.audience}\n"
                    f"Research notes:\n{state.research_notes or 'No research notes.'}\n"
                    f"Analysis notes:\n{state.analysis_notes or 'No analysis notes.'}\n"
                    f"Sources:\n{self._format_source_references(state.sources)}"
                ),
            )
            state.final_answer = self._append_references(response.content, state.sources)
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.WRITER,
                    content=state.final_answer,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )
            state.add_trace_event("writer.run", span)
        return state

    @staticmethod
    def _format_source_references(sources: list[SourceDocument]) -> str:
        if not sources:
            return "No source references available."
        return "\n".join(
            f"[{index}] {source.title}: {source.url or source.snippet}"
            for index, source in enumerate(sources, start=1)
        )

    @staticmethod
    def _append_references(content: str, sources: list[SourceDocument]) -> str:
        if not sources or "References:" in content:
            return content
        references = "\n".join(
            f"[{index}] {source.title} - {source.url or 'local note'}"
            for index, source in enumerate(sources, start=1)
        )
        return f"{content.strip()}\n\nReferences:\n{references}"
