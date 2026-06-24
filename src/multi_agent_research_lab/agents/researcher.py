"""Researcher agent – collects sources and writes research notes."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a meticulous Research Specialist. Your job is to synthesize
information from provided sources into clear, factual research notes.

Guidelines:
- Cite sources by their title when referencing facts.
- Flag any conflicting information between sources.
- Keep notes dense but readable (300-500 words).
- Use bullet points for key findings.
- Note any significant gaps in the available sources.

Output format:
## Research Notes: {topic}

**Key Findings:**
- ...

**Source Analysis:**
- ...

**Gaps / Limitations:**
- ...
"""


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(
        self,
        llm: LLMClient | None = None,
        search: SearchClient | None = None,
    ) -> None:
        self._llm = llm or LLMClient()
        self._search = search or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate state.sources and state.research_notes."""
        query = state.request.query

        with trace_span("researcher.search", {"query": query}) as span:
            try:
                sources = self._search.search(query, max_results=state.request.max_sources)
                state.sources = sources
                span["attributes"]["num_sources"] = len(sources)
                logger.info("Researcher found %d sources", len(sources))
            except Exception as exc:
                err_msg = f"Search failed: {exc}"
                logger.error(err_msg)
                state.errors.append(err_msg)
                sources = []

        if not sources:
            raise AgentExecutionError("Researcher: no sources found after search")

        # Build context for LLM synthesis
        sources_text = "\n\n".join(
            f"[{i+1}] **{s.title}**\nURL: {s.url or 'N/A'}\n{s.snippet}"
            for i, s in enumerate(sources)
        )

        user_prompt = (
            f"Research Query: {query}\n\n"
            f"Audience: {state.request.audience}\n\n"
            f"Available Sources:\n{sources_text}\n\n"
            "Please synthesize these sources into comprehensive research notes."
        )

        with trace_span("researcher.synthesize") as span:
            try:
                response = self._llm.complete(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                )
                state.research_notes = response.content
                span["attributes"]["output_tokens"] = response.output_tokens
            except Exception as exc:
                err_msg = f"LLM synthesis failed: {exc}"
                logger.error(err_msg)
                state.errors.append(err_msg)
                # Fallback: concatenate snippets
                state.research_notes = f"## Research Notes: {query}\n\n" + "\n\n".join(
                    f"**{s.title}**: {s.snippet}" for s in sources
                )

        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=state.research_notes,
                metadata={"num_sources": len(sources)},
            )
        )
        state.add_trace_event("researcher.done", {"notes_len": len(state.research_notes or "")})
        logger.info("Researcher done | notes=%d chars", len(state.research_notes or ""))
        return state