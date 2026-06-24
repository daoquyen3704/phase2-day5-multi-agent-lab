"""Analyst agent – turns research notes into structured insights."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a rigorous Analytical Specialist. Your job is to critically evaluate
research notes and extract structured insights.

Your analysis MUST include:
1. **Core Claims** – the 3-5 most important factual claims with evidence quality ratings (Strong/Moderate/Weak).
2. **Contrasting Viewpoints** – where sources disagree or present different perspectives.
3. **Evidence Quality** – assess which claims are well-supported vs speculative.
4. **Key Implications** – what this research means for the audience.
5. **Confidence Level** – your overall confidence in the research (High/Medium/Low) with justification.

Be critical. Flag weak evidence. Do not repeat the research notes verbatim.
Output 300-400 words of structured analysis.
"""


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate state.analysis_notes."""
        if not state.research_notes:
            raise AgentExecutionError("Analyst: research_notes is empty – run Researcher first")

        user_prompt = (
            f"Original Query: {state.request.query}\n\n"
            f"Audience: {state.request.audience}\n\n"
            f"Research Notes to Analyse:\n{state.research_notes}\n\n"
            "Provide your structured analysis."
        )

        with trace_span("analyst.analyze") as span:
            try:
                response = self._llm.complete(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                )
                state.analysis_notes = response.content
                span["attributes"]["output_tokens"] = response.output_tokens
            except Exception as exc:
                err_msg = f"Analyst LLM failed: {exc}"
                logger.error(err_msg)
                state.errors.append(err_msg)
                # Fallback: basic extraction
                state.analysis_notes = (
                    f"## Analysis Notes\n\n"
                    f"Analysis of research on: {state.request.query}\n\n"
                    f"[Automated fallback] Research notes contain {len(state.research_notes)} characters. "
                    "LLM analysis unavailable due to error."
                )

        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=state.analysis_notes,
                metadata={"research_len": len(state.research_notes)},
            )
        )
        state.add_trace_event("analyst.done", {"analysis_len": len(state.analysis_notes or "")})
        logger.info("Analyst done | analysis=%d chars", len(state.analysis_notes or ""))
        return state