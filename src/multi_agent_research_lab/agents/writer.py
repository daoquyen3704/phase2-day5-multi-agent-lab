"""Writer agent – produces the final answer."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a skilled Technical Writer. Your job is to synthesize research and
analysis into a clear, well-structured response for the target audience.

Requirements:
- Write approximately 500 words (±100 words).
- Start with a brief executive summary (2-3 sentences).
- Use clear section headers (##).
- Cite sources inline where relevant using [Source Title] notation.
- End with a "## Key Takeaways" section (3-5 bullet points).
- Tailor language to the stated audience.
- Do NOT invent facts not present in the research/analysis notes.
- If the research has gaps, acknowledge them honestly.

Tone: authoritative but accessible, no unnecessary jargon.
"""


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate state.final_answer."""
        if not state.research_notes:
            raise AgentExecutionError("Writer: research_notes is empty")

        # Build comprehensive context
        sources_summary = ""
        if state.sources:
            sources_summary = "**Available Sources:**\n" + "\n".join(
                f"- {s.title}" + (f" ({s.url})" if s.url else "")
                for s in state.sources
            )

        user_prompt = (
            f"Original Query: {state.request.query}\n\n"
            f"Audience: {state.request.audience}\n\n"
            f"{sources_summary}\n\n"
            f"Research Notes:\n{state.research_notes}\n\n"
            f"Analysis Notes:\n{state.analysis_notes or '(no analysis available)'}\n\n"
            "Please write the final response."
        )

        with trace_span("writer.compose") as span:
            try:
                response = self._llm.complete(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                )
                state.final_answer = response.content
                span["attributes"]["output_tokens"] = response.output_tokens

                # Track token cost in agent result metadata
                cost = response.cost_usd
                total_input = sum(
                    (r.metadata.get("input_tokens") or 0) for r in state.agent_results
                )
            except Exception as exc:
                err_msg = f"Writer LLM failed: {exc}"
                logger.error(err_msg)
                state.errors.append(err_msg)
                # Fallback: stitch notes together
                state.final_answer = (
                    f"# {state.request.query}\n\n"
                    f"{state.research_notes}\n\n"
                    f"---\n\n"
                    f"{state.analysis_notes or ''}"
                )
                cost = None

        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=state.final_answer,
                metadata={
                    "answer_len": len(state.final_answer),
                    "cost_usd": cost,
                },
            )
        )
        state.add_trace_event("writer.done", {"answer_len": len(state.final_answer)})
        logger.info("Writer done | answer=%d chars", len(state.final_answer))
        return state