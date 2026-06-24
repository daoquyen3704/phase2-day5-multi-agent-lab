"""Critic agent – optional fact-checking and quality review."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a rigorous Quality Critic. Your job is to review the final answer
and flag any issues before delivery.

Check for:
1. **Factual consistency** – does the answer contradict the research notes?
2. **Citation coverage** – are key claims backed by sources?
3. **Hallucination signals** – any specific numbers, dates, or names not in the research?
4. **Completeness** – does the answer address the original query?
5. **Audience fit** – is the tone and depth appropriate?

Output format:
## Critic Review

**Overall Quality**: [Excellent / Good / Acceptable / Needs Revision]
**Issues Found**: (list or "None")
**Suggested Fix**: (brief or "None")
**Approved**: [YES / NO]
"""


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings to trace."""
        if not state.final_answer:
            logger.warning("Critic skipped: no final_answer to review")
            return state

        user_prompt = (
            f"Original Query: {state.request.query}\n\n"
            f"Research Notes:\n{state.research_notes or '(none)'}\n\n"
            f"Final Answer to Review:\n{state.final_answer}\n\n"
            "Please provide your quality review."
        )

        with trace_span("critic.review") as span:
            try:
                response = self._llm.complete(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                )
                review = response.content
                approved = "approved: yes" in review.lower()
                span["attributes"]["approved"] = approved

                state.agent_results.append(
                    AgentResult(
                        agent=AgentName.CRITIC,
                        content=review,
                        metadata={"approved": approved},
                    )
                )
                state.add_trace_event("critic.review", {
                    "approved": approved,
                    "review_len": len(review),
                })
                logger.info("Critic review done | approved=%s", approved)

                # Append review note to final answer if issues found
                if not approved:
                    state.final_answer = (
                        state.final_answer
                        + "\n\n---\n*[Quality review flagged potential issues. "
                        "See trace for details.]*"
                    )
            except Exception as exc:
                logger.warning("Critic failed (non-fatal): %s", exc)
                state.errors.append(f"Critic non-fatal: {exc}")

        return state