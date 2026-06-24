"""Optional critic agent skeleton for bonus work."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings.

        Performs a lightweight citation and completeness check.
        """

        findings: list[str] = []
        if not state.final_answer:
            findings.append("Final answer is missing.")
        if state.sources and state.final_answer and "References:" not in state.final_answer:
            findings.append("Final answer has sources but no reference section.")
        if not state.analysis_notes:
            findings.append("Analysis notes are missing.")
        if not findings:
            findings.append("No blocking quality issues found.")

        content = "\n".join(f"- {finding}" for finding in findings)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=content,
                metadata={"finding_count": len(findings)},
            )
        )
        with trace_span("critic.run", {"finding_count": len(findings)}) as span:
            state.add_trace_event("critic.run", span)
        return state
