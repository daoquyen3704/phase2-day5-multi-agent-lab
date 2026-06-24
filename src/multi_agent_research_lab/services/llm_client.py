"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from dataclasses import dataclass
from importlib import import_module
from time import sleep
from typing import Any

from multi_agent_research_lab.core.config import Settings, get_settings


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client skeleton."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion.

        Uses OpenAI when `OPENAI_API_KEY` is configured. When no key is present,
        returns a deterministic local response so tests and classroom demos still run.
        """

        if not self._has_openai_api_key():
            return self._fallback_complete(system_prompt, user_prompt)

        for attempt in range(3):
            try:
                return self._complete_with_openai(system_prompt, user_prompt)
            except Exception:  # pragma: no cover - depends on external provider
                if attempt < 2:
                    sleep(0.5 * (2**attempt))

        fallback = self._fallback_complete(system_prompt, user_prompt)
        return LLMResponse(
            content=(
                f"{fallback.content}\n\n"
                "Provider warning: real LLM call failed after retries; used local fallback."
            ),
            input_tokens=fallback.input_tokens,
            output_tokens=fallback.output_tokens,
            cost_usd=fallback.cost_usd,
        )

    def _complete_with_openai(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        openai_module = import_module("openai")
        openai_client_class = openai_module.__dict__["OpenAI"]
        client = openai_client_class(
            api_key=self.settings.openai_api_key,
            timeout=self.settings.timeout_seconds,
        )
        response = client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        choice = response.choices[0]
        content = choice.message.content or ""
        usage: Any = getattr(response, "usage", None)
        input_tokens = self._usage_int(usage, "prompt_tokens")
        output_tokens = self._usage_int(usage, "completion_tokens")
        return LLMResponse(
            content=content.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=self._estimate_cost_usd(input_tokens, output_tokens),
        )

    def _has_openai_api_key(self) -> bool:
        key = (self.settings.openai_api_key or "").strip()
        return key.startswith("sk-")

    @staticmethod
    def _usage_int(usage: Any, field_name: str) -> int | None:
        if usage is None:
            return None
        value = getattr(usage, field_name, None)
        return value if isinstance(value, int) else None

    @staticmethod
    def _estimate_cost_usd(input_tokens: int | None, output_tokens: int | None) -> float | None:
        if input_tokens is None and output_tokens is None:
            return None
        prompt_tokens = input_tokens or 0
        completion_tokens = output_tokens or 0
        return (prompt_tokens * 0.00000015) + (completion_tokens * 0.0000006)

    @staticmethod
    def _fallback_complete(system_prompt: str, user_prompt: str) -> LLMResponse:
        role_prompt = system_prompt.lower()
        topic = user_prompt.strip().splitlines()[0][:120] if user_prompt.strip() else "the topic"

        if "writer agent" in role_prompt or "final answer" in role_prompt:
            content = (
                "Final answer:\n"
                "A reliable research workflow should separate source collection, analysis, "
                "and synthesis. Use citations for claims, keep routing explicit, and benchmark "
                "the multi-agent result against a single-agent baseline."
            )
        elif "analyst agent" in role_prompt or "analysis" in role_prompt:
            content = (
                "Analysis notes:\n"
                "- Strong evidence should be tied to named sources.\n"
                "- Multi-agent workflows help when tasks need separation of concerns.\n"
                "- Main risks are latency, cost, coordination errors, and weak validation."
            )
        elif "researcher agent" in role_prompt or "researcher" in role_prompt:
            content = (
                f"Research notes for {topic}:\n"
                "- Define the problem, target users, and expected decision.\n"
                "- Compare at least two approaches and capture trade-offs.\n"
                "- Keep citations attached to claims so analysis can verify evidence."
            )
        else:
            content = (
                f"Baseline answer for {topic}: collect relevant context, identify key claims, "
                "state trade-offs, and provide a concise recommendation with source references."
            )

        approx_tokens = max(1, len(system_prompt.split()) + len(user_prompt.split()))
        return LLMResponse(
            content=content,
            input_tokens=approx_tokens,
            output_tokens=max(1, len(content.split())),
            cost_usd=0.0,
        )
