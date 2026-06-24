"""LLM client abstraction – OpenAI API.

Reads OPENAI_API_KEY and OPENAI_MODEL from environment / .env file.
"""

import logging
import os
from dataclasses import dataclass, field

from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


@dataclass
class LLMClient:
    """OpenAI LLM client."""

    model: str = ""
    api_key: str = ""
    max_tokens: int = 2048
    timeout_seconds: int = 60

    def __post_init__(self) -> None:
        from openai import OpenAI

        if not self.api_key:
            key = os.environ.get("OPENAI_API_KEY", "").strip()
            if not key:
                raise ValueError("OPENAI_API_KEY not set in environment or .env")
            object.__setattr__(self, "api_key", key)

        if not self.model:
            model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()
            object.__setattr__(self, "model", model)

        logger.info("LLMClient ready | model=%s", self.model)
        self._client = OpenAI(api_key=self.api_key, timeout=self.timeout_seconds)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion."""
        from openai import AuthenticationError

        logger.debug("LLM call | model=%s", self.model)
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self.max_tokens,
            )
        except AuthenticationError as exc:
            logger.error("OpenAI authentication failed. Check OPENAI_API_KEY.")
            raise

        content = response.choices[0].message.content or ""
        usage = response.usage
        in_tok = usage.prompt_tokens if usage else None
        out_tok = usage.completion_tokens if usage else None
        cost: float | None = None
        if in_tok and out_tok:
            # gpt-4o-mini pricing: $0.15/1M input, $0.60/1M output
            cost = (in_tok * 0.00000015) + (out_tok * 0.0000006)

        logger.debug("LLM done | in=%s out=%s cost=$%s", in_tok, out_tok,
                     f"{cost:.6f}" if cost else "N/A")
        return LLMResponse(content=content, input_tokens=in_tok,
                           output_tokens=out_tok, cost_usd=cost)