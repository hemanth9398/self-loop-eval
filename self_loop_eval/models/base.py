"""Base LLM wrapper with provider-agnostic interface."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from self_loop_eval.config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Structured response from an LLM call."""

    content: str
    reasoning: str = ""
    score: float | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "reasoning": self.reasoning,
            "score": self.score,
            "metadata": self.metadata,
        }


class LLMWrapper(ABC):
    """Abstract base wrapper for LLM providers.

    Subclasses implement the actual API call; this base provides
    common prompt formatting and response parsing.
    """

    def __init__(self, config: ModelConfig):
        self.config = config

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Send a prompt to the LLM and return the raw text response.

        Args:
            prompt: The user prompt.
            system_prompt: An optional system/role prompt.

        Returns:
            The model's text response.
        """

    def generate_structured(
        self, prompt: str, system_prompt: str = ""
    ) -> LLMResponse:
        """Generate a response and attempt to parse structured fields.

        The model is asked to respond in JSON with 'content', 'reasoning',
        and optionally 'score' fields. Falls back to unstructured if parsing fails.

        Args:
            prompt: The user prompt.
            system_prompt: System prompt.

        Returns:
            An LLMResponse with parsed fields.
        """
        structured_prompt = (
            f"{prompt}\n\n"
            "Respond in JSON format with the following fields:\n"
            '- "content": your solution or answer\n'
            '- "reasoning": your step-by-step reasoning / chain of thought\n'
            '- "score": (optional) a self-assessed score from 0.0 to 1.0\n'
        )

        raw = self.generate(structured_prompt, system_prompt)
        return self._parse_response(raw)

    @staticmethod
    def _parse_response(raw: str) -> LLMResponse:
        """Parse a raw LLM response into an LLMResponse object."""
        # Try to extract JSON from the response
        try:
            # Handle ```json ... ``` blocks
            if "```json" in raw:
                json_str = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                json_str = raw.split("```")[1].split("```")[0].strip()
            else:
                json_str = raw.strip()

            data = json.loads(json_str)
            return LLMResponse(
                content=data.get("content", raw),
                reasoning=data.get("reasoning", ""),
                score=data.get("score"),
                metadata={"raw": raw, "parsed": True},
            )
        except (json.JSONDecodeError, IndexError):
            logger.debug("Could not parse structured response, using raw text.")
            return LLMResponse(
                content=raw,
                reasoning="",
                score=None,
                metadata={"raw": raw, "parsed": False},
            )
