"""Student model wrapper — the learner in the self-eval loop."""

from __future__ import annotations

import logging

from self_loop_eval.config import ModelConfig
from self_loop_eval.models.base import LLMResponse, LLMWrapper
from self_loop_eval.models.prompts import (
    SELF_CORRECT_PROMPT,
    SELF_REFLECT_PROMPT,
    STUDENT_SYSTEM_PROMPT,
    TASK_SOLVE_PROMPT,
)

logger = logging.getLogger(__name__)


class StudentModel(LLMWrapper):
    """Student model that solves tasks, self-evaluates, and self-corrects.

    The student is a smaller/medium LLM that learns through the self-eval loop.
    It produces solutions with chain-of-thought reasoning and can critique
    and improve its own work.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._client = None

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate a response from the student model.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt (defaults to student persona).

        Returns:
            The model's text response.
        """
        if not system_prompt:
            system_prompt = STUDENT_SYSTEM_PROMPT

        client = self._get_client()
        response = client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content or ""

    def solve_task(self, task_prompt: str) -> LLMResponse:
        """Produce an initial solution for a task.

        Args:
            task_prompt: The formatted task description.

        Returns:
            LLMResponse with the solution and reasoning.
        """
        prompt = TASK_SOLVE_PROMPT.format(task=task_prompt)
        return self.generate_structured(prompt, STUDENT_SYSTEM_PROMPT)

    def self_reflect(self, task_prompt: str, solution: LLMResponse) -> LLMResponse:
        """Self-evaluate a previous solution.

        The student reviews its own answer and reasoning, identifies mistakes,
        and produces a self-critique with a score.

        Args:
            task_prompt: The original task description.
            solution: The student's previous solution.

        Returns:
            LLMResponse with the self-critique and score.
        """
        prompt = SELF_REFLECT_PROMPT.format(
            task=task_prompt,
            solution=solution.content,
            reasoning=solution.reasoning,
        )
        return self.generate_structured(prompt, STUDENT_SYSTEM_PROMPT)

    def self_correct(
        self, task_prompt: str, solution: LLMResponse, critique: LLMResponse
    ) -> LLMResponse:
        """Produce an improved solution based on self-critique.

        Args:
            task_prompt: The original task description.
            solution: The previous solution.
            critique: The self-critique from self_reflect.

        Returns:
            LLMResponse with the improved solution.
        """
        prompt = SELF_CORRECT_PROMPT.format(
            task=task_prompt,
            solution=solution.content,
            reasoning=solution.reasoning,
            critique=critique.content,
        )
        return self.generate_structured(prompt, STUDENT_SYSTEM_PROMPT)

    def _get_client(self):
        """Lazily create the OpenAI client."""
        if self._client is None:
            import openai

            kwargs = {}
            if self.config.api_key:
                kwargs["api_key"] = self.config.api_key
            if self.config.api_base:
                kwargs["base_url"] = self.config.api_base
            self._client = openai.OpenAI(**kwargs)
        return self._client
