"""Teacher model wrapper — the guide/evaluator in the self-eval loop."""

from __future__ import annotations

import logging

from self_loop_eval.config import ModelConfig
from self_loop_eval.models.base import LLMResponse, LLMWrapper
from self_loop_eval.models.prompts import (
    TEACHER_COMPARE_PROMPT,
    TEACHER_EVALUATE_PROMPT,
    TEACHER_SYSTEM_PROMPT,
    TEACHER_THINKING_PROMPT,
)

logger = logging.getLogger(__name__)


class TeacherModel(LLMWrapper):
    """Teacher model that evaluates, compares, and provides thinking scaffolds.

    The teacher is a larger, more capable LLM. It does NOT give answers directly
    but provides evaluation, comparison, and thinking strategies to help the
    student improve.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._client = None

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate a response from the teacher model.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.

        Returns:
            The model's text response.
        """
        if not system_prompt:
            system_prompt = TEACHER_SYSTEM_PROMPT

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

    def evaluate_student(
        self,
        task_prompt: str,
        student_rounds: list[dict],
        ground_truth: str | None = None,
    ) -> LLMResponse:
        """Independently evaluate the student's work across all rounds.

        Args:
            task_prompt: The original task.
            student_rounds: List of dicts with 'solution' and 'critique' per round.
            ground_truth: Optional ground truth answer.

        Returns:
            LLMResponse with teacher's evaluation and score.
        """
        rounds_text = self._format_rounds(student_rounds)
        gt_text = f"\n\nGround Truth:\n{ground_truth}" if ground_truth else ""

        prompt = TEACHER_EVALUATE_PROMPT.format(
            task=task_prompt,
            student_rounds=rounds_text,
            ground_truth=gt_text,
        )
        return self.generate_structured(prompt, TEACHER_SYSTEM_PROMPT)

    def inject_thinking(
        self,
        task_prompt: str,
        student_solution: str,
        student_critique: str,
    ) -> LLMResponse:
        """Generate a thinking scaffold — NOT an answer, but a strategy hint.

        This is the key teacher intervention: provide reasoning approaches
        and perspectives the student hasn't considered.

        Args:
            task_prompt: The original task.
            student_solution: The student's current best solution.
            student_critique: The student's self-critique.

        Returns:
            LLMResponse with thinking hints and strategies.
        """
        prompt = TEACHER_THINKING_PROMPT.format(
            task=task_prompt,
            solution=student_solution,
            critique=student_critique,
        )
        return self.generate_structured(prompt, TEACHER_SYSTEM_PROMPT)

    def compare_rounds(
        self,
        task_prompt: str,
        first_solution: str,
        final_solution: str,
        ground_truth: str | None = None,
    ) -> LLMResponse:
        """Compare the student's first and final solutions.

        Determines if the student is genuinely improving or just rephrasing.

        Args:
            task_prompt: The original task.
            first_solution: Round 1 solution.
            final_solution: Final round solution.
            ground_truth: Optional ground truth.

        Returns:
            LLMResponse with comparison analysis and improvement score.
        """
        gt_text = f"\n\nGround Truth:\n{ground_truth}" if ground_truth else ""
        prompt = TEACHER_COMPARE_PROMPT.format(
            task=task_prompt,
            first_solution=first_solution,
            final_solution=final_solution,
            ground_truth=gt_text,
        )
        return self.generate_structured(prompt, TEACHER_SYSTEM_PROMPT)

    @staticmethod
    def _format_rounds(rounds: list[dict]) -> str:
        """Format student rounds into a readable text block."""
        parts = []
        for i, r in enumerate(rounds):
            parts.append(f"--- Round {i + 1} ---")
            parts.append(f"Solution:\n{r.get('solution', 'N/A')}")
            if r.get("critique"):
                parts.append(f"Self-Critique:\n{r['critique']}")
            if r.get("score") is not None:
                parts.append(f"Self-Score: {r['score']}")
            parts.append("")
        return "\n".join(parts)

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
