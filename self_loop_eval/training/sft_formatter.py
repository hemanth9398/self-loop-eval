"""SFT data formatter — converts training tuples into supervised fine-tuning format."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SFTFormatter:
    """Formats training data into SFT (Supervised Fine-Tuning) pairs.

    Produces instruction/response pairs where:
    - Instruction = task + previous solution + critique context
    - Response = improved solution with teacher-enhanced reasoning

    The goal is to teach the student to think like the teacher evaluated,
    incorporating the teacher's thinking scaffolds into the student's reasoning.
    """

    def format_for_sft(self, training_data: list[dict]) -> list[dict]:
        """Convert training tuples to SFT instruction/response format.

        Args:
            training_data: List of training dicts from TrainingDataCollector.

        Returns:
            List of SFT-formatted dicts with 'instruction' and 'response' fields.
        """
        sft_pairs = []

        for entry in training_data:
            # Basic self-correction pair
            sft_pair = self._make_self_correction_pair(entry)
            sft_pairs.append(sft_pair)

            # If teacher thought is available, add teacher-enhanced pair
            if entry.get("teacher_thought"):
                teacher_pair = self._make_teacher_thinking_pair(entry)
                sft_pairs.append(teacher_pair)

        logger.info("Formatted %d SFT pairs from %d training entries",
                     len(sft_pairs), len(training_data))
        return sft_pairs

    def save_sft_data(
        self, sft_pairs: list[dict], output_path: str | Path
    ) -> Path:
        """Save SFT pairs to a JSON file for training.

        Args:
            sft_pairs: List of SFT-formatted dicts.
            output_path: Path to save the JSON file.

        Returns:
            Path to the saved file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(sft_pairs, f, indent=2)

        logger.info("Saved %d SFT pairs to %s", len(sft_pairs), output_path)
        return output_path

    @staticmethod
    def _make_self_correction_pair(entry: dict) -> dict:
        """Create an SFT pair that teaches self-correction.

        Instruction: "Here's your previous solution and your self-critique.
                      Produce an improved solution."
        Response: The actual improved solution the student produced.
        """
        instruction = (
            f"You are solving a coding task. Here is the task and your previous attempt.\n\n"
            f"## Task\n{entry['task_prompt']}\n\n"
            f"## Your Previous Solution\n{entry['solution_v1']}\n\n"
            f"## Your Self-Critique\n{entry['self_critique']}\n\n"
            f"Based on your self-evaluation, produce an improved solution with reasoning."
        )

        # Build the ideal response — the improved solution with reasoning
        response_parts = []
        if entry.get("reasoning_v2"):
            response_parts.append(f"Reasoning: {entry['reasoning_v2']}")
        response_parts.append(f"Solution:\n{entry['solution_v2']}")

        return {
            "instruction": instruction,
            "response": "\n\n".join(response_parts),
            "metadata": {
                "type": "self_correction",
                "task_id": entry["task_id"],
                "round_pair": entry["round_pair"],
                "improvement": entry.get("improvement", 0),
            },
        }

    @staticmethod
    def _make_teacher_thinking_pair(entry: dict) -> dict:
        """Create an SFT pair that teaches teacher-level thinking.

        This pair injects the teacher's reasoning scaffold into the training:
        Instruction: task + student solution + teacher hint
        Response: improved solution that incorporates the teacher's thinking
        """
        instruction = (
            f"You are solving a coding task. A mentor has given you a thinking hint.\n\n"
            f"## Task\n{entry['task_prompt']}\n\n"
            f"## Your Current Solution\n{entry['solution_v1']}\n\n"
            f"## Mentor's Thinking Hint\n{entry['teacher_thought']}\n\n"
            f"Use this hint to improve your solution. Show your reasoning."
        )

        response_parts = []
        if entry.get("reasoning_v2"):
            response_parts.append(f"Reasoning: {entry['reasoning_v2']}")
        response_parts.append(f"Solution:\n{entry['solution_v2']}")

        return {
            "instruction": instruction,
            "response": "\n\n".join(response_parts),
            "metadata": {
                "type": "teacher_thinking",
                "task_id": entry["task_id"],
                "round_pair": entry["round_pair"],
                "improvement": entry.get("improvement", 0),
            },
        }
