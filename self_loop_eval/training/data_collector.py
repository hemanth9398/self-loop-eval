"""Training data collector — gathers self-eval loop results into training tuples."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from self_loop_eval.config import TrainingConfig
from self_loop_eval.eval_loop.round_state import LoopResult

logger = logging.getLogger(__name__)


class TrainingDataCollector:
    """Collects self-eval loop results and structures them as training data.

    Each loop result produces one or more training tuples of the form:
    (task, student_answer_v1, self_critique, student_answer_v2,
     teacher_feedback, teacher_thought, ground_truth)
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self._data: list[dict] = []

    def collect(self, loop_result: LoopResult) -> list[dict]:
        """Extract training tuples from a completed loop result.

        Args:
            loop_result: A completed LoopResult from the self-eval loop.

        Returns:
            List of training data dicts.
        """
        tuples = []

        rounds = loop_result.rounds
        if len(rounds) < 2:
            # Need at least 2 rounds for a meaningful training pair
            logger.debug("Skipping task %s: only %d round(s)", loop_result.task_id, len(rounds))
            return tuples

        for i in range(len(rounds) - 1):
            current = rounds[i]
            next_round = rounds[i + 1]

            entry = {
                "task_id": loop_result.task_id,
                "task_prompt": loop_result.task_prompt,
                "round_pair": f"{current.round_number}->{next_round.round_number}",
                "solution_v1": current.solution.content,
                "reasoning_v1": current.solution.reasoning,
                "self_score_v1": current.self_score,
                "env_score_v1": current.env_score,
                "self_critique": next_round.critique.content if next_round.critique else "",
                "solution_v2": next_round.solution.content,
                "reasoning_v2": next_round.solution.reasoning,
                "self_score_v2": next_round.self_score,
                "env_score_v2": next_round.env_score,
                "teacher_feedback": (
                    next_round.teacher_eval.content if next_round.teacher_eval else ""
                ),
                "teacher_thought": (
                    next_round.teacher_thought.content if next_round.teacher_thought else ""
                ),
                "ground_truth": loop_result.ground_truth or "",
                "improvement": (
                    (next_round.env_score or 0) - (current.env_score or 0)
                ),
                "timestamp": datetime.utcnow().isoformat(),
            }
            tuples.append(entry)

        self._data.extend(tuples)
        return tuples

    def collect_batch(self, loop_results: list[LoopResult]) -> list[dict]:
        """Collect training data from multiple loop results.

        Args:
            loop_results: List of completed LoopResults.

        Returns:
            All training data dicts collected.
        """
        all_tuples = []
        for result in loop_results:
            all_tuples.extend(self.collect(result))
        return all_tuples

    def save(self, filename: str | None = None) -> Path:
        """Save collected training data to a JSONL file.

        Args:
            filename: Optional filename. Defaults to timestamped file.

        Returns:
            Path to the saved file.
        """
        output_dir = Path(self.config.training_data_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"training_data_{ts}.jsonl"

        output_path = output_dir / filename
        with open(output_path, "w") as f:
            for entry in self._data:
                f.write(json.dumps(entry) + "\n")

        logger.info("Saved %d training entries to %s", len(self._data), output_path)
        return output_path

    def clear(self) -> None:
        """Clear collected data."""
        self._data.clear()

    @property
    def size(self) -> int:
        """Return the number of collected training entries."""
        return len(self._data)
