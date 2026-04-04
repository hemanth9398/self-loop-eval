"""Convergence detection for the self-evaluation loop."""

from __future__ import annotations

import logging
from difflib import SequenceMatcher

from self_loop_eval.config import EvalLoopConfig
from self_loop_eval.eval_loop.round_state import RoundState

logger = logging.getLogger(__name__)


class ConvergenceDetector:
    """Detects when the student has stopped improving and the loop should end.

    Convergence is detected when:
    1. The score improvement between rounds drops below a threshold.
    2. The solution text similarity between rounds is very high (just rephrasing).
    3. The student has been stuck at the same score for multiple rounds (plateau).
    """

    def __init__(self, config: EvalLoopConfig):
        self.config = config
        self._plateau_count = 0
        self._prev_score: float | None = None

    def reset(self) -> None:
        """Reset state for a new loop."""
        self._plateau_count = 0
        self._prev_score = None

    def check_convergence(self, rounds: list[RoundState]) -> bool:
        """Check if the loop has converged based on recent rounds.

        Args:
            rounds: All rounds completed so far.

        Returns:
            True if the loop should stop (converged or stuck).
        """
        if len(rounds) < 2:
            return False

        current = rounds[-1]
        previous = rounds[-2]

        # Check score improvement
        if current.env_score is not None and previous.env_score is not None:
            improvement = current.env_score - previous.env_score
            if improvement < self.config.min_score_improvement:
                self._plateau_count += 1
                logger.info(
                    "Plateau detected (round %d): improvement=%.4f, plateau_count=%d",
                    current.round_number,
                    improvement,
                    self._plateau_count,
                )
            else:
                self._plateau_count = 0

        # Check text similarity (are solutions too similar = just rephrasing?)
        similarity = self._text_similarity(
            current.solution.content, previous.solution.content
        )
        if similarity > (1.0 - self.config.convergence_threshold):
            logger.info(
                "Solution text converged (round %d): similarity=%.4f",
                current.round_number,
                similarity,
            )
            return True

        # Check plateau
        if self._plateau_count >= self.config.stuck_plateau_rounds:
            logger.info(
                "Score plateau detected (round %d): stuck for %d rounds",
                current.round_number,
                self._plateau_count,
            )
            return True

        return False

    def is_stuck(self, rounds: list[RoundState]) -> bool:
        """Determine if the student is stuck and needs teacher intervention.

        The student is stuck if the score has not improved for multiple rounds
        and the self-scores are low.

        Args:
            rounds: All rounds completed so far.

        Returns:
            True if teacher intervention should be triggered.
        """
        if len(rounds) < 2:
            return False

        # Stuck if plateau detected
        if self._plateau_count >= self.config.stuck_plateau_rounds:
            return True

        # Stuck if the latest score is still low despite multiple attempts
        latest = rounds[-1]
        if latest.env_score is not None and latest.env_score < 0.5 and len(rounds) >= 3:
            return True

        return False

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        """Compute text similarity ratio between two strings."""
        return SequenceMatcher(None, a, b).ratio()
