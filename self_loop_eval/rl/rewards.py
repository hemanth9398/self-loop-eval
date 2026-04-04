"""Reward function construction for the RL layer."""

from __future__ import annotations

import logging

from self_loop_eval.config import RLConfig
from self_loop_eval.eval_loop.round_state import LoopResult

logger = logging.getLogger(__name__)


class RewardFunction:
    """Constructs reward signals from self-eval loop results.

    The reward combines two components:
    1. Self-improvement: Did the student's score improve between rounds?
    2. Teacher alignment: Does the student's self-assessment match the teacher's?
    """

    def __init__(self, config: RLConfig):
        self.config = config

    def compute_reward(self, loop_result: LoopResult) -> float:
        """Compute the total reward for a completed self-eval loop.

        Args:
            loop_result: A completed LoopResult.

        Returns:
            A scalar reward value.
        """
        self_improvement = self._self_improvement_reward(loop_result)
        teacher_alignment = self._teacher_alignment_reward(loop_result)

        reward = (
            self.config.reward_self_improvement_weight * self_improvement
            + self.config.reward_teacher_alignment_weight * teacher_alignment
        )

        logger.debug(
            "Reward for %s: self_improvement=%.4f, teacher_alignment=%.4f, total=%.4f",
            loop_result.task_id,
            self_improvement,
            teacher_alignment,
            reward,
        )

        return reward

    def compute_batch_rewards(
        self, loop_results: list[LoopResult]
    ) -> list[dict]:
        """Compute rewards for a batch of loop results.

        Args:
            loop_results: List of completed LoopResults.

        Returns:
            List of dicts with task_id, reward, and component scores.
        """
        rewards = []
        for result in loop_results:
            reward = self.compute_reward(result)
            rewards.append({
                "task_id": result.task_id,
                "reward": reward,
                "self_improvement": self._self_improvement_reward(result),
                "teacher_alignment": self._teacher_alignment_reward(result),
                "num_rounds": result.num_rounds,
                "final_score": result.final_env_score,
            })
        return rewards

    @staticmethod
    def _self_improvement_reward(loop_result: LoopResult) -> float:
        """Reward based on score improvement from first to last round.

        Returns a value in [-1.0, 1.0]:
        - Positive if the student improved
        - Zero if no change
        - Negative if the student got worse
        """
        first = loop_result.first_score
        final = loop_result.final_score
        if first is None or final is None:
            return 0.0
        return max(-1.0, min(1.0, final - first))

    @staticmethod
    def _teacher_alignment_reward(loop_result: LoopResult) -> float:
        """Reward based on alignment between student self-assessment and teacher eval.

        Measures how close the student's self-score is to the teacher's score.
        A perfectly calibrated student gets reward 1.0.

        Returns a value in [0.0, 1.0].
        """
        # Find the last round with both self-score and teacher eval
        for r in reversed(loop_result.rounds):
            if r.self_score is not None and r.teacher_eval and r.teacher_eval.score is not None:
                gap = abs(r.self_score - r.teacher_eval.score)
                # Convert gap to reward: 0 gap = 1.0 reward, 1.0 gap = 0.0 reward
                return max(0.0, 1.0 - gap)

        # No teacher eval available
        return 0.0
