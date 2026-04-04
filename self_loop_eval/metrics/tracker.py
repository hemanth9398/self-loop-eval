"""Metrics tracker — monitors improvement, teacher dependency, and self-eval accuracy."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from self_loop_eval.config import MetricsConfig
from self_loop_eval.eval_loop.round_state import LoopResult

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Tracks and persists metrics across training cycles.

    Metrics tracked:
    1. Improvement over cycles: Does the student get better over time?
    2. Teacher dependency: Does the student need the teacher less?
    3. Self-eval accuracy: Does the student's self-assessment match reality?
    4. Per-task performance trends
    """

    def __init__(self, config: MetricsConfig):
        self.config = config
        self._metrics_dir = Path(config.metrics_dir)
        self._metrics_dir.mkdir(parents=True, exist_ok=True)
        self._history: list[dict] = []
        self._load_history()

    def record_cycle(self, loop_results: list[LoopResult]) -> dict:
        """Record metrics for a complete training cycle.

        Args:
            loop_results: All loop results from this cycle.

        Returns:
            Summary dict with computed metrics.
        """
        timestamp = datetime.now(tz=timezone.utc).isoformat()

        cycle_metrics = {
            "timestamp": timestamp,
            "cycle_number": len(self._history) + 1,
            "tasks_processed": len(loop_results),
            "improvement": self._compute_improvement_metrics(loop_results),
            "teacher_dependency": self._compute_teacher_dependency(loop_results),
            "self_eval_accuracy": self._compute_self_eval_accuracy(loop_results),
            "per_task": self._compute_per_task_metrics(loop_results),
        }

        self._history.append(cycle_metrics)
        self._save_history()

        logger.info(
            "Cycle %d metrics: avg_improvement=%.4f, teacher_rate=%.2f, "
            "self_eval_accuracy=%.4f",
            cycle_metrics["cycle_number"],
            cycle_metrics["improvement"]["avg_improvement"],
            cycle_metrics["teacher_dependency"]["teacher_intervention_rate"],
            cycle_metrics["self_eval_accuracy"]["avg_accuracy"],
        )

        return cycle_metrics

    def get_trend(self, metric_name: str, last_n: int = 10) -> list[float]:
        """Get the trend of a metric over recent cycles.

        Args:
            metric_name: One of 'improvement', 'teacher_dependency', 'self_eval_accuracy'.
            last_n: Number of recent cycles to include.

        Returns:
            List of metric values over time.
        """
        key_map = {
            "improvement": ("improvement", "avg_improvement"),
            "teacher_dependency": ("teacher_dependency", "teacher_intervention_rate"),
            "self_eval_accuracy": ("self_eval_accuracy", "avg_accuracy"),
        }

        if metric_name not in key_map:
            return []

        category, field = key_map[metric_name]
        history = self._history[-last_n:]
        return [h[category][field] for h in history if category in h]

    def get_summary(self) -> dict:
        """Get a full summary of all tracked metrics.

        Returns:
            Summary dict with trends and latest values.
        """
        if not self._history:
            return {"cycles": 0, "message": "No cycles recorded yet"}

        latest = self._history[-1]
        return {
            "total_cycles": len(self._history),
            "latest_cycle": latest["cycle_number"],
            "latest_improvement": latest["improvement"]["avg_improvement"],
            "latest_teacher_rate": latest["teacher_dependency"][
                "teacher_intervention_rate"
            ],
            "latest_self_eval_accuracy": latest["self_eval_accuracy"]["avg_accuracy"],
            "improvement_trend": self.get_trend("improvement"),
            "teacher_dependency_trend": self.get_trend("teacher_dependency"),
            "self_eval_accuracy_trend": self.get_trend("self_eval_accuracy"),
        }

    @staticmethod
    def _compute_improvement_metrics(results: list[LoopResult]) -> dict:
        """Compute improvement metrics across all tasks."""
        improvements = []
        final_scores = []

        for r in results:
            r.compute_improvement()
            improvements.append(r.improvement)
            final_scores.append(r.final_env_score)

        avg_improvement = sum(improvements) / len(improvements) if improvements else 0.0
        avg_final_score = sum(final_scores) / len(final_scores) if final_scores else 0.0
        tasks_improved = sum(1 for i in improvements if i > 0)

        return {
            "avg_improvement": avg_improvement,
            "avg_final_score": avg_final_score,
            "tasks_improved": tasks_improved,
            "tasks_total": len(results),
            "improvement_rate": tasks_improved / len(results) if results else 0.0,
        }

    @staticmethod
    def _compute_teacher_dependency(results: list[LoopResult]) -> dict:
        """Compute how much the student relied on the teacher."""
        teacher_interventions = sum(1 for r in results if r.teacher_intervened)
        total = len(results)

        return {
            "teacher_intervention_rate": (
                teacher_interventions / total if total > 0 else 0.0
            ),
            "teacher_interventions": teacher_interventions,
            "total_tasks": total,
        }

    @staticmethod
    def _compute_self_eval_accuracy(results: list[LoopResult]) -> dict:
        """Compute how well the student's self-assessment matches reality.

        Compares self-assigned scores with environment evaluation scores.
        """
        gaps = []

        for r in results:
            for round_state in r.rounds:
                if round_state.self_score is not None and round_state.env_score is not None:
                    gap = abs(round_state.self_score - round_state.env_score)
                    gaps.append(gap)

        avg_gap = sum(gaps) / len(gaps) if gaps else 1.0
        accuracy = max(0.0, 1.0 - avg_gap)

        return {
            "avg_accuracy": accuracy,
            "avg_gap": avg_gap,
            "num_comparisons": len(gaps),
        }

    @staticmethod
    def _compute_per_task_metrics(results: list[LoopResult]) -> dict:
        """Compute per-task metrics."""
        per_task: dict[str, dict] = {}
        for r in results:
            per_task[r.task_id] = {
                "final_score": r.final_env_score,
                "improvement": r.improvement,
                "num_rounds": r.num_rounds,
                "converged": r.converged,
                "teacher_intervened": r.teacher_intervened,
            }
        return per_task

    def _load_history(self) -> None:
        """Load metrics history from disk."""
        history_file = self._metrics_dir / "history.json"
        if history_file.exists():
            try:
                self._history = json.loads(history_file.read_text())
            except (json.JSONDecodeError, OSError):
                logger.warning("Could not load metrics history, starting fresh")
                self._history = []

    def _save_history(self) -> None:
        """Save metrics history to disk."""
        history_file = self._metrics_dir / "history.json"
        history_file.write_text(json.dumps(self._history, indent=2))
