"""Training scheduler — orchestrates nightly training cycles."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

from self_loop_eval.config import SystemConfig
from self_loop_eval.environments.base import TaskEnvironment
from self_loop_eval.eval_loop.loop import SelfEvalLoop
from self_loop_eval.models.student import StudentModel
from self_loop_eval.models.teacher import TeacherModel
from self_loop_eval.training.data_collector import TrainingDataCollector
from self_loop_eval.training.sft_formatter import SFTFormatter

logger = logging.getLogger(__name__)


class TrainingScheduler:
    """Orchestrates the full training cycle: eval loop → collect data → train.

    This is the nightly training scheduler that:
    1. Runs the self-eval loop on a batch of tasks
    2. Collects training data from the loop results
    3. Formats data for SFT
    4. (Optionally) triggers LoRA fine-tuning
    """

    def __init__(
        self,
        config: SystemConfig,
        student: StudentModel,
        teacher: TeacherModel,
        environment: TaskEnvironment,
    ):
        self.config = config
        self.student = student
        self.teacher = teacher
        self.environment = environment
        self.eval_loop = SelfEvalLoop(config, student, teacher, environment)
        self.data_collector = TrainingDataCollector(config.training)
        self.sft_formatter = SFTFormatter()

    def run_cycle(self) -> dict:
        """Run one complete training cycle.

        Returns:
            Summary dict with cycle results.
        """
        cycle_start = datetime.utcnow()
        logger.info("Starting training cycle at %s", cycle_start.isoformat())

        # Step 1: Run self-eval loop on all tasks
        loop_results = self.eval_loop.run_all_tasks()
        logger.info("Completed %d task loops", len(loop_results))

        # Step 2: Collect training data
        self.data_collector.clear()
        training_data = self.data_collector.collect_batch(loop_results)
        logger.info("Collected %d training entries", len(training_data))

        # Step 3: Save raw training data
        raw_data_path = self.data_collector.save()

        # Step 4: Format for SFT
        sft_pairs = self.sft_formatter.format_for_sft(training_data)

        # Step 5: Save SFT data
        sft_dir = Path(self.config.training.training_data_path)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        sft_path = self.sft_formatter.save_sft_data(
            sft_pairs, sft_dir / f"sft_data_{ts}.json"
        )

        cycle_end = datetime.utcnow()

        summary = {
            "cycle_start": cycle_start.isoformat(),
            "cycle_end": cycle_end.isoformat(),
            "duration_seconds": (cycle_end - cycle_start).total_seconds(),
            "tasks_processed": len(loop_results),
            "training_entries": len(training_data),
            "sft_pairs": len(sft_pairs),
            "raw_data_path": str(raw_data_path),
            "sft_data_path": str(sft_path),
            "results": [r.to_dict() for r in loop_results],
        }

        logger.info(
            "Cycle complete: %d tasks, %d training pairs, %.1fs",
            len(loop_results),
            len(sft_pairs),
            summary["duration_seconds"],
        )

        return summary

    def run_continuous(self, interval_hours: float = 24.0) -> None:
        """Run training cycles continuously at a fixed interval.

        Args:
            interval_hours: Hours between cycles (default: 24 for nightly).
        """
        logger.info("Starting continuous training (interval: %.1f hours)", interval_hours)
        interval_seconds = interval_hours * 3600

        while True:
            try:
                summary = self.run_cycle()
                logger.info(
                    "Cycle %s complete. Next cycle in %.1f hours.",
                    summary["cycle_start"],
                    interval_hours,
                )
            except Exception:
                logger.exception("Error in training cycle")

            time.sleep(interval_seconds)
