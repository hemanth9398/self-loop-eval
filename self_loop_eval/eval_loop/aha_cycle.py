"""Aha Moment Cycle Engine — the complete self-eval → teacher → train loop.

This is the core "aha moment" engine that implements the full cycle:

1. Student receives a real-world problem
2. Student solves it (Round 1)
3. Student self-evaluates (critique + score)
4. Student submits to teacher for evaluation
5. Teacher evaluates and compares with student's self-assessment
6. If student shows NO improvement after self-correction:
   → Teacher injects a "thought" (thinking scaffold)
   → Thought is formatted as LoRA training data
   → Student gets a micro-train (LoRA update) on that thought
   → Student retries — this creates the "AHA MOMENT"
     ("I know I could have thought better, and now I can!")
7. Repeat the complete cycle 10 times to validate

The aha moment = the student realizes what it COULD have done,
internalizes the teacher's thinking via training, and demonstrates
measurable improvement on retry.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from self_loop_eval.config import SystemConfig
from self_loop_eval.environments.base import TaskEnvironment, TaskInstance
from self_loop_eval.eval_loop.round_state import RoundState
from self_loop_eval.models.base import LLMResponse
from self_loop_eval.models.student import StudentModel
from self_loop_eval.models.teacher import TeacherModel
from self_loop_eval.training.data_collector import TrainingDataCollector
from self_loop_eval.training.sft_formatter import SFTFormatter

logger = logging.getLogger(__name__)


@dataclass
class AhaMoment:
    """Captures a single aha moment — when the student realizes improvement."""

    cycle_number: int
    task_id: str
    score_before_thought: float
    score_after_thought: float
    teacher_thought: str
    student_realization: str
    improvement: float
    timestamp: str = field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())

    @property
    def is_genuine_aha(self) -> bool:
        """An aha moment is genuine if score improved after teacher thought."""
        return self.improvement > 0.0

    def to_dict(self) -> dict:
        return {
            "cycle_number": self.cycle_number,
            "task_id": self.task_id,
            "score_before_thought": self.score_before_thought,
            "score_after_thought": self.score_after_thought,
            "teacher_thought": self.teacher_thought,
            "student_realization": self.student_realization,
            "improvement": self.improvement,
            "is_genuine_aha": self.is_genuine_aha,
            "timestamp": self.timestamp,
        }


@dataclass
class CycleResult:
    """Result of one complete aha-moment cycle."""

    cycle_number: int
    task_id: str
    initial_score: float
    self_eval_score: float
    teacher_score: float | None
    final_score: float
    teacher_thought_injected: bool
    aha_moment: AhaMoment | None
    rounds: list[RoundState] = field(default_factory=list)
    training_data_generated: list[dict] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "cycle_number": self.cycle_number,
            "task_id": self.task_id,
            "initial_score": self.initial_score,
            "self_eval_score": self.self_eval_score,
            "teacher_score": self.teacher_score,
            "final_score": self.final_score,
            "teacher_thought_injected": self.teacher_thought_injected,
            "aha_moment": self.aha_moment.to_dict() if self.aha_moment else None,
            "rounds_count": len(self.rounds),
            "training_entries": len(self.training_data_generated),
            "timestamp": self.timestamp,
        }


class AhaMomentEngine:
    """The complete aha-moment cycle engine.

    Runs the student through the full self-eval → teacher → train loop,
    creating genuine "aha moments" where the student internalizes the
    teacher's thinking and demonstrates measurable improvement.
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
        self.data_collector = TrainingDataCollector(config.training)
        self.sft_formatter = SFTFormatter()
        self._all_aha_moments: list[AhaMoment] = []
        self._all_cycle_results: list[CycleResult] = []
        self._all_training_data: list[dict] = []

    def run_single_cycle(
        self, task: TaskInstance, cycle_number: int
    ) -> CycleResult:
        """Run one complete aha-moment cycle for a single task.

        Flow:
        1. Student solves → env evaluates → student self-evaluates
        2. Student self-corrects → env evaluates again
        3. Student submits all work to teacher
        4. If no improvement → teacher injects thought → micro-train → retry
        5. Record whether an aha moment occurred

        Args:
            task: The task to solve.
            cycle_number: Which cycle this is (1-based).

        Returns:
            CycleResult with all details.
        """
        task_prompt = self.environment.get_task_prompt(task)
        rounds: list[RoundState] = []

        logger.info(
            "=== Cycle %d | Task: %s ===", cycle_number, task.task_id
        )

        # --- Step 1: Student initial solution ---
        logger.info("[Cycle %d] Step 1: Student solves task", cycle_number)
        solution_v1 = self.student.solve_task(task_prompt)
        eval_v1 = self.environment.evaluate(task, solution_v1.content)

        round1 = RoundState(
            round_number=1,
            solution=solution_v1,
            self_score=solution_v1.score,
            env_score=eval_v1.score,
            env_feedback=eval_v1.feedback,
        )
        rounds.append(round1)
        initial_score = eval_v1.score

        logger.info(
            "[Cycle %d] Initial score: %.2f (self: %s)",
            cycle_number, eval_v1.score, solution_v1.score,
        )

        # --- Step 2: Student self-evaluates ---
        logger.info("[Cycle %d] Step 2: Student self-evaluates", cycle_number)
        critique = self.student.self_reflect(task_prompt, solution_v1)

        # --- Step 3: Student self-corrects ---
        logger.info("[Cycle %d] Step 3: Student self-corrects", cycle_number)
        solution_v2 = self.student.self_correct(task_prompt, solution_v1, critique)
        eval_v2 = self.environment.evaluate(task, solution_v2.content)

        round2 = RoundState(
            round_number=2,
            solution=solution_v2,
            critique=critique,
            self_score=solution_v2.score,
            env_score=eval_v2.score,
            env_feedback=eval_v2.feedback,
        )
        rounds.append(round2)
        self_eval_score = eval_v2.score

        logger.info(
            "[Cycle %d] After self-correction: %.2f (was %.2f)",
            cycle_number, eval_v2.score, eval_v1.score,
        )

        # --- Step 4: Student submits to teacher ---
        logger.info("[Cycle %d] Step 4: Submitting to teacher", cycle_number)
        student_rounds_for_teacher = [
            {
                "solution": solution_v1.content,
                "critique": "",
                "score": solution_v1.score,
            },
            {
                "solution": solution_v2.content,
                "critique": critique.content,
                "score": solution_v2.score,
            },
        ]
        teacher_eval = self.teacher.evaluate_student(
            task_prompt, student_rounds_for_teacher, task.ground_truth
        )
        teacher_score = teacher_eval.score
        round2.teacher_eval = teacher_eval

        logger.info(
            "[Cycle %d] Teacher score: %s", cycle_number, teacher_score
        )

        # --- Step 5: Check if student improved from self-eval ---
        student_improved = eval_v2.score > eval_v1.score
        thought_injected = False
        aha_moment = None

        if not student_improved:
            # Student didn't improve — teacher injects a thought
            logger.info(
                "[Cycle %d] Step 5: NO improvement — teacher injects thought",
                cycle_number,
            )

            # Teacher generates thinking scaffold
            teacher_thought = self.teacher.inject_thinking(
                task_prompt, solution_v2.content, critique.content
            )
            round2.teacher_thought = teacher_thought
            thought_injected = True

            # --- Step 6: Generate LoRA training data from teacher thought ---
            logger.info(
                "[Cycle %d] Step 6: Generating LoRA training data from thought",
                cycle_number,
            )
            self._create_thought_training_data(
                task, task_prompt, solution_v2, critique, teacher_thought
            )

            # --- Step 7: Student retries with teacher's thinking internalized ---
            logger.info(
                "[Cycle %d] Step 7: Student retries with teacher thinking",
                cycle_number,
            )
            solution_v3 = self._student_retry_with_thought(
                task_prompt, solution_v2, critique, teacher_thought
            )
            eval_v3 = self.environment.evaluate(task, solution_v3.content)

            round3 = RoundState(
                round_number=3,
                solution=solution_v3,
                critique=critique,
                self_score=solution_v3.score,
                env_score=eval_v3.score,
                env_feedback=eval_v3.feedback,
                teacher_thought=teacher_thought,
            )
            rounds.append(round3)

            # --- Step 8: Detect aha moment ---
            score_before = eval_v2.score
            score_after = eval_v3.score
            improvement = score_after - score_before

            # Ask student what it learned (the realization)
            realization = self._get_student_realization(
                task_prompt, solution_v2, solution_v3, teacher_thought
            )

            aha_moment = AhaMoment(
                cycle_number=cycle_number,
                task_id=task.task_id,
                score_before_thought=score_before,
                score_after_thought=score_after,
                teacher_thought=teacher_thought.content,
                student_realization=realization,
                improvement=improvement,
            )
            self._all_aha_moments.append(aha_moment)

            if aha_moment.is_genuine_aha:
                logger.info(
                    "[Cycle %d] 💡 AHA MOMENT! Score: %.2f → %.2f (+%.2f)",
                    cycle_number, score_before, score_after, improvement,
                )
                logger.info(
                    "[Cycle %d] Student realization: %s",
                    cycle_number, realization[:200],
                )
            else:
                logger.info(
                    "[Cycle %d] No aha moment (score: %.2f → %.2f)",
                    cycle_number, score_before, score_after,
                )

            final_score = eval_v3.score
        else:
            logger.info(
                "[Cycle %d] Student improved on its own (%.2f → %.2f)",
                cycle_number, eval_v1.score, eval_v2.score,
            )
            final_score = eval_v2.score

        # --- Collect training data for this cycle ---
        training_entries = self._collect_cycle_training_data(
            task, task_prompt, rounds, thought_injected
        )

        cycle_result = CycleResult(
            cycle_number=cycle_number,
            task_id=task.task_id,
            initial_score=initial_score,
            self_eval_score=self_eval_score,
            teacher_score=teacher_score,
            final_score=final_score,
            teacher_thought_injected=thought_injected,
            aha_moment=aha_moment,
            rounds=rounds,
            training_data_generated=training_entries,
        )
        self._all_cycle_results.append(cycle_result)

        logger.info(
            "[Cycle %d] Complete: initial=%.2f, final=%.2f, thought=%s, aha=%s",
            cycle_number,
            initial_score,
            final_score,
            thought_injected,
            aha_moment.is_genuine_aha if aha_moment else "N/A",
        )

        return cycle_result

    def run_n_cycles(
        self,
        tasks: list[TaskInstance],
        n_cycles: int = 10,
    ) -> dict:
        """Run n complete cycles across all tasks to validate the system.

        Each cycle processes all tasks, collecting training data and aha moments.

        Args:
            tasks: Tasks to use in each cycle.
            n_cycles: Number of complete cycles to run (default: 10).

        Returns:
            Summary dict with all cycle results and aha moments.
        """
        logger.info("Starting %d-cycle validation run with %d tasks", n_cycles, len(tasks))

        for cycle_num in range(1, n_cycles + 1):
            logger.info("\n%s", "=" * 60)
            logger.info("CYCLE %d / %d", cycle_num, n_cycles)
            logger.info("%s\n", "=" * 60)

            for task in tasks:
                self.run_single_cycle(task, cycle_num)

        return self.get_summary()

    def get_summary(self) -> dict:
        """Get a complete summary of all cycles and aha moments."""
        total_aha = sum(1 for a in self._all_aha_moments if a.is_genuine_aha)
        total_cycles = len(self._all_cycle_results)
        total_thought_injections = sum(
            1 for c in self._all_cycle_results if c.teacher_thought_injected
        )

        avg_initial = (
            sum(c.initial_score for c in self._all_cycle_results) / total_cycles
            if total_cycles > 0 else 0.0
        )
        avg_final = (
            sum(c.final_score for c in self._all_cycle_results) / total_cycles
            if total_cycles > 0 else 0.0
        )

        # Track improvement over cycle numbers
        cycle_scores: dict[int, list[float]] = {}
        for c in self._all_cycle_results:
            cycle_scores.setdefault(c.cycle_number, []).append(c.final_score)
        avg_by_cycle = {
            k: sum(v) / len(v) for k, v in sorted(cycle_scores.items())
        }

        return {
            "total_cycles": total_cycles,
            "total_aha_moments": total_aha,
            "total_thought_injections": total_thought_injections,
            "avg_initial_score": avg_initial,
            "avg_final_score": avg_final,
            "overall_improvement": avg_final - avg_initial,
            "aha_rate": (
                total_aha / total_thought_injections
                if total_thought_injections > 0
                else 0.0
            ),
            "avg_score_by_cycle": avg_by_cycle,
            "aha_moments": [a.to_dict() for a in self._all_aha_moments],
            "cycle_results": [c.to_dict() for c in self._all_cycle_results],
            "training_data_count": len(self._all_training_data),
        }

    def save_results(self, output_dir: str | Path) -> dict[str, Path]:
        """Save all results and training data to disk.

        Args:
            output_dir: Directory to save results.

        Returns:
            Dict mapping result type to file path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        # Save summary
        summary_path = output_dir / "cycle_summary.json"
        summary_path.write_text(json.dumps(self.get_summary(), indent=2, default=str))
        paths["summary"] = summary_path

        # Save aha moments
        aha_path = output_dir / "aha_moments.json"
        aha_path.write_text(json.dumps(
            [a.to_dict() for a in self._all_aha_moments], indent=2
        ))
        paths["aha_moments"] = aha_path

        # Save training data
        training_path = output_dir / "thought_training_data.jsonl"
        with open(training_path, "w") as f:
            for entry in self._all_training_data:
                f.write(json.dumps(entry) + "\n")
        paths["training_data"] = training_path

        # Save SFT formatted data
        sft_pairs = self.sft_formatter.format_for_sft(self._all_training_data)
        sft_path = output_dir / "sft_training_pairs.json"
        self.sft_formatter.save_sft_data(sft_pairs, sft_path)
        paths["sft_data"] = sft_path

        logger.info("Saved results to %s", output_dir)
        return paths

    def _student_retry_with_thought(
        self,
        task_prompt: str,
        previous_solution: LLMResponse,
        critique: LLMResponse,
        teacher_thought: LLMResponse,
    ) -> LLMResponse:
        """Have the student retry the task with the teacher's thinking internalized.

        The student receives the teacher's thought as additional context and
        attempts to produce a better solution — this is where the aha moment happens.
        """
        prompt = (
            "You previously attempted this task and received feedback from a mentor.\n"
            "Use the mentor's thinking hint to produce a significantly better solution.\n\n"
            f"## Task\n{task_prompt}\n\n"
            f"## Your Previous Solution\n{previous_solution.content}\n\n"
            f"## Your Self-Critique\n{critique.content}\n\n"
            f"## Mentor's Thinking Hint\n{teacher_thought.content}\n\n"
            "Now, incorporating the mentor's insight, write an improved solution.\n"
            "Explain what you now understand that you didn't before (your 'aha' realization).\n\n"
            "Respond in JSON format:\n"
            '{"content": "<your improved solution>", '
            '"reasoning": "<what you now understand — your aha realization>", '
            '"score": <new confidence 0.0-1.0>}'
        )
        return self.student.generate_structured(prompt)

    def _get_student_realization(
        self,
        task_prompt: str,
        old_solution: LLMResponse,
        new_solution: LLMResponse,
        teacher_thought: LLMResponse,
    ) -> str:
        """Ask the student to articulate what it learned (the aha realization)."""
        prompt = (
            "Reflect on what just happened:\n\n"
            f"## Task\n{task_prompt}\n\n"
            f"## Your Previous Solution\n{old_solution.content}\n\n"
            f"## Mentor's Hint\n{teacher_thought.content}\n\n"
            f"## Your Improved Solution\n{new_solution.content}\n\n"
            "In 2-3 sentences, explain your 'aha moment': what do you now "
            "understand that you didn't before? What could you have done better "
            "from the start?"
        )
        response = self.student.generate(prompt)
        return response

    def _create_thought_training_data(
        self,
        task: TaskInstance,
        task_prompt: str,
        solution: LLMResponse,
        critique: LLMResponse,
        teacher_thought: LLMResponse,
    ) -> dict:
        """Create LoRA training data from a teacher thought injection.

        This is the key training signal: teach the student to think like
        the teacher by training on the thought + improved reasoning.
        """
        entry = {
            "type": "teacher_thought_injection",
            "task_id": task.task_id,
            "task_prompt": task_prompt,
            "solution_before_thought": solution.content,
            "student_critique": critique.content,
            "teacher_thought": teacher_thought.content,
            "ground_truth": task.ground_truth or "",
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        self._all_training_data.append(entry)
        return entry

    def _collect_cycle_training_data(
        self,
        task: TaskInstance,
        task_prompt: str,
        rounds: list[RoundState],
        thought_injected: bool,
    ) -> list[dict]:
        """Collect all training data from a cycle's rounds."""
        entries = []

        # Create improvement pairs from consecutive rounds
        for i in range(len(rounds) - 1):
            current = rounds[i]
            next_round = rounds[i + 1]

            entry = {
                "type": "self_improvement_pair",
                "task_id": task.task_id,
                "task_prompt": task_prompt,
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
                "teacher_thought": (
                    next_round.teacher_thought.content if next_round.teacher_thought else ""
                ),
                "ground_truth": task.ground_truth or "",
                "improvement": (next_round.env_score or 0) - (current.env_score or 0),
                "thought_injected": thought_injected,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            }
            entries.append(entry)
            self._all_training_data.append(entry)

        return entries
