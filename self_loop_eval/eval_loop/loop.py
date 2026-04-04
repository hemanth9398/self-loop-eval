"""The main self-evaluation loop — orchestrates student, teacher, and environment."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from self_loop_eval.config import SystemConfig
from self_loop_eval.environments.base import TaskEnvironment, TaskInstance
from self_loop_eval.eval_loop.convergence import ConvergenceDetector
from self_loop_eval.eval_loop.round_state import LoopResult, RoundState
from self_loop_eval.models.student import StudentModel
from self_loop_eval.models.teacher import TeacherModel

logger = logging.getLogger(__name__)


class SelfEvalLoop:
    """The core iterative self-evaluation loop.

    Flow per task:
    1. Student solves the task (Round 1)
    2. Environment evaluates the solution
    3. Student self-reflects (critique)
    4. Student self-corrects (improved solution)
    5. Repeat 2-4 until convergence or max rounds
    6. If stuck → teacher intervenes with thinking scaffold
    7. Collect all rounds into a LoopResult
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
        self.convergence = ConvergenceDetector(config.eval_loop)

    def run_task(self, task: TaskInstance) -> LoopResult:
        """Run the full self-eval loop for a single task.

        Args:
            task: The task instance to solve.

        Returns:
            LoopResult with all rounds and summary metrics.
        """
        self.convergence.reset()
        task_prompt = self.environment.get_task_prompt(task)

        result = LoopResult(
            task_id=task.task_id,
            task_prompt=task_prompt,
            ground_truth=task.ground_truth,
        )

        logger.info("Starting self-eval loop for task: %s", task.task_id)

        # Round 1: Initial solution
        solution = self.student.solve_task(task_prompt)
        env_eval = self.environment.evaluate(task, solution.content)

        round_state = RoundState(
            round_number=1,
            solution=solution,
            self_score=solution.score,
            env_score=env_eval.score,
            env_feedback=env_eval.feedback,
        )
        result.rounds.append(round_state)

        logger.info(
            "Round 1: env_score=%.2f, self_score=%s",
            env_eval.score,
            solution.score,
        )

        # If already perfect, no need to loop
        if env_eval.passed:
            logger.info("Task solved perfectly in Round 1!")
            result.final_env_score = env_eval.score
            result.completed_at = datetime.now(tz=timezone.utc).isoformat()
            return result

        # Rounds 2..N: Self-eval → self-correct loop
        for round_num in range(2, self.config.eval_loop.max_rounds + 1):
            # Self-reflect
            critique = self.student.self_reflect(task_prompt, solution)

            # Check if teacher intervention is needed
            if (
                self.config.eval_loop.enable_teacher
                and self.convergence.is_stuck(result.rounds)
            ):
                logger.info("Student is stuck — triggering teacher intervention")
                teacher_thought = self.teacher.inject_thinking(
                    task_prompt, solution.content, critique.content
                )
                # Append teacher thought to critique so student can use it
                critique.content += f"\n\n[Teacher Hint]: {teacher_thought.content}"
                result.teacher_intervened = True

                # Store teacher eval on the previous round
                prev_round = result.rounds[-1]
                prev_round.teacher_thought = teacher_thought

            # Self-correct
            solution = self.student.self_correct(task_prompt, solution, critique)

            # Evaluate new solution
            env_eval = self.environment.evaluate(task, solution.content)

            round_state = RoundState(
                round_number=round_num,
                solution=solution,
                critique=critique,
                self_score=solution.score,
                env_score=env_eval.score,
                env_feedback=env_eval.feedback,
            )
            result.rounds.append(round_state)

            logger.info(
                "Round %d: env_score=%.2f, self_score=%s",
                round_num,
                env_eval.score,
                solution.score,
            )

            # Check convergence
            if self.convergence.check_convergence(result.rounds):
                result.converged = True
                logger.info("Loop converged at round %d", round_num)
                break

            # If perfect, stop
            if env_eval.passed:
                logger.info("Task solved perfectly in Round %d!", round_num)
                break

        # Final teacher evaluation (if enabled)
        if self.config.eval_loop.enable_teacher:
            self._run_teacher_eval(result)

        # Finalize
        result.final_env_score = result.rounds[-1].env_score or 0.0
        result.compute_improvement()
        result.completed_at = datetime.now(tz=timezone.utc).isoformat()

        logger.info(
            "Loop complete: %d rounds, improvement=%.4f, teacher=%s",
            result.num_rounds,
            result.improvement,
            result.teacher_intervened,
        )

        return result

    def run_all_tasks(self) -> list[LoopResult]:
        """Run the self-eval loop for all tasks in the environment.

        Returns:
            List of LoopResults, one per task.
        """
        tasks = self.environment.load_tasks()
        results = []
        for task in tasks:
            result = self.run_task(task)
            results.append(result)
        return results

    def _run_teacher_eval(self, result: LoopResult) -> None:
        """Run teacher evaluation and comparison on completed rounds."""
        student_rounds = [
            {
                "solution": r.solution.content,
                "critique": r.critique.content if r.critique else "",
                "score": r.self_score,
            }
            for r in result.rounds
        ]

        teacher_eval = self.teacher.evaluate_student(
            result.task_prompt,
            student_rounds,
            result.ground_truth,
        )

        # Store teacher eval on the final round
        if result.rounds:
            result.rounds[-1].teacher_eval = teacher_eval

        # Run comparison between first and last
        if len(result.rounds) >= 2:
            self.teacher.compare_rounds(
                result.task_prompt,
                result.rounds[0].solution.content,
                result.rounds[-1].solution.content,
                result.ground_truth,
            )
