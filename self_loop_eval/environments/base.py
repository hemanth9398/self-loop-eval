"""Abstract base classes for task environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class TaskInstance:
    """A single task for the student to solve."""

    task_id: str
    description: str
    input_data: dict = field(default_factory=dict)
    ground_truth: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result of evaluating a solution against ground truth."""

    score: float  # 0.0 to 1.0
    passed: bool
    feedback: str
    details: dict = field(default_factory=dict)


class TaskEnvironment(ABC):
    """Abstract base for task environments.

    A task environment defines a domain (coding, prediction, reasoning, etc.)
    and provides methods to load tasks and evaluate solutions objectively.
    """

    @property
    @abstractmethod
    def domain(self) -> str:
        """Return the domain name (e.g., 'coding', 'prediction')."""

    @abstractmethod
    def load_tasks(self) -> list[TaskInstance]:
        """Load and return all available tasks."""

    @abstractmethod
    def evaluate(self, task: TaskInstance, solution: str) -> EvalResult:
        """Evaluate a student's solution against the task's ground truth.

        Args:
            task: The task instance with ground truth.
            solution: The student's solution string.

        Returns:
            EvalResult with score, pass/fail, and feedback.
        """

    def get_task_prompt(self, task: TaskInstance) -> str:
        """Format a task into a prompt string for the student model.

        Args:
            task: The task instance.

        Returns:
            A formatted prompt string.
        """
        return task.description
