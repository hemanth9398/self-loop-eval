"""Task environments for self-loop-eval."""

from self_loop_eval.environments.base import TaskEnvironment, TaskInstance, EvalResult
from self_loop_eval.environments.coding import CodingTaskEnvironment

__all__ = ["TaskEnvironment", "TaskInstance", "EvalResult", "CodingTaskEnvironment"]
