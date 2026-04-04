"""Task environments for self-loop-eval."""

from self_loop_eval.environments.base import EvalResult, TaskEnvironment, TaskInstance
from self_loop_eval.environments.coding import CodingTaskEnvironment
from self_loop_eval.environments.real_world_problems import get_real_world_problems

__all__ = [
    "TaskEnvironment",
    "TaskInstance",
    "EvalResult",
    "CodingTaskEnvironment",
    "get_real_world_problems",
]
