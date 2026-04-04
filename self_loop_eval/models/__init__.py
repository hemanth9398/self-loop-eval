"""Model wrappers for student and teacher LLMs."""

from self_loop_eval.models.base import LLMWrapper
from self_loop_eval.models.student import StudentModel
from self_loop_eval.models.teacher import TeacherModel

__all__ = ["LLMWrapper", "StudentModel", "TeacherModel"]
