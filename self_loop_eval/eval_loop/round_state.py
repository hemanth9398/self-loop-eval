"""State tracking for each round of the self-evaluation loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from self_loop_eval.models.base import LLMResponse


@dataclass
class RoundState:
    """State of a single self-evaluation round.

    Captures the solution, self-critique, scores, and optional teacher
    intervention for one iteration of the loop.
    """

    round_number: int
    solution: LLMResponse
    critique: LLMResponse | None = None
    self_score: float | None = None
    env_score: float | None = None
    env_feedback: str = ""
    teacher_eval: LLMResponse | None = None
    teacher_thought: LLMResponse | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())

    def to_dict(self) -> dict:
        """Serialize the round state to a dictionary."""
        return {
            "round_number": self.round_number,
            "solution": self.solution.to_dict(),
            "critique": self.critique.to_dict() if self.critique else None,
            "self_score": self.self_score,
            "env_score": self.env_score,
            "env_feedback": self.env_feedback,
            "teacher_eval": self.teacher_eval.to_dict() if self.teacher_eval else None,
            "teacher_thought": (
                self.teacher_thought.to_dict() if self.teacher_thought else None
            ),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RoundState:
        """Deserialize a round state from a dictionary."""
        cls._validate_required_fields(data, "round state", ["round_number", "solution"])
        return cls(
            round_number=data["round_number"],
            solution=LLMResponse(**data["solution"]),
            critique=LLMResponse(**data["critique"]) if data.get("critique") else None,
            self_score=data.get("self_score"),
            env_score=data.get("env_score"),
            env_feedback=data.get("env_feedback", ""),
            teacher_eval=(
                LLMResponse(**data["teacher_eval"]) if data.get("teacher_eval") else None
            ),
            teacher_thought=(
                LLMResponse(**data["teacher_thought"])
                if data.get("teacher_thought")
                else None
            ),
            timestamp=data.get("timestamp", ""),
        )

    @staticmethod
    def _validate_required_fields(
        data: dict, label: str, fields: list[str]
    ) -> None:
        missing = [field for field in fields if field not in data]
        if missing:
            missing_fields = ", ".join(sorted(missing))
            raise ValueError(f"Missing required {label} fields: {missing_fields}")


@dataclass
class LoopResult:
    """Complete result of a self-evaluation loop for one task.

    Contains all rounds, the final outcome, and summary metrics.
    """

    task_id: str
    task_prompt: str
    rounds: list[RoundState] = field(default_factory=list)
    converged: bool = False
    teacher_intervened: bool = False
    final_env_score: float = 0.0
    improvement: float = 0.0
    ground_truth: str | None = None
    started_at: str = field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())
    completed_at: str | None = None

    @property
    def num_rounds(self) -> int:
        return len(self.rounds)

    @property
    def first_score(self) -> float | None:
        if self.rounds:
            return self.rounds[0].env_score
        return None

    @property
    def final_score(self) -> float | None:
        if self.rounds:
            return self.rounds[-1].env_score
        return None

    def compute_improvement(self) -> float:
        """Compute the score improvement from first to last round."""
        first = self.first_score
        final = self.final_score
        if first is not None and final is not None:
            self.improvement = final - first
        return self.improvement

    def to_dict(self) -> dict:
        """Serialize the loop result to a dictionary."""
        return {
            "task_id": self.task_id,
            "task_prompt": self.task_prompt,
            "rounds": [r.to_dict() for r in self.rounds],
            "converged": self.converged,
            "teacher_intervened": self.teacher_intervened,
            "final_env_score": self.final_env_score,
            "improvement": self.improvement,
            "ground_truth": self.ground_truth,
            "num_rounds": self.num_rounds,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> LoopResult:
        """Deserialize a loop result from a dictionary."""
        RoundState._validate_required_fields(data, "loop result", ["task_id", "task_prompt"])
        return cls(
            task_id=data["task_id"],
            task_prompt=data["task_prompt"],
            rounds=[RoundState.from_dict(r) for r in data.get("rounds", [])],
            converged=data.get("converged", False),
            teacher_intervened=data.get("teacher_intervened", False),
            final_env_score=data.get("final_env_score", 0.0),
            improvement=data.get("improvement", 0.0),
            ground_truth=data.get("ground_truth"),
            started_at=data.get("started_at", ""),
            completed_at=data.get("completed_at"),
        )
