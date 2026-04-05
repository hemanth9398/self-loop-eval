"""Tests for the aha moment cycle engine — the complete self-eval → teacher → train loop."""


import pytest

from self_loop_eval.config import EvalLoopConfig, ModelConfig, SystemConfig, TrainingConfig
from self_loop_eval.environments.coding import CodingTask, CodingTaskEnvironment
from self_loop_eval.environments.real_world_problems import get_real_world_problems
from self_loop_eval.eval_loop.aha_cycle import AhaMoment, AhaMomentEngine, CycleResult
from self_loop_eval.eval_loop.convergence import ConvergenceDetector
from self_loop_eval.eval_loop.round_state import LoopResult, RoundState
from self_loop_eval.metrics.tracker import MetricsTracker
from self_loop_eval.models.base import LLMResponse
from self_loop_eval.models.student import StudentModel
from self_loop_eval.models.teacher import TeacherModel
from self_loop_eval.rl.rewards import RewardFunction
from self_loop_eval.training.data_collector import TrainingDataCollector
from self_loop_eval.training.sft_formatter import SFTFormatter

# ---------------------------------------------------------------------------
# Mock models — simulate LLM behavior without API calls
# ---------------------------------------------------------------------------

class MockStudentModel(StudentModel):
    """Mock student that returns controlled responses for testing.

    Simulates a student that:
    - Round 1: Produces a partially correct solution
    - After self-reflection: Recognizes some issues
    - After self-correction: Improves slightly (or not, to trigger teacher)
    - After teacher thought: Produces a significantly better solution
    """

    def __init__(self, improve_on_self_correct: bool = False):
        config = ModelConfig(provider="mock", model_name="mock-student")
        super().__init__(config)
        self._improve_on_self_correct = improve_on_self_correct
        self._call_count = 0

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        self._call_count += 1
        if "aha moment" in prompt.lower() or "realization" in prompt.lower():
            return (
                "I now realize I should have used a hash map for O(1) lookups "
                "instead of a nested loop. The teacher's hint about data structures "
                "made me see the inefficiency in my approach."
            )
        return '{"content": "mock response", "reasoning": "mock reasoning", "score": 0.5}'

    def generate_structured(self, prompt: str, system_prompt: str = "") -> LLMResponse:
        self._call_count += 1

        # Detect what kind of prompt this is
        prompt_lower = prompt.lower()

        if "mentor's thinking hint" in prompt_lower or "mentor's insight" in prompt_lower:
            # Student retry with teacher thought — should be MUCH better
            return LLMResponse(
                content=self._get_improved_solution(),
                reasoning=(
                    "After the mentor's hint, I realized I should "
                    "use a hash map. This gives O(1) lookups."
                ),
                score=0.9,
            )
        elif "improved solution" in prompt_lower or "self-evaluation" in prompt_lower:
            # Self-correction
            if self._improve_on_self_correct:
                return LLMResponse(
                    content=self._get_improved_solution(),
                    reasoning="I fixed the edge cases I missed.",
                    score=0.8,
                )
            else:
                # No improvement — should trigger teacher
                return LLMResponse(
                    content=self._get_initial_solution(),
                    reasoning="I tried to improve but couldn't find a better approach.",
                    score=0.5,
                )
        elif "evaluate your solution" in prompt_lower or "critically evaluate" in prompt_lower:
            # Self-reflection
            return LLMResponse(
                content=(
                    "My solution handles basic cases but may miss "
                    "edge cases. Time complexity could be better."
                ),
                reasoning="I see potential issues with empty inputs and duplicates.",
                score=0.5,
            )
        else:
            # Initial solve
            return LLMResponse(
                content=self._get_initial_solution(),
                reasoning="Using a brute force approach with nested loops.",
                score=0.6,
            )

    def solve_task(self, task_prompt: str) -> LLMResponse:
        self._call_count += 1
        return LLMResponse(
            content=self._get_initial_solution(),
            reasoning="Initial brute force approach.",
            score=0.6,
        )

    def self_reflect(self, task_prompt: str, solution: LLMResponse) -> LLMResponse:
        self._call_count += 1
        return LLMResponse(
            content=(
                "My solution uses O(n^2) time. I could use a "
                "hash map for O(n). Edge cases might fail."
            ),
            reasoning="Analyzing time complexity and edge cases.",
            score=0.5,
        )

    def self_correct(
        self, task_prompt: str, solution: LLMResponse, critique: LLMResponse
    ) -> LLMResponse:
        self._call_count += 1
        if self._improve_on_self_correct:
            return LLMResponse(
                content=self._get_improved_solution(),
                reasoning="Fixed the approach based on self-critique.",
                score=0.8,
            )
        return LLMResponse(
            content=self._get_initial_solution(),
            reasoning="Tried to improve but stuck on the same approach.",
            score=0.5,
        )

    @staticmethod
    def _get_initial_solution() -> str:
        # Deliberately buggy: doesn't sort first, so unsorted inputs fail
        return (
            "def merge_intervals(intervals):\n"
            "    if not intervals:\n"
            "        return []\n"
            "    merged = [intervals[0]]\n"
            "    for i in range(1, len(intervals)):\n"
            "        if intervals[i][0] <= merged[-1][1]:\n"
            "            merged[-1][1] = max(merged[-1][1], intervals[i][1])\n"
            "        else:\n"
            "            merged.append(intervals[i])\n"
            "    return merged\n"
        )

    @staticmethod
    def _get_improved_solution() -> str:
        return (
            "def merge_intervals(intervals):\n"
            "    if not intervals:\n"
            "        return []\n"
            "    intervals.sort(key=lambda x: x[0])\n"
            "    merged = [intervals[0]]\n"
            "    for start, end in intervals[1:]:\n"
            "        if start <= merged[-1][1]:\n"
            "            merged[-1][1] = max(merged[-1][1], end)\n"
            "        else:\n"
            "            merged.append([start, end])\n"
            "    return merged\n"
        )


class MockTeacherModel(TeacherModel):
    """Mock teacher that returns controlled evaluation and thinking scaffolds."""

    def __init__(self):
        config = ModelConfig(provider="mock", model_name="mock-teacher")
        super().__init__(config)

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        return '{"content": "teacher response", "reasoning": "teacher reasoning", "score": 0.7}'

    def evaluate_student(
        self, task_prompt: str, student_rounds: list[dict], ground_truth: str | None = None
    ) -> LLMResponse:
        return LLMResponse(
            content="The student shows understanding but misses the optimal sort-first approach.",
            reasoning="Student uses O(n^2), optimal is O(n log n) with sorting.",
            score=0.6,
        )

    def inject_thinking(
        self, task_prompt: str, student_solution: str, student_critique: str
    ) -> LLMResponse:
        return LLMResponse(
            content=(
                "Consider sorting the intervals by start time first. "
                "Then you only need a single pass to merge overlapping intervals. "
                "Think about: what property makes merging easy after sorting?"
            ),
            reasoning="Guiding toward the sort-first strategy.",
            score=None,
        )

    def compare_rounds(
        self, task_prompt: str, first: str, final: str, ground_truth: str | None = None
    ) -> LLMResponse:
        return LLMResponse(
            content="Student moved from brute force to sorted approach — genuine improvement.",
            reasoning="The core algorithm changed, not just rephrasing.",
            score=0.8,
        )


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _make_merge_intervals_task() -> CodingTask:
    """Return the merge_intervals task from real-world problems."""
    problems = get_real_world_problems()
    return next(p for p in problems if p.task_id == "merge_intervals")


def _make_config() -> SystemConfig:
    return SystemConfig(
        eval_loop=EvalLoopConfig(max_rounds=5, enable_teacher=True),
        training=TrainingConfig(training_data_path="/tmp/self_loop_eval_test/training"),
    )


# ---------------------------------------------------------------------------
# Tests: AhaMoment dataclass
# ---------------------------------------------------------------------------

class TestAhaMoment:
    def test_genuine_aha_on_improvement(self):
        aha = AhaMoment(
            cycle_number=1,
            task_id="test",
            score_before_thought=0.3,
            score_after_thought=0.8,
            teacher_thought="Use sorting",
            student_realization="I should sort first!",
            improvement=0.5,
        )
        assert aha.is_genuine_aha is True

    def test_no_aha_on_no_improvement(self):
        aha = AhaMoment(
            cycle_number=1,
            task_id="test",
            score_before_thought=0.5,
            score_after_thought=0.5,
            teacher_thought="Use sorting",
            student_realization="I'm not sure what changed.",
            improvement=0.0,
        )
        assert aha.is_genuine_aha is False

    def test_no_aha_on_regression(self):
        aha = AhaMoment(
            cycle_number=1,
            task_id="test",
            score_before_thought=0.6,
            score_after_thought=0.4,
            teacher_thought="Use sorting",
            student_realization="Something went wrong.",
            improvement=-0.2,
        )
        assert aha.is_genuine_aha is False

    def test_serialization(self):
        aha = AhaMoment(
            cycle_number=2,
            task_id="merge",
            score_before_thought=0.3,
            score_after_thought=0.9,
            teacher_thought="Sort first",
            student_realization="Aha!",
            improvement=0.6,
        )
        d = aha.to_dict()
        assert d["cycle_number"] == 2
        assert d["is_genuine_aha"] is True
        assert d["improvement"] == 0.6


# ---------------------------------------------------------------------------
# Tests: AhaMomentEngine — the complete cycle
# ---------------------------------------------------------------------------

class TestAhaMomentEngine:
    """Tests for the complete aha moment cycle."""

    def test_single_cycle_no_improvement_triggers_teacher(self):
        """When student doesn't improve, teacher thought is injected."""
        config = _make_config()
        student = MockStudentModel(improve_on_self_correct=False)
        teacher = MockTeacherModel()
        env = CodingTaskEnvironment()
        task = _make_merge_intervals_task()

        engine = AhaMomentEngine(config, student, teacher, env)
        result = engine.run_single_cycle(task, cycle_number=1)

        assert isinstance(result, CycleResult)
        assert result.teacher_thought_injected is True
        assert result.aha_moment is not None
        assert len(result.rounds) == 3  # initial + self-correct + retry with thought

    def test_single_cycle_student_improves_no_teacher(self):
        """When student improves on its own, no teacher intervention."""
        config = _make_config()
        student = MockStudentModel(improve_on_self_correct=True)
        teacher = MockTeacherModel()
        env = CodingTaskEnvironment()
        task = _make_merge_intervals_task()

        engine = AhaMomentEngine(config, student, teacher, env)
        result = engine.run_single_cycle(task, cycle_number=1)

        assert result.teacher_thought_injected is False
        assert result.aha_moment is None
        assert len(result.rounds) == 2  # initial + self-correct only

    def test_aha_moment_detected_on_teacher_intervention(self):
        """After teacher thought injection, aha moment is detected."""
        config = _make_config()
        student = MockStudentModel(improve_on_self_correct=False)
        teacher = MockTeacherModel()
        env = CodingTaskEnvironment()
        task = _make_merge_intervals_task()

        engine = AhaMomentEngine(config, student, teacher, env)
        result = engine.run_single_cycle(task, cycle_number=1)

        # The mock student improves after teacher thought
        assert result.aha_moment is not None
        assert result.aha_moment.is_genuine_aha is True
        assert result.aha_moment.score_after_thought > result.aha_moment.score_before_thought

    def test_training_data_generated(self):
        """Cycle generates training data for LoRA fine-tuning."""
        config = _make_config()
        student = MockStudentModel(improve_on_self_correct=False)
        teacher = MockTeacherModel()
        env = CodingTaskEnvironment()
        task = _make_merge_intervals_task()

        engine = AhaMomentEngine(config, student, teacher, env)
        result = engine.run_single_cycle(task, cycle_number=1)

        assert len(result.training_data_generated) > 0
        # Should have improvement pairs
        for entry in result.training_data_generated:
            assert "task_id" in entry
            assert "solution_v1" in entry or "teacher_thought" in entry

    def test_teacher_thought_in_training_data(self):
        """Teacher thought injection creates specific training entries."""
        config = _make_config()
        student = MockStudentModel(improve_on_self_correct=False)
        teacher = MockTeacherModel()
        env = CodingTaskEnvironment()
        task = _make_merge_intervals_task()

        engine = AhaMomentEngine(config, student, teacher, env)
        engine.run_single_cycle(task, cycle_number=1)

        # Check that thought training data exists
        thought_entries = [
            e for e in engine._all_training_data
            if e.get("type") == "teacher_thought_injection"
        ]
        assert len(thought_entries) >= 1
        assert thought_entries[0]["teacher_thought"] != ""

    def test_cycle_result_serialization(self):
        """CycleResult can be serialized to dict."""
        config = _make_config()
        student = MockStudentModel(improve_on_self_correct=False)
        teacher = MockTeacherModel()
        env = CodingTaskEnvironment()
        task = _make_merge_intervals_task()

        engine = AhaMomentEngine(config, student, teacher, env)
        result = engine.run_single_cycle(task, cycle_number=1)

        d = result.to_dict()
        assert d["cycle_number"] == 1
        assert d["task_id"] == "merge_intervals"
        assert "initial_score" in d
        assert "final_score" in d
        assert "aha_moment" in d

    def test_student_submits_to_teacher(self):
        """Student's work is explicitly submitted to teacher for evaluation."""
        config = _make_config()
        student = MockStudentModel(improve_on_self_correct=False)
        teacher = MockTeacherModel()
        env = CodingTaskEnvironment()
        task = _make_merge_intervals_task()

        engine = AhaMomentEngine(config, student, teacher, env)
        result = engine.run_single_cycle(task, cycle_number=1)

        # Teacher should have evaluated
        assert result.teacher_score is not None

    def test_get_summary(self):
        """Engine provides a complete summary after cycles."""
        config = _make_config()
        student = MockStudentModel(improve_on_self_correct=False)
        teacher = MockTeacherModel()
        env = CodingTaskEnvironment()
        task = _make_merge_intervals_task()

        engine = AhaMomentEngine(config, student, teacher, env)
        engine.run_single_cycle(task, cycle_number=1)
        engine.run_single_cycle(task, cycle_number=2)

        summary = engine.get_summary()
        assert summary["total_cycles"] == 2
        assert "total_aha_moments" in summary
        assert "avg_initial_score" in summary
        assert "avg_final_score" in summary
        assert "aha_rate" in summary


# ---------------------------------------------------------------------------
# Tests: 10-cycle validation run
# ---------------------------------------------------------------------------

class TestTenCycleValidation:
    """Test the complete 10-cycle validation run."""

    def test_run_10_cycles(self):
        """Run the complete 10-cycle validation and verify results."""
        config = _make_config()
        student = MockStudentModel(improve_on_self_correct=False)
        teacher = MockTeacherModel()
        env = CodingTaskEnvironment()

        # Use merge_intervals as the real-world problem
        tasks = [_make_merge_intervals_task()]

        engine = AhaMomentEngine(config, student, teacher, env)
        summary = engine.run_n_cycles(tasks, n_cycles=10)

        # Verify 10 cycles completed
        assert summary["total_cycles"] == 10

        # Verify aha moments were detected
        assert summary["total_aha_moments"] > 0

        # Verify thought injections happened (student never self-improves in this mock)
        assert summary["total_thought_injections"] == 10

        # Verify training data was generated
        assert summary["training_data_count"] > 0

        # Verify per-cycle scores exist
        assert len(summary["avg_score_by_cycle"]) == 10

    def test_run_10_cycles_with_self_improving_student(self):
        """Student that self-improves should have fewer teacher interventions."""
        config = _make_config()
        student = MockStudentModel(improve_on_self_correct=True)
        teacher = MockTeacherModel()
        env = CodingTaskEnvironment()

        tasks = [_make_merge_intervals_task()]
        engine = AhaMomentEngine(config, student, teacher, env)
        summary = engine.run_n_cycles(tasks, n_cycles=10)

        assert summary["total_cycles"] == 10
        # Self-improving student should have NO teacher interventions
        assert summary["total_thought_injections"] == 0
        assert summary["total_aha_moments"] == 0

    def test_run_10_cycles_multiple_tasks(self):
        """10 cycles across multiple real-world problems."""
        config = _make_config()
        student = MockStudentModel(improve_on_self_correct=False)
        teacher = MockTeacherModel()
        env = CodingTaskEnvironment()

        # Use 2 tasks to verify multi-task handling
        problems = get_real_world_problems()
        tasks = problems[:2]

        engine = AhaMomentEngine(config, student, teacher, env)
        summary = engine.run_n_cycles(tasks, n_cycles=10)

        # 10 cycles × 2 tasks = 20 total cycle results
        assert summary["total_cycles"] == 20


# ---------------------------------------------------------------------------
# Tests: Convergence detector
# ---------------------------------------------------------------------------

class TestConvergenceDetector:
    def test_no_convergence_first_round(self):
        config = EvalLoopConfig()
        detector = ConvergenceDetector(config)
        rounds = [
            RoundState(
                round_number=1,
                solution=LLMResponse(content="v1"),
                env_score=0.5,
            )
        ]
        assert detector.check_convergence(rounds) is False

    def test_convergence_on_similar_text(self):
        config = EvalLoopConfig(convergence_threshold=0.05)
        detector = ConvergenceDetector(config)
        rounds = [
            RoundState(
                round_number=1,
                solution=LLMResponse(content="def solve(x): return x + 1"),
                env_score=0.5,
            ),
            RoundState(
                round_number=2,
                solution=LLMResponse(content="def solve(x): return x + 1"),
                env_score=0.5,
            ),
        ]
        assert detector.check_convergence(rounds) is True

    def test_convergence_on_plateau(self):
        config = EvalLoopConfig(stuck_plateau_rounds=2, min_score_improvement=0.01)
        detector = ConvergenceDetector(config)

        rounds = []
        for i in range(4):
            rounds.append(
                RoundState(
                    round_number=i + 1,
                    solution=LLMResponse(content=f"solution v{i+1} unique text {i*100}"),
                    env_score=0.5,  # Same score = plateau
                )
            )
            if len(rounds) >= 2:
                if detector.check_convergence(rounds):
                    break

        assert detector.is_stuck(rounds) is True


# ---------------------------------------------------------------------------
# Tests: Training data collection & SFT formatting
# ---------------------------------------------------------------------------

class TestTrainingPipeline:
    def test_data_collector_basic(self):
        config = TrainingConfig(training_data_path="/tmp/self_loop_eval_test/td")
        collector = TrainingDataCollector(config)

        loop_result = LoopResult(
            task_id="test",
            task_prompt="Solve this",
            rounds=[
                RoundState(
                    round_number=1,
                    solution=LLMResponse(content="v1", reasoning="r1"),
                    env_score=0.3,
                    self_score=0.4,
                ),
                RoundState(
                    round_number=2,
                    solution=LLMResponse(content="v2", reasoning="r2"),
                    critique=LLMResponse(content="needs work"),
                    env_score=0.7,
                    self_score=0.6,
                ),
            ],
        )

        entries = collector.collect(loop_result)
        assert len(entries) == 1
        assert entries[0]["solution_v1"] == "v1"
        assert entries[0]["solution_v2"] == "v2"
        assert entries[0]["improvement"] == pytest.approx(0.4)

    def test_data_collector_single_round_skipped(self):
        config = TrainingConfig()
        collector = TrainingDataCollector(config)

        loop_result = LoopResult(
            task_id="test",
            task_prompt="Solve",
            rounds=[
                RoundState(round_number=1, solution=LLMResponse(content="v1")),
            ],
        )
        entries = collector.collect(loop_result)
        assert len(entries) == 0

    def test_sft_formatter_basic(self):
        formatter = SFTFormatter()
        training_data = [
            {
                "task_id": "test",
                "task_prompt": "Solve this problem",
                "round_pair": "1->2",
                "solution_v1": "v1 code",
                "reasoning_v1": "first try",
                "self_critique": "I made mistakes",
                "solution_v2": "v2 code",
                "reasoning_v2": "improved approach",
                "teacher_thought": "",
                "improvement": 0.3,
            }
        ]
        pairs = formatter.format_for_sft(training_data)
        assert len(pairs) == 1  # No teacher thought = only self-correction pair
        assert "instruction" in pairs[0]
        assert "response" in pairs[0]

    def test_sft_formatter_with_teacher_thought(self):
        formatter = SFTFormatter()
        training_data = [
            {
                "task_id": "test",
                "task_prompt": "Solve this",
                "round_pair": "1->2",
                "solution_v1": "v1",
                "reasoning_v1": "r1",
                "self_critique": "needs work",
                "solution_v2": "v2",
                "reasoning_v2": "r2",
                "teacher_thought": "Use a hash map for O(1) lookups",
                "improvement": 0.5,
            }
        ]
        pairs = formatter.format_for_sft(training_data)
        # Should have 2 pairs: self-correction + teacher thinking
        assert len(pairs) == 2
        teacher_pair = next(p for p in pairs if p["metadata"]["type"] == "teacher_thinking")
        assert "hash map" in teacher_pair["instruction"]


# ---------------------------------------------------------------------------
# Tests: Reward function
# ---------------------------------------------------------------------------

class TestRewardFunction:
    def test_positive_reward_on_improvement(self):
        from self_loop_eval.config import RLConfig
        rf = RewardFunction(RLConfig())

        result = LoopResult(
            task_id="test", task_prompt="solve",
            rounds=[
                RoundState(round_number=1, solution=LLMResponse(content="v1"), env_score=0.3),
                RoundState(round_number=2, solution=LLMResponse(content="v2"), env_score=0.8),
            ],
        )
        reward = rf.compute_reward(result)
        assert reward > 0

    def test_zero_reward_no_improvement(self):
        from self_loop_eval.config import RLConfig
        rf = RewardFunction(RLConfig())

        result = LoopResult(
            task_id="test", task_prompt="solve",
            rounds=[
                RoundState(round_number=1, solution=LLMResponse(content="v1"), env_score=0.5),
                RoundState(round_number=2, solution=LLMResponse(content="v2"), env_score=0.5),
            ],
        )
        reward = rf.compute_reward(result)
        assert reward == 0.0

    def test_teacher_alignment_reward(self):
        from self_loop_eval.config import RLConfig
        rf = RewardFunction(RLConfig())

        result = LoopResult(
            task_id="test", task_prompt="solve",
            rounds=[
                RoundState(
                    round_number=1, solution=LLMResponse(content="v1"),
                    env_score=0.5, self_score=0.5,
                    teacher_eval=LLMResponse(content="ok", score=0.5),
                ),
            ],
        )
        # Perfect alignment: self_score == teacher_score
        reward = rf._teacher_alignment_reward(result)
        assert reward == 1.0


# ---------------------------------------------------------------------------
# Tests: Metrics tracker
# ---------------------------------------------------------------------------

class TestMetricsTracker:
    def test_loop_result_from_dict_requires_core_fields(self):
        with pytest.raises(ValueError, match="task_id"):
            LoopResult.from_dict({"task_prompt": "solve"})

    def test_record_cycle_from_serialized_loop_result_preserves_metrics(self, tmp_path):
        from self_loop_eval.config import MetricsConfig

        metrics_dir = str(tmp_path / "metrics_serialized")
        tracker = MetricsTracker(MetricsConfig(metrics_dir=metrics_dir))

        original = LoopResult(
            task_id="t1",
            task_prompt="solve",
            converged=True,
            teacher_intervened=True,
            final_env_score=0.8,
            improvement=0.3,
            rounds=[
                RoundState(
                    round_number=1,
                    solution=LLMResponse(content="v1", score=0.6),
                    env_score=0.5,
                    self_score=0.6,
                ),
                RoundState(
                    round_number=2,
                    solution=LLMResponse(content="v2", score=0.7),
                    critique=LLMResponse(content="fix edge case"),
                    env_score=0.8,
                    self_score=0.7,
                    teacher_eval=LLMResponse(content="better", score=0.8),
                ),
            ],
        )

        restored = LoopResult.from_dict(original.to_dict())
        metrics = tracker.record_cycle([restored])

        assert metrics["improvement"]["avg_final_score"] == 0.8
        assert metrics["self_eval_accuracy"]["num_comparisons"] == 2
        assert metrics["per_task"]["t1"]["final_score"] == 0.8
        assert metrics["per_task"]["t1"]["num_rounds"] == 2
        assert metrics["per_task"]["t1"]["converged"] is True

    def test_record_cycle(self, tmp_path):
        from self_loop_eval.config import MetricsConfig
        metrics_dir = str(tmp_path / "metrics")
        tracker = MetricsTracker(MetricsConfig(metrics_dir=metrics_dir))

        results = [
            LoopResult(
                task_id="t1", task_prompt="solve",
                final_env_score=0.8, improvement=0.3,
                teacher_intervened=True,
                rounds=[
                    RoundState(
                        round_number=1, solution=LLMResponse(content="v1"),
                        env_score=0.5, self_score=0.6,
                    ),
                    RoundState(
                        round_number=2, solution=LLMResponse(content="v2"),
                        env_score=0.8, self_score=0.7,
                    ),
                ],
            ),
        ]

        metrics = tracker.record_cycle(results)
        assert metrics["cycle_number"] == 1
        assert metrics["improvement"]["avg_improvement"] > 0
        assert metrics["teacher_dependency"]["teacher_intervention_rate"] == 1.0

    def test_summary_no_cycles(self, tmp_path):
        from self_loop_eval.config import MetricsConfig
        metrics_dir = str(tmp_path / "metrics2")
        tracker = MetricsTracker(MetricsConfig(metrics_dir=metrics_dir))
        summary = tracker.get_summary()
        assert summary["cycles"] == 0
