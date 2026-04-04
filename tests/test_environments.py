"""Tests for the coding task environment and real-world problems."""

from self_loop_eval.environments.coding import CodingTask, CodingTaskEnvironment, TestCase
from self_loop_eval.environments.real_world_problems import get_real_world_problems


class TestCodingTaskEnvironment:
    """Tests for CodingTaskEnvironment."""

    def test_default_tasks_load(self):
        env = CodingTaskEnvironment()
        tasks = env.load_tasks()
        assert len(tasks) >= 3
        assert all(isinstance(t, CodingTask) for t in tasks)

    def test_domain_is_coding(self):
        env = CodingTaskEnvironment()
        assert env.domain == "coding"

    def test_evaluate_correct_solution(self):
        env = CodingTaskEnvironment()
        task = CodingTask(
            task_id="add",
            description="Add two numbers",
            function_name="add",
            test_cases=[
                TestCase(input="2, 3", expected_output="5"),
                TestCase(input="0, 0", expected_output="0"),
            ],
        )
        solution = "def add(a, b):\n    return a + b"
        result = env.evaluate(task, solution)
        assert result.score == 1.0
        assert result.passed is True

    def test_evaluate_wrong_solution(self):
        env = CodingTaskEnvironment()
        task = CodingTask(
            task_id="add",
            description="Add two numbers",
            function_name="add",
            test_cases=[
                TestCase(input="2, 3", expected_output="5"),
                TestCase(input="0, 0", expected_output="0"),
            ],
        )
        solution = "def add(a, b):\n    return a * b"
        result = env.evaluate(task, solution)
        assert result.score < 1.0
        assert result.passed is False

    def test_evaluate_partial_solution(self):
        env = CodingTaskEnvironment()
        task = CodingTask(
            task_id="abs_val",
            description="Return absolute value",
            function_name="abs_val",
            test_cases=[
                TestCase(input="5", expected_output="5"),
                TestCase(input="-3", expected_output="3"),
                TestCase(input="0", expected_output="0"),
            ],
        )
        # This works for positive but not negative
        solution = "def abs_val(x):\n    return x"
        result = env.evaluate(task, solution)
        # Should pass 2 out of 3 (5 and 0 but not -3)
        assert 0.0 < result.score < 1.0

    def test_evaluate_syntax_error(self):
        env = CodingTaskEnvironment()
        task = CodingTask(
            task_id="add",
            description="Add",
            function_name="add",
            test_cases=[TestCase(input="1, 2", expected_output="3")],
        )
        solution = "def add(a, b:\n    return a + b"  # syntax error
        result = env.evaluate(task, solution)
        assert result.score == 0.0
        assert result.passed is False

    def test_evaluate_timeout(self):
        env = CodingTaskEnvironment()
        task = CodingTask(
            task_id="infinite",
            description="Infinite loop",
            function_name="infinite",
            test_cases=[TestCase(input="", expected_output="done")],
        )
        solution = "def infinite():\n    while True: pass"
        result = env.evaluate(task, solution)
        assert result.score == 0.0
        assert result.passed is False

    def test_task_prompt_format(self):
        env = CodingTaskEnvironment()
        task = CodingTask(
            task_id="test",
            description="Test task description",
            function_name="solve",
            test_cases=[TestCase(input="1", expected_output="2")],
        )
        prompt = env.get_task_prompt(task)
        assert "Test task description" in prompt
        assert "solve" in prompt
        assert "Example" in prompt

    def test_load_from_tasks_list(self):
        tasks_data = [
            {
                "task_id": "custom_add",
                "description": "Add numbers",
                "function_name": "add",
                "test_cases": [
                    {"input": "1, 2", "expected_output": "3"},
                ],
            }
        ]
        env = CodingTaskEnvironment(tasks=tasks_data)
        tasks = env.load_tasks()
        assert len(tasks) == 1
        assert tasks[0].task_id == "custom_add"
        assert len(tasks[0].test_cases) == 1

    def test_no_test_cases(self):
        env = CodingTaskEnvironment()
        task = CodingTask(
            task_id="empty",
            description="No tests",
            function_name="noop",
            test_cases=[],
        )
        result = env.evaluate(task, "def noop(): pass")
        assert result.score == 0.0
        assert result.passed is False


class TestRealWorldProblems:
    """Tests for real-world coding problems — validate ground truths pass."""

    def setup_method(self):
        self.env = CodingTaskEnvironment()
        self.problems = get_real_world_problems()

    def test_problems_exist(self):
        assert len(self.problems) >= 5

    def test_all_have_test_cases(self):
        for p in self.problems:
            assert len(p.test_cases) >= 2, f"{p.task_id} needs >= 2 test cases"

    def test_all_have_ground_truth(self):
        for p in self.problems:
            assert p.ground_truth, f"{p.task_id} missing ground truth"

    def test_ground_truth_lru_cache(self):
        task = next(p for p in self.problems if p.task_id == "lru_cache")
        result = self.env.evaluate(task, task.ground_truth)
        assert result.passed, f"LRU cache ground truth failed: {result.feedback}"

    def test_ground_truth_rate_limiter(self):
        task = next(p for p in self.problems if p.task_id == "rate_limiter")
        result = self.env.evaluate(task, task.ground_truth)
        assert result.passed, f"Rate limiter ground truth failed: {result.feedback}"

    def test_ground_truth_merge_intervals(self):
        task = next(p for p in self.problems if p.task_id == "merge_intervals")
        result = self.env.evaluate(task, task.ground_truth)
        assert result.passed, f"Merge intervals ground truth failed: {result.feedback}"

    def test_ground_truth_flatten_dict(self):
        task = next(p for p in self.problems if p.task_id == "flatten_nested_dict")
        result = self.env.evaluate(task, task.ground_truth)
        assert result.passed, f"Flatten dict ground truth failed: {result.feedback}"

    def test_ground_truth_task_scheduler(self):
        task = next(p for p in self.problems if p.task_id == "task_scheduler")
        result = self.env.evaluate(task, task.ground_truth)
        assert result.passed, f"Task scheduler ground truth failed: {result.feedback}"

    def test_all_ground_truths_pass(self):
        """Meta-test: every real-world problem's ground truth must pass all its test cases."""
        for task in self.problems:
            result = self.env.evaluate(task, task.ground_truth)
            assert result.passed, (
                f"Ground truth for '{task.task_id}' failed:\n{result.feedback}"
            )
