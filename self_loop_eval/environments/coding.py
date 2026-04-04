"""Coding task environment — evaluates code solutions via test cases."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from self_loop_eval.environments.base import EvalResult, TaskEnvironment, TaskInstance


@dataclass
class TestCase:
    """A single test case for a coding problem."""

    input: str
    expected_output: str
    description: str = ""


@dataclass
class CodingTask(TaskInstance):
    """A coding task with test cases."""

    test_cases: list[TestCase] = field(default_factory=list)
    language: str = "python"
    function_name: str = ""


class CodingTaskEnvironment(TaskEnvironment):
    """Environment for coding tasks evaluated by running test cases.

    Tasks are loaded from a JSON file with the following structure:
    ```json
    [
        {
            "task_id": "two_sum",
            "description": "Write a function ...",
            "function_name": "two_sum",
            "test_cases": [
                {"input": "[2,7,11,15], 9", "expected_output": "[0, 1]"}
            ],
            "ground_truth": "def two_sum(nums, target): ..."
        }
    ]
    ```
    """

    def __init__(self, tasks_file: str | Path | None = None, tasks: list[dict] | None = None):
        """Initialize with either a tasks file path or a list of task dicts.

        Args:
            tasks_file: Path to a JSON file containing tasks.
            tasks: A list of task dictionaries (used if tasks_file is None).
        """
        self._tasks_file = Path(tasks_file) if tasks_file else None
        self._raw_tasks = tasks or []

    @property
    def domain(self) -> str:
        return "coding"

    def load_tasks(self) -> list[CodingTask]:
        """Load coding tasks from the JSON file or raw task list."""
        raw: list[dict] = []
        if self._tasks_file and self._tasks_file.exists():
            raw = json.loads(self._tasks_file.read_text())
        elif self._raw_tasks:
            raw = self._raw_tasks
        else:
            return self._default_tasks()

        tasks: list[CodingTask] = []
        for item in raw:
            test_cases = [
                TestCase(
                    input=tc["input"],
                    expected_output=tc["expected_output"],
                    description=tc.get("description", ""),
                )
                for tc in item.get("test_cases", [])
            ]
            tasks.append(
                CodingTask(
                    task_id=item["task_id"],
                    description=item["description"],
                    input_data=item.get("input_data", {}),
                    ground_truth=item.get("ground_truth"),
                    metadata=item.get("metadata", {}),
                    test_cases=test_cases,
                    language=item.get("language", "python"),
                    function_name=item.get("function_name", ""),
                )
            )
        return tasks

    def evaluate(self, task: TaskInstance, solution: str) -> EvalResult:
        """Evaluate a coding solution by running it against test cases.

        Args:
            task: A CodingTask instance with test cases.
            solution: The student's code as a string.

        Returns:
            EvalResult with score based on fraction of test cases passed.
        """
        if not isinstance(task, CodingTask):
            return EvalResult(
                score=0.0,
                passed=False,
                feedback="Task is not a CodingTask instance.",
            )

        if not task.test_cases:
            return EvalResult(
                score=0.0,
                passed=False,
                feedback="No test cases defined for this task.",
            )

        passed_count = 0
        total = len(task.test_cases)
        details: list[dict] = []

        for i, tc in enumerate(task.test_cases):
            result = self._run_test_case(solution, task.function_name, tc)
            details.append(result)
            if result["passed"]:
                passed_count += 1

        score = passed_count / total if total > 0 else 0.0
        all_passed = passed_count == total

        feedback_parts = []
        for i, d in enumerate(details):
            status = "✓" if d["passed"] else "✗"
            feedback_parts.append(
                f"Test {i + 1} {status}: {d.get('message', '')}"
            )

        return EvalResult(
            score=score,
            passed=all_passed,
            feedback="\n".join(feedback_parts),
            details={"test_results": details, "passed": passed_count, "total": total},
        )

    def get_task_prompt(self, task: TaskInstance) -> str:
        """Format a coding task into a prompt."""
        if not isinstance(task, CodingTask):
            return task.description

        prompt = f"## Coding Task\n\n{task.description}\n\n"
        if task.function_name:
            prompt += f"Write a Python function named `{task.function_name}`.\n\n"
        if task.test_cases:
            prompt += "### Examples\n\n"
            for i, tc in enumerate(task.test_cases[:3]):  # Show up to 3 examples
                prompt += f"**Example {i + 1}:**\n"
                prompt += f"- Input: `{tc.input}`\n"
                prompt += f"- Expected Output: `{tc.expected_output}`\n\n"
        prompt += (
            "Provide your solution as a Python function. "
            "Include your reasoning as comments in the code.\n"
        )
        return prompt

    def _run_test_case(
        self, solution: str, function_name: str, test_case: TestCase
    ) -> dict:
        """Run a single test case against the student's solution.

        Args:
            solution: The student's code.
            function_name: The function to call.
            test_case: The test case to run.

        Returns:
            Dict with 'passed', 'message', 'actual_output', 'expected_output'.
        """
        test_code = self._build_test_script(solution, function_name, test_case)

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as f:
                f.write(test_code)
                f.flush()
                tmp_path = f.name

            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=10,
            )

            Path(tmp_path).unlink(missing_ok=True)

            if result.returncode == 0:
                actual = result.stdout.strip()
                expected = test_case.expected_output.strip()
                if actual == expected:
                    return {
                        "passed": True,
                        "message": "Output matches expected.",
                        "actual_output": actual,
                        "expected_output": expected,
                    }
                else:
                    return {
                        "passed": False,
                        "message": f"Expected '{expected}', got '{actual}'.",
                        "actual_output": actual,
                        "expected_output": expected,
                    }
            else:
                return {
                    "passed": False,
                    "message": f"Runtime error: {result.stderr.strip()[:500]}",
                    "actual_output": None,
                    "expected_output": test_case.expected_output,
                }
        except subprocess.TimeoutExpired:
            Path(tmp_path).unlink(missing_ok=True)
            return {
                "passed": False,
                "message": "Execution timed out (10s limit).",
                "actual_output": None,
                "expected_output": test_case.expected_output,
            }
        except Exception as e:
            return {
                "passed": False,
                "message": f"Error running test: {e}",
                "actual_output": None,
                "expected_output": test_case.expected_output,
            }

    @staticmethod
    def _build_test_script(
        solution: str, function_name: str, test_case: TestCase
    ) -> str:
        """Build a Python script that runs the student's code and prints the result."""
        return (
            f"{solution}\n\n"
            f"if __name__ == '__main__':\n"
            f"    result = {function_name}({test_case.input})\n"
            f"    print(result)\n"
        )

    @staticmethod
    def _default_tasks() -> list[CodingTask]:
        """Return a set of built-in default coding tasks."""
        return [
            CodingTask(
                task_id="two_sum",
                description=(
                    "Given a list of integers `nums` and an integer `target`, "
                    "return the indices of the two numbers that add up to `target`. "
                    "You may assume each input has exactly one solution."
                ),
                function_name="two_sum",
                test_cases=[
                    TestCase(
                        input="[2, 7, 11, 15], 9",
                        expected_output="[0, 1]",
                    ),
                    TestCase(
                        input="[3, 2, 4], 6",
                        expected_output="[1, 2]",
                    ),
                    TestCase(
                        input="[3, 3], 6",
                        expected_output="[0, 1]",
                    ),
                ],
                ground_truth=(
                    "def two_sum(nums, target):\n"
                    "    seen = {}\n"
                    "    for i, n in enumerate(nums):\n"
                    "        comp = target - n\n"
                    "        if comp in seen:\n"
                    "            return [seen[comp], i]\n"
                    "        seen[n] = i\n"
                    "    return []\n"
                ),
            ),
            CodingTask(
                task_id="fizzbuzz",
                description=(
                    "Write a function `fizzbuzz(n)` that returns a list of strings from 1 to n. "
                    "For multiples of 3 use 'Fizz', for multiples of 5 use 'Buzz', "
                    "for multiples of both use 'FizzBuzz', otherwise the number as a string."
                ),
                function_name="fizzbuzz",
                test_cases=[
                    TestCase(
                        input="5",
                        expected_output="['1', '2', 'Fizz', '4', 'Buzz']",
                    ),
                    TestCase(
                        input="15",
                        expected_output=(
                            "['1', '2', 'Fizz', '4', 'Buzz', 'Fizz', '7', '8', "
                            "'Fizz', 'Buzz', '11', 'Fizz', '13', '14', 'FizzBuzz']"
                        ),
                    ),
                ],
                ground_truth=(
                    "def fizzbuzz(n):\n"
                    "    result = []\n"
                    "    for i in range(1, n + 1):\n"
                    "        if i % 15 == 0:\n"
                    "            result.append('FizzBuzz')\n"
                    "        elif i % 3 == 0:\n"
                    "            result.append('Fizz')\n"
                    "        elif i % 5 == 0:\n"
                    "            result.append('Buzz')\n"
                    "        else:\n"
                    "            result.append(str(i))\n"
                    "    return result\n"
                ),
            ),
            CodingTask(
                task_id="reverse_string",
                description=(
                    "Write a function `reverse_string(s)` that returns the reversed version "
                    "of the input string `s`."
                ),
                function_name="reverse_string",
                test_cases=[
                    TestCase(input="'hello'", expected_output="olleh"),
                    TestCase(input="'Python'", expected_output="nohtyP"),
                    TestCase(input="''", expected_output=""),
                ],
                ground_truth=(
                    "def reverse_string(s):\n"
                    "    return s[::-1]\n"
                ),
            ),
        ]
