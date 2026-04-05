"""CLI entry point for self-loop-eval."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from self_loop_eval.config import SystemConfig
from self_loop_eval.environments.coding import CodingTaskEnvironment
from self_loop_eval.eval_loop.loop import SelfEvalLoop
from self_loop_eval.metrics.tracker import MetricsTracker
from self_loop_eval.models.student import StudentModel
from self_loop_eval.models.teacher import TeacherModel
from self_loop_eval.training.scheduler import TrainingScheduler

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="self-loop-eval",
        description="Self-Evaluation & Self-Training Loop System",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a JSON config file",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command — single cycle
    run_parser = subparsers.add_parser("run", help="Run one training cycle")
    run_parser.add_argument(
        "--tasks-file",
        type=str,
        default=None,
        help="Path to a JSON file with coding tasks",
    )

    # Eval command — just run eval loop without training
    eval_parser = subparsers.add_parser("eval", help="Run eval loop only (no training)")
    eval_parser.add_argument(
        "--tasks-file",
        type=str,
        default=None,
        help="Path to a JSON file with coding tasks",
    )

    # Metrics command — show metrics summary
    subparsers.add_parser("metrics", help="Show metrics summary")

    # Continuous command
    cont_parser = subparsers.add_parser("continuous", help="Run continuous training")
    cont_parser.add_argument(
        "--interval",
        type=float,
        default=24.0,
        help="Hours between training cycles",
    )

    args = parser.parse_args(argv)

    # Load config
    config = _load_config(args.config)

    # Set up logging
    logging.basicConfig(
        level=getattr(logging, config.metrics.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if args.command == "run":
        _cmd_run(config, args)
    elif args.command == "eval":
        _cmd_eval(config, args)
    elif args.command == "metrics":
        _cmd_metrics(config)
    elif args.command == "continuous":
        _cmd_continuous(config, args)
    else:
        parser.print_help()
        sys.exit(1)


def _load_config(config_path: str | None) -> SystemConfig:
    """Load system config from a JSON file or use defaults."""
    if config_path:
        path = Path(config_path)
        if path.exists():
            data = json.loads(path.read_text())
            return SystemConfig.from_dict(data)
        else:
            logger.warning("Config file not found: %s, using defaults", config_path)
    return SystemConfig()


def _build_components(
    config: SystemConfig, tasks_file: str | None = None
) -> tuple[StudentModel, TeacherModel, CodingTaskEnvironment]:
    """Build the student, teacher, and environment."""
    student = StudentModel(config.student)
    teacher = TeacherModel(config.teacher)
    env = CodingTaskEnvironment(tasks_file=tasks_file)
    config.ensure_dirs()
    return student, teacher, env


def _cmd_run(config: SystemConfig, args: argparse.Namespace) -> None:
    """Run one complete training cycle."""
    student, teacher, env = _build_components(config, args.tasks_file)
    scheduler = TrainingScheduler(config, student, teacher, env)
    metrics_tracker = MetricsTracker(config.metrics)

    summary = scheduler.run_cycle()

    # Record metrics
    from self_loop_eval.eval_loop.round_state import LoopResult

    loop_results = [LoopResult.from_dict(r_data) for r_data in summary["results"]]

    metrics_tracker.record_cycle(loop_results)

    print(json.dumps(summary, indent=2, default=str))


def _cmd_eval(config: SystemConfig, args: argparse.Namespace) -> None:
    """Run eval loop only (no training)."""
    student, teacher, env = _build_components(config, args.tasks_file)
    eval_loop = SelfEvalLoop(config, student, teacher, env)

    results = eval_loop.run_all_tasks()
    for result in results:
        print(json.dumps(result.to_dict(), indent=2, default=str))


def _cmd_metrics(config: SystemConfig) -> None:
    """Show metrics summary."""
    tracker = MetricsTracker(config.metrics)
    summary = tracker.get_summary()
    print(json.dumps(summary, indent=2))


def _cmd_continuous(config: SystemConfig, args: argparse.Namespace) -> None:
    """Run continuous training."""
    student, teacher, env = _build_components(config)
    scheduler = TrainingScheduler(config, student, teacher, env)
    scheduler.run_continuous(interval_hours=args.interval)


if __name__ == "__main__":
    main()
