"""Self-evaluation loop — the core iterative improvement engine."""

from self_loop_eval.eval_loop.loop import SelfEvalLoop
from self_loop_eval.eval_loop.convergence import ConvergenceDetector
from self_loop_eval.eval_loop.round_state import RoundState, LoopResult

__all__ = ["SelfEvalLoop", "ConvergenceDetector", "RoundState", "LoopResult"]
