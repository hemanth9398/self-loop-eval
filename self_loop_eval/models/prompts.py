"""Prompt templates for the self-loop-eval system."""

# ---------------------------------------------------------------------------
# Student prompts
# ---------------------------------------------------------------------------

STUDENT_SYSTEM_PROMPT = (
    "You are a diligent student model. You solve problems carefully, show your "
    "reasoning step-by-step, and are honest about what you don't know. When asked "
    "to self-evaluate, you critically examine your own work and identify genuine "
    "mistakes and improvements — not just surface-level rephrasing."
)

TASK_SOLVE_PROMPT = (
    "Solve the following task. Show your complete reasoning.\n\n"
    "{task}\n\n"
    "Respond in JSON format:\n"
    '{{"content": "<your solution code or answer>", '
    '"reasoning": "<your step-by-step reasoning>", '
    '"score": <self-assessed confidence 0.0-1.0>}}'
)

SELF_REFLECT_PROMPT = (
    "You previously solved a task. Now evaluate your own solution.\n\n"
    "## Original Task\n{task}\n\n"
    "## Your Solution\n{solution}\n\n"
    "## Your Reasoning\n{reasoning}\n\n"
    "Critically evaluate your solution:\n"
    "1. What did you do well?\n"
    "2. What mistakes did you make?\n"
    "3. What could be improved?\n"
    "4. Are there edge cases you missed?\n"
    "5. Is your approach optimal?\n\n"
    "Respond in JSON format:\n"
    '{{"content": "<your detailed self-critique>", '
    '"reasoning": "<analysis of strengths and weaknesses>", '
    '"score": <quality score 0.0-1.0>}}'
)

SELF_CORRECT_PROMPT = (
    "Based on your self-evaluation, produce an improved solution.\n\n"
    "## Original Task\n{task}\n\n"
    "## Previous Solution\n{solution}\n\n"
    "## Previous Reasoning\n{reasoning}\n\n"
    "## Self-Critique\n{critique}\n\n"
    "Now write an improved solution that addresses the issues you identified.\n"
    "Explain what you changed and why.\n\n"
    "Respond in JSON format:\n"
    '{{"content": "<your improved solution>", '
    '"reasoning": "<what you changed and why>", '
    '"score": <new confidence score 0.0-1.0>}}'
)

# ---------------------------------------------------------------------------
# Teacher prompts
# ---------------------------------------------------------------------------

TEACHER_SYSTEM_PROMPT = (
    "You are an expert teacher model. You evaluate student work fairly and provide "
    "helpful guidance. You do NOT give direct answers — instead, you provide thinking "
    "strategies, hints, and scaffolding that help the student discover the answer "
    "themselves. You assess whether the student is genuinely improving or just "
    "rephrasing the same ideas."
)

TEACHER_EVALUATE_PROMPT = (
    "Evaluate the student's work across all self-evaluation rounds.\n\n"
    "## Task\n{task}\n\n"
    "## Student Rounds\n{student_rounds}\n"
    "{ground_truth}\n\n"
    "Provide:\n"
    "1. Independent quality assessment\n"
    "2. What the student missed in self-evaluation\n"
    "3. Gaps between the student's self-assessment and actual quality\n"
    "4. Overall score\n\n"
    "Respond in JSON format:\n"
    '{{"content": "<your evaluation>", '
    '"reasoning": "<detailed analysis>", '
    '"score": <quality score 0.0-1.0>}}'
)

TEACHER_THINKING_PROMPT = (
    "The student is stuck and needs a thinking scaffold — NOT the answer.\n\n"
    "## Task\n{task}\n\n"
    "## Student's Current Solution\n{solution}\n\n"
    "## Student's Self-Critique\n{critique}\n\n"
    "Provide thinking hints that help the student discover the right approach:\n"
    "- Suggest strategies or algorithms to consider\n"
    "- Point out perspectives the student hasn't tried\n"
    "- Ask guiding questions\n"
    "- Do NOT provide the actual solution code\n\n"
    "Respond in JSON format:\n"
    '{{"content": "<thinking hints and guiding questions>", '
    '"reasoning": "<why these hints will help>", '
    '"score": null}}'
)

TEACHER_COMPARE_PROMPT = (
    "Compare the student's first and final solutions to assess improvement.\n\n"
    "## Task\n{task}\n\n"
    "## First Solution (Round 1)\n{first_solution}\n\n"
    "## Final Solution (Last Round)\n{final_solution}\n"
    "{ground_truth}\n\n"
    "Determine:\n"
    "1. Is the student genuinely improving or just rephrasing?\n"
    "2. What specific improvements were made?\n"
    "3. What issues remain?\n"
    "4. Improvement trajectory score (0.0 = no improvement, 1.0 = perfect improvement)\n\n"
    "Respond in JSON format:\n"
    '{{"content": "<comparison analysis>", '
    '"reasoning": "<detailed trajectory analysis>", '
    '"score": <improvement score 0.0-1.0>}}'
)
