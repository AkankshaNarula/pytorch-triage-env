"""
RFC 004-compliant rubric system for pytorch_triage_env.

Two rubric components:
1. TrajectoryRubric: dense per-step rewards for investigative behavior
2. LLMJudgeRubric: terminal reward multiplier based on explanation quality

The real openenv.core.rubrics.LLMJudge API (from official release notes):
  from openenv.core.rubrics import LLMJudge
  judge = LLMJudge(
      prompt_template="...",
      client=openai_client,
  )
  score = await judge(action, observation)   # float in [0, 1]

We implement graceful fallback for environments where rubrics aren't available.
"""
from __future__ import annotations
import os
from typing import Optional

# RFC 004 Rubric import — three-level graceful fallback
try:
    from openenv.core.rubrics import LLMJudge as _OpenEnvLLMJudge
    _JUDGE_CLASS_AVAILABLE = True
except ImportError:
    try:
        from openenv.core.rubrics.base import LLMJudge as _OpenEnvLLMJudge
        _JUDGE_CLASS_AVAILABLE = True
    except ImportError:
        _JUDGE_CLASS_AVAILABLE = False
        _OpenEnvLLMJudge = None

# Base Rubric class
try:
    from openenv.core.rubrics import Rubric as _RubricBase
except ImportError:
    try:
        from openenv.core.rubrics.base import Rubric as _RubricBase
    except ImportError:
        class _RubricBase:
            """Duck-type fallback Rubric base class."""
            def forward(self, action, observation) -> float:
                raise NotImplementedError


# ── Per-step signal file map ──────────────────────────────────────────────────

_SIGNAL_FILES = {
    "oom_graph_leak":          ["train.py"],
    "fsdp_collective_deadlock":["train.py"],
    "compile_graph_break":     ["train.py"],
    "ddp_gradient_hang":       ["train.py", "model.py"],
}

_DIAGNOSTIC_COMMANDS = {
    "oom_graph_leak":          None,
    "fsdp_collective_deadlock":None,
    "compile_graph_break":     "TORCH_LOGS=dynamo",
    "ddp_gradient_hang":       None,
}


class TrajectoryRubric(_RubricBase):
    """
    Dense per-step reward shaping.

    ReadFileAction:
      +0.08  reading a signal file (directly relevant to the bug)
      +0.02  reading any other file (general investigation)

    ExecuteBashAction:
      +0.12  running with the correct diagnostic flag for this task
      +0.05  running training (python train.py / torchrun ...) — shows verification
      -0.03  running with a SyntaxError (Python can't even parse the edited file)
      -0.01  running an unrecognized command

    EditFileAction:
      +0.05  making any edit (attempt)
      -0.08  introducing a Python SyntaxError

    ViewGitDiffAction:
      +0.02  tracking changes (good engineering practice)

    SubmitFixAction:
      0.00  (terminal reward handled by LLMJudgeRubric + verify_fix in environment)
    """

    def forward(self, action, observation) -> float:
        action_type  = getattr(action, "action_type", None)
        task_name    = getattr(observation, "task_name", "")
        run_status   = getattr(observation, "run_status", "not_run")
        signal_files = _SIGNAL_FILES.get(task_name, [])
        diag_flag    = _DIAGNOSTIC_COMMANDS.get(task_name)

        if action_type == "read_file":
            filename = getattr(action, "filename", "")
            return 0.08 if filename in signal_files else 0.02

        if action_type == "edit_file":
            new_str = getattr(action, "new_str", "")
            if self._has_syntax_error(getattr(action, "filename", ""), new_str):
                return -0.08
            return 0.05

        if action_type == "execute_bash":
            command = getattr(action, "command", "")
            if self._has_syntax_error_in_output(run_status):
                return -0.03
            if diag_flag and diag_flag in command:
                return 0.12   # used the right diagnostic flag
            if "python" in command or "torchrun" in command:
                return 0.05
            return -0.01

        if action_type == "view_git_diff":
            return 0.02

        if action_type == "submit_fix":
            return 0.0   # handled by terminal rubric

        return 0.0

    def _has_syntax_error(self, filename: str, new_str: str) -> bool:
        if not filename.endswith(".py"):
            return False
        try:
            compile(new_str, filename, "exec")
            return False
        except SyntaxError:
            return True

    def _has_syntax_error_in_output(self, run_status: str) -> bool:
        return run_status == "syntax_error"


# ── LLM Judge Rubric ──────────────────────────────────────────────────────────

_JUDGE_PROMPT = """\
You are evaluating an ML engineer's explanation of a PyTorch training bug fix.

TASK: {task_name}
FIX APPLIED: {diff_summary}
ENGINEER'S EXPLANATION: {explanation}

Rate the technical depth of the explanation on a scale of 0.0 to 1.0:

1.0 — Complete understanding:
  - Correctly identifies ROOT CAUSE (not just symptom)
  - Explains WHY the fix works at the framework/CUDA/compiler level
  - Mentions how to prevent this class of bug in future

0.7 — Good understanding:
  - Identifies the correct root cause
  - Explains what changed and approximately why it helps
  - Some technical depth

0.5 — Surface understanding:
  - Identifies that something was wrong and changed it
  - But explains only WHAT changed, not WHY it was wrong
  - Equivalent to "I changed X to Y and it worked"

0.2 — Minimal understanding:
  - Vague or incorrect explanation
  - Does not understand why the original code was wrong

0.0 — No understanding / incorrect:
  - Wrong explanation of the bug
  - Could not explain the fix at all

Respond with ONLY a decimal number between 0.0 and 1.0. Nothing else."""


class LLMJudgeRubric:
    """
    Terminal reward component based on explanation quality.

    Uses openenv.core.rubrics.LLMJudge if available.
    Falls back to a keyword-based heuristic judge if not.

    Final terminal reward formula:
      base_score = 1.0 (if fix verified) or partial_credit
      judge_multiplier = LLMJudge score (0.5 to 1.0)
      terminal_reward = base_score * judge_multiplier
    """

    JUDGE_MODEL    = os.getenv("JUDGE_MODEL")
    JUDGE_API_KEY  = os.getenv("HF_TOKEN") or os.getenv("JUDGE_API_KEY", "")
    JUDGE_API_BASE = os.getenv("JUDGE_API_BASE", "https://router.huggingface.co/v1")

    def score_explanation(
        self,
        task_name: str,
        explanation: str,
        diff_summary: str,
    ) -> float:
        """
        Returns a float in [0.0, 1.0] rating the quality of the fix explanation.
        Uses LLMJudge if JUDGE_MODEL is set, otherwise falls back to keyword heuristic.
        """
        if self.JUDGE_MODEL and explanation:
            return self._llm_judge(task_name, explanation, diff_summary)
        return self._keyword_judge(task_name, explanation)

    def _llm_judge(self, task_name: str, explanation: str, diff_summary: str) -> float:
        try:
            from openai import OpenAI
            client = OpenAI(base_url=self.JUDGE_API_BASE, api_key=self.JUDGE_API_KEY)
            prompt = _JUDGE_PROMPT.format(
                task_name=task_name,
                diff_summary=diff_summary[:500],
                explanation=explanation[:1000],
            )
            resp = client.chat.completions.create(
                model=self.JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=8,
            )
            score_str = (resp.choices[0].message.content or "0.5").strip()
            score = float(score_str)
            return round(max(0.0, min(1.0, score)), 3)
        except Exception:
            return 0.5   # fallback to neutral on any error

    def _keyword_judge(self, task_name: str, explanation: str) -> float:
        """Keyword-based heuristic judge when LLM is not available."""
        explanation_lower = explanation.lower()
        deep_keywords = {
            "oom_graph_leak":           ["computation graph", "grad_fn", "tensor reference", "garbage collect", "detach"],
            "fsdp_collective_deadlock": ["collective operation", "all ranks", "nccl", "barrier", "rank guard", "broadcast"],
            "compile_graph_break":      ["graph break", "dynamo", "data-dependent", "recompilation", "eager mode", "python scalar"],
            "ddp_gradient_hang":        ["find_unused_parameters", "gradient sync", "unused param", "backward hook", "conditional branch"],
        }
        shallow_keywords = {
            "oom_graph_leak":           [".item()", "memory", "oom"],
            "fsdp_collective_deadlock": ["if rank", "all_reduce", "moved"],
            "compile_graph_break":      ["compiler.disable", "graph", "compile"],
            "ddp_gradient_hang":        ["find_unused", "true", "ddp"],
        }

        task_deep    = deep_keywords.get(task_name, [])
        task_shallow = shallow_keywords.get(task_name, [])

        deep_hits    = sum(1 for kw in task_deep    if kw in explanation_lower)
        shallow_hits = sum(1 for kw in task_shallow if kw in explanation_lower)

        if deep_hits >= 2:    return 1.0
        if deep_hits == 1:    return 0.7
        if shallow_hits >= 2: return 0.5
        if shallow_hits >= 1: return 0.3
        return 0.2
