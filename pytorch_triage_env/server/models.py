"""
Pydantic v2 models for pytorch_triage_env.

Key design decisions:
- TriageAction is a Pydantic v2 discriminated union on action_type
- TriageObservation does NOT declare done or reward (inherited from openenv)
- All action subtypes are strict, typed, and self-documenting
"""
from __future__ import annotations
from typing import Annotated, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    # Fallback stubs when openenv is not installed
    class Action(BaseModel):  # type: ignore[no-redef]
        action_type: str = ""

    class Observation(BaseModel):  # type: ignore[no-redef]
        done: bool = False
        reward: float = 0.0

    class State(BaseModel):  # type: ignore[no-redef]
        episode_id: str = ""
        step_count: int = 0


# ── Action sub-types ──────────────────────────────────────────────────────────

class ReadFileAction(Action):
    """Read a file from the virtual workspace."""
    action_type: Literal["read_file"] = "read_file"
    filename: str = Field(
        description="Filename to read. One of: train.py, model.py, config.py, data_loader.py"
    )


class EditFileAction(Action):
    """
    Apply a targeted string replacement to a file.
    old_str MUST match file content exactly — copy it verbatim from read_file output.
    """
    action_type: Literal["edit_file"] = "edit_file"
    filename: str = Field(description="File to edit")
    old_str: str = Field(description="Exact string to replace (must match file content character-for-character)")
    new_str: str = Field(description="Replacement string")


class ExecuteBashAction(Action):
    """
    Run a command against the mock execution engine.
    Examples: 'python train.py', 'torchrun --nproc_per_node=4 train.py',
              'TORCH_LOGS=dynamo python train.py', 'python -c "import torch; print(torch.__version__)"'
    """
    action_type: Literal["execute_bash"] = "execute_bash"
    command: str = Field(
        description="Shell command to execute against the mock engine"
    )


class ViewGitDiffAction(Action):
    """Show a unified diff of all changes made so far versus the original files."""
    action_type: Literal["view_git_diff"] = "view_git_diff"
    filename: Optional[str] = Field(
        default=None,
        description="Show diff for a specific file (None = all changed files)"
    )


class SubmitFixAction(Action):
    """
    Submit the final fix. Triggers full verification by mock engine and LLMJudge rubric.
    The explanation field is evaluated by the LLMJudge — explain WHY the bug occurred,
    not just what you changed. Shallow explanations receive a 0.5× score multiplier.
    """
    action_type: Literal["submit_fix"] = "submit_fix"
    explanation: str = Field(
        description=(
            "Detailed technical explanation of: (1) the root cause of the bug, "
            "(2) why the fix works, (3) how to prevent this class of bug in future. "
            "This is evaluated by an LLM judge — shallow explanations are penalized."
        )
    )


# ── Discriminated union ───────────────────────────────────────────────────────

TriageAction = Annotated[
    Union[
        ReadFileAction,
        EditFileAction,
        ExecuteBashAction,
        ViewGitDiffAction,
        SubmitFixAction,
    ],
    Field(discriminator="action_type"),
]


# ── Observation ───────────────────────────────────────────────────────────────

class TriageObservation(Observation):
    """
    Agent observation at each step.

    CRITICAL: `done` and `reward` are INHERITED from openenv.core.env_server.types.Observation.
    DO NOT re-declare them here. Pydantic v2 will raise a field conflict.
    """
    task_name: str = Field(
        description="Current task: oom_graph_leak | fsdp_collective_deadlock | compile_graph_break | ddp_gradient_hang"
    )
    task_description: str = Field(
        description="High-level description of the failing training run (the 'incident report')"
    )
    terminal_output: str = Field(
        description="Output from the last action (file content, bash output, diff, or error trace)"
    )
    current_files: Dict[str, str] = Field(
        description="Current state of all workspace files (may have been edited)"
    )
    run_status: Literal["not_run", "failing", "partial", "passing", "diagnostic", "syntax_error"] = Field(
        default="not_run",
        description="Result of the last ExecuteBashAction run"
    )
    system_status: str = Field(
        default="idle",
        description="System status: idle | running | training_failed | training_passed"
    )
    step_number: int = Field(description="Current step (1-indexed)")
    max_steps: int = Field(description="Maximum steps this episode")
    budget_remaining: int = Field(description="Steps remaining")
    actions_taken: List[str] = Field(
        default_factory=list,
        description="List of action_types taken so far"
    )
    hint: Optional[str] = Field(
        default=None,
        description="Diagnostic hint (appears after 3+ failed ExecuteBash actions)"
    )
    instructions: str = Field(description="Task-specific agent instructions")


# ── State ─────────────────────────────────────────────────────────────────────

class TriageState(State):
    """
    Episode state. Extends openenv State (provides episode_id and step_count).
    """
    task_name: str = Field(default="oom_graph_leak")
    run_status: str = Field(default="not_run")
    files_edited: List[str] = Field(default_factory=list)
    cumulative_reward: float = Field(default=0.0)
    fix_submitted: bool = Field(default=False)
    fix_verified: bool = Field(default=False)
    bash_run_count: int = Field(default=0)
    llmjudge_score: Optional[float] = Field(default=None)
