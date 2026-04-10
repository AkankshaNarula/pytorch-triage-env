"""
PyTorchTriageEnv: the core OpenEnv environment.
Dual-import pattern throughout (in-repo vs Docker).
"""
from __future__ import annotations
from uuid import uuid4
from typing import List, Optional

try:
    from ..server.models import (
        TriageObservation, TriageState,
        ReadFileAction, EditFileAction, ExecuteBashAction,
        ViewGitDiffAction, SubmitFixAction,
    )
    from .mock_execution_engine import MockExecutionEngine, TASK_CONFIGS
    from .rubrics import TrajectoryRubric, LLMJudgeRubric
    from .virtual_fs import VirtualFileSystem
except ImportError:
    from server.models import (
        TriageObservation, TriageState,
        ReadFileAction, EditFileAction, ExecuteBashAction,
        ViewGitDiffAction, SubmitFixAction,
    )
    from server.mock_execution_engine import MockExecutionEngine, TASK_CONFIGS
    from server.rubrics import TrajectoryRubric, LLMJudgeRubric
    from server.virtual_fs import VirtualFileSystem

try:
    from openenv.core.env_server.interfaces import Environment
except ImportError:
    class Environment:  # type: ignore[no-redef]
        """Stub Environment base class when openenv is not installed."""
        pass


_TASK_DESCRIPTIONS = {
    "oom_graph_leak": """\
Production alert: training job `resnet-imagenet-v3` is crashing at epoch 3 with CUDA OOM.
Memory usage grows each epoch even though the model size is constant.
The job was running fine last week. A team member updated train.py yesterday.
""",
    "fsdp_collective_deadlock": """\
Production alert: distributed training run `llm-sft-4node` has been frozen for 30 minutes.
All 4 processes appear alive (ps shows them running) but no output since step 0.
This is a 4×GPU job using FSDP. NCCL Watchdog timeout imminent.
""",
    "compile_graph_break": """\
Performance regression: training throughput on `vit-b16-imagenet` dropped 40%.
We enabled torch.compile last week expecting a 2.4× speedup.
Instead it's running slower than eager mode. No crashes. Loss is converging normally.
""",
    "ddp_gradient_hang": """\
Production alert: multi-GPU training job `multitask-cls-4gpu` hangs after step 0.
Uses DDP with 4 GPUs. The model has both a main head and an auxiliary head.
Training uses auxiliary head every 5th step. Job appears frozen since yesterday.
""",
}

_TASK_INSTRUCTIONS = {
    "oom_graph_leak": """\
Debug a CUDA OOM crash in a PyTorch training loop.
Files: train.py, model.py, config.py, data_loader.py
Actions: read_file, edit_file, execute_bash, view_git_diff, submit_fix

Strategy:
1. execute_bash("python train.py") to see the error trace
2. read_file("train.py") to examine the training loop
3. Find the accumulator that retains the computation graph
4. edit_file to apply the fix
5. execute_bash("python train.py") to verify
6. submit_fix with a deep explanation of WHY the bug causes OOM
""",
    "fsdp_collective_deadlock": """\
Debug an FSDP distributed training deadlock.
Files: train.py, model.py, config.py, data_loader.py
Actions: read_file, edit_file, execute_bash, view_git_diff, submit_fix

Strategy:
1. execute_bash("torchrun --nproc_per_node=4 train.py") to see the NCCL error
2. read_file("train.py") to examine the distributed training loop
3. Locate the collective operation that is behind a rank guard
4. edit_file to move it outside the conditional
5. Verify the fix
6. submit_fix explaining why collective ops cannot be inside rank conditionals
""",
    "compile_graph_break": """\
Debug a torch.compile performance regression (silent graph break).
Files: train.py, model.py, config.py, data_loader.py
Actions: read_file, edit_file, execute_bash, view_git_diff, submit_fix

Strategy:
1. execute_bash("python train.py") — confirm slower than eager
2. execute_bash("TORCH_LOGS=dynamo python train.py") — find the graph break
3. read_file("train.py") — locate the data-dependent branch
4. edit_file — apply @torch.compiler.disable or restructure
5. execute_bash("TORCH_LOGS=dynamo python train.py") — verify 0 graph breaks
6. submit_fix explaining data-dependent control flow and Dynamo limitations
""",
    "ddp_gradient_hang": """\
Debug a DDP gradient synchronization hang.
Files: train.py, model.py, config.py, data_loader.py
Actions: read_file, edit_file, execute_bash, view_git_diff, submit_fix

Strategy:
1. execute_bash("torchrun --nproc_per_node=4 train.py") — see the hang/timeout
2. read_file("model.py") — examine the forward method for conditional branches
3. read_file("train.py") — check DDP wrapper configuration
4. edit_file — add find_unused_parameters=True
5. Verify and submit with explanation of DDP gradient sync requirements
""",
}

_MAX_STEPS = {
    "oom_graph_leak":           8,
    "fsdp_collective_deadlock": 9,
    "compile_graph_break":      10,
    "ddp_gradient_hang":        9,
}


class PyTorchTriageEnv(Environment):
    """PyTorch Training Infrastructure Triage RL Environment."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, task_name: str = "oom_graph_leak"):
        self._task_name = task_name
        self._cfg = TASK_CONFIGS.get(task_name, TASK_CONFIGS["oom_graph_leak"])
        self._vfs = VirtualFileSystem(self._cfg["files"])
        self._engine = MockExecutionEngine()

        # RFC 004: rubrics assigned on the environment
        self.rubric       = TrajectoryRubric()
        self.judge_rubric = LLMJudgeRubric()

        self._step_count:       int   = 0
        self._done:             bool  = False
        self._run_status:       str   = "not_run"
        self._actions_taken:    List[str] = []
        self._cumulative_reward: float = 0.0
        self._fix_submitted:    bool  = False
        self._fix_verified:     bool  = False
        self._episode_id:       str   = str(uuid4())
        self._max_steps:        int   = _MAX_STEPS.get(task_name, 9)
        self._bash_run_count:   int   = 0
        self._llmjudge_score:   Optional[float] = None

    def reset(self, seed=None, episode_id=None, **kwargs) -> TriageObservation:
        task = kwargs.get("task", self._task_name)
        self._task_name          = task
        self._cfg                = TASK_CONFIGS.get(task, TASK_CONFIGS["oom_graph_leak"])
        self._vfs.reset(self._cfg["files"])
        self._step_count         = 0
        self._done               = False
        self._run_status         = "not_run"
        self._actions_taken      = []
        self._cumulative_reward  = 0.0
        self._fix_submitted      = False
        self._fix_verified       = False
        self._episode_id         = episode_id or str(uuid4())
        self._max_steps          = _MAX_STEPS.get(task, 9)
        self._bash_run_count     = 0
        self._llmjudge_score     = None

        return self._make_obs(
            reward=0.0,
            terminal_output=(
                f"PyTorch Triage Session initialized.\n"
                f"Task: {task}\n"
                f"Available files: {', '.join(sorted(self._cfg['files'].keys()))}\n\n"
                f"INCIDENT REPORT:\n{_TASK_DESCRIPTIONS.get(task, '')}\n"
                f"Run `execute_bash` with 'python train.py' to see the error."
            ),
        )

    def step(self, action) -> TriageObservation:
        """Execute action. Returns TriageObservation with done/reward set on it."""
        if self._done:
            return self._make_obs(reward=0.0, terminal_output="Episode complete.")

        self._step_count += 1
        action_type = getattr(action, "action_type", "unknown")
        self._actions_taken.append(action_type)
        terminal_output = ""

        # ── Dispatch action ────────────────────────────────────────────────

        if isinstance(action, ReadFileAction):
            terminal_output = self._vfs.read(action.filename)

        elif isinstance(action, EditFileAction):
            result = self._vfs.edit(action.filename, action.old_str, action.new_str)
            terminal_output = result
            if "ERROR:" in result:
                self._run_status = "syntax_error"
            else:
                self._run_status = "not_run"   # reset — need to run to verify

        elif isinstance(action, ExecuteBashAction):
            self._bash_run_count += 1
            status, output = self._engine.simulate(
                self._task_name, self._vfs.files, action.command
            )
            self._run_status = status
            terminal_output  = output

        elif isinstance(action, ViewGitDiffAction):
            terminal_output = self._vfs.git_diff(action.filename)

        elif isinstance(action, SubmitFixAction):
            verified = self._engine.verify_fix(self._task_name, self._vfs.files)
            self._fix_submitted = True
            self._fix_verified  = verified
            self._done          = True
            terminal_output = (
                f"{'✓ Fix verified!' if verified else '✗ Fix not verified.'}\n"
                + (self._cfg["success_trace"] if verified else self._cfg["buggy_trace"][:400])
            )

        # Auto-terminate at max steps
        if self._step_count >= self._max_steps and not self._done:
            self._done = True
            terminal_output += f"\n\n[Max steps ({self._max_steps}) reached — episode terminated]"

        # ── Compute rewards ─────────────────────────────────────────────────

        obs = self._make_obs(reward=0.0, terminal_output=terminal_output)
        step_reward = self.rubric.forward(action, obs)

        # Terminal reward component
        if isinstance(action, SubmitFixAction):
            diff_summary = self._vfs.git_diff()
            judge_score  = self.judge_rubric.score_explanation(
                self._task_name,
                getattr(action, "explanation", ""),
                diff_summary,
            )
            self._llmjudge_score = judge_score

            if self._fix_verified:
                # Base: 1.0 × judge multiplier (floor 0.5 so correct fix always scores >0)
                judge_multiplier = max(0.5, judge_score)
                step_reward += 1.0 * judge_multiplier
                # Efficiency bonus
                efficiency   = max(0.0, 1.0 - self._step_count / self._max_steps)
                step_reward += 0.1 * efficiency
            else:
                # Partial credit based on fix signature matches
                sigs = self._cfg.get("fix_signatures", [])
                if sigs:
                    matched      = sum(
                        1 for fname, sig in sigs
                        if sig in self._vfs.files.get(fname, "")
                    )
                    partial_base = 0.30 * (matched / len(sigs))
                    step_reward += partial_base * max(0.3, judge_score)

        self._cumulative_reward += step_reward
        return self._make_obs(reward=step_reward, terminal_output=terminal_output)

    @property
    def state(self) -> TriageState:
        return TriageState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_name=self._task_name,
            run_status=self._run_status,
            files_edited=self._vfs.changed_files,
            cumulative_reward=self._cumulative_reward,
            fix_submitted=self._fix_submitted,
            fix_verified=self._fix_verified,
            bash_run_count=self._bash_run_count,
            llmjudge_score=self._llmjudge_score,
        )

    def _make_obs(self, reward: float, terminal_output: str) -> TriageObservation:
        budget = max(0, self._max_steps - self._step_count)
        hint   = None
        if self._bash_run_count >= 3 and self._run_status == "failing":
            hint = self._cfg.get("hint")
        return TriageObservation(
            task_name=self._task_name,
            task_description=_TASK_DESCRIPTIONS.get(self._task_name, ""),
            terminal_output=terminal_output,
            current_files=self._vfs.files,
            run_status=self._run_status,
            system_status="training_failed" if self._run_status == "failing" else "idle",
            step_number=self._step_count,
            max_steps=self._max_steps,
            budget_remaining=budget,
            actions_taken=list(self._actions_taken),
            hint=hint,
            instructions=_TASK_INSTRUCTIONS.get(self._task_name, ""),
            done=self._done,
            reward=reward,
        )
