"""
tests/test_environment.py — PyTorchTriageEnv integration tests (no server needed).

Run: python tests/test_environment.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Patch openenv imports before importing environment
import types

def _make_openenv_stubs():
    """Inject minimal openenv stubs so the environment can import without openenv installed."""
    import pydantic

    class _Action(pydantic.BaseModel):
        action_type: str = ""

    class _Observation(pydantic.BaseModel):
        done:   bool  = False
        reward: float = 0.0

    class _State(pydantic.BaseModel):
        episode_id: str  = ""
        step_count: int  = 0

    class _Environment:
        pass

    # Inject module hierarchy
    for mod_name, attrs in [
        ("openenv",                                  {}),
        ("openenv.core",                             {}),
        ("openenv.core.env_server",                  {}),
        ("openenv.core.env_server.types",            {"Action": _Action, "Observation": _Observation, "State": _State}),
        ("openenv.core.env_server.interfaces",       {"Environment": _Environment}),
        ("openenv.core.rubrics",                     {}),
    ]:
        if mod_name not in sys.modules:
            m = types.ModuleType(mod_name)
            for k, v in attrs.items():
                setattr(m, k, v)
            sys.modules[mod_name] = m
        else:
            for k, v in attrs.items():
                if not hasattr(sys.modules[mod_name], k):
                    setattr(sys.modules[mod_name], k, v)

_make_openenv_stubs()

from pytorch_triage_env.server.environment import PyTorchTriageEnv
from pytorch_triage_env.server.models import (
    ReadFileAction, EditFileAction, ExecuteBashAction,
    ViewGitDiffAction, SubmitFixAction,
)

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
_failures = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  {PASS} {label}")
    else:
        msg = f"  {FAIL} {label}" + (f" — {detail}" if detail else "")
        print(msg)
        _failures.append(label)


def test_reset():
    print("\n[test_environment] reset():")
    env = PyTorchTriageEnv(task_name="oom_graph_leak")
    obs = env.reset(task="oom_graph_leak")

    check("obs.task_name == 'oom_graph_leak'", obs.task_name == "oom_graph_leak")
    check("obs.step_number == 0", obs.step_number == 0)
    check("obs.done is False", obs.done is False)
    check("obs.reward == 0.0", obs.reward == 0.0)
    check("obs.current_files has 4 files", len(obs.current_files) == 4)
    check("obs.run_status == 'not_run'", obs.run_status == "not_run")
    check("obs.budget_remaining == max_steps", obs.budget_remaining == obs.max_steps)
    check("state.step_count == 0", env.state.step_count == 0)


def test_read_file_action():
    print("\n[test_environment] ReadFileAction:")
    env = PyTorchTriageEnv(task_name="oom_graph_leak")
    env.reset(task="oom_graph_leak")

    obs = env.step(ReadFileAction(filename="train.py"))
    check("step_number incremented to 1", obs.step_number == 1)
    check("terminal_output contains train.py content", "epoch_loss" in obs.terminal_output)
    check("reward > 0 for signal file read", obs.reward > 0)

    obs = env.step(ReadFileAction(filename="config.py"))
    check("reading non-signal file gives small positive reward", 0 < obs.reward <= 0.08)


def test_execute_bash_failing():
    print("\n[test_environment] ExecuteBashAction (unfixed):")
    env = PyTorchTriageEnv(task_name="oom_graph_leak")
    env.reset(task="oom_graph_leak")

    obs = env.step(ExecuteBashAction(command="python train.py"))
    check("run_status is 'failing' before fix", obs.run_status == "failing")
    check("terminal_output contains OOM error", "OutOfMemoryError" in obs.terminal_output or "out of memory" in obs.terminal_output.lower())
    check("state.bash_run_count == 1", env.state.bash_run_count == 1)


def test_edit_and_verify_oom():
    print("\n[test_environment] EditFileAction + verify (oom_graph_leak):")
    env = PyTorchTriageEnv(task_name="oom_graph_leak")
    env.reset(task="oom_graph_leak")

    obs = env.step(EditFileAction(
        filename="train.py",
        old_str="epoch_loss += loss  # BUG: retains computation graph! Should be loss.item()",
        new_str="epoch_loss += loss.item()",
    ))
    check("edit succeeds", "✓" in obs.terminal_output)
    check("run_status reset to 'not_run' after edit", obs.run_status == "not_run")
    check("state.files_edited contains train.py", "train.py" in env.state.files_edited)

    obs = env.step(ExecuteBashAction(command="python train.py"))
    check("run_status is 'passing' after fix", obs.run_status == "passing", f"got '{obs.run_status}'")
    check("terminal_output has success trace", "stable" in obs.terminal_output.lower() or "completed" in obs.terminal_output.lower())


def test_submit_fix_verified():
    print("\n[test_environment] SubmitFixAction (verified fix):")
    env = PyTorchTriageEnv(task_name="oom_graph_leak")
    env.reset(task="oom_graph_leak")

    env.step(EditFileAction(
        filename="train.py",
        old_str="epoch_loss += loss  # BUG: retains computation graph! Should be loss.item()",
        new_str="epoch_loss += loss.item()",
    ))

    deep_explanation = (
        "The root cause is that epoch_loss += loss retains a reference to the entire "
        "computation graph through the grad_fn chain. Each batch appends a new graph node, "
        "preventing garbage collection of intermediate tensors, causing O(N) memory growth. "
        "Using loss.item() extracts a Python scalar and detaches from the computation graph, "
        "allowing PyTorch to free memory after each backward() call."
    )
    obs = env.step(SubmitFixAction(explanation=deep_explanation))

    check("done is True after submit", obs.done is True)
    check("reward > 0.5 for verified fix + deep explanation", obs.reward > 0.5, f"got {obs.reward:.3f}")
    check("fix_verified is True", env.state.fix_verified is True)
    check("terminal_output shows verification success", "✓" in obs.terminal_output or "verified" in obs.terminal_output.lower())


def test_submit_fix_unverified():
    print("\n[test_environment] SubmitFixAction (unverified — bug not fixed):")
    env = PyTorchTriageEnv(task_name="oom_graph_leak")
    env.reset(task="oom_graph_leak")

    obs = env.step(SubmitFixAction(explanation="I think the fix is done."))
    check("done is True", obs.done is True)
    check("fix_verified is False", env.state.fix_verified is False)
    check("reward < 0.5 for unverified fix", obs.reward < 0.5, f"got {obs.reward:.3f}")


def test_view_git_diff():
    print("\n[test_environment] ViewGitDiffAction:")
    env = PyTorchTriageEnv(task_name="oom_graph_leak")
    env.reset(task="oom_graph_leak")

    obs = env.step(ViewGitDiffAction(filename=None))
    check("no-change diff says so", "No changes" in obs.terminal_output)

    env.step(EditFileAction(
        filename="train.py",
        old_str="epoch_loss += loss  # BUG: retains computation graph! Should be loss.item()",
        new_str="epoch_loss += loss.item()",
    ))
    obs = env.step(ViewGitDiffAction(filename=None))
    check("diff shows the change", "loss.item()" in obs.terminal_output)
    check("diff uses unified diff format", "---" in obs.terminal_output or "+++" in obs.terminal_output)


def test_max_steps_terminates():
    print("\n[test_environment] Max steps termination:")
    env = PyTorchTriageEnv(task_name="oom_graph_leak")
    env.reset(task="oom_graph_leak")
    max_steps = env._max_steps

    obs = None
    for _ in range(max_steps):
        obs = env.step(ExecuteBashAction(command="python train.py"))

    check("episode terminates at max_steps", obs.done is True)


def test_hint_appears_after_3_failures():
    print("\n[test_environment] Hint after 3 failed bash runs:")
    env = PyTorchTriageEnv(task_name="oom_graph_leak")
    env.reset(task="oom_graph_leak")

    for _ in range(3):
        obs = env.step(ExecuteBashAction(command="python train.py"))

    check("hint appears after 3 failing runs", obs.hint is not None)
    check("hint is non-empty string", isinstance(obs.hint, str) and len(obs.hint) > 5)


def test_all_tasks_reset():
    print("\n[test_environment] All tasks can reset and run:")
    tasks = ["oom_graph_leak", "fsdp_collective_deadlock", "compile_graph_break", "ddp_gradient_hang"]
    for task in tasks:
        env = PyTorchTriageEnv(task_name=task)
        obs = env.reset(task=task)
        check(f"{task}: reset returns valid obs", obs.task_name == task)
        check(f"{task}: has 4 files", len(obs.current_files) == 4)


def test_state_property():
    print("\n[test_environment] state @property:")
    env = PyTorchTriageEnv(task_name="oom_graph_leak")
    env.reset(task="oom_graph_leak")

    state = env.state
    check("state.task_name == 'oom_graph_leak'", state.task_name == "oom_graph_leak")
    check("state.step_count == 0", state.step_count == 0)
    check("state.fix_submitted is False", state.fix_submitted is False)
    check("state is a property (not a method call)", not callable(type(env).state))


if __name__ == "__main__":
    test_reset()
    test_read_file_action()
    test_execute_bash_failing()
    test_edit_and_verify_oom()
    test_submit_fix_verified()
    test_submit_fix_unverified()
    test_view_git_diff()
    test_max_steps_terminates()
    test_hint_appears_after_3_failures()
    test_all_tasks_reset()
    test_state_property()

    print()
    if _failures:
        print(f"\033[91mFAILED: {len(_failures)} test(s)\033[0m")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\033[92mAll environment tests passed.\033[0m")
