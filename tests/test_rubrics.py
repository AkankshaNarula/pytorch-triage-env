"""
tests/test_rubrics.py — TrajectoryRubric and LLMJudgeRubric unit tests.

Run: python tests/test_rubrics.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pytorch_triage_env.server.rubrics import TrajectoryRubric, LLMJudgeRubric

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


# ── Minimal stub classes so we don't need openenv installed ──────────────────

class _Action:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

class _Obs:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


# ── TrajectoryRubric ──────────────────────────────────────────────────────────

def test_trajectory_read_signal_file():
    print("\n[test_rubrics] TrajectoryRubric — read_file rewards:")
    rubric = TrajectoryRubric()

    action = _Action(action_type="read_file", filename="train.py")
    obs    = _Obs(task_name="oom_graph_leak", run_status="not_run")
    reward = rubric.forward(action, obs)
    check("reading signal file (train.py) gives 0.08", reward == 0.08, f"got {reward}")

    action = _Action(action_type="read_file", filename="config.py")
    reward = rubric.forward(action, obs)
    check("reading non-signal file gives 0.02", reward == 0.02, f"got {reward}")


def test_trajectory_execute_bash_rewards():
    print("\n[test_rubrics] TrajectoryRubric — execute_bash rewards:")
    rubric = TrajectoryRubric()

    # Diagnostic flag for compile_graph_break
    action = _Action(action_type="execute_bash", command="TORCH_LOGS=dynamo python train.py")
    obs    = _Obs(task_name="compile_graph_break", run_status="failing")
    reward = rubric.forward(action, obs)
    check("correct diagnostic flag gives 0.12", reward == 0.12, f"got {reward}")

    # Plain python run
    action = _Action(action_type="execute_bash", command="python train.py")
    obs    = _Obs(task_name="oom_graph_leak", run_status="failing")
    reward = rubric.forward(action, obs)
    check("plain python run gives 0.05", reward == 0.05, f"got {reward}")

    # torchrun
    action = _Action(action_type="execute_bash", command="torchrun --nproc_per_node=4 train.py")
    reward = rubric.forward(action, obs)
    check("torchrun gives 0.05", reward == 0.05, f"got {reward}")

    # Unrecognized command
    action = _Action(action_type="execute_bash", command="ls -la")
    reward = rubric.forward(action, obs)
    check("unrecognized command gives -0.01", reward == -0.01, f"got {reward}")

    # SyntaxError in output
    action = _Action(action_type="execute_bash", command="python train.py")
    obs    = _Obs(task_name="oom_graph_leak", run_status="syntax_error")
    reward = rubric.forward(action, obs)
    check("syntax_error run gives -0.03", reward == -0.03, f"got {reward}")


def test_trajectory_edit_file_rewards():
    print("\n[test_rubrics] TrajectoryRubric — edit_file rewards:")
    rubric = TrajectoryRubric()

    # Valid edit
    action = _Action(action_type="edit_file", filename="train.py",
                     old_str="epoch_loss += loss", new_str="epoch_loss += loss.item()")
    obs    = _Obs(task_name="oom_graph_leak", run_status="not_run")
    reward = rubric.forward(action, obs)
    check("valid edit gives 0.05", reward == 0.05, f"got {reward}")

    # Edit that introduces a syntax error
    action = _Action(action_type="edit_file", filename="train.py",
                     old_str="def main():", new_str="def main(:\n    pass")
    reward = rubric.forward(action, obs)
    check("syntax-error edit gives -0.08", reward == -0.08, f"got {reward}")


def test_trajectory_view_git_diff():
    print("\n[test_rubrics] TrajectoryRubric — view_git_diff reward:")
    rubric = TrajectoryRubric()
    action = _Action(action_type="view_git_diff", filename=None)
    obs    = _Obs(task_name="oom_graph_leak", run_status="not_run")
    reward = rubric.forward(action, obs)
    check("view_git_diff gives 0.02", reward == 0.02, f"got {reward}")


def test_trajectory_submit_fix_zero():
    print("\n[test_rubrics] TrajectoryRubric — submit_fix gives 0 (terminal handled elsewhere):")
    rubric = TrajectoryRubric()
    action = _Action(action_type="submit_fix", explanation="Some explanation.")
    obs    = _Obs(task_name="oom_graph_leak", run_status="passing")
    reward = rubric.forward(action, obs)
    check("submit_fix step reward is 0.0", reward == 0.0, f"got {reward}")


# ── LLMJudgeRubric keyword judge ──────────────────────────────────────────────

def test_llmjudge_keyword_deep():
    print("\n[test_rubrics] LLMJudgeRubric — keyword judge deep explanations:")
    judge = LLMJudgeRubric()

    explanation = (
        "The root cause is that accumulating a tensor directly retains the entire "
        "computation graph through the grad_fn chain. Every batch appends to the "
        "epoch_loss node, preventing garbage collection of intermediate tensors."
    )
    score = judge._keyword_judge("oom_graph_leak", explanation)
    check("2+ deep keywords → score 1.0", score == 1.0, f"got {score}")


def test_llmjudge_keyword_shallow():
    print("\n[test_rubrics] LLMJudgeRubric — keyword judge shallow explanations:")
    judge = LLMJudgeRubric()

    explanation = "I changed loss.item() and the OOM went away."
    score = judge._keyword_judge("oom_graph_leak", explanation)
    check("shallow keywords → score ≥ 0.3 and < 1.0", 0.3 <= score < 1.0, f"got {score}")


def test_llmjudge_keyword_fsdp():
    print("\n[test_rubrics] LLMJudgeRubric — fsdp deep explanation:")
    judge = LLMJudgeRubric()

    explanation = (
        "NCCL collective operations require all ranks to participate simultaneously. "
        "Placing all_reduce inside a rank guard means only rank 0 enters the barrier; "
        "other ranks never call it, so NCCL hangs."
    )
    score = judge._keyword_judge("fsdp_collective_deadlock", explanation)
    check("fsdp deep keywords → score 1.0", score == 1.0, f"got {score}")


def test_llmjudge_keyword_empty():
    print("\n[test_rubrics] LLMJudgeRubric — empty explanation:")
    judge = LLMJudgeRubric()
    score = judge._keyword_judge("oom_graph_leak", "")
    check("empty explanation → low score (≤ 0.3)", score <= 0.3, f"got {score}")


def test_llmjudge_compile_graph_break():
    print("\n[test_rubrics] LLMJudgeRubric — compile_graph_break:")
    judge = LLMJudgeRubric()

    explanation = (
        "The graph break occurs because loss.item() is data-dependent control flow. "
        "Dynamo cannot trace through a Python scalar branch, causing a recompilation "
        "on every step and falling back to eager mode."
    )
    score = judge._keyword_judge("compile_graph_break", explanation)
    check("compile deep keywords → score ≥ 0.7", score >= 0.7, f"got {score}")


def test_llmjudge_ddp_gradient_hang():
    print("\n[test_rubrics] LLMJudgeRubric — ddp_gradient_hang:")
    judge = LLMJudgeRubric()

    explanation = (
        "DDP's gradient sync uses backward hooks on all parameters. With "
        "find_unused_parameters=False, DDP waits for gradients from every parameter. "
        "The conditional branch means aux_head parameters never receive gradients "
        "on non-auxiliary steps, so DDP stalls indefinitely."
    )
    score = judge._keyword_judge("ddp_gradient_hang", explanation)
    check("ddp deep keywords → score ≥ 0.7", score >= 0.7, f"got {score}")


if __name__ == "__main__":
    test_trajectory_read_signal_file()
    test_trajectory_execute_bash_rewards()
    test_trajectory_edit_file_rewards()
    test_trajectory_view_git_diff()
    test_trajectory_submit_fix_zero()
    test_llmjudge_keyword_deep()
    test_llmjudge_keyword_shallow()
    test_llmjudge_keyword_fsdp()
    test_llmjudge_keyword_empty()
    test_llmjudge_compile_graph_break()
    test_llmjudge_ddp_gradient_hang()

    print()
    if _failures:
        print(f"\033[91mFAILED: {len(_failures)} test(s)\033[0m")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\033[92mAll rubric tests passed.\033[0m")
