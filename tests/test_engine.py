"""
tests/test_engine.py — MockExecutionEngine unit tests.

Run: python tests/test_engine.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pytorch_triage_env.server.mock_execution_engine import MockExecutionEngine, TASK_CONFIGS
from pytorch_triage_env.server.virtual_fs import VirtualFileSystem

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


def test_all_tasks_fail_unmodified():
    print("\n[test_engine] All tasks fail before fix:")
    engine = MockExecutionEngine()
    for task_name, cfg in TASK_CONFIGS.items():
        vfs = VirtualFileSystem(cfg["files"])
        status, trace = engine.simulate(task_name, vfs.files)
        check(
            f"{task_name}: fails before fix (got '{status}')",
            status == "failing",
            f"status={status}",
        )
        check(
            f"{task_name}: error trace non-empty",
            len(trace) > 100,
        )


def test_oom_graph_leak_fix():
    print("\n[test_engine] oom_graph_leak fix detection:")
    engine = MockExecutionEngine()
    cfg    = TASK_CONFIGS["oom_graph_leak"]
    vfs    = VirtualFileSystem(cfg["files"])

    # Apply the fix
    result = vfs.edit(
        "train.py",
        "epoch_loss += loss  # BUG: retains computation graph! Should be loss.item()",
        "epoch_loss += loss.item()",
    )
    check("edit succeeds", "✓" in result, result)

    status, trace = engine.simulate("oom_graph_leak", vfs.files)
    check("status is passing after fix", status == "passing", f"got '{status}'")
    check("success trace mentions stable memory", "stable" in trace.lower() or "Memory" in trace)

    verified = engine.verify_fix("oom_graph_leak", vfs.files)
    check("verify_fix returns True", verified)


def test_oom_graph_leak_still_fails_without_fix():
    print("\n[test_engine] oom_graph_leak partial / no fix:")
    engine = MockExecutionEngine()
    cfg    = TASK_CONFIGS["oom_graph_leak"]
    vfs    = VirtualFileSystem(cfg["files"])

    status, _ = engine.simulate("oom_graph_leak", vfs.files)
    check("unmodified → failing", status == "failing")

    verified = engine.verify_fix("oom_graph_leak", vfs.files)
    check("verify_fix returns False on unmodified", not verified)


def test_fsdp_deadlock_fix():
    print("\n[test_engine] fsdp_collective_deadlock fix detection:")
    engine = MockExecutionEngine()
    cfg    = TASK_CONFIGS["fsdp_collective_deadlock"]
    vfs    = VirtualFileSystem(cfg["files"])

    # Move all_reduce outside the if rank == 0 block
    # The fix: remove the indented all_reduce and add it before the if block
    old = (
        "        if rank == 0:\n"
        "            loss_tensor = loss.detach().clone()\n"
        "            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)  # BUG: collective inside rank guard\n"
        "            avg_loss = loss_tensor.item() / world_size\n"
        "            if step % 10 == 0:\n"
        "                print(f\"Step {step}: avg_loss={avg_loss:.4f}\")"
    )
    new = (
        "        loss_tensor = loss.detach().clone()\n"
        "        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)\n"
        "        avg_loss = loss_tensor.item() / world_size\n"
        "        if rank == 0:\n"
        "            if step % 10 == 0:\n"
        "                print(f\"Step {step}: avg_loss={avg_loss:.4f}\")"
    )
    result = vfs.edit("train.py", old, new)
    check("edit applied", "ERROR" not in result, result)

    status, trace = engine.simulate("fsdp_collective_deadlock", vfs.files)
    check("status is passing after fix", status == "passing", f"got '{status}'")
    check("success trace shows training progress", "Step" in trace or "synchronized" in trace)

    verified = engine.verify_fix("fsdp_collective_deadlock", vfs.files)
    check("verify_fix returns True", verified)


def test_compile_graph_break_fix():
    print("\n[test_engine] compile_graph_break fix detection:")
    engine = MockExecutionEngine()
    cfg    = TASK_CONFIGS["compile_graph_break"]
    vfs    = VirtualFileSystem(cfg["files"])

    # Apply fix: add @torch.compiler.disable before the function
    result = vfs.edit(
        "train.py",
        "@torch.compile\ndef training_step",
        "@torch.compiler.disable\ndef training_step",
    )
    check("edit applied", "ERROR" not in result, result)

    status, trace = engine.simulate("compile_graph_break", vfs.files)
    check("status is passing after fix", status == "passing", f"got '{status}'")
    check("success trace mentions 0 graph breaks", "0 graph break" in trace or "No recompilation" in trace)

    verified = engine.verify_fix("compile_graph_break", vfs.files)
    check("verify_fix returns True", verified)


def test_compile_graph_break_dynamo_diagnostic():
    print("\n[test_engine] compile_graph_break diagnostic trace:")
    engine = MockExecutionEngine()
    cfg    = TASK_CONFIGS["compile_graph_break"]
    vfs    = VirtualFileSystem(cfg["files"])

    status, trace = engine.simulate(
        "compile_graph_break", vfs.files, command="TORCH_LOGS=dynamo python train.py"
    )
    check("diagnostic command returns 'diagnostic' status", status == "diagnostic", f"got '{status}'")
    check("dynamo trace mentions graph break", "graph break" in trace.lower() or "Graph break" in trace)


def test_ddp_gradient_hang_fix():
    print("\n[test_engine] ddp_gradient_hang fix detection:")
    engine = MockExecutionEngine()
    cfg    = TASK_CONFIGS["ddp_gradient_hang"]
    vfs    = VirtualFileSystem(cfg["files"])

    result = vfs.edit(
        "train.py",
        "model = DDP(model, device_ids=[rank], find_unused_parameters=False)  # BUG",
        "model = DDP(model, device_ids=[rank], find_unused_parameters=True)",
    )
    check("edit applied", "ERROR" not in result, result)

    status, trace = engine.simulate("ddp_gradient_hang", vfs.files)
    check("status is passing after fix", status == "passing", f"got '{status}'")
    check("success trace shows training completed", "completed" in trace.lower() or "Training" in trace)

    verified = engine.verify_fix("ddp_gradient_hang", vfs.files)
    check("verify_fix returns True", verified)


def test_virtual_fs_edit_errors():
    print("\n[test_engine] VirtualFileSystem error cases:")
    cfg = TASK_CONFIGS["oom_graph_leak"]
    vfs = VirtualFileSystem(cfg["files"])

    result = vfs.edit("train.py", "THIS_STRING_DOES_NOT_EXIST_IN_FILE", "replacement")
    check("edit with missing old_str returns ERROR", "ERROR" in result)

    result = vfs.read("nonexistent.py")
    check("read nonexistent file returns ERROR", "ERROR" in result)

    diff = vfs.git_diff()
    check("git_diff on unmodified returns no-change message", "No changes" in diff)


def test_all_task_files_present():
    print("\n[test_engine] All task files are populated:")
    for task_name, cfg in TASK_CONFIGS.items():
        files = cfg["files"]
        for fname in ["train.py", "model.py", "config.py", "data_loader.py"]:
            check(f"{task_name}/{fname} present and non-empty", fname in files and len(files[fname]) > 50)


if __name__ == "__main__":
    test_all_tasks_fail_unmodified()
    test_oom_graph_leak_fix()
    test_oom_graph_leak_still_fails_without_fix()
    test_fsdp_deadlock_fix()
    test_compile_graph_break_fix()
    test_compile_graph_break_dynamo_diagnostic()
    test_ddp_gradient_hang_fix()
    test_virtual_fs_edit_errors()
    test_all_task_files_present()

    print()
    if _failures:
        print(f"\033[91mFAILED: {len(_failures)} test(s)\033[0m")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\033[92mAll engine tests passed.\033[0m")
