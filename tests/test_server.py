"""
tests/test_server.py — HTTP server integration tests.

Requires a running server. Set ENV_URL env var or use default http://localhost:7860.
Run: ENV_URL=http://localhost:7860 python tests/test_server.py
"""
import sys
import os
import json

try:
    import requests
except ImportError:
    print("requests not installed — pip install requests")
    sys.exit(1)

BASE_URL = os.getenv("ENV_URL", "http://localhost:7860").rstrip("/")

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


def post(path: str, body: dict) -> dict:
    r = requests.post(f"{BASE_URL}{path}", json=body, timeout=10)
    r.raise_for_status()
    return r.json()


def get(path: str) -> dict:
    r = requests.get(f"{BASE_URL}{path}", timeout=10)
    r.raise_for_status()
    return r.json()


def test_health():
    print("\n[test_server] GET /health:")
    data = get("/health")
    check("status == 'ok'", data.get("status") == "ok", str(data))
    check("env field present", "env" in data)
    check("tasks list has 4 items", len(data.get("tasks", [])) == 4)
    check("all 4 task names present", set(data.get("tasks", [])) == {
        "oom_graph_leak", "fsdp_collective_deadlock", "compile_graph_break", "ddp_gradient_hang"
    })


def test_schema():
    print("\n[test_server] GET /schema:")
    data = get("/schema")
    check("observation schema present", "observation" in data)
    check("action_types list present", "action_types" in data)
    check("5 action types", len(data.get("action_types", [])) == 5)


def test_reset_all_tasks():
    print("\n[test_server] POST /reset for all tasks:")
    for task in ["oom_graph_leak", "fsdp_collective_deadlock", "compile_graph_break", "ddp_gradient_hang"]:
        data = post("/reset", {"task": task})
        obs  = data.get("observation", {})
        check(f"{task}: reset returns observation", bool(obs))
        check(f"{task}: task_name matches", obs.get("task_name") == task, f"got '{obs.get('task_name')}'")
        check(f"{task}: 4 files in current_files", len(obs.get("current_files", {})) == 4)
        check(f"{task}: step_number == 0", obs.get("step_number") == 0)
        check(f"{task}: done is False", obs.get("done") is False)


def test_full_oom_episode():
    print("\n[test_server] Full OOM episode via HTTP:")

    # Reset
    data = post("/reset", {"task": "oom_graph_leak"})
    obs  = data["observation"]
    check("reset: task_name == 'oom_graph_leak'", obs["task_name"] == "oom_graph_leak")

    # Step 1: execute_bash to see error
    data = post("/step", {"action_type": "execute_bash", "command": "python train.py"})
    obs  = data["observation"]
    check("step 1: run_status is 'failing'", obs["run_status"] == "failing", obs["run_status"])
    check("step 1: reward returned", "reward" in data)

    # Step 2: read train.py
    data = post("/step", {"action_type": "read_file", "filename": "train.py"})
    obs  = data["observation"]
    check("step 2: train.py content in output", "epoch_loss" in obs["terminal_output"])
    check("step 2: reward > 0", data["reward"] > 0)

    # Step 3: apply fix
    data = post("/step", {
        "action_type": "edit_file",
        "filename":    "train.py",
        "old_str":     "epoch_loss += loss  # BUG: retains computation graph! Should be loss.item()",
        "new_str":     "epoch_loss += loss.item()",
    })
    obs = data["observation"]
    check("step 3: edit succeeds", "✓" in obs["terminal_output"] or "Edit applied" in obs["terminal_output"])

    # Step 4: verify
    data = post("/step", {"action_type": "execute_bash", "command": "python train.py"})
    obs  = data["observation"]
    check("step 4: run_status is 'passing'", obs["run_status"] == "passing", obs["run_status"])

    # Step 5: submit
    data = post("/step", {
        "action_type": "submit_fix",
        "explanation": (
            "The root cause is that epoch_loss += loss retains the entire computation "
            "graph through the grad_fn chain. Each iteration appends a new graph node, "
            "preventing garbage collection of intermediate tensors, growing memory O(N). "
            "Using loss.item() extracts a Python scalar, breaking the tensor reference "
            "and allowing the graph to be freed after each backward() call."
        ),
    })
    obs = data["observation"]
    check("step 5: done is True", obs["done"] is True)
    check("step 5: reward > 0.5", data["reward"] > 0.5, f"got {data['reward']:.3f}")
    check("step 5: fix_verified in state", data.get("state", {}).get("fix_verified") is True)


def test_state_endpoint():
    print("\n[test_server] GET /state:")
    post("/reset", {"task": "ddp_gradient_hang"})
    state = get("/state")
    check("task_name in state", "task_name" in state)
    check("step_count in state", "step_count" in state)
    check("fix_verified in state", "fix_verified" in state)


def test_invalid_action_type():
    print("\n[test_server] Invalid action type handling:")
    post("/reset", {"task": "oom_graph_leak"})
    try:
        r = requests.post(
            f"{BASE_URL}/step",
            json={"action_type": "nonexistent_action"},
            timeout=10,
        )
        check("server returns 4xx for invalid action", 400 <= r.status_code < 500, f"got {r.status_code}")
    except requests.exceptions.HTTPError as e:
        check("server returns 4xx for invalid action", True)


def test_compile_graph_break_diagnostic():
    print("\n[test_server] compile_graph_break diagnostic command:")
    post("/reset", {"task": "compile_graph_break"})

    data = post("/step", {"action_type": "execute_bash", "command": "TORCH_LOGS=dynamo python train.py"})
    obs  = data["observation"]
    check("run_status is 'diagnostic'", obs["run_status"] == "diagnostic", obs["run_status"])
    check("dynamo output in terminal", "dynamo" in obs["terminal_output"].lower() or "graph break" in obs["terminal_output"].lower())


if __name__ == "__main__":
    print(f"Testing server at {BASE_URL}")
    try:
        get("/health")
    except Exception as e:
        print(f"\033[91mCannot reach server at {BASE_URL}: {e}\033[0m")
        print("Start the server first: uvicorn pytorch_triage_env.server.app:app --port 7860")
        sys.exit(1)

    test_health()
    test_schema()
    test_reset_all_tasks()
    test_full_oom_episode()
    test_state_endpoint()
    test_invalid_action_type()
    test_compile_graph_break_diagnostic()

    print()
    if _failures:
        print(f"\033[91mFAILED: {len(_failures)} test(s)\033[0m")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\033[92mAll server tests passed.\033[0m")
