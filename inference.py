"""
inference.py — Baseline agent for pytorch_triage_env
REQUIRED env vars (injected by validator):
    API_BASE_URL      LiteLLM proxy endpoint (default: HF router)
    MODEL_NAME        Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN          API key — injected by validator, NO default
    LOCAL_IMAGE_NAME  Docker image name (optional, for from_docker_image())

IMPORTANT:
  - HF_TOKEN has NO default — validator injects it. Matches checklist exactly.
  - A pre-flight LLM test call is made at startup to verify proxy connectivity.
  - The env server is warmed up (with retries) before episodes begin.
"""
import asyncio
import json
import os
import re
import textwrap
import time
from typing import List, Optional

import requests as _requests
from openai import OpenAI

# ── Optional package import ────────────────────────────────────────────────────
try:
    from pytorch_triage_env.server.models import (
        ReadFileAction, EditFileAction, ExecuteBashAction,
        ViewGitDiffAction, SubmitFixAction, TriageAction,
    )
    from pydantic import TypeAdapter
    _action_adapter = TypeAdapter(TriageAction)

    def make_action(d: dict):
        return _action_adapter.validate_python(d)

    PACKAGE_AVAILABLE = True
except ImportError:
    PACKAGE_AVAILABLE = False

    def make_action(d: dict):
        return d


# ── Lightweight HTTP env client ────────────────────────────────────────────────

class _Obs:
    def __init__(self, d):
        [setattr(self, k, _Obs(v) if isinstance(v, dict) else (
            [_Obs(i) if isinstance(i, dict) else i for i in v]
            if isinstance(v, list) else v
        )) for k, v in (d or {}).items()]


class _StepResult:
    def __init__(self, d):
        self.observation = _Obs(d.get("observation", d))
        self.reward = float(d.get("reward", 0.0))
        self.done   = bool(d.get("done", False))


class _HTTPEnvClient:
    def __init__(self, base_url: str, timeout: int = 90):
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout

    async def reset(self, **kw):
        r = _requests.post(
            f"{self.base_url}/reset", json=kw, timeout=self.timeout
        )
        r.raise_for_status()
        return _StepResult(r.json())

    async def step(self, action):
        payload = (
            action if isinstance(action, dict)
            else action.model_dump() if hasattr(action, "model_dump")
            else vars(action)
        )
        r = _requests.post(
            f"{self.base_url}/step", json=payload, timeout=self.timeout
        )
        r.raise_for_status()
        return _StepResult(r.json())

    async def close(self):
        pass

    @classmethod
    async def from_docker_image(cls, image_name: str):
        return cls(os.environ.get("ENV_URL", "https://an8136-pytorch-triage-env.hf.space"))


# ── Config — MATCHES SUBMISSION CHECKLIST EXACTLY ──────────────────────────────
# Defaults ONLY for API_BASE_URL and MODEL_NAME. HF_TOKEN has NO default.
API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN         = os.getenv("HF_TOKEN")          # injected by validator — no default
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # optional: for from_docker_image()

ENV_URL          = os.getenv("ENV_URL", "https://an8136-pytorch-triage-env.hf.space")
IMAGE_NAME       = LOCAL_IMAGE_NAME or os.getenv("IMAGE_NAME")
BENCHMARK        = "pytorch_triage_env"
SUCCESS_THRESHOLD = 0.50

TASKS = [
    "oom_graph_leak",
    "fsdp_collective_deadlock",
    "compile_graph_break",
    "ddp_gradient_hang",
]

MAX_STEPS_MAP = {
    "oom_graph_leak":           8,
    "fsdp_collective_deadlock": 9,
    "compile_graph_break":      10,
    "ddp_gradient_hang":        9,
}


# ── LLM client ─────────────────────────────────────────────────────────────────

def make_llm_client() -> OpenAI:
    """Create OpenAI client via validator-injected API_BASE_URL and HF_TOKEN."""
    print(
        f"[CONFIG] base_url={API_BASE_URL}  model={MODEL_NAME}  "
        f"hf_token_set={bool(HF_TOKEN)}  env_url={ENV_URL}",
        flush=True,
    )
    # Use HF_TOKEN as api_key per submission checklist
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def preflight_llm_test(client: OpenAI) -> bool:
    """
    MANDATORY pre-flight test: make one LLM API call through the validator proxy
    before any environment interaction. This proves connectivity and ensures at
    least one call is observed on the provided API key.
    """
    print("[PREFLIGHT] Testing LLM proxy connectivity...", flush=True)
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system",  "content": "You are a test assistant."},
                {"role": "user",    "content": "Reply with the single word: READY"},
            ],
            max_tokens=10,
            temperature=0.0,
        )
        answer = (resp.choices[0].message.content or "").strip()
        print(f"[PREFLIGHT] LLM proxy OK — response={answer!r}  model={MODEL_NAME}", flush=True)
        return True
    except Exception as exc:
        print(f"[PREFLIGHT] LLM proxy error: {exc}", flush=True)
        return False


# ── Env warm-up ────────────────────────────────────────────────────────────────

def warmup_env(url: str, max_wait: int = 120, poll_interval: int = 6) -> bool:
    """
    Wait for the environment server to be healthy.
    HF Spaces can take 30-90 seconds to wake from sleep — we poll /health
    until it responds or we time out.
    Returns True if the server is ready, False on timeout.
    """
    print(f"[WARMUP] Waiting for env server at {url} (max {max_wait}s)...", flush=True)
    deadline = time.time() + max_wait
    attempt  = 0
    while time.time() < deadline:
        attempt += 1
        try:
            r = _requests.get(f"{url}/health", timeout=10)
            if r.status_code == 200:
                print(f"[WARMUP] Env server ready after {attempt} attempts.", flush=True)
                return True
        except Exception as exc:
            print(f"[WARMUP] Attempt {attempt}: {exc}", flush=True)
        time.sleep(poll_interval)
    print(f"[WARMUP] Env server not ready after {max_wait}s — proceeding anyway.", flush=True)
    return False


# ── Prompts ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are a Staff ML Infrastructure Engineer at a major AI lab.
    You are debugging a failing PyTorch training run.
    You have access to a virtual workspace. Use the tools to triage and fix the bug.

    AVAILABLE ACTIONS (respond with ONLY a JSON object):

    Read a file:
    {"action_type": "read_file", "filename": "train.py"}

    Edit a file (old_str must EXACTLY match file content — copy verbatim):
    {"action_type": "edit_file", "filename": "train.py", "old_str": "...", "new_str": "..."}

    Run a command (python train.py, torchrun ..., TORCH_LOGS=dynamo python train.py, etc.):
    {"action_type": "execute_bash", "command": "python train.py"}

    View your changes:
    {"action_type": "view_git_diff", "filename": null}

    Submit final fix (REQUIRED: deep explanation of root cause, not just what you changed):
    {"action_type": "submit_fix", "explanation": "The root cause is X because Y. The fix works because Z. To prevent this class of bug in future, W."}

    WORKFLOW:
    1. execute_bash to see the error
    2. read_file to understand the bug
    3. edit_file to apply the fix
    4. execute_bash to verify the fix works
    5. submit_fix with a technically deep explanation

    The explanation is evaluated by an LLM judge. "I changed X to Y" scores 0.3.
    "The root cause is A because B, and the fix works because C" scores 1.0.

    Respond ONLY with the JSON object. No prose.
""").strip()


def build_prompt(obs) -> str:
    files_listing = "\n".join(
        f"  {fname}: {len(content)} chars"
        for fname, content in (getattr(obs, "current_files", None) or {}).items()
    )
    hint_block = ""
    if getattr(obs, "hint", None):
        hint_block = f"\nHINT (after multiple failed runs): {obs.hint}\n"

    return textwrap.dedent(f"""
        TASK: {getattr(obs, 'task_name', '')}
        STEP: {getattr(obs, 'step_number', 0)}/{getattr(obs, 'max_steps', 9)} | BUDGET: {getattr(obs, 'budget_remaining', '?')} | STATUS: {getattr(obs, 'run_status', 'not_run')}

        INCIDENT REPORT:
        {getattr(obs, 'task_description', '')}

        WORKSPACE:
        {files_listing}
        {hint_block}
        INSTRUCTIONS:
        {getattr(obs, 'instructions', '')}

        LAST OUTPUT:
        {getattr(obs, 'terminal_output', '(none)')[:2000]}

        What is your next action? Reply with ONLY a JSON object.
    """).strip()


def get_action(client: OpenAI, obs) -> dict:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_prompt(obs)},
            ],
            temperature=0.2,
            max_tokens=600,
        )
        text = (resp.choices[0].message.content or "").strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$",          "", text)
        m = re.search(r"\{.*?\}", text, re.DOTALL)
        if m:
            return json.loads(m.group())
        print(f"[DEBUG] Could not parse JSON from LLM response: {text[:200]}", flush=True)
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
    return {
        "action_type": "submit_fix",
        "explanation": (
            "Fallback submission — LLM response could not be parsed. "
            "The root cause is likely a computation graph retention issue "
            "or a misconfigured distributed training collective operation."
        ),
    }


# ── Logging (mandatory format) ─────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    e = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={e}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    r_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={r_str}",
        flush=True,
    )


# ── Episode runner ─────────────────────────────────────────────────────────────

async def run_episode(
    task_name:  str,
    llm_client: OpenAI,
    image_name: Optional[str],
) -> tuple:
    env_client = (
        await _HTTPEnvClient.from_docker_image(image_name)
        if image_name
        else _HTTPEnvClient(base_url=ENV_URL, timeout=90)
    )
    rewards:     List[float] = []
    steps_taken: int         = 0
    max_steps    = MAX_STEPS_MAP.get(task_name, 9)

    try:
        result = await env_client.reset(task=task_name)
        obs    = result.observation

        for step in range(1, max_steps + 1):
            if result.done:
                break

            action_dict = get_action(llm_client, obs)
            action_str  = json.dumps(action_dict).replace('"', "'")[:120]

            try:
                action = make_action(action_dict)
                result = await env_client.step(action)
                obs    = result.observation
                reward = float(result.reward or 0.0)
                done   = bool(result.done)
                error  = None
            except Exception as ex:
                reward = 0.0
                done   = True
                error  = str(ex)[:80]
                print(f"[DEBUG] step() failed: {ex}", flush=True)

            rewards.append(reward)
            steps_taken = step
            log_step(step, action_str, reward, done, error)

            if done:
                break

    finally:
        await env_client.close()

    total = sum(rewards)
    score = round(min(max(total, 0.0), 1.0), 3)
    return score, rewards, steps_taken


# ── Main ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    # 1. Create LLM client using strictly validator-injected env vars
    llm_client = make_llm_client()

    # 2. MANDATORY pre-flight: make one guaranteed API call through the proxy
    preflight_llm_test(llm_client)

    # 3. Warm up env server (handles HF Space cold-start delays)
    warmup_env(ENV_URL, max_wait=120)

    # 4. Run all task episodes
    all_scores: List[float] = []

    for task_name in TASKS:
        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
        score, rewards, steps = 0.0, [], 0
        success = False
        try:
            score, rewards, steps = await run_episode(task_name, llm_client, IMAGE_NAME)
            success = score >= SUCCESS_THRESHOLD
        except Exception as exc:
            print(f"[DEBUG] Episode error ({task_name}): {exc}", flush=True)

        log_end(success=success, steps=steps, score=score, rewards=rewards)
        all_scores.append(score)

    avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"[SUMMARY] tasks={len(TASKS)} avg_score={avg:.3f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
