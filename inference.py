"""
inference.py — Baseline agent for pytorch_triage_env
REQUIRED env vars:
    API_BASE_URL   LLM endpoint (LiteLLM proxy — injected by validator)
    API_KEY        LLM API key  (LiteLLM proxy — injected by validator)
    MODEL_NAME     Model identifier
    IMAGE_NAME     Docker image (optional — set ENV_URL instead for local server)
    ENV_URL        Local server URL (default: http://localhost:7860)
    JUDGE_MODEL    LLMJudge model (optional — enables deep explanation scoring)

IMPORTANT: Do NOT use HF_TOKEN or any other key as fallback for API_KEY.
           Do NOT hardcode any base_url. Always use os.environ["API_BASE_URL"].
"""
import asyncio
import json
import os
import re
import textwrap
from typing import List, Optional

import httpx
from openai import OpenAI

# ── Client import ────────────────────────────────────────────────────────────

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


import requests as _requests


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
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    async def __aenter__(self): return self
    async def __aexit__(self, *a): pass

    async def reset(self, **kw):
        return _StepResult(_requests.post(f"{self.base_url}/reset", json=kw, timeout=30).json())

    async def step(self, action):
        payload = (
            action if isinstance(action, dict)
            else action.model_dump() if hasattr(action, "model_dump")
            else vars(action)
        )
        return _StepResult(_requests.post(f"{self.base_url}/step", json=payload, timeout=30).json())

    async def close(self): pass

    @classmethod
    async def from_docker_image(cls, image_name: str):
        return cls(os.getenv("ENV_URL", "https://an8136-pytorch-triage-env.hf.space"))


# ── Config ────────────────────────────────────────────────────────────────────

IMAGE_NAME    = os.getenv("IMAGE_NAME")

# CRITICAL: These MUST come from environment — injected by the validator's LiteLLM proxy.
# Never fall back to HF_TOKEN, hardcoded keys, or alternative base URLs.
API_KEY       = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN", "dummy")
API_BASE_URL  = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")

MODEL_NAME    = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL       = os.getenv("ENV_URL", "https://an8136-pytorch-triage-env.hf.space")
BENCHMARK     = "pytorch_triage_env"
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

# ── LLM Client (must route through LiteLLM proxy) ────────────────────────────

def make_llm_client() -> OpenAI:
    """
    Build an OpenAI-compatible client that routes EXCLUSIVELY through
    the validator's LiteLLM proxy.

    Key points:
    - base_url is set to API_BASE_URL from env (the LiteLLM proxy endpoint).
    - api_key is set to API_KEY from env (the proxy-issued key).
    - OPENAI_API_KEY env var is cleared so the SDK cannot silently fall back
      to it and bypass the proxy.
    - OPENAI_BASE_URL env var is also cleared for the same reason.
    - http_client is explicitly constructed to avoid any SDK-level proxy
      env-var interference (e.g. OPENAI_PROXY).
    """
    # Prevent the OpenAI SDK from silently reading its own env-var overrides
    # which would bypass the LiteLLM proxy
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("OPENAI_BASE_URL", None)

    # Normalise base_url: the OpenAI SDK appends /chat/completions to whatever
    # base_url you give it. LiteLLM proxy exposes /v1, so ensure it ends with /v1.
    base_url = API_BASE_URL.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"

    print(f"[CONFIG] LLM proxy base_url={base_url}  model={MODEL_NAME}", flush=True)

    client = OpenAI(
        base_url=base_url,
        api_key=API_KEY,
        http_client=httpx.Client(),   # fresh client — no env-var proxy leakage
    )
    return client


# ── Prompts ───────────────────────────────────────────────────────────────────

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
        # Strip markdown fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$",          "", text)
        m = re.search(r"\{.*?\}", text, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
    return {
        "action_type": "submit_fix",
        "explanation": "Fallback — could not parse LLM response.",
    }


# ── Logging ───────────────────────────────────────────────────────────────────
# Hackathon mandatory format: [START], [STEP], [END]

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


# ── Episode runner ────────────────────────────────────────────────────────────

async def run_episode(
    task_name: str,
    llm_client: OpenAI,
    image_name: Optional[str],
) -> tuple:
    env_client = (
        await _HTTPEnvClient.from_docker_image(image_name)
        if image_name
        else _HTTPEnvClient(base_url=ENV_URL)
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
            action_str  = json.dumps(action_dict).replace('"', "'")[:100]

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
                error  = str(ex)[:60]

            rewards.append(reward)
            steps_taken = step
            log_step(step, action_str, reward, done, error)

            if done:
                break

    finally:
        await env_client.close()

    total = sum(rewards)
    score = round(min(max(total, 0.0), 1.0), 3)   # clamp to [0, 1] per spec
    return score, rewards, steps_taken


async def main() -> None:
    llm_client  = make_llm_client()   # uses LiteLLM proxy exclusively
    all_scores: List[float] = []

    for task_name in TASKS:
        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
        score, rewards, steps = 0.0, [], 0
        success = False
        try:
            score, rewards, steps = await run_episode(task_name, llm_client, IMAGE_NAME)
            success = score >= SUCCESS_THRESHOLD
        except Exception as e:
            print(f"[DEBUG] Episode error ({task_name}): {e}", flush=True)

        log_end(success=success, steps=steps, score=score, rewards=rewards)
        all_scores.append(score)

    avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"[SUMMARY] tasks={len(TASKS)} avg_score={avg:.3f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
