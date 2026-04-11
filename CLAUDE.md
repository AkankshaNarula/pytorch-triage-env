# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

An OpenEnv RL environment where an LLM agent debugs simulated PyTorch training failures. The agent reads virtual files, edits code, runs mock training (producing realistic PyTorch error traces), and submits a fix with a technical explanation scored by an LLM judge. Built for the OpenEnv Hackathon 2026.

There are 3 debug tasks: `oom_graph_leak` (easy), `fsdp_collective_deadlock` (medium), `ddp_gradient_hang` (hard). A fourth task (`compile_graph_break`) exists in `mock_execution_engine.py` but is not exposed in the active task list.

## Commands

```bash
# Install (from repo root)
cd pytorch_triage_env && pip install -e . && cd ..

# Run the FastAPI server (port 7860)
uvicorn pytorch_triage_env.server.app:app --port 7860

# Run the baseline agent (requires HF_TOKEN)
HF_TOKEN=hf_... ENV_URL=http://localhost:7860 python inference.py

# Docker
docker build -f pytorch_triage_env/Dockerfile -t pytorch-triage-env ./pytorch_triage_env
docker run -p 7860:7860 pytorch-triage-env

# Tests — all use plain `python`, no pytest
python tests/test_engine.py        # Unit: engine + VFS (no server needed)
python tests/test_rubrics.py       # Unit: rubric scoring
python tests/test_environment.py   # Environment logic
# Integration test (needs running server):
uvicorn pytorch_triage_env.server.app:app --port 7860 &
ENV_URL=http://localhost:7860 python tests/test_server.py

# OpenEnv validation
cd pytorch_triage_env && openenv validate && cd ..
```

## Architecture

Two main components communicate over HTTP:

**1. FastAPI server** (`pytorch_triage_env/server/`) — the RL environment, runs on port 7860:
- `app.py` — manual FastAPI server (not using `openenv.create_app`; see comment in file for why). Endpoints: `/reset`, `/step`, `/state`, `/health`, `/schema`.
- `environment.py` — `PyTorchTriageEnv` core logic. Orchestrates VFS, mock engine, and rubrics. `reset()` returns a `TriageObservation` at the top level (not nested); `step()` returns `{observation, reward, done, state}`.
- `models.py` — Pydantic v2 discriminated union for actions (`TriageAction` = union of 5 action types keyed on `action_type`). `TriageObservation` inherits `done`/`reward` from the openenv `Observation` base — do not re-declare them.
- `mock_execution_engine.py` — returns pre-written error traces based on whether fix signatures are present in the VFS files. Does NOT import torch. The FSDP task uses structural checking (indentation analysis) rather than simple substring matching.
- `rubrics.py` — `TrajectoryRubric` (dense per-step rewards) + `LLMJudgeRubric` (terminal reward). The LLM judge falls back to keyword heuristics when `JUDGE_MODEL` env var is not set.
- `virtual_fs.py` — in-memory file system with git-diff tracking. `edit()` uses exact substring matching (`str.replace`), not regex.

**2. Baseline agent** (`inference.py` at repo root) — required by the OpenEnv validator to be at the root:
- Uses the OpenAI SDK pointed at the HuggingFace router (`API_BASE_URL`).
- Multi-turn conversation: `history` list accumulates user/assistant turns across steps within an episode.
- Has anti-loop directives (budget-critical warnings, repeated-action detection).
- Scores clamped to open interval (0.001, 0.999) — the validator rejects exactly 0.0 or 1.0.
- `_Obs` / `_StepResult` / `_HTTPEnvClient` are lightweight wrappers to avoid depending on the server package at import time.

## Key Constraints

- **No PyTorch dependency**: the server/Docker image does not install PyTorch. `mock_execution_engine.py` must never import torch — it parses file strings and returns canned traces.
- **Dual-import pattern**: server modules use try/except import chains (relative → absolute → bare) to work both as an installed package and when run directly from the `pytorch_triage_env/` directory inside Docker.
- **OpenEnv RFC 004 compliance**: the openenv package may or may not be installed. All base classes (`Environment`, `Action`, `Observation`, `State`, `Rubric`, `LLMJudge`) have local stub fallbacks.
- **Validator score range**: scores must be strictly in (0, 1) — not 0.0, not 1.0.
- **`/reset` response format**: returns observation fields at top level, not nested under `"observation"` key. `/step` returns `{observation, reward, done, state}`.


## Hackathon Context: MetaAIxOpenEnv Round 1

This repository is an active submission for Round 1 of the MetaAIxOpenEnv hackathon. All code modifications, environment logic, and infrastructure setups must strictly adhere to the following constraints to pass automated validation and agentic evaluation. 

### 1. Domain & Environment Design
* **Real-World Utility (30% of Score):** The environment must simulate a genuine, practical task that humans perform, such as code review, data cleaning, or email triage. Do not build games or artificial toy problems, as these will receive a 0 in utility evaluation.
* **Strict OpenEnv Compliance:** You must implement the full OpenEnv interface using Pydantic models for `Observation`, `Action`, and `State`. The environment must correctly expose `step()`, `reset()`, and `state()` endpoints.
* **Documentation:** A `README.md` must be maintained containing the environment description, action/observation space definitions, task descriptions (with expected difficulty), setup instructions, and baseline scores.

### 2. Tasks & Grading Mechanics
* **Difficulty Progression (25% of Score):** The environment must contain a minimum of 3 defined tasks that scale in difficulty (easy → medium → hard). The hardest task must genuinely challenge frontier AI models.
* **Deterministic Scoring:** Each task must feature a programmatic grader with clear, deterministic success and failure criteria. Graders must return scores strictly in the `0.0` to `1.0` range.
* **Dense Reward Function:** The reward signal must measure partial progress toward task completion throughout the trajectory, rather than relying on a sparse, binary end-of-episode reward. 
* **Behavioral Penalties:** The reward function must actively penalize undesirable agent behaviors, such as infinite loops or destructive actions.

### 3. Baseline Inference Script
* **Placement & Naming:** A script named `inference.py` must exist at the root directory of the project.
* **Client Configuration:** The script must use the OpenAI API client to interact with the LLM, dynamically reading `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` from environment variables.
* **Strict Logging Format:** The inference script must emit structured stdout logs exactly matching the required formats: `[START]`, `[STEP]`, and `[END]`. Any deviation in field names or ordering will cause the automated evaluator to fail.
* **Resource Limits:** The inference script must complete within 20 minutes and successfully execute on a machine with 2 vCPUs and 8GB of memory.

### 4. Infrastructure & Validation
* **Deployment:** The project must run as a containerized Hugging Face Space tagged with `openenv`. It must respond with an HTTP 200 status code to a `/reset` ping.
* **Dockerization:** A working `Dockerfile` must be included, allowing the environment to start cleanly using standard `docker build` and `docker run` commands.
* **Validation Passing:** The codebase must successfully pass the `openenv validate` command. A valid `openenv.yaml` file must be present.

### 5. Disqualification Triggers (CRITICAL)
Avoid the following under all circumstances to prevent automatic disqualification during Phase 1 judging:
* The environment fails to deploy or respond.
* The environment is plagiarized or trivially modified from an existing OpenEnv environment.
* The graders are broken and repeatedly return the exact same score.
* The `inference.py` baseline script is missing.