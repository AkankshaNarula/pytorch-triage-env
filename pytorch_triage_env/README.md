# pytorch_triage_env

**PyTorch Training Infrastructure Triage** — OpenEnv Hackathon India 2026

An RL environment where an LLM agent acts as a Staff ML Engineer debugging
real-world PyTorch training failures. The agent reads virtual files, edits
code, runs mock training (with authentic error traces), and submits a
verified fix with a technical explanation scored by an LLM judge.

---

## Tasks

| Task | Difficulty | Bug Pattern |
|---|---|---|
| `oom_graph_leak` | Medium | `epoch_loss += loss` retains computation graph → OOM at epoch 3 |
| `fsdp_collective_deadlock` | Hard | `dist.all_reduce` inside `if rank == 0` → NCCL Watchdog timeout |
| `compile_graph_break` | Hard | `loss.item()` inside `@torch.compile` function → silent 3.4× slowdown |
| `ddp_gradient_hang` | Expert | `find_unused_parameters=False` with conditional forward → DDP hang |

---

## Quick Start

```bash
# Install
pip install -e pytorch_triage_env/

# Start server
uvicorn pytorch_triage_env.server.app:app --host 0.0.0.0 --port 7860

# Health check
curl http://localhost:7860/health

# Reset to a task
curl -X POST http://localhost:7860/reset \
  -H 'Content-Type: application/json' \
  -d '{"task": "oom_graph_leak"}'

# Run the baseline inference agent
export HF_TOKEN="hf_..."
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export ENV_URL="http://localhost:7860"
python inference.py
```

## Docker

```bash
docker build -f pytorch_triage_env/Dockerfile -t pytorch-triage-env ./pytorch_triage_env
docker run -p 7860:7860 pytorch-triage-env
```

## Scoring

| Component | Weight | Description |
|---|---|---|
| Trajectory rewards | ~0.3 | Per-step rewards for reading relevant files, using diagnostic flags, making edits |
| Terminal reward | 1.0 × judge | Base 1.0 if fix verified, multiplied by LLMJudge score (0.5–1.0) |
| Efficiency bonus | +0.1 | Bonus for solving with steps to spare |
| Partial credit | 0–0.3 | Awarded for partially-correct fixes |

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check — returns 200 with task list |
| `/reset` | POST | Start new episode. Body: `{"task": "<task_name>"}` |
| `/step` | POST | Execute action. Body: action JSON object |
| `/state` | GET | Current episode state |
| `/schema` | GET | JSON schema for observation and state |

## Action Types

```json
{"action_type": "read_file",    "filename": "train.py"}
{"action_type": "edit_file",    "filename": "train.py", "old_str": "...", "new_str": "..."}
{"action_type": "execute_bash", "command": "python train.py"}
{"action_type": "view_git_diff","filename": null}
{"action_type": "submit_fix",   "explanation": "Root cause: ... Fix: ... Prevention: ..."}
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `HF_TOKEN` | Yes | HuggingFace API key |
| `MODEL_NAME` | Yes | LLM model (e.g. `Qwen/Qwen2.5-72B-Instruct`) |
| `ENV_URL` | No | Server URL (default: `http://localhost:7860`) |
| `JUDGE_MODEL` | No | LLM for explanation scoring (uses keyword fallback if unset) |
| `JUDGE_API_BASE` | No | Judge API base URL |
