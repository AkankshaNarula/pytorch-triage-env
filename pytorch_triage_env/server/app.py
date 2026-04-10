"""
FastAPI app for pytorch_triage_env.
Tries the official openenv create_app first; falls back to manual FastAPI server.
"""
from __future__ import annotations
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import TypeAdapter, ValidationError

try:
    from .models import TriageAction, TriageObservation, TriageState
    from .environment import PyTorchTriageEnv
except ImportError:
    try:
        from pytorch_triage_env.server.models import TriageAction, TriageObservation, TriageState
        from pytorch_triage_env.server.environment import PyTorchTriageEnv
    except ImportError:
        from server.models import TriageAction, TriageObservation, TriageState
        from server.environment import PyTorchTriageEnv

# Always use the manual server so that /health and /schema are fully under our
# control.  create_app (openenv) registers its own /health (returning
# {"status":"healthy"}) and a /schema handler that calls
# action_cls.model_json_schema() — which fails for a discriminated Union alias.
# Our manual server returns the correct formats for both endpoints.
_use_create_app = False

if _use_create_app:  # pragma: no cover — kept for reference only
    try:
        from openenv.core.env_server import create_app
    except ImportError:
        from openenv.core.env_server.app import create_app
    app = create_app(
        PyTorchTriageEnv,
        TriageAction,
        TriageObservation,
        env_name="pytorch-triage-env",
        max_concurrent_envs=4,
    )
else:
    # ── Manual fallback server ────────────────────────────────────────────────
    app = FastAPI(title="pytorch-triage-env", version="1.0.0")
    _env = PyTorchTriageEnv()
    _action_adapter = TypeAdapter(TriageAction)

    @app.post("/reset")
    async def reset(request: Request):
        global _env
        body: dict = {}
        try:
            body = await request.json()
        except Exception:
            pass
        task = body.get("task", "oom_graph_leak")
        _env = PyTorchTriageEnv(task_name=task)
        obs  = _env.reset(**body)
        # OpenEnv validator expects the observation fields at the TOP LEVEL
        # (not nested under an "observation" key)
        return obs.model_dump()

    @app.post("/step")
    async def step(request: Request):
        body = await request.json()
        data = body.get("action", body)
        try:
            action = _action_adapter.validate_python(data)
        except ValidationError as exc:
            return JSONResponse(status_code=422, content={"detail": exc.errors()})
        obs = _env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": obs.reward,
            "done":   obs.done,
            "state":  _env.state.model_dump(),
        }

    @app.get("/state")
    async def get_state():
        return _env.state.model_dump()

    @app.get("/metadata")
    async def metadata():
        return {
            "name": "pytorch_triage_env",
            "version": "1.0.0",
            "description": "PyTorch Training Infrastructure Triage RL Environment",
        }


# ── Always-present endpoints (health + schema) ────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "env":    "pytorch-triage-env",
        "version": "1.0.0",
        "tasks": [
            "oom_graph_leak",
            "fsdp_collective_deadlock",
            "ddp_gradient_hang",
        ],
    }


@app.get("/schema")
async def schema():
    return {
        "observation":   TriageObservation.model_json_schema(),
        "state":         TriageState.model_json_schema(),
        "action_types": [
            "read_file", "edit_file", "execute_bash",
            "view_git_diff", "submit_fix",
        ],
    }


def main():
    import uvicorn
    uvicorn.run(
        "pytorch_triage_env.server.app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
    )


if __name__ == "__main__":
    main()
