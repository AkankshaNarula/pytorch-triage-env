# CLAUDE.md — pytorch_triage_env Build Instructions

## SOURCE OF TRUTH
ALL file content is in the context document. Read the entire context doc before creating any file.

## CREATION ORDER
1.  pytorch_triage_env/server/__init__.py
2.  pytorch_triage_env/server/virtual_fs.py
3.  pytorch_triage_env/server/mock_execution_engine.py  (copy ALL 4 tasks verbatim — do not truncate)
4.  pytorch_triage_env/server/rubrics.py
5.  pytorch_triage_env/server/models.py                 (discriminated union action)
6.  pytorch_triage_env/server/environment.py
7.  pytorch_triage_env/server/app.py                    (try primary, use fallback if needed)
8.  pytorch_triage_env/__init__.py
9.  pytorch_triage_env/openenv.yaml
10. pytorch_triage_env/pyproject.toml
11. pytorch_triage_env/server/requirements.txt
12. pytorch_triage_env/Dockerfile                       (port 7860, NOT 8000)
13. pytorch_triage_env/README.md
14. inference.py                                        (REPO ROOT — MANDATORY)
15. tests/ (all 4 test files)
16. .gitignore

## CRITICAL: DISCRIMINATED UNION
When deserializing TriageAction in app.py fallback:
  from pydantic import TypeAdapter
  _adapter = TypeAdapter(TriageAction)
  action = _adapter.validate_python(request_body)

## OPENENV IMPORTS (try in order)
  from openenv.core.env_server.types import Action, Observation, State
  from openenv.core.env_server.interfaces import Environment
  from openenv.core.env_server import create_app        # alt: .app import create_app
  from openenv.core.rubrics import LLMJudge             # real RFC 004 class
  from openenv.core.rubrics import Rubric               # base class, alt: .base import Rubric
  from openenv.core.env_client import EnvClient
  from openenv.core.client_types import StepResult

## ABSOLUTE DO-NOTS
- Do NOT import torch in mock_execution_engine.py
- Do NOT use port 8000 — use 7860
- Do NOT put inference.py inside pytorch_triage_env/
- Do NOT re-declare done or reward on TriageObservation
- Do NOT return a tuple from step()
- Do NOT make state a regular method (must be @property)
- Do NOT instantiate PyTorchTriageEnv() when calling create_app

## AFTER CREATION
```bash
cd pytorch_triage_env && pip install -e . && cd ..
python tests/test_engine.py
python tests/test_rubrics.py
python tests/test_environment.py
uvicorn pytorch_triage_env.server.app:app --port 7860 &
sleep 3
ENV_URL=http://localhost:7860 python tests/test_server.py
cd pytorch_triage_env && openenv validate && cd ..
docker build -f pytorch_triage_env/Dockerfile -t pytorch-triage-env ./pytorch_triage_env
export HF_TOKEN=hf_... MODEL_NAME=Qwen/Qwen2.5-72B-Instruct ENV_URL=http://localhost:7860
python inference.py
```
