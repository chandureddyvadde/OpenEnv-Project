"""
DevOps Incident Response OpenEnv – FastAPI server
Entry point: server/app.py
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from env import get_env
from tasks import list_tasks

app = FastAPI(
    title="DevOps Incident Response OpenEnv",
    description="OpenEnv environment for DevOps / SRE incident response simulation",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> Dict[str, str]:
    """Liveness probe – must return exactly {\"status\": \"ok\"}."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@app.get("/tasks")
async def get_tasks() -> Dict[str, Any]:
    """Return a description of all available tasks."""
    return {"tasks": list_tasks()}


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

@app.post("/reset")
async def reset(request: Request) -> Dict[str, Any]:
    """
    Reset the environment to the beginning of a task.

    Accepts an optional JSON body: {"task_id": "<id>"}
    If the body is absent or empty, the default task is used.
    """
    task_id: Optional[str] = None

    # Safely parse optional body
    try:
        raw_body = await request.body()
        if raw_body:
            try:
                body = json.loads(raw_body)
                if isinstance(body, dict):
                    task_id = body.get("task_id")
            except (json.JSONDecodeError, ValueError):
                pass  # Ignore malformed bodies; use default task
    except Exception:
        pass

    env = get_env()
    result = env.reset(task_id=task_id)

    # result is always a plain dict
    return {
        "observation": result.get("observation", ""),
        "task_id": result.get("task_id", ""),
        "difficulty": result.get("difficulty", ""),
        "max_steps": result.get("max_steps", 0),
        "info": result.get("info", {}),
    }


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

@app.post("/step")
async def step(request: Request) -> Dict[str, Any]:
    """
    Execute one action in the environment.

    Expected body: {"action": "<command string>"}
    """
    try:
        raw_body = await request.body()
        body = json.loads(raw_body)
        action: str = body.get("action", "")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid request body: {exc}")

    if not isinstance(action, str) or not action.strip():
        raise HTTPException(status_code=422, detail="Field 'action' must be a non-empty string.")

    env = get_env()
    result = env.step(action=action.strip())

    return {
        "observation": result.get("observation", ""),
        "reward": result.get("reward", 0.0),
        "done": result.get("done", False),
        "info": result.get("info", {}),
    }


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@app.get("/state")
async def state() -> Dict[str, Any]:
    """Return the current environment state (read-only snapshot)."""
    env = get_env()
    return env.state()


# ---------------------------------------------------------------------------
# Required entrypoint helpers
# ---------------------------------------------------------------------------

def main():
    """Return the FastAPI app object (required by pyproject.toml script)."""
    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)
