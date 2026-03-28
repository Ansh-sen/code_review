"""FastAPI server exposing the Code Review Environment over HTTP."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from code_review_env import CodeReviewEnv, Action

app = FastAPI(
    title="Code Review Environment API",
    description="OpenEnv-compliant RL environment for AI code review.",
    version="0.1.0",
)

# Global environment instance
env = CodeReviewEnv(max_steps=3)


class ResetRequest(BaseModel):
    task_name: Optional[str] = None


@app.post("/reset")
def reset_endpoint(req: ResetRequest):
    """Reset the environment and start a new episode."""
    try:
        obs = env.reset(task_name=req.task_name)
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step_endpoint(action: Action):
    """Submit an action and receive the step result."""
    try:
        result = env.step(action)
        return result.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state_endpoint():
    """Return the current episode state."""
    return env.state()


@app.get("/tasks")
def tasks_endpoint():
    """Return available tasks with difficulty and description."""
    from code_review_env.tasks import TASKS

    return {
        name: {
            "difficulty": (
                "easy" if name == "syntax_error_detection"
                else "medium" if name == "logic_bug_detection"
                else "hard"
            ),
            "description": task["description"],
        }
        for name, task in TASKS.items()
    }


@app.get("/health")
def health_endpoint():
    """Health check endpoint."""
    return {
        "status": "ok",
        "env": "code-review-env",
        "version": "0.1.0",
    }
