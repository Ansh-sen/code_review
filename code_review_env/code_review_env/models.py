"""Pydantic models for the Code Review Environment."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Observation(BaseModel):
    """Observation returned to the agent after reset() or step()."""

    code_snippet: str = Field(..., description="Python code to review")
    task_name: str = Field(..., description="One of the task keys")
    task_description: str = Field(..., description="Human-readable task instructions")
    step_number: int = Field(default=0, description="Current step in episode")
    max_steps: int = Field(default=3, description="Maximum steps allowed")
    previous_feedback: Optional[str] = Field(
        default=None, description="Feedback from last step"
    )


class Action(BaseModel):
    """Action submitted by the agent."""

    bug_found: bool = Field(..., description="Did the agent find a bug?")
    bug_line: Optional[int] = Field(
        default=None, description="1-indexed line number, or None"
    )
    explanation: str = Field(..., description="Agent's explanation")
    fixed_code: Optional[str] = Field(
        default=None, description="Corrected code, or None"
    )


class Reward(BaseModel):
    """Reward returned after grading an action."""

    value: float = Field(..., ge=0.0, le=1.0, description="Reward value 0.0–1.0")
    breakdown: Dict[str, float] = Field(..., description="Per-criterion scores")
    feedback: str = Field(..., description="Human-readable feedback string")


class StepResult(BaseModel):
    """Result returned by step()."""

    observation: Observation
    reward: Reward
    done: bool
    info: Dict = Field(..., description="Step metadata")


class EpisodeState(BaseModel):
    """Internal state of a running episode."""

    task_name: str
    code_snippet: str
    expected_bug_line: Optional[int]
    expected_explanation_keywords: List[str]
    expected_fixed_code: Optional[str]
    step_number: int = 0
    max_steps: int = 3
    done: bool = False
    cumulative_reward: float = 0.0
