"""CodeReviewEnv — the main RL environment class."""

from __future__ import annotations

import json
from typing import Optional

from .models import Action, EpisodeState, Observation, StepResult
from .tasks import TASKS, grade_action


class CodeReviewEnv:
    """OpenEnv-compliant RL environment for AI code review.

    The agent receives a Python code snippet, identifies bugs,
    and suggests fixes over up to ``max_steps`` interactions.
    """

    def __init__(self, max_steps: int = 3) -> None:
        self.max_steps = max_steps
        self._state: Optional[EpisodeState] = None

    # ── public API ────────────────────────────────────────────────────

    def reset(self, task_name: Optional[str] = None) -> Observation:
        """Start a new episode.

        Args:
            task_name: Key from ``TASKS``. Defaults to the first task.

        Returns:
            Initial :class:`Observation`.

        Raises:
            ValueError: If *task_name* is not in ``TASKS``.
        """
        if task_name is None:
            task_name = next(iter(TASKS))

        if task_name not in TASKS:
            raise ValueError(
                f"Unknown task '{task_name}'. "
                f"Available tasks: {list(TASKS.keys())}"
            )

        task = TASKS[task_name]
        self._state = EpisodeState(
            task_name=task_name,
            code_snippet=task["code_snippet"],
            expected_bug_line=task["expected_bug_line"],
            expected_explanation_keywords=task["expected_explanation_keywords"],
            expected_fixed_code=task["expected_fixed_code"],
            step_number=0,
            max_steps=self.max_steps,
            done=False,
            cumulative_reward=0.0,
        )
        return self._make_observation()

    def step(self, action: Action) -> StepResult:
        """Execute one step.

        Args:
            action: The agent's :class:`Action`.

        Returns:
            A :class:`StepResult` with observation, reward, done flag, and info.

        Raises:
            RuntimeError: If :meth:`reset` has not been called or the episode
                is already done.
        """
        if self._state is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")
        if self._state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        # Increment step before grading
        self._state.step_number += 1

        # Grade the action
        reward = grade_action(action, self._state)

        # Update cumulative reward
        self._state.cumulative_reward += reward.value

        # Check termination
        perfect = reward.value >= 0.95
        done = perfect or self._state.step_number >= self._state.max_steps
        self._state.done = done

        # Build observation with feedback
        observation = self._make_observation(previous_feedback=reward.feedback)

        info = {
            "step": self._state.step_number,
            "cumulative_reward": round(self._state.cumulative_reward, 3),
            "task": self._state.task_name,
            "perfect": perfect,
        }

        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
        )

    def state(self) -> dict:
        """Return the current episode state as a plain dict."""
        if self._state is None:
            return {"status": "not_started"}
        return self._state.model_dump()

    @staticmethod
    def parse_action(json_str: str) -> Action:
        """Parse a raw JSON string (e.g. from an LLM) into an :class:`Action`.

        Args:
            json_str: JSON string matching the Action schema.

        Returns:
            Parsed :class:`Action` instance.
        """
        data = json.loads(json_str)
        return Action(**data)

    # ── private helpers ───────────────────────────────────────────────

    def _make_observation(
        self, previous_feedback: Optional[str] = None
    ) -> Observation:
        """Build an Observation from the current episode state."""
        assert self._state is not None
        task = TASKS[self._state.task_name]
        return Observation(
            code_snippet=self._state.code_snippet,
            task_name=self._state.task_name,
            task_description=task["description"],
            step_number=self._state.step_number,
            max_steps=self._state.max_steps,
            previous_feedback=previous_feedback,
        )
