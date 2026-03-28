#!/usr/bin/env python3
"""Baseline inference script using the OpenAI Python client.

Environment variables:
  OPENAI_API_KEY   (required)
  OPENAI_BASE_URL  (optional, defaults to https://api.openai.com/v1)
  OPENAI_MODEL     (optional, defaults to gpt-4o-mini)
"""

import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from code_review_env import CodeReviewEnv, Action
from code_review_env.tasks import TASKS


SYSTEM_PROMPT = """\
You are a senior Python code reviewer. You will be given a Python code snippet \
and a task description. Respond ONLY with valid JSON matching this schema — \
no markdown, no preamble, no explanation outside the JSON:

{
  "bug_found": true/false,
  "bug_line": <int or null>,
  "explanation": "<your explanation>",
  "fixed_code": "<corrected code or null>"
}
"""


def build_user_prompt(obs) -> str:
    """Build the user prompt from an observation."""
    prompt = f"## Task\n{obs.task_description}\n\n"
    prompt += f"## Code to Review\n```python\n{obs.code_snippet}\n```\n"
    if obs.previous_feedback:
        prompt += (
            f"\n## Previous Feedback (step {obs.step_number})\n"
            f"{obs.previous_feedback}\n"
            f"\nPlease improve your answer based on this feedback."
        )
    return prompt


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        print("Usage: OPENAI_API_KEY=sk-... python baseline.py")
        sys.exit(1)

    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    env = CodeReviewEnv(max_steps=3)

    print("=" * 60)
    print(f"  Baseline Inference — model: {model}")
    print("=" * 60)

    results: dict = {}

    for task_name in TASKS:
        print(f"\n── {task_name} ──")
        obs = env.reset(task_name)
        best_score = 0.0
        scores: list[float] = []

        for step_idx in range(env.max_steps):
            user_prompt = build_user_prompt(obs)

            try:
                response = client.chat.completions.create(
                    model=model,
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                raw = response.choices[0].message.content.strip()
                # Strip markdown fences if LLM wraps its output
                if raw.startswith("```"):
                    lines = raw.splitlines()
                    lines = [l for l in lines if not l.startswith("```")]
                    raw = "\n".join(lines)

                action = env.parse_action(raw)
            except Exception as e:
                print(f"  Step {step_idx + 1}: parse error — {e}")
                action = Action(
                    bug_found=False,
                    bug_line=None,
                    explanation="Failed to parse response",
                    fixed_code=None,
                )

            result = env.step(action)
            score = result.reward.value
            scores.append(score)
            best_score = max(best_score, score)

            print(f"  Step {step_idx + 1}: score={score:.3f}  done={result.done}")

            if result.done:
                break

            obs = result.observation

        avg_score = sum(scores) / len(scores) if scores else 0.0
        results[task_name] = {
            "best_score": round(best_score, 3),
            "avg_score": round(avg_score, 3),
            "num_steps": len(scores),
            "scores": [round(s, 3) for s in scores],
        }

    # ── Summary table ─────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  {'Task':<30} {'Best':>8} {'Avg':>8}")
    print(f"  {'─' * 30} {'─' * 8} {'─' * 8}")
    total_best = 0.0
    for task_name, data in results.items():
        print(f"  {task_name:<30} {data['best_score']:>8.3f} {data['avg_score']:>8.3f}")
        total_best += data["best_score"]

    mean_best = total_best / len(results) if results else 0.0
    print(f"  {'─' * 30} {'─' * 8}")
    print(f"  {'Overall mean-best':<30} {mean_best:>8.3f}")
    print(f"  Model: {model}")
    print(f"{'=' * 60}")

    # ── Save JSON report ──────────────────────────────────────────
    report = {
        "model": model,
        "mean_best_score": round(mean_best, 3),
        "tasks": results,
    }
    with open("baseline_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nResults saved to baseline_results.json")


if __name__ == "__main__":
    main()
