#!/usr/bin/env python3
"""Baseline inference script using the Google Gemini API.

Environment variables:
  GEMINI_API_KEY   (required)
  GEMINI_MODEL     (optional, defaults to gemini-2.0-flash)
"""

import json
import os
import sys
import time

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
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable is not set.")
        print("Usage: GEMINI_API_KEY=... python baseline.py")
        sys.exit(1)

    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=SYSTEM_PROMPT,
        generation_config=genai.GenerationConfig(temperature=0.0),
    )

    env = CodeReviewEnv(max_steps=3)

    print("=" * 60)
    print(f"  Baseline Inference — model: {model_name}")
    print("=" * 60)

    results: dict = {}

    for task_idx, task_name in enumerate(TASKS):
        if task_idx > 0:
            print("  (waiting 10s between tasks to avoid rate limits...)")
            time.sleep(10)
        print(f"\n── {task_name} ──")
        obs = env.reset(task_name)
        best_score = 0.0
        scores: list[float] = []

        for step_idx in range(env.max_steps):
            user_prompt = build_user_prompt(obs)

            try:
                # Retry logic for free-tier rate limits
                raw = None
                for attempt in range(3):
                    try:
                        response = model.generate_content(user_prompt)
                        raw = response.text.strip()
                        break
                    except Exception as api_err:
                        if attempt < 2:
                            wait = 60 * (attempt + 1)
                            print(f"  Step {step_idx + 1}: rate limit, retrying in {wait}s...")
                            import time
                            time.sleep(wait)
                        else:
                            raise api_err

                # Strip markdown fences (```json ... ``` or ``` ... ```)
                if "```" in raw:
                    import re
                    match = re.search(r"```(?:json)?\s*\n?(.*?)```", raw, re.DOTALL)
                    if match:
                        raw = match.group(1).strip()

                # Try to extract JSON object if there's extra text
                if not raw.startswith("{"):
                    start = raw.find("{")
                    end = raw.rfind("}") + 1
                    if start != -1 and end > start:
                        raw = raw[start:end]

                action = env.parse_action(raw)
            except Exception as e:
                print(f"  Step {step_idx + 1}: parse error — {e}")
                try:
                    print(f"  Raw response: {response.text[:200]}")
                except Exception:
                    pass
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
    print(f"  Model: {model_name}")
    print(f"{'=' * 60}")

    # ── Save JSON report ──────────────────────────────────────────
    report = {
        "model": model_name,
        "mean_best_score": round(mean_best, 3),
        "tasks": results,
    }
    with open("baseline_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nResults saved to baseline_results.json")


if __name__ == "__main__":
    main()
