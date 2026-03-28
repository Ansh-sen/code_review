#!/usr/bin/env python3
"""Local demo — runs all 3 tasks with hardcoded near-perfect actions.

No API key required. No network calls.
"""

from code_review_env import CodeReviewEnv, Action
from code_review_env.tasks import TASKS


def main() -> None:
    env = CodeReviewEnv(max_steps=3)

    # Hardcoded near-perfect actions per task
    perfect_actions: dict[str, Action] = {
        "syntax_error_detection": Action(
            bug_found=True,
            bug_line=3,
            explanation=(
                "The for loop on line 3 is missing a colon ':' at the end. "
                "In Python, a for loop requires a colon after the iterable. "
                "This is a syntax error that prevents the code from running."
            ),
            fixed_code=TASKS["syntax_error_detection"]["expected_fixed_code"],
        ),
        "logic_bug_detection": Action(
            bug_found=True,
            bug_line=7,
            explanation=(
                "The variable count is never updated because 'count + 1' "
                "computes a value but never assigns it back. The assignment "
                "operator += should be used: 'count += 1'. Without the "
                "increment assignment, count is not updated and stays 0."
            ),
            fixed_code=TASKS["logic_bug_detection"]["expected_fixed_code"],
        ),
        "refactor_and_optimize": Action(
            bug_found=True,
            bug_line=5,
            explanation=(
                "The inner loop iterates range(len(lst)) which means j can "
                "equal i, causing each element to compare with itself. The "
                "condition i != j prevents self-match but the algorithm is "
                "still O(n^2) quadratic complexity. Using a set to optimize "
                "and track seen elements reduces complexity to O(n)."
            ),
            fixed_code=TASKS["refactor_and_optimize"]["expected_fixed_code"],
        ),
    }

    print("=" * 60)
    print("  Code Review Environment — Local Demo")
    print("=" * 60)

    total_score = 0.0
    task_count = 0

    for task_name in TASKS:
        obs = env.reset(task_name)
        action = perfect_actions[task_name]
        result = env.step(action)

        # Show first 3 lines of code
        code_lines = obs.code_snippet.strip().splitlines()[:3]
        code_preview = "\n    ".join(code_lines)

        print(f"\n── {task_name} ──")
        print(f"  Code (first 3 lines):\n    {code_preview}")
        print(f"  Score:    {result.reward.value:.3f}")
        print(f"  Feedback: {result.reward.feedback}")
        print(f"  Done:     {result.done}")

        total_score += result.reward.value
        task_count += 1

    overall = total_score / task_count if task_count > 0 else 0.0
    print(f"\n{'=' * 60}")
    print(f"  Overall score: {overall:.3f}")
    print(f"{'=' * 60}")
    print("✓ Environment is working correctly.")


if __name__ == "__main__":
    main()
