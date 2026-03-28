#!/usr/bin/env python3
"""57-check OpenEnv compliance validation script.

Run:  python validate.py
All 57 checks must pass (exit code 0).
"""

import sys
import yaml

from code_review_env import CodeReviewEnv, Action, Observation, Reward, StepResult, TASKS


passed = 0
failed = 0
total_checks = 57


def check(description: str, condition: bool) -> None:
    """Record a single check result."""
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✓ {description}")
    else:
        failed += 1
        print(f"  ✗ {description}")


def section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def main() -> None:
    env = CodeReviewEnv(max_steps=3)

    # ══════════════════════════════════════════════════════════════
    # Section 1: openenv.yaml validation (7 checks)
    # ══════════════════════════════════════════════════════════════
    section("1. openenv.yaml validation")

    try:
        with open("openenv.yaml", "r") as f:
            manifest = yaml.safe_load(f)
        check("openenv.yaml exists and is valid YAML", True)
    except Exception:
        check("openenv.yaml exists and is valid YAML", False)
        manifest = {}

    check("manifest has 'name' field", "name" in manifest)
    check("manifest has 'version' field", "version" in manifest)
    check("manifest has 'tasks' field with >= 3 entries",
          isinstance(manifest.get("tasks"), list) and len(manifest.get("tasks", [])) >= 3)
    check("manifest has 'observation_space' field", "observation_space" in manifest)
    check("manifest has 'action_space' field", "action_space" in manifest)
    check("manifest has 'reward_range' [0.0, 1.0]",
          manifest.get("reward_range") == [0.0, 1.0])

    # ══════════════════════════════════════════════════════════════
    # Section 2: reset() validation (12 checks)
    # ══════════════════════════════════════════════════════════════
    section("2. reset() validation")

    task_names = ["syntax_error_detection", "logic_bug_detection", "refactor_and_optimize"]

    for task_name in task_names:
        obs = env.reset(task_name)
        check(f"reset('{task_name}') returns Observation",
              isinstance(obs, Observation))
        check(f"  step_number == 0", obs.step_number == 0)
        check(f"  task_name matches", obs.task_name == task_name)
        check(f"  code_snippet is non-empty",
              isinstance(obs.code_snippet, str) and len(obs.code_snippet) > 0)

    # ══════════════════════════════════════════════════════════════
    # Section 3: step() validation (12 checks)
    # ══════════════════════════════════════════════════════════════
    section("3. step() validation")

    for task_name in task_names:
        env.reset(task_name)
        action = Action(
            bug_found=True,
            bug_line=1,
            explanation="test explanation",
            fixed_code="test code",
        )
        result = env.step(action)
        check(f"step('{task_name}') returns StepResult",
              isinstance(result, StepResult))
        check(f"  observation/reward/done types correct",
              isinstance(result.observation, Observation)
              and isinstance(result.reward, Reward)
              and isinstance(result.done, bool))
        check(f"  info is dict with required keys",
              isinstance(result.info, dict)
              and "step" in result.info
              and "cumulative_reward" in result.info
              and "task" in result.info)
        check(f"  reward value in [0.0, 1.0]",
              0.0 <= result.reward.value <= 1.0)

    # ══════════════════════════════════════════════════════════════
    # Section 4: Reward range — bad actions (3 checks)
    # ══════════════════════════════════════════════════════════════
    section("4. Reward range — bad actions")

    for task_name in task_names:
        env.reset(task_name)
        bad_action = Action(
            bug_found=False,
            bug_line=None,
            explanation="",
            fixed_code=None,
        )
        result = env.step(bad_action)
        check(f"bad action '{task_name}' reward in [0.0, 1.0]",
              0.0 <= result.reward.value <= 1.0)

    # ══════════════════════════════════════════════════════════════
    # Section 4b: Reward range — good actions (3 checks)
    # ══════════════════════════════════════════════════════════════
    section("4b. Reward range — good actions")

    good_actions = {
        "syntax_error_detection": Action(
            bug_found=True, bug_line=3,
            explanation="missing colon : in for loop syntax error",
            fixed_code=TASKS["syntax_error_detection"]["expected_fixed_code"],
        ),
        "logic_bug_detection": Action(
            bug_found=True, bug_line=7,
            explanation="count += assignment increment not updated never",
            fixed_code=TASKS["logic_bug_detection"]["expected_fixed_code"],
        ),
        "refactor_and_optimize": Action(
            bug_found=True, bug_line=5,
            explanation="itself O(n quadratic i == j set optimize complexity",
            fixed_code=TASKS["refactor_and_optimize"]["expected_fixed_code"],
        ),
    }

    for task_name in task_names:
        env.reset(task_name)
        result = env.step(good_actions[task_name])
        check(f"good action '{task_name}' reward in (0.5, 1.0]",
              0.5 < result.reward.value <= 1.0)

    # ══════════════════════════════════════════════════════════════
    # Section 5: Partial rewards (6 checks)
    # ══════════════════════════════════════════════════════════════
    section("5. Partial reward validation")

    for task_name in task_names:
        env.reset(task_name)
        partial_action = Action(
            bug_found=True,
            bug_line=99,
            explanation="something might be wrong",
            fixed_code=None,
        )
        result = env.step(partial_action)
        check(f"partial '{task_name}' gives 0.0 < reward < 1.0",
              0.0 < result.reward.value < 1.0)
        check(f"partial '{task_name}' breakdown has multiple keys",
              len(result.reward.breakdown) > 1)

    # ══════════════════════════════════════════════════════════════
    # Section 6: state() validation (6 checks)
    # ══════════════════════════════════════════════════════════════
    section("6. state() validation")

    env.reset("syntax_error_detection")
    state = env.state()
    check("state() returns dict", isinstance(state, dict))
    check("state has 'task_name'", "task_name" in state)
    check("state has 'step_number'", "step_number" in state)
    check("state has 'done'", "done" in state)
    check("state has 'cumulative_reward'", "cumulative_reward" in state)

    # state before reset
    env_fresh = CodeReviewEnv(max_steps=3)
    check("state() before reset returns {'status': 'not_started'}",
          env_fresh.state() == {"status": "not_started"})

    # ══════════════════════════════════════════════════════════════
    # Section 7: done flag validation (2 checks)
    # ══════════════════════════════════════════════════════════════
    section("7. done flag validation")

    # Done on max_steps exhaustion
    env.reset("syntax_error_detection")
    bad = Action(bug_found=True, bug_line=99, explanation="wrong", fixed_code=None)
    for _ in range(3):
        result = env.step(bad)
    check("done=True after max_steps exhaustion", result.done is True)

    # Done on near-perfect score
    env.reset("syntax_error_detection")
    result = env.step(good_actions["syntax_error_detection"])
    check("done=True on near-perfect score (>= 0.95)", result.done is True)

    # ══════════════════════════════════════════════════════════════
    # Section 8: Error handling (2 checks)
    # ══════════════════════════════════════════════════════════════
    section("8. Error handling")

    try:
        env.reset("nonexistent_task")
        check("ValueError on unknown task", False)
    except ValueError:
        check("ValueError on unknown task", True)

    env2 = CodeReviewEnv(max_steps=3)
    try:
        env2.step(Action(bug_found=True, bug_line=1,
                         explanation="test", fixed_code=None))
        check("RuntimeError before reset()", False)
    except RuntimeError:
        check("RuntimeError before reset()", True)

    # ══════════════════════════════════════════════════════════════
    # Section 9: Minimum 3 tasks (4 checks)
    # ══════════════════════════════════════════════════════════════
    section("9. Minimum 3 tasks")

    check("len(TASKS) >= 3", len(TASKS) >= 3)
    check("'syntax_error_detection' in TASKS",
          "syntax_error_detection" in TASKS)
    check("'logic_bug_detection' in TASKS",
          "logic_bug_detection" in TASKS)
    check("'refactor_and_optimize' in TASKS",
          "refactor_and_optimize" in TASKS)

    # ══════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print(f"  ✓ Passed: {passed}/{total_checks}")
    if failed == 0:
        print("  All checks passed.")
    else:
        print(f"  ✗ Failed: {failed}/{total_checks}")
    print(f"{'=' * 60}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
