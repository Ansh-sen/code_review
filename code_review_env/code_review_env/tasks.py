"""Task definitions and grading logic for the Code Review Environment."""

from typing import TYPE_CHECKING

from .models import Action, Reward

if TYPE_CHECKING:
    from .models import EpisodeState

# ── Task 1: syntax_error_detection (EASY) ────────────────────────────
_SYNTAX_CODE = """\
def calculate_average(numbers):
    total = 0
    for num in numbers
        total += num
    return total / len(numbers)"""

_SYNTAX_FIXED = """\
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)"""

# ── Task 2: logic_bug_detection (MEDIUM) ─────────────────────────────
_LOGIC_CODE = """\
def count_vowels(text):
    \"\"\"Count the number of vowels in a string.\"\"\"
    vowels = 'aeiouAEIOU'
    count = 0
    for char in text:
        if char in vowels:
            count + 1
    return count

# Example usage:
# result = count_vowels("Hello World")
# Expected: 3
# Actual:   0  (bug!)

# The function runs without errors but always returns 0
# because 'count + 1' computes a value that is never stored."""

_LOGIC_FIXED = """\
def count_vowels(text):
    \"\"\"Count the number of vowels in a string.\"\"\"
    vowels = 'aeiouAEIOU'
    count = 0
    for char in text:
        if char in vowels:
            count += 1
    return count

# Example usage:
# result = count_vowels("Hello World")
# Expected: 3
# Actual:   0  (bug!)

# The function runs without errors but always returns 0
# because 'count + 1' computes a value that is never stored."""

# ── Task 3: refactor_and_optimize (HARD) ─────────────────────────────
_REFACTOR_CODE = """\
def find_duplicates(lst):
    \"\"\"Find all duplicate elements in a list.\"\"\"
    duplicates = []
    for i in range(len(lst)):
        for j in range(len(lst)):
            if lst[i] == lst[j] and i != j:
                if lst[i] not in duplicates:
                    duplicates.append(lst[i])
    return duplicates

# This has O(n^2) complexity and also a subtle bug:
# j iterates range(len(lst)) instead of range(i+1, len(lst)),
# causing redundant comparisons and potential issues."""

_REFACTOR_FIXED = """\
def find_duplicates(lst):
    \"\"\"Find all duplicate elements in a list.\"\"\"
    seen = set()
    duplicates = set()
    for item in lst:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    return list(duplicates)"""

# ── TASKS dictionary ──────────────────────────────────────────────────
TASKS: dict = {
    "syntax_error_detection": {
        "description": (
            "The code below contains a Python syntax error. Find the line "
            "with the error and explain what is wrong."
        ),
        "code_snippet": _SYNTAX_CODE,
        "expected_bug_line": 3,
        "expected_explanation_keywords": [
            "colon", ":", "syntax", "for loop", "missing",
        ],
        "expected_fixed_code": _SYNTAX_FIXED,
    },
    "logic_bug_detection": {
        "description": (
            "The function below runs without errors but produces wrong results. "
            "Find the logical bug and explain why the output is incorrect."
        ),
        "code_snippet": _LOGIC_CODE,
        "expected_bug_line": 7,
        "expected_explanation_keywords": [
            "+=", "count", "assignment", "increment", "not updated", "never",
        ],
        "expected_fixed_code": _LOGIC_FIXED,
    },
    "refactor_and_optimize": {
        "description": (
            "The function below is algorithmically incorrect AND inefficient. "
            "Identify the bug, explain the complexity issue, and provide a "
            "corrected, optimized version."
        ),
        "code_snippet": _REFACTOR_CODE,
        "expected_bug_line": 5,
        "expected_explanation_keywords": [
            "itself", "O(n", "quadratic", "i == j", "set", "optimize",
            "complexity",
        ],
        "expected_fixed_code": _REFACTOR_FIXED,
    },
}


def grade_action(action: Action, state: "EpisodeState") -> Reward:
    """Grade an agent's action against the expected answer.

    Scoring breakdown (total capped at 1.0):
      - bug_found:  0.20  (binary match)
      - bug_line:   0.25  (correct if abs(actual - expected) <= 1)
      - explanation: 0.30  (partial keyword match)
      - fixed_code: 0.25  (exact strip match → 0.25, key line → 0.12)

    Special rule: if bug_found is False, total is capped at 0.05.
    Step penalty: if step > 1 AND reward < 0.30, subtract 0.05*(step-1).
    """
    breakdown: dict[str, float] = {}
    feedback_parts: list[str] = []

    # ── 1. bug_found (0.20) ───────────────────────────────────────────
    expected_bug = state.expected_bug_line is not None
    if action.bug_found == expected_bug:
        breakdown["bug_found"] = 0.20
        feedback_parts.append("✓ bug_found correct")
    else:
        breakdown["bug_found"] = 0.00
        feedback_parts.append("✗ bug_found incorrect")

    # ── 2. bug_line (0.25) ────────────────────────────────────────────
    if (
        action.bug_line is not None
        and state.expected_bug_line is not None
        and abs(action.bug_line - state.expected_bug_line) <= 1
    ):
        breakdown["bug_line"] = 0.25
        feedback_parts.append("✓ bug_line correct")
    elif action.bug_line is not None and state.expected_bug_line is not None:
        breakdown["bug_line"] = 0.00
        feedback_parts.append("✗ bug_line incorrect")
    else:
        breakdown["bug_line"] = 0.00
        feedback_parts.append("✗ bug_line missing or not expected")

    # ── 3. explanation (0.30, partial keyword match) ──────────────────
    keywords = state.expected_explanation_keywords
    if keywords:
        explanation_lower = action.explanation.lower()
        matched = sum(1 for kw in keywords if kw.lower() in explanation_lower)
        ratio = matched / len(keywords)
        breakdown["explanation"] = round(ratio * 0.30, 3)
        if ratio >= 0.8:
            feedback_parts.append(f"✓ explanation ({matched}/{len(keywords)} keywords)")
        elif ratio >= 0.4:
            feedback_parts.append(f"~ explanation ({matched}/{len(keywords)} keywords)")
        else:
            feedback_parts.append(f"✗ explanation ({matched}/{len(keywords)} keywords)")
    else:
        breakdown["explanation"] = 0.00
        feedback_parts.append("✗ explanation: no keywords to match")

    # ── 4. fixed_code (0.25) ──────────────────────────────────────────
    if action.fixed_code is not None and state.expected_fixed_code is not None:
        if action.fixed_code.strip() == state.expected_fixed_code.strip():
            breakdown["fixed_code"] = 0.25
            feedback_parts.append("✓ fixed_code exact match")
        else:
            # Partial credit: check if the key fix line is present
            expected_lines = state.expected_fixed_code.strip().splitlines()
            actual_lines = action.fixed_code.strip().splitlines()
            # Find lines that differ between expected fix and original code
            task_data = TASKS.get(state.task_name, {})
            original_lines = task_data.get("code_snippet", "").strip().splitlines()
            key_fix_lines = [
                line.strip()
                for line in expected_lines
                if line.strip() not in [ol.strip() for ol in original_lines]
            ]
            if key_fix_lines and any(
                kfl in " ".join(al.strip() for al in actual_lines)
                for kfl in key_fix_lines
            ):
                breakdown["fixed_code"] = 0.12
                feedback_parts.append("~ fixed_code partial match (key fix present)")
            else:
                breakdown["fixed_code"] = 0.00
                feedback_parts.append("✗ fixed_code incorrect")
    elif action.fixed_code is None and state.expected_fixed_code is not None:
        breakdown["fixed_code"] = 0.00
        feedback_parts.append("✗ fixed_code missing")
    else:
        breakdown["fixed_code"] = 0.00
        feedback_parts.append("✗ fixed_code not applicable")

    # ── Total calculation ─────────────────────────────────────────────
    total = sum(breakdown.values())

    # Special rule: if agent said no bug, cap at 0.05
    if not action.bug_found:
        total = min(total, 0.05)
        breakdown = {k: 0.0 for k in breakdown}
        breakdown["bug_found"] = min(0.05, total)
        feedback_parts = ["✗ Agent said no bug found — reward capped at 0.05"]

    # Step penalty
    if state.step_number > 1 and total < 0.30:
        penalty = 0.05 * (state.step_number - 1)
        total = max(0.0, total - penalty)
        feedback_parts.append(f"⚠ Step penalty: -{penalty:.2f}")

    total = min(1.0, round(total, 3))

    return Reward(
        value=total,
        breakdown=breakdown,
        feedback=" | ".join(feedback_parts),
    )
