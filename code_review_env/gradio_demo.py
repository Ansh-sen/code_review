#!/usr/bin/env python3
"""Gradio 4.x interactive demo for the Code Review Environment."""

import gradio as gr

from code_review_env import CodeReviewEnv, Action
from code_review_env.tasks import TASKS

# ── Shared environment instance ───────────────────────────────────────
env = CodeReviewEnv(max_steps=3)

# ── Task label mapping ────────────────────────────────────────────────
TASK_LABELS = {
    "🟢 Easy — Syntax Error Detection": "syntax_error_detection",
    "🟡 Medium — Logic Bug Detection": "logic_bug_detection",
    "🔴 Hard — Refactor & Optimize": "refactor_and_optimize",
}


def start_task(task_label: str):
    """Reset the environment for the selected task."""
    task_name = TASK_LABELS.get(task_label)
    if task_name is None:
        return "⚠ Please select a task.", ""

    obs = env.reset(task_name)
    desc_md = f"### 📋 {task_label}\n\n{obs.task_description}"
    code_md = f"```python\n{obs.code_snippet}\n```"
    return desc_md, code_md


def submit_review(
    bug_found: bool,
    bug_line: float | None,
    explanation: str,
    fixed_code: str,
):
    """Submit the agent's review and return formatted feedback."""
    try:
        action = Action(
            bug_found=bug_found,
            bug_line=int(bug_line) if bug_line is not None and bug_line > 0 else None,
            explanation=explanation or "",
            fixed_code=fixed_code.strip() if fixed_code and fixed_code.strip() else None,
        )
        result = env.step(action)
    except RuntimeError as e:
        return f"⚠ {e}"

    # Build score bar
    score = result.reward.value
    filled = int(score * 23)
    bar = "█" * filled + "░" * (23 - filled)

    md = f"## Score: {score:.3f}  [{bar}]\n\n"
    md += "### Breakdown\n"
    for key, val in result.reward.breakdown.items():
        md += f"- **{key}**: {val:.3f}\n"
    md += f"\n### Feedback\n{result.reward.feedback}\n"

    if result.done:
        md += "\n✅ **Episode complete!** Start a new task above."

    return md


# ── Gradio UI ─────────────────────────────────────────────────────────
def create_demo() -> gr.Blocks:
    """Build and return the Gradio Blocks demo."""
    with gr.Blocks(theme=gr.themes.Soft(), title="Code Review Environment") as demo:
        gr.Markdown("# 🔍 Code Review Environment\n*OpenEnv RL environment for AI code review*")

        with gr.Row():
            with gr.Column(scale=1):
                task_dropdown = gr.Dropdown(
                    choices=list(TASK_LABELS.keys()),
                    label="Select Task",
                    value=list(TASK_LABELS.keys())[0],
                )
                start_btn = gr.Button("▶ Start Task", variant="primary")
                task_desc = gr.Markdown("Select a task and click **▶ Start Task**.")
            with gr.Column(scale=2):
                code_display = gr.Markdown("*Code will appear here after starting a task.*")

        gr.Markdown("---")
        gr.Markdown("### ✍️ Your Review")

        with gr.Row():
            with gr.Column():
                bug_found = gr.Checkbox(label="Bug found?", value=True)
                bug_line = gr.Number(label="Bug line number", precision=0)
            with gr.Column():
                explanation = gr.Textbox(label="Explanation", lines=3)
                fixed_code = gr.Textbox(label="Fixed code", lines=5)

        submit_btn = gr.Button("📤 Submit Review", variant="primary")
        feedback_output = gr.Markdown("*Submit your review to see results.*")

        # Wire up events
        start_btn.click(
            fn=start_task,
            inputs=[task_dropdown],
            outputs=[task_desc, code_display],
        )
        submit_btn.click(
            fn=submit_review,
            inputs=[bug_found, bug_line, explanation, fixed_code],
            outputs=[feedback_output],
        )

    return demo


demo = create_demo()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
