---
title: Code Review Environment
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - code-review
  - real-world
  - rl-environment
  - meta-scalar-hackathon
license: mit
short_description: OpenEnv RL environment for AI code review agents
---

# 🔍 Code Review Environment

An **OpenEnv-compliant** RL environment for AI code review agents. The agent reviews Python code, identifies bugs, and suggests fixes.

## 🌐 Live Demo

Try the interactive Gradio UI above, or use the REST API directly.

## 📡 API Endpoints

| Method | Endpoint  | Description                  |
|--------|-----------|------------------------------|
| POST   | `/reset`  | Start a new episode          |
| POST   | `/step`   | Submit a review action       |
| GET    | `/state`  | Get current episode state    |
| GET    | `/tasks`  | List available tasks         |
| GET    | `/health` | Health check                 |

## ⚡ Quick Usage

```python
import httpx

BASE = "https://your-space.hf.space"

# Reset
obs = httpx.post(f"{BASE}/reset", json={"task_name": "syntax_error_detection"}).json()
print(obs["code_snippet"])

# Step
result = httpx.post(f"{BASE}/step", json={
    "bug_found": True,
    "bug_line": 3,
    "explanation": "Missing colon after for loop",
    "fixed_code": "for num in numbers:"
}).json()
print(f"Score: {result['reward']['value']}")
```

## 📊 Tasks

| Task                      | Difficulty | Description                              |
|---------------------------|------------|------------------------------------------|
| `syntax_error_detection`  | 🟢 Easy   | Find Python syntax errors                |
| `logic_bug_detection`     | 🟡 Medium | Identify logic bugs                      |
| `refactor_and_optimize`   | 🔴 Hard   | Fix inefficiency + optimize              |

## 🏆 Reward Breakdown

| Criterion     | Weight | Logic                          |
|--------------|--------|--------------------------------|
| `bug_found`  | 0.20   | Binary match                   |
| `bug_line`   | 0.25   | Correct within ±1 line         |
| `explanation`| 0.30   | Partial keyword matching       |
| `fixed_code` | 0.25   | Exact or partial match         |

Total reward: 0.0–1.0, with step penalties for repeated low scores.
