# 🔍 Code Review Environment

An **OpenEnv-compliant** reinforcement learning environment where an AI agent reviews Python code snippets, identifies bugs, and suggests fixes.

Built for the **Meta Scalar × OpenEnv Hackathon**.

---

## ✨ Features

- **Real-world task**: Simulates professional code review — not a toy/game
- **3 difficulty levels**: Easy (syntax errors), Medium (logic bugs), Hard (refactor & optimize)
- **Partial rewards**: Graded scoring across 4 criteria (0.0–1.0)
- **OpenEnv compliant**: Typed Pydantic models, REST API, `openenv.yaml`
- **Interactive Gradio UI**: Try it in your browser
- **Baseline agent**: OpenAI-powered inference script included

## 📂 Project Structure

```
code_review_env/
├── code_review_env/       # Core RL environment package
│   ├── __init__.py
│   ├── models.py          # Pydantic models
│   ├── environment.py     # CodeReviewEnv class
│   └── tasks.py           # Tasks + grading logic
├── server/
│   ├── __init__.py
│   └── app.py             # FastAPI REST API
├── demo.py                # Local demo (no API key)
├── validate.py            # 57-check compliance test
├── baseline.py            # OpenAI baseline agent
├── gradio_demo.py         # Gradio interactive UI
├── launch.py              # Combined launcher (port 7860)
├── openenv.yaml           # OpenEnv manifest
├── Dockerfile             # HuggingFace Spaces ready
├── requirements.txt
├── setup.py
└── README.md
```

## 🚀 Quick Start

### Install

```bash
pip install -r requirements.txt
pip install -e .
```

### Run Demo (no API key needed)

```bash
python demo.py
```

### Run Validation (57 checks)

```bash
python validate.py
```

### Run Baseline Agent

```bash
OPENAI_API_KEY=sk-... python baseline.py
```

### Launch Server + UI

```bash
python launch.py
# Open http://localhost:7860
```

## 🌐 API Endpoints

| Method | Endpoint  | Description                  |
|--------|-----------|------------------------------|
| POST   | `/reset`  | Start a new episode          |
| POST   | `/step`   | Submit an action             |
| GET    | `/state`  | Get current episode state    |
| GET    | `/tasks`  | List available tasks         |
| GET    | `/health` | Health check                 |

### Example Usage

```python
import httpx

# Start a new episode
resp = httpx.post("http://localhost:7860/reset", json={"task_name": "syntax_error_detection"})
obs = resp.json()
print(obs["code_snippet"])

# Submit a review
action = {
    "bug_found": True,
    "bug_line": 3,
    "explanation": "Missing colon in for loop",
    "fixed_code": "def calculate_average(numbers):\n    total = 0\n    for num in numbers:\n        total += num\n    return total / len(numbers)"
}
resp = httpx.post("http://localhost:7860/step", json=action)
result = resp.json()
print(f"Score: {result['reward']['value']}")
```

## 📊 Tasks

| Task                      | Difficulty | Description                              |
|---------------------------|------------|------------------------------------------|
| `syntax_error_detection`  | 🟢 Easy   | Find and fix Python syntax errors        |
| `logic_bug_detection`     | 🟡 Medium | Identify logic bugs causing wrong output |
| `refactor_and_optimize`   | 🔴 Hard   | Fix bugs AND optimize algorithm          |

## 🏆 Reward Breakdown

| Criterion    | Weight | Logic                                    |
|-------------|--------|------------------------------------------|
| `bug_found` | 0.20   | Binary match                             |
| `bug_line`  | 0.25   | Correct if within ±1 line                |
| `explanation`| 0.30  | Partial keyword matching                 |
| `fixed_code`| 0.25   | Exact match (0.25) or key fix (0.12)     |

**Special rules:**
- If `bug_found=False` → reward capped at 0.05
- Step penalty: −0.05 × (step − 1) if reward < 0.30 and step > 1

## 📄 License

MIT
