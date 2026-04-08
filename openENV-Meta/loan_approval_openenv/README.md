---
title: AI Loan Approval Evaluator
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
python_version: "3.10"
app_file: app.py
pinned: false
---

# AI Loan Approval and Risk Optimization Environment

## Problem Description
This environment simulates a real-world bank loan approval system. An AI agent evaluates customer financial variables (income, credit score, loan amount, employment years) and makes a decision to safely maximize profit while analyzing risk profiles and potentially fraudulent applications.

## Environment Explanation
The environment provides randomized, realistic customer data. The state is presented to an AI agent which must return a discrete action.
Depending on the decision quality relative to the specific task difficulty, the environment calculates and returns a reward, representing the agent's performance.

### Observation Space
The state is represented as a dictionary with the following variables:
- `income`: integer between 20,000 and 100,000
- `credit_score`: integer between 300 and 850
- `loan_amount`: integer between 5,000 and 50,000
- `employment_years`: integer between 0 and 10

### Action Space
Discrete text actions:
- `approve`: Authorize the loan.
- `reject`: Decline the loan.
- `review`: Recommend human review.

### Reward Function
Rewards are strictly bound between `0.0` and `1.0`.
- **Correct safe approval:** `1.0`
- **Risky approval:** `0.2` - `0.5`
- **Bad approval leading to default:** `0.0`
- **Correct rejection of risky applicant:** `1.0`
- **Fraud prevented:** `1.0`
- **Partial progress:** (e.g., flagging a mid-risk score for review) returns rewards between `0.3` and `0.7`.

### Tasks Description
- **Task 1 — Easy**: Uses simple rule-based evaluation mapped cleanly from `credit_score`.
- **Task 2 — Medium**: Applies multiple features including Debt-to-Income evaluations, weighing `credit_score`, `income`, and `loan_amount`. 
- **Task 3 — Hard**: Injects rigorous systemic reviews for fraud and major risk detection, introducing `employment_years` alongside extremely high loan bounds and low credit scores.

## Setup Instructions

### How to Run Locally
1. Clone or download this project.
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. Run the baseline agent to see example environment interactions and scores:
```bash
python run_agent.py
```
4. Evaluate all configured tasks with the built-in grading logic:
```bash
python tasks.py
```
5. Execute the automated LLM inference script (required for the Hackathon Submission Portal):
```bash
python inference.py
```

### How to Run with Docker
1. Build the Docker image:
```bash
docker build -t loan-env .
```
2. Run the image (Defaults to the baseline agent):
```bash
docker run loan-env
```

### How to Deploy to Hugging Face
This repository is pre-configured with a user-friendly Gradio web application explicitly designed for **Hugging Face Spaces**.
1. Navigate to your Hugging Face account and create a new Space.
2. Choose **Gradio** as the Space SDK.
3. Upload all the files inside this repository to your Space.
4. The space will build the environment and automatically map it to the interactive `app.py` script. The UI displays the current state, action interactions, and subsequent reward logic securely!
