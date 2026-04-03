# 🚀 SamaarthyaSetu — AI Employment Agent for Inclusive Hiring

### सामर्थ्यसेतु — Bridge of Enablement

An explainable AI system that fixes the broken employment pipeline for persons with disabilities.

---

## 🌍 The Problem

India has 40–90 million persons with disabilities, yet less than 15% are formally employed.

After training programs:

* NGOs mark candidates as “placed”
* Employers often have no record
* Candidates are still unemployed

The system lacks:

* accountability
* intelligence
* follow-up

---

## 💡 Our Solution

SamaarthyaSetu is an AI agent that actively navigates and fixes this pipeline.

It:

* Matches candidates to jobs
* Finds eligible government schemes
* Resolves placement conflicts
* Schedules interviews
* Updates dashboards

This is not a chatbot.
This is a decision-making AI agent with actions and reasoning.

---

## 🎥 Live Demo

App: https://kalimx03-samaarthyasetu.hf.space

API Docs: https://kalimx03-samaarthyasetu.hf.space/docs

GitHub: https://github.com/kalimx03/samaarthyasetu


---

## ⚙️ What the Agent Can Do

### 🟢 Task 1 — Job Matching

Matches candidate to job using:

* skills
* accessibility fit
* language

---

### 🟡 Task 2 — Scheme Navigator

* Finds eligible government schemes
* Generates checklist
* Prioritises benefits

---

### 🔴 Task 3 — Placement Reconciliation

* Detects NGO vs employer mismatch
* Resolves conflict
* Schedules interview
* Updates dashboard

---

## 🧠 Explainable AI (Core Feature)

Every decision is visible:

🚀 ACTION: get_candidate
📊 ENV: Candidate retrieved

🚀 ACTION: check_scheme_eligibility
📊 ENV: 9 schemes eligible

🚀 ACTION: generate_checklist
📊 ENV: Checklist generated

---

## 🧩 Modes

Rules  → Deterministic baseline (no API needed)
Hybrid → Rules + AI fallback
LLM    → Fully AI-driven (Groq powered)

---

## 🏗️ Architecture

Agent → Environment → Actions → Reward → State → Agent

---

## 📊 Results

Task 1: 0.97
Task 2: 1.00
Task 3: 1.00

Average: 0.99

---

## ⚡ Tech Stack

Frontend: Gradio
Backend: FastAPI
AI: Groq LLM (LLaMA 3)
ML: PyTorch
Validation: Pydantic
Testing: Pytest

---

## 🚀 Quickstart

git clone https://github.com/kalimx03/samaarthyasetu
cd samaarthyasetu
pip install -r requirements.txt

Run baseline:
python inference.py --mode rules

Run AI agent:
export GROQ_API_KEY=your_key
python inference.py --mode llm --task task_001

---

## 🌱 Real-World Impact

Aligned with:

SDG 8 — Decent Work
SDG 10 — Reduced Inequalities
SDG 11 — Sustainable Cities

---

## 🧪 Engineering Quality

* Deterministic environment
* Fully reproducible
* Typed models
* Modular architecture
* API + UI + CLI

---

## 🏆 Why This Matters

Most AI projects generate answers.
SamaarthyaSetu takes actions.

---

## 👨‍💻 Built For

Meta × Scaler OpenEnv Hackathon 2026

---

## ❤️ Final Thought

No trained differently-abled person should be left invisible in the system.

---

## 🔗 Links

Live App: https://kalimx03-samaarthyasetu.hf.space

API Docs: https://kalimx03-samaarthyasetu.hf.space/docs

GitHub: https://github.com/kalimx03/samaarthyasetu

---
