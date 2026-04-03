<div align="center">

<img src="https://img.shields.io/badge/Meta_×_Scaler-OpenEnv_Hackathon_2026-0064E0?style=for-the-badge&logo=meta&logoColor=white"/>
<img src="https://img.shields.io/badge/CMR-Cause_2026-FF6B35?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Score-0.99_avg-2B6B4A?style=for-the-badge&logo=checkmarx&logoColor=white"/>
<img src="https://img.shields.io/badge/Tasks-3_×_PERFECT-gold?style=for-the-badge"/>

<br/><br/>

# SamaarthyaSetu 🌉
### *सामर्थ्यसेतु — Bridge of Enablement*

**An OpenEnv reinforcement-learning environment where an AI agent navigates the real, broken employment pipeline for persons with disabilities in Bangalore, India.**

<br/>

[![Live API](https://img.shields.io/badge/🚀_Live_API-kalimx03--samaarthyasetu.hf.space-2B6B4A?style=flat-square)](https://kalimx03-samaarthyasetu.hf.space)
[![API Docs](https://img.shields.io/badge/📖_Swagger_Docs-/docs-D97B3A?style=flat-square)](https://kalimx03-samaarthyasetu.hf.space/docs)
[![GitHub](https://img.shields.io/badge/GitHub-kalimx03%2Fsamaarthyasetu-181717?style=flat-square&logo=github)](https://github.com/kalimx03/samaarthyasetu)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

</div>

---

## The Problem This Solves

> India has **40–90 million persons with disabilities.** Fewer than **15%** are formally employed.

Every year, NGOs like Samarthanam and Enable India train hundreds of differently-abled people in Bangalore. Then training ends — and candidates fall into a complete void. No structured job matching. No scheme navigation. No employer accountability. No follow-up. The placement counsellor marks them "placed" in a spreadsheet. The employer has no record. The candidate is still searching.

**SamaarthyaSetu forces an AI agent to fix this, one episode at a time.**

---

## What the Agent Does

The environment models three real failure modes in Bangalore's disability employment pipeline:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  TASK 1 🟢  Job Verification Match          (Easy   · ≤15 steps)       │
│  ─────────────────────────────────────────────────────────────────      │
│  Match cand_003 (Arjun Nair, locomotor disability, React/JS)            │
│  to job_002 (Infosys BPM Junior Frontend Developer).                    │
│  Score = 0.60 × skill_jaccard + 0.25 × accommodation + 0.15 × lang     │
│  Baseline: 6 steps · Score 0.97                                         │
│                                                                         │
│  TASK 2 🟡  Scheme Navigator               (Medium  · ≤20 steps)       │
│  ─────────────────────────────────────────────────────────────────      │
│  For cand_005 (Suresh Patil, speech disability), find all 4 eligible    │
│  government schemes and generate a prioritised document checklist.      │
│  Baseline: 4 steps · Score 1.00                                         │
│                                                                         │
│  TASK 3 🔴  Placement Reconciliation       (Hard    · ≤30 steps)       │
│  ─────────────────────────────────────────────────────────────────      │
│  NGO says cand_011 is "placed". Employer has no record.                 │
│  Candidate says still searching. Classify conflict, resolve it,         │
│  schedule fresh interview with emp_001, update the ward dashboard.      │
│  Baseline: 6 steps · Score 1.00                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Results

<div align="center">

| Task | Difficulty | Steps | Score | Grade |
|------|-----------|-------|-------|-------|
| Verified Job Match | 🟢 Easy | 6 / 15 | **0.97** | ✅ PASS |
| Scheme Navigator | 🟡 Medium | 4 / 20 | **1.00** | ✅ PERFECT |
| Placement Reconciliation | 🔴 Hard | 6 / 30 | **1.00** | ✅ PERFECT |
| **Average** | | | **0.99** | |

</div>

*All scores from rule-based baseline — fully reproducible without any API key.*

---

## 60-Second Quickstart

```bash
git clone https://github.com/kalimx03/samaarthyasetu && cd samaarthyasetu
pip install -r requirements.txt

# No API key needed — run the full baseline immediately
python inference.py --mode rules

# LLM agent solving tasks independently (requires API key)
export HF_TOKEN="your-key"
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
python inference.py --mode llm --task task_001 --demo

# Full test suite (44 tests, 4 modules)
python -m pytest tests/ -v

# Start the API server locally
make run-api   # → http://localhost:7860/docs
```

---

## Architecture

```
samaarthyasetu/
│
├── samaarthya_ops_env/          ← The OpenEnv core (the evaluatable artifact)
│   ├── environment.py           # reset() · step() · state()
│   ├── models.py                # Pydantic v2: Action, Observation, State
│   ├── matching_engine.py       # Jaccard composite scorer (60/25/15 weights)
│   ├── reward_shaping.py        # 12 partial signals + cosine efficiency decay
│   ├── data/seed_data.py        # 15 candidates · 10 jobs · 10 schemes · 5 NGOs
│   ├── tasks/task_definitions.py
│   └── graders/task_graders.py  # Deterministic — no randomness
│
├── server/app.py                # FastAPI: /reset /step /state /candidates ...
├── ml/model.py                  # PyTorch Siamese skill-matching network
├── inference.py                 # CLI: --mode rules | llm | hybrid
├── openenv.yaml                 # OpenEnv v2.0.0 specification
├── Dockerfile                   # Port 7860 · HuggingFace Spaces ready
├── tests/                       # 44 tests across 4 modules
└── docs/                        # REWARD_DESIGN.md · JUDGE_QA.md · DEMO_SCRIPT.md
```

**Request flow:**
```
Agent  →  POST /step  →  environment.py  →  graders  →  reward  →  Observation  →  Agent
```

---

## OpenEnv Interface

```python
from samaarthya_ops_env import SamaarthyaSetuEnvironment, Action

env = SamaarthyaSetuEnvironment("task_001")
obs = env.reset()

# obs.progress           → 0.0
# obs.available_actions  → 11 action types
# obs.data               → task description, required_actions, reward_design

obs, reward, done, info = env.step(
    Action(
        action_type="match_candidate_to_job",
        parameters={"candidate_id": "cand_003", "job_id": "job_002"}
    )
)

# info["partial_signals_awarded"]  → which of the 12 signals fired this step
# info["grade_details"]            → accuracy breakdown per sub-score

state = env.state()
# Returns typed snapshot: steps_taken, progress_score, selected_candidate_id, ...
```

### Action Space (11 actions)

| Category | Actions |
|----------|---------|
| **Information** | `list_candidates` · `get_candidate` · `list_jobs` · `get_job` |
| **Matching** | `match_candidate_to_job` |
| **Schemes** | `check_scheme_eligibility` · `generate_scheme_checklist` |
| **Placement** | `schedule_interview` · `resolve_placement_conflict` · `publish_dashboard_update` |
| **Control** | `finalize_task` |

### REST API

```
POST  /reset            {"task_id": "task_001"}              → Observation
POST  /step             {"task_id": "...", "action": {...}}  → {obs, reward, done, info}
GET   /state/{task_id}  → Typed state snapshot
GET   /candidates       → All 15 candidates with disability profiles
GET   /match/{cand_id}  → Top-3 job matches with composite scores
GET   /dashboard        → Ward-level placement analytics (10 Bangalore wards)
```

---

## Reward Design

```
R(τ) = 0.50 × progress_fraction
     + 0.35 × accuracy_signal
     + 0.15 × step_efficiency(steps, max_steps)
     + Σ partial_signal_bonus          ← 12 signals, fire at most once each
     − Σ penalties
```

**Step efficiency uses cosine decay** — smooth gradient from 1.0 at step 0 to 0.0 at max_steps:
```
efficiency = 0.5 × (1 + cos(π × steps / max_steps))
```

**Partial signals:**

| Signal | Bonus | When it fires |
|--------|-------|---------------|
| `candidate_loaded` | +0.04 | Correct candidate retrieved |
| `match_computed` | +0.08 | Match score successfully computed |
| `high_quality_match` | +0.04 | Composite score ≥ 0.90 |
| `all_schemes_found` | +0.05 | 100% scheme recall |
| `interview_confirmed` | +0.09 | Correct employer + correct candidate |
| `ward_dashboard_updated` | +0.05 | Ward dashboard reflects new status |
| *(+ 6 more)* | | |

**Penalties:** `-0.05` invalid action · `-0.02` repeated action · `-0.10` premature finalize

Full derivation with justification: [`docs/REWARD_DESIGN.md`](docs/REWARD_DESIGN.md)

---

## Three Inference Modes

```bash
# Mode 1: Rules — deterministic baseline, zero API calls, always reproducible
python inference.py --mode rules

# Mode 2: LLM — the model drives every single step, no rule scaffolding
python inference.py --mode llm --task task_001 --demo

# Mode 3: Hybrid — LLM primary, rule fallback only on JSON parse failure
python inference.py --mode hybrid --verbose
```

Every step is labelled `[llm]`, `[rules]`, or `[llm-safety-fallback]` — verifiable at a glance.

---

## PyTorch Model

`ml/model.py` — **Siamese neural network** for candidate-job compatibility:

```
Candidate features ─→ Encoder (shared weights) ─→ 64-dim embedding ─┐
                                                                       ├─→ Compatibility Head ─→ Score ∈ [0,1]
Job features ──────→ Encoder (shared weights) ─→ 64-dim embedding ─┘
```

- Compatibility head: `Linear(128→64)` → `Linear(64→32)` → `Linear(32→1)` → `Sigmoid`
- Training: 150 synthetic pairs · `MSELoss` · `AdamW` · `CosineAnnealingLR`

```bash
python ml/model.py   # trains locally, saves to ml/skill_matching_model.pt
```

---

## Seed Data

All entities modelled on real Bangalore organisations and government schemes:

| Entity | Count | Examples |
|--------|-------|---------|
| Candidates | 15 | Locomotor · Visual · Hearing · Speech · Intellectual |
| Employers | 8 | Infosys BPM · Wipro GreenTech · HDFC Bank · Myntra · BigBasket |
| NGOs | 5 | Samarthanam Trust · Enable India · Spastics Society Karnataka |
| Gov. Schemes | 10 | DKVY · Vishwas Karnataka · NHFDC Loan · Section 80U |
| Wards | 10 | Koramangala · Whitefield · HSR Layout · Malleshwaram · … |
| Jobs | 10 | React developer · Data entry · Bakery assistant · … |

---

## SDG Alignment

| Goal | How SamaarthyaSetu addresses it |
|------|--------------------------------|
| **SDG 8** — Decent Work | Eliminates the post-training employment void for PwD |
| **SDG 10** — Reduced Inequalities | Accommodation-fit scoring ensures PwD reach employers who can support them |
| **SDG 11** — Sustainable Cities | Ward-level analytics make invisible placement data visible to policymakers |
| **SDG 17** — Partnerships | Unifies NGO, employer, and government data in one accountable pipeline |

---

## Engineering Quality

```
✅  Pydantic v2 typed models throughout
✅  44 tests across 4 modules (environment · tasks · reward · api)
✅  Deterministic graders — no randomness, reproducible on every run
✅  GitHub Actions CI: lint + tests + Docker smoke test on every push
✅  OpenEnv v2.0.0 spec compliant (openenv.yaml)
✅  HuggingFace Spaces Dockerfile — port 7860, live and responding
✅  Three inference modes with labelled step output
✅  docs/JUDGE_QA.md — answers to every technical question, prepared in advance
```

---

## Further Reading

- [`docs/REWARD_DESIGN.md`](docs/REWARD_DESIGN.md) — full reward function derivation with tables
- [`docs/JUDGE_QA.md`](docs/JUDGE_QA.md) — pre-prepared answers to every technical question
- [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md) — 90-second and 3-minute live demo scripts

---

<div align="center">

**MIT License · Open source · Built for social impact**

*"No trained differently-abled person should be left unemployed and invisible."*

<br/>

[![Live API](https://img.shields.io/badge/🚀_Live_API-Running-2B6B4A?style=for-the-badge)](https://kalimx03-samaarthyasetu.hf.space)
[![API Docs](https://img.shields.io/badge/📖_Swagger-/docs-D97B3A?style=for-the-badge)](https://kalimx03-samaarthyasetu.hf.space/docs)

</div>
