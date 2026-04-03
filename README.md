---
title: SamaarthyaSetu
emoji: 🌉
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# SamaarthyaSetu 🌉
### *Sanskrit: "Bridge of Enablement"*

**Meta × Scaler PyTorch/OpenEnv Hackathon 2026 · CMR Cause 2026**

🔗 **GitHub:** https://github.com/kalimx03/samaarthyasetu  
🚀 **Live API:** https://kalimx03-samaarthyasetu.hf.space  
📖 **API Docs:** https://kalimx03-samaarthyasetu.hf.space/docs

---

## One Paragraph

India has 40–90 million persons with disabilities. Fewer than 15% are formally employed. Every year, NGOs like Samarthanam and Enable India train hundreds of differently-abled youth in Bangalore — then watch them fall into a complete void: no job matching, no scheme navigation, no employer accountability, no follow-up. SamaarthyaSetu is an OpenEnv reinforcement learning environment that forces an AI agent to navigate the exact system these candidates face. The agent must match candidates to jobs using disability-aware composite scoring, navigate 10 real government schemes to generate prioritised document checklists, and resolve three-way placement conflicts between NGOs, employers, and follow-up records — all across 10 real Bangalore wards.

---

## Why This Wins

- **Domain specificity:** This is not another grid world or traffic light. Every candidate, employer, NGO, scheme, and ward is modelled on a real Bangalore entity. The placement conflict task is based on a real, documented failure mode in India's disability employment sector.
- **Dense rewards from day one:** 12 partial-progress signals fire throughout each episode. The cosine-decay efficiency function provides gradient signal at every step — no sparse reward problem.
- **LLM solves tasks independently:** Run `python inference.py --mode llm --task task_001 --demo` — every step shows `[llm]` as the source. No rule-based scaffolding. A real agent, reasoning in real time.
- **Three-mode inference:** `rules` (reproducible baseline), `llm` (pure LLM agent), `hybrid` (LLM with parse-error fallback). Switch with a single CLI flag.
- **Production-grade engineering:** Pydantic v2 models, deterministic graders, 44 tests, CI with Docker smoke test, HuggingFace Spaces ready.

---

## Quick Start (60 seconds)

```bash
git clone https://github.com/kalimx03/samaarthyasetu && cd samaarthyasetu
pip install -r requirements.txt

# Rule-based baseline — no API key needed
python inference.py --mode rules

# LLM agent solving a task independently
export HF_TOKEN="your-key"
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
python inference.py --mode llm --task task_001 --demo

# Run the full test suite
python -m pytest tests/ -v

# Start the API server
make run-api  # → http://localhost:7860/docs
```

---

## Architecture

```
samaarthyasetu/
├── samaarthya_ops_env/          ← OpenEnv core (the evaluatable artifact)
│   ├── environment.py           # reset() / step() / state()
│   ├── models.py                # Pydantic v2: Action, Observation, State
│   ├── matching_engine.py       # 60/25/15 weighted composite scorer
│   ├── reward_shaping.py        # dense signals + cosine efficiency (importable)
│   ├── data/seed_data.py        # 15 candidates, 10 jobs, 10 schemes, 5 NGOs, 8 employers
│   ├── tasks/task_definitions.py
│   └── graders/task_graders.py  # deterministic, no randomness
├── server/app.py                # FastAPI REST: /reset /step /state /candidates ...
├── ml/model.py                  # PyTorch Siamese skill-matching network
├── inference.py                 # ✅ --mode rules | llm | hybrid
├── openenv.yaml                 # ✅ OpenEnv v2.0.0 spec
├── Dockerfile                   # ✅ port 7860, HuggingFace Spaces compatible
├── tests/                       # 44 tests across 4 modules
└── docs/                        # REWARD_DESIGN.md, JUDGE_QA.md, DEMO_SCRIPT.md
```

**Data flow:**
```
Agent → POST /step → environment.py → graders → reward → Observation → Agent
```

---

## OpenEnv Interface

### Three required methods

```python
from samaarthya_ops_env import SamaarthyaSetuEnvironment, Action

env = SamaarthyaSetuEnvironment("task_001")

obs = env.reset()
# obs.progress == 0.0
# obs.available_actions == [11 action types]
# obs.data includes task description, required_actions, reward_design

obs, reward, done, info = env.step(
    Action(action_type="match_candidate_to_job",
           parameters={"candidate_id": "cand_003", "job_id": "job_002"})
)
# reward includes partial signal bonuses, penalties
# info["partial_signals_awarded"] shows what fired this step

state = env.state()
# typed snapshot: steps_taken, progress_score, selected_candidate_id, ...
```

### Action space (11 actions)

```
list_candidates  get_candidate  list_jobs  get_job
match_candidate_to_job  check_scheme_eligibility  generate_scheme_checklist
schedule_interview  resolve_placement_conflict  publish_dashboard_update
finalize_task
```

### REST API (for evaluation harness)

```
POST /reset          {"task_id": "task_001"} → Observation
POST /step           {"task_id": "...", "action": {...}} → {obs, reward, done, info}
GET  /state/{id}     → State snapshot
GET  /candidates     → 15 candidates
GET  /match/{id}     → top-3 job matches for candidate
GET  /dashboard      → ward-level placement analytics
```

---

## Tasks

### Task 1 — Verified Job Match  🟢 Easy  (≤ 15 steps)

**The problem it represents:** A newly trained candidate needs to be matched to the right job — not just by keyword, but by actual skill overlap, accommodation fit, and language compatibility.

**What the agent must do:** List candidates → list jobs → retrieve cand_003 (Arjun Nair, locomotor disability, React/JS developer) → retrieve job_002 (Infosys BPM Junior Frontend Developer) → call `match_candidate_to_job` → finalize.

**Why it's non-trivial:** Match score = 0.60 × skill_jaccard + 0.25 × accommodation_coverage + 0.15 × language_overlap. cand_003 scores **0.95** against job_002; the next-best job scores 0.23. An agent that guesses randomly has a ~10% chance of finding the right job.

**Baseline:** 6 steps, score **0.98**

---

### Task 2 — Scheme Navigator  🟡 Medium  (≤ 20 steps)

**The problem it represents:** A person with a speech disability in Karnataka is entitled to multiple central and state government schemes, each requiring a different document set. No one explains this in a navigable way.

**What the agent must do:** Retrieve cand_005 (Suresh Patil, speech disability) → `check_scheme_eligibility` across 10 schemes → `generate_scheme_checklist` with prioritised document list → finalize.

**Why it's non-trivial:** The grader checks whether the agent found all 4 eligible schemes (recall = 1.0) and whether the checklist contains both a document list and a priority order. Missing either incurs accuracy penalty.

**Baseline:** 4 steps, score **0.97**

---

### Task 3 — Placement Reconciliation  🔴 Hard  (≤ 30 steps)

**The problem it represents:** NGOs are incentivised to inflate placement rates. A common failure: NGO marks cand_011 as placed; employer has no hire record; follow-up finds candidate still searching. Three conflicting records, no audit trail.

**What the agent must do:** Retrieve cand_011 (Deepak Gowda) and job_002 → `resolve_placement_conflict` with correct conflict type and resolution → `schedule_interview` with emp_001 (Infosys BPM) → `publish_dashboard_update` for Malleshwaram ward → finalize.

**Why it's non-trivial:** The grader checks the correct employer (not just any employer), the correct conflict type classification, and that the dashboard was updated — 4 independent accuracy sub-scores. Partial credit for each.

**Baseline:** 6 steps, score **1.00**

---

## Reward Design

```
R = 0.50 × progress_fraction
  + 0.35 × accuracy_signal
  + 0.15 × step_efficiency(steps, max_steps)
  + Σ partial_signal_bonus       ← 12 signals, each fires at most once
  − Σ penalties
```

**Step efficiency:** `0.5 × (1 + cos(π × steps/max_steps))` — smooth cosine decay from 1.0 (0 steps) to 0.0 (max_steps). Never a hard cliff.

**Partial signals (selected):**

| Signal | Bonus | Tier |
|--------|-------|------|
| `candidate_loaded` | +0.04 | Information retrieval |
| `match_computed` | +0.08 | Computation |
| `high_quality_match` | +0.04 | Decision quality |
| `interview_confirmed` | +0.09 | Decision quality |
| `all_schemes_found` | +0.05 | Completeness |

Full design and justification: `docs/REWARD_DESIGN.md`

---

## Inference Modes

```bash
# Mode 1: Rule-based — reproducible, no API key, used for baseline evaluation
python inference.py --mode rules

# Mode 2: LLM-only — every step is LLM-generated, no rule scaffolding
python inference.py --mode llm --task task_001 --demo

# Mode 3: Hybrid — LLM primary, rule fallback only on JSON parse failure
python inference.py --mode hybrid

# Single task, verbose LLM trace
python inference.py --mode llm --task task_002 --verbose
```

Output includes a `[source]` label on every step: `[llm]`, `[rules]`, or `[llm-safety-fallback]`, so it is verifiable which mode is actually controlling each action.

---

## Evaluation Summary

All scores from rule-based baseline (reproducible without API key):

| Task | Steps | Score | Completed |
|------|-------|-------|-----------|
| task_001 Verified Job Match | 6 | **0.97** | ✅ |
| task_002 Scheme Navigator | 4 | **1.00** | ✅ |
| task_003 Placement Reconciliation | 6 | **1.00** | ✅ |
| **Average** | | **0.99** | |

LLM-only mode (`--mode llm`) with gpt-4o-mini solves task_001 and task_002 independently in our testing. Scores vary slightly with model temperature but remain > 0.75.

---

## PyTorch Model

`ml/model.py` implements a **Siamese neural network** for candidate-job compatibility scoring:
- Dual encoder: candidate features and job features encoded independently with shared weights
- Compatibility head: concatenated embeddings → Linear(128→64) → Linear(64→32) → Linear(32→1) → Sigmoid
- Trained on 150 synthetic pairs with MSELoss + AdamW + CosineAnnealingLR

```bash
python ml/model.py   # trains and saves to ml/skill_matching_model.pt
```

---

## Seed Data

All entities are Bangalore-specific and real:

| Entity | Count | Examples |
|--------|-------|---------|
| Candidates | 15 | Locomotor, visual, hearing, speech, intellectual disabilities |
| Employers | 8 | Infosys BPM, Wipro GreenTech, HDFC Bank, Myntra, BigBasket |
| NGOs | 5 | Samarthanam Trust, Enable India, Spastics Society Karnataka |
| Schemes | 10 | DKVY, Vishwas Karnataka, NHFDC loan, Section 80U |
| Wards | 10 | Koramangala, Whitefield, HSR Layout, Malleshwaram, … |
| Jobs | 10 | React developer, data entry analyst, bakery assistant, … |

---

## Setup

```bash
# Install
pip install -r requirements.txt

# Validate the OpenEnv interface
make validate

# Run all tests
make test

# Start API server
make run-api          # → http://localhost:7860
                      # → http://localhost:7860/docs (Swagger)

# Docker
docker build -t samaarthyasetu . && docker run -p 7860:7860 samaarthyasetu
```

## Deploy to HuggingFace Spaces

```bash
git remote add hf https://huggingface.co/spaces/kalimx03/samaarthyasetu
git push hf main
```

Set Secrets in HuggingFace Space Settings: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`.  
The Dockerfile exposes port 7860 automatically.

---

## SDG Alignment (CMR Cause 2026)

| SDG | How |
|-----|-----|
| **SDG 8** — Decent Work | Direct employment pipeline removes the post-training void |
| **SDG 10** — Reduced Inequalities | Accommodation fit scoring matches PwD to employers who can actually support them |
| **SDG 11** — Sustainable Cities | Ward-level analytics make invisible placement data visible |
| **SDG 17** — Partnerships | Unifies NGO, employer, and government data in one accountable system |

---

## Further Reading

- `docs/REWARD_DESIGN.md` — full reward function derivation with tables
- `docs/JUDGE_QA.md` — pre-prepared answers to every technical question
- `docs/DEMO_SCRIPT.md` — 90-second and 3-minute live demo scripts

---

**MIT License — Open source, built for social impact.**

*"No trained differently-abled person should be left unemployed and invisible."*
