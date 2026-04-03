# SamaarthyaSetu — Reward Function Design

## Overview

SamaarthyaSetu uses a **meaningful, multi-component reward function with dense partial-progress signals** — not a sparse binary reward. This makes the environment trainable with RL from scratch, without reward shaping tricks or hand-engineered curricula.

---

## Reward Formula

```
R_total = W_PROGRESS  × progress_fraction
         + W_ACCURACY  × accuracy_signal
         + W_EFFICIENCY× step_efficiency(steps, max_steps)
         + Σ partial_signal_bonus(signal_id)   ← dense per-step credit
         − Σ penalty(reason)

Where:
  W_PROGRESS   = 0.50
  W_ACCURACY   = 0.35
  W_EFFICIENCY = 0.15
```

---

## Components

### 1. Progress (W = 0.50)

```
progress = |required_actions ∩ actions_taken| / |required_actions|
```

A task-specific set of required actions is defined. As the agent completes each required action type, progress grows from 0 → 1. This is a continuous signal that fires whenever the agent takes a new required action.

### 2. Accuracy (W = 0.35)

Evaluated at episode end by the task grader. Measures:
- **Task 1**: Did the agent match the correct candidate to the correct job?
- **Task 2**: Did the agent find all eligible schemes for the correct candidate?
- **Task 3**: Did the agent correctly identify the conflict, resolve it, schedule an interview with the right employer, and publish the dashboard update?

Partial accuracy credit is awarded (not binary) — see `graders/task_graders.py`.

### 3. Step Efficiency (W = 0.15)

```python
efficiency = 0.5 × (1 + cos(π × steps / max_steps))
```

A **smooth cosine decay** from 1.0 (0 steps) to 0.0 (max_steps). This:
- Rewards agents that solve tasks concisely
- Provides gradient signal throughout the episode (not a hard cliff)
- Differentiates top performers from average ones

| Steps used | Efficiency (task_001, max=15) |
|-----------|-------------------------------|
| 4         | 0.91                          |
| 6         | 0.79                          |
| 8         | 0.65                          |
| 10        | 0.50                          |
| 15        | 0.00                          |

### 4. Partial Progress Signals (dense, once-per-episode)

Each signal fires **at most once per episode** when a meaningful sub-goal is first achieved. This creates a **dense reward landscape** — the agent receives gradient information at every step, not just at episode end.

| Signal | Tier | Bonus | Description |
|--------|------|-------|-------------|
| `candidate_loaded` | 1 | +0.04 | Candidate profile fetched |
| `job_loaded` | 1 | +0.03 | Job record fetched |
| `match_computed` | 2 | +0.08 | Composite match score computed (60/25/15 weights) |
| `scheme_checked` | 2 | +0.07 | Scheme eligibility evaluated |
| `top_match_found` | 3 | +0.06 | Optimal candidate-job pair identified |
| `high_quality_match` | 3 | +0.04 | Match score ≥ 0.75 |
| `conflict_identified` | 3 | +0.05 | Conflict type correctly classified |
| `conflict_resolved` | 3 | +0.08 | Resolution strategy executed |
| `interview_confirmed` | 3 | +0.09 | Interview booked with correct employer |
| `checklist_generated` | 3 | +0.08 | Document checklist generated with priority |
| `all_schemes_found` | 4 | +0.05 | All expected schemes found (recall = 1.0) |
| `ward_dashboard_updated` | 4 | +0.05 | Ward dashboard published |

**Maximum achievable partial bonus: ~0.72**

### 5. Penalties

| Reason | Amount | When |
|--------|--------|------|
| `invalid_action` | −0.05 | Invalid parameters or unknown entity ID |
| `repeated_action` | −0.02 | Same action type called twice (except `finalize_task`) |
| `early_finalize` | −0.10 | `finalize_task` called with < 50% required actions done |
| `wrong_target` | −0.02 | Fetched wrong candidate/job for this task (soft) |

---

## Why Dense Partial Signals?

Standard RL environments often give binary reward (0 or 1) only at episode end. This creates the **sparse reward problem**: an agent taking random actions receives no learning signal until it accidentally completes the task, which may never happen in large action spaces.

SamaarthyaSetu's partial signals solve this by:

1. **Tier 1 signals** fire immediately on information retrieval — the agent learns to look things up before acting
2. **Tier 2 signals** reward correct computation — the agent learns to score candidates, not just list them
3. **Tier 3 signals** reward decision quality — the agent learns which candidate/job/resolution is correct
4. **Tier 4 signals** reward completeness — the agent learns to complete the full task, not just the first step

This mirrors how RL researchers shape rewards for complex planning tasks (e.g. robot manipulation, dialogue systems), where intermediate sub-goals provide the gradient information needed to learn efficiently.

---

## Per-Task Reward Profiles

### Task 1 — Verified Job Match (Easy)
- **Max partial bonus**: ~0.31 (signals: candidate_loaded, job_loaded, match_computed, top_match_found, high_quality_match)
- **Expected optimal reward**: ~0.85
- **Pass threshold**: ≥ 0.70

### Task 2 — Scheme Navigator (Medium)
- **Max partial bonus**: ~0.24 (signals: candidate_loaded, scheme_checked, all_schemes_found, checklist_generated)
- **Expected optimal reward**: ~0.80
- **Pass threshold**: ≥ 0.65

### Task 3 — Placement Reconciliation (Hard)
- **Max partial bonus**: ~0.40 (signals: candidate_loaded, job_loaded, conflict_identified, conflict_resolved, interview_confirmed, ward_dashboard_updated)
- **Expected optimal reward**: ~0.90
- **Pass threshold**: ≥ 0.60

---

## Implementation

The reward function is implemented across three files:

- `samaarthya_ops_env/environment.py` — reward computation in `step()`
- `samaarthya_ops_env/reward_shaping.py` — standalone utilities + catalogue
- `samaarthya_ops_env/graders/task_graders.py` — accuracy component for each task

All components are unit tested in `tests/test_reward_shaping.py` (37 tests).

---

## SDG Alignment

The reward design directly encodes the UN SDG 8 (Decent Work) and SDG 10 (Reduced Inequalities) goals:

- **Accuracy rewards** are highest for tasks that directly improve employment outcomes (interview scheduling, conflict resolution)
- **Scheme checklist signals** incentivize the agent to navigate government benefits — key for SDG 1 (No Poverty)
- **Dashboard update signal** rewards transparency and accountability — key for SDG 16 (Strong Institutions)
