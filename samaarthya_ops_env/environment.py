"""
SamaarthyaSetu OpenEnv Environment  (v2)
Implements the standard OpenEnv interface: reset(), step(), state()

Reward Design — Partial Progress Signals
-----------------------------------------
  total_reward = W_PROGRESS  * progress_fraction
               + W_ACCURACY  * accuracy_signal
               + W_EFFICIENCY* step_efficiency
               + Σ partial_bonuses              ← dense per-step credit
               - Σ penalties

Partial signals fire ONCE on the first step where a meaningful sub-goal
is achieved, giving the agent a dense reward gradient rather than a
single sparse end-of-episode signal.

This makes the reward function meaningful even for failed episodes and
allows RL training to make progress from the very first few steps.
"""
from __future__ import annotations
import copy
import math
from typing import Any

from samaarthya_ops_env.models import Action, Observation, State
from samaarthya_ops_env.data import CANDIDATES, JOBS, SCHEMES, NGOS, EMPLOYERS
from samaarthya_ops_env.tasks import TASKS
from samaarthya_ops_env.graders import GRADERS
from samaarthya_ops_env.matching_engine import (
    compute_match_score, get_top_matches,
    check_scheme_eligibility, generate_scheme_checklist,
)

AVAILABLE_ACTIONS = [
    "list_candidates", "get_candidate", "list_jobs", "get_job",
    "match_candidate_to_job", "check_scheme_eligibility",
    "generate_scheme_checklist", "schedule_interview",
    "resolve_placement_conflict", "publish_dashboard_update", "finalize_task",
]

# ── Reward component weights ──────────────────────────────────────────────────
W_PROGRESS   = 0.50   # fraction of required actions completed
W_ACCURACY   = 0.35   # correctness of key decisions
W_EFFICIENCY = 0.15   # step economy (fewer steps = higher score)

# ── Partial-progress signal definitions ──────────────────────────────────────
# (bonus_reward, human-readable description)
# Each signal fires AT MOST ONCE per episode.
PARTIAL_SIGNALS: dict[str, tuple[float, str]] = {
    "candidate_loaded"       : (0.04, "Candidate record retrieved"),
    "job_loaded"             : (0.03, "Job record retrieved"),
    "match_computed"         : (0.08, "Composite match score computed (60/25/15 weights)"),
    "top_match_found"        : (0.06, "Optimal candidate-job pair identified"),
    "high_quality_match"     : (0.04, "Match score ≥ 0.75 — high-confidence placement"),
    "scheme_checked"         : (0.07, "Scheme eligibility evaluated for candidate"),
    "all_schemes_found"      : (0.05, "All expected eligible schemes identified"),
    "checklist_generated"    : (0.08, "Prioritised document checklist generated"),
    "conflict_identified"    : (0.05, "Placement conflict type correctly classified"),
    "conflict_resolved"      : (0.08, "Conflict resolution strategy executed"),
    "interview_confirmed"    : (0.09, "Fresh interview scheduled with correct employer"),
    "ward_dashboard_updated" : (0.05, "Ward-level dashboard record published"),
}

# ── Penalty constants ─────────────────────────────────────────────────────────
PENALTY_INVALID      = -0.05   # action with invalid / missing parameters
PENALTY_REPEATED     = -0.02   # repeating the same action type
PENALTY_EARLY_FIN    = -0.10   # finalise with < 50 % of required actions done
PENALTY_WRONG_TARGET = -0.02   # soft penalty for fetching a non-target entity


class SamaarthyaSetuEnvironment:
    """
    SamaarthyaSetu OpenEnv environment.

    Simulates the inclusive employment operating system for persons with
    disabilities in Bangalore.  Three tasks of increasing difficulty:

      task_001  Verified Job Match         Easy    ≤ 15 steps
      task_002  Scheme Navigator           Medium  ≤ 20 steps
      task_003  Placement Reconciliation   Hard    ≤ 30 steps
    """

    def __init__(self, task_id: str = "task_001"):
        if task_id not in TASKS:
            raise ValueError(
                f"Unknown task_id '{task_id}'. Valid: {list(TASKS.keys())}"
            )
        self._task_id = task_id
        self._state: dict[str, Any] = {}
        self._history: list[dict] = []
        self._partial_fired: set[str] = set()
        self._candidates = {c.id: c for c in CANDIDATES}
        self._jobs       = {j.id: j for j in JOBS}
        self._schemes    = {s.id: s for s in SCHEMES}
        self._ngos       = {n.id: n for n in NGOS}
        self._employers  = {e.id: e for e in EMPLOYERS}

    # ── Public OpenEnv interface ───────────────────────────────────────────────

    def reset(self) -> Observation:
        """Reset the environment for the configured task. Must be called before step()."""
        task = TASKS[self._task_id]
        self._partial_fired = set()
        self._state = {
            "current_task_id"        : self._task_id,
            "selected_candidate_id"  : None,
            "selected_job_id"        : None,
            "steps_taken"            : 0,
            "progress_score"         : 0.0,
            "completed"              : False,
            "actions_taken"          : [],
            "reward_history"         : [],
            "partial_signals_fired"  : [],
            "last_match_score"       : 0.0,
            "eligible_scheme_ids"    : [],
            "generated_checklist"    : {},
            "conflict_resolved"      : False,
            "interview_scheduled"    : False,
            "interview_employer_id"  : None,
            "dashboard_updated"      : False,
        }
        self._history = []
        return Observation(
            message=(
                f"Environment reset. Task: [{task.difficulty.upper()}] "
                f"{task.name}. {task.description}"
            ),
            data={
                "task_id"         : self._task_id,
                "task_name"       : task.name,
                "difficulty"      : task.difficulty,
                "max_steps"       : task.max_steps,
                "required_actions": task.required_actions,
                "reward_design"   : {
                    "formula"         : "W_PROGRESS*progress + W_ACCURACY*accuracy + W_EFFICIENCY*efficiency + partial_bonuses - penalties",
                    "weights"         : {"progress": W_PROGRESS, "accuracy": W_ACCURACY, "efficiency": W_EFFICIENCY},
                    "partial_signals" : {k: v[0] for k, v in PARTIAL_SIGNALS.items()},
                    "penalties"       : {
                        "invalid_action" : PENALTY_INVALID,
                        "repeated_action": PENALTY_REPEATED,
                        "early_finalize" : PENALTY_EARLY_FIN,
                    },
                },
            },
            available_actions=AVAILABLE_ACTIONS,
            progress=0.0,
        )

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        """
        Execute one action.
        Returns: (Observation, reward: float, done: bool, info: dict)
        """
        if self._state.get("completed"):
            return (
                Observation(
                    message="Episode already completed.",
                    data={},
                    available_actions=[],
                    progress=1.0,
                ),
                0.0, True, {"error": "already_done"},
            )

        self._state["steps_taken"] += 1
        atype  = action.action_type
        params = action.parameters
        info: dict[str, Any] = {
            "step"            : self._state["steps_taken"],
            "action"          : atype,
            "partial_signals" : [],
        }
        reward = 0.0

        # ── Penalty: repeated action ──────────────────────────────────────────
        if atype in self._state["actions_taken"] and atype != "finalize_task":
            reward += PENALTY_REPEATED
            info.setdefault("penalties", []).append(
                {"reason": "repeated_action", "amount": PENALTY_REPEATED}
            )

        # ── Dispatch to handler ───────────────────────────────────────────────
        try:
            obs_data, action_reward, fired = self._dispatch(atype, params)
            reward += action_reward
            for sig_id in fired:
                if sig_id not in self._partial_fired:
                    bonus, desc = PARTIAL_SIGNALS[sig_id]
                    reward += bonus
                    self._partial_fired.add(sig_id)
                    self._state["partial_signals_fired"].append(sig_id)
                    info["partial_signals"].append(
                        {"signal": sig_id, "bonus": bonus, "description": desc}
                    )
        except ValueError as exc:
            reward += PENALTY_INVALID
            obs_data = {
                "error"  : str(exc),
                "message": f"Action failed: {exc}",
            }
            info.setdefault("penalties", []).append(
                {"reason": "invalid_action", "error": str(exc), "amount": PENALTY_INVALID}
            )

        # ── Track which action types have been called ─────────────────────────
        if atype not in self._state["actions_taken"]:
            self._state["actions_taken"].append(atype)

        # ── Progress = required actions completed / total required ────────────
        task     = TASKS[self._task_id]
        required = set(task.required_actions)
        taken    = set(self._state["actions_taken"])
        self._state["progress_score"] = len(required & taken) / len(required)

        # ── Compute step efficiency (cosine decay) ────────────────────────────
        efficiency = self._step_efficiency(
            self._state["steps_taken"], task.max_steps
        )
        info["step_efficiency"] = round(efficiency, 4)

        # ── Episode termination ───────────────────────────────────────────────
        done = False
        if atype == "finalize_task":
            if len(required & taken) < len(required) * 0.5:
                reward += PENALTY_EARLY_FIN
                info.setdefault("penalties", []).append(
                    {"reason": "early_finalize", "amount": PENALTY_EARLY_FIN}
                )

            final_score, grade_details = GRADERS[self._task_id](
                self._history, self._state
            )
            # Blend efficiency bonus into final score
            boosted = min(1.0, final_score + efficiency * W_EFFICIENCY * 0.3)
            reward += boosted * 0.5

            info["grade_details"]          = grade_details
            info["final_score"]            = round(boosted, 4)
            info["raw_grader_score"]       = round(final_score, 4)
            info["efficiency_bonus"]       = round(efficiency * W_EFFICIENCY * 0.3, 4)
            info["partial_signals_fired"]  = list(self._partial_fired)
            info["partial_signals_total"]  = len(self._partial_fired)
            self._state["completed"]       = True
            done                           = True

        self._state["reward_history"].append(round(reward, 4))
        self._history.append(copy.deepcopy(self._state))

        obs = Observation(
            message=obs_data.get("message", f"Action '{atype}' executed."),
            data={
                **obs_data,
                "_partial_this_step"   : info["partial_signals"],
                "_partial_total"       : len(self._partial_fired),
                "_step_efficiency"     : round(efficiency, 4),
                "_progress"            : round(self._state["progress_score"], 4),
                "_cumulative_reward"   : round(sum(self._state["reward_history"]), 4),
            },
            available_actions=[] if done else AVAILABLE_ACTIONS,
            progress=self._state["progress_score"],
        )
        return obs, round(reward, 4), done, info

    def state(self) -> State:
        """Return a typed snapshot of the current environment state."""
        return State(**{
            k: self._state[k]
            for k in State.model_fields
            if k in self._state
        })

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _step_efficiency(steps: int, max_steps: int) -> float:
        """
        Smooth cosine-decay efficiency ∈ [0, 1].
        At steps = 0          → 1.00  (maximum)
        At steps = max_steps  → 0.00  (minimum)
        The cosine ensures meaningful gradient throughout the episode.
        """
        if max_steps <= 0:
            return 0.0
        ratio = min(steps / max_steps, 1.0)
        return round(0.5 * (1.0 + math.cos(math.pi * ratio)), 4)

    def _dispatch(
        self, atype: str, params: dict
    ) -> tuple[dict, float, list[str]]:
        """Route to the appropriate handler.  Returns (data, reward, signals)."""
        handlers = {
            "list_candidates"           : self._list_candidates,
            "get_candidate"             : self._get_candidate,
            "list_jobs"                 : self._list_jobs,
            "get_job"                   : self._get_job,
            "match_candidate_to_job"    : self._match_candidate_to_job,
            "check_scheme_eligibility"  : self._check_scheme_eligibility,
            "generate_scheme_checklist" : self._generate_scheme_checklist,
            "schedule_interview"        : self._schedule_interview,
            "resolve_placement_conflict": self._resolve_placement_conflict,
            "publish_dashboard_update"  : self._publish_dashboard_update,
            "finalize_task"             : self._finalize_task,
        }
        if atype not in handlers:
            raise ValueError(f"Unknown action: {atype}")
        return handlers[atype](params)

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _list_candidates(self, p: dict) -> tuple[dict, float, list[str]]:
        cands = list(self._candidates.values())
        if ward := p.get("ward"):
            cands = [c for c in cands if c.ward == ward]
        if dtype := p.get("disability_type"):
            cands = [c for c in cands if c.disability_type == dtype]
        return {
            "message"   : f"Found {len(cands)} candidates.",
            "candidates": [
                {"id": c.id, "name": c.name,
                 "disability_type": c.disability_type,
                 "skills": c.skills, "ward": c.ward}
                for c in cands
            ],
            "total": len(cands),
        }, 0.04, []

    def _get_candidate(self, p: dict) -> tuple[dict, float, list[str]]:
        cid = p.get("candidate_id")
        if not cid or cid not in self._candidates:
            raise ValueError(f"Candidate '{cid}' not found.")
        c = self._candidates[cid]
        self._state["selected_candidate_id"] = cid
        # Soft penalty for wrong target
        task = TASKS[self._task_id]
        target = task.success_criteria.get(
            "target_candidate_id",
            task.success_criteria.get("candidate_id"),
        )
        r = 0.08
        if target and cid != target:
            r += PENALTY_WRONG_TARGET
        return {
            "message"  : f"Retrieved candidate {c.name}.",
            "candidate": c.model_dump(),
        }, r, ["candidate_loaded"]

    def _list_jobs(self, p: dict) -> tuple[dict, float, list[str]]:
        jobs = list(self._jobs.values())
        if ward := p.get("ward"):
            jobs = [j for j in jobs if j.ward == ward]
        if p.get("disability_friendly_only"):
            jobs = [j for j in jobs if j.disability_friendly]
        return {
            "message": f"Found {len(jobs)} open jobs.",
            "jobs"   : [
                {"id": j.id, "title": j.title,
                 "employer": j.employer_name,
                 "ward": j.ward, "salary_range": j.salary_range}
                for j in jobs
            ],
            "total": len(jobs),
        }, 0.04, []

    def _get_job(self, p: dict) -> tuple[dict, float, list[str]]:
        jid = p.get("job_id")
        if not jid or jid not in self._jobs:
            raise ValueError(f"Job '{jid}' not found.")
        j = self._jobs[jid]
        self._state["selected_job_id"] = jid
        return {
            "message": f"Retrieved job {j.title}.",
            "job"    : j.model_dump(),
        }, 0.06, ["job_loaded"]

    def _match_candidate_to_job(self, p: dict) -> tuple[dict, float, list[str]]:
        cid = p.get("candidate_id") or self._state.get("selected_candidate_id")
        jid = p.get("job_id")
        if not cid or cid not in self._candidates:
            raise ValueError("Valid candidate_id required.")
        c = self._candidates[cid]
        self._state["selected_candidate_id"] = cid
        signals = ["match_computed"]
        task    = TASKS[self._task_id]

        if jid:
            if jid not in self._jobs:
                raise ValueError(f"Job '{jid}' not found.")
            j = self._jobs[jid]
            score = compute_match_score(c, j)
            self._state["selected_job_id"]   = jid
            self._state["last_match_score"]  = score
            # Partial: optimal pair
            t_cid = task.success_criteria.get("target_candidate_id")
            t_jid = task.success_criteria.get("target_job_id")
            if t_cid and t_jid and cid == t_cid and jid == t_jid:
                signals.append("top_match_found")
            if score >= 0.75:
                signals.append("high_quality_match")
            return {
                "message"        : f"Match score {c.name} → {j.title}: {score:.2%}",
                "candidate_id"   : cid,
                "job_id"         : jid,
                "match_score"    : score,
                "score_breakdown": {"skill": 0.60, "accommodation": 0.25, "language": 0.15},
            }, 0.08 + 0.10 * score, signals
        else:
            top = get_top_matches(c, list(self._jobs.values()), top_n=3)
            if top:
                best = top[0]
                self._state["selected_job_id"]  = best["job_id"]
                self._state["last_match_score"] = best["overall_score"]
                signals.append("top_match_found")
                if best["overall_score"] >= 0.75:
                    signals.append("high_quality_match")
            return {
                "message"     : f"Top 3 matches for {c.name}.",
                "candidate_id": cid,
                "top_matches" : top,
            }, 0.10, signals

    def _check_scheme_eligibility(self, p: dict) -> tuple[dict, float, list[str]]:
        cid = p.get("candidate_id") or self._state.get("selected_candidate_id")
        if not cid or cid not in self._candidates:
            raise ValueError("Valid candidate_id required.")
        c = self._candidates[cid]
        self._state["selected_candidate_id"] = cid
        results     = check_scheme_eligibility(c, list(self._schemes.values()))
        eligible_ids = [r["scheme_id"] for r in results if r["eligible"]]
        self._state["eligible_scheme_ids"] = eligible_ids
        signals = ["scheme_checked"]
        task    = TASKS[self._task_id]
        expected = set(task.success_criteria.get("expected_eligible_schemes", []))
        if expected and expected.issubset(set(eligible_ids)):
            signals.append("all_schemes_found")
        return {
            "message"           : f"{len(eligible_ids)} schemes eligible for {c.name}.",
            "candidate_id"      : cid,
            "results"           : results,
            "eligible_scheme_ids": eligible_ids,
            "eligible_count"    : len(eligible_ids),
        }, 0.08, signals

    def _generate_scheme_checklist(self, p: dict) -> tuple[dict, float, list[str]]:
        cid = p.get("candidate_id") or self._state.get("selected_candidate_id")
        if not cid:
            raise ValueError("candidate_id required. Run check_scheme_eligibility first.")
        if not self._state.get("eligible_scheme_ids"):
            raise ValueError("No eligible schemes. Run check_scheme_eligibility first.")
        c         = self._candidates[cid]
        results   = check_scheme_eligibility(c, list(self._schemes.values()))
        checklist = generate_scheme_checklist(results)
        self._state["generated_checklist"] = checklist
        has_docs  = bool(checklist.get("documents"))
        has_prio  = bool(checklist.get("priority_order"))
        r         = 0.08 + (0.04 if has_docs else 0) + (0.04 if has_prio else 0)
        return {
            "message"          : f"Scheme checklist generated for {cid}.",
            "checklist"        : checklist,
            "has_documents"    : has_docs,
            "has_priority_order": has_prio,
        }, r, ["checklist_generated"]

    def _schedule_interview(self, p: dict) -> tuple[dict, float, list[str]]:
        cid = p.get("candidate_id") or self._state.get("selected_candidate_id")
        eid = p.get("employer_id")
        dt  = p.get("date", "2026-04-15")
        if not cid or not eid:
            raise ValueError("candidate_id and employer_id are required.")
        if eid not in self._employers:
            raise ValueError(f"Employer '{eid}' not found.")
        self._state["interview_scheduled"]  = True
        self._state["interview_employer_id"] = eid
        emp  = self._employers[eid]
        c    = self._candidates.get(cid)
        task = TASKS[self._task_id]
        r    = 0.12 if eid == task.success_criteria.get("target_employer_id") else 0.06
        return {
            "message"        : (
                f"Interview scheduled: "
                f"{c.name if c else cid} with {emp.name} on {dt}."
            ),
            "candidate_id"   : cid,
            "employer_id"    : eid,
            "employer_name"  : emp.name,
            "date"           : dt,
            "confirmation_id": f"INTVW-{cid[-3:]}-{eid[-3:]}-{dt.replace('-','')}",
        }, r, ["interview_confirmed"]

    def _resolve_placement_conflict(self, p: dict) -> tuple[dict, float, list[str]]:
        cid           = p.get("candidate_id") or self._state.get("selected_candidate_id")
        conflict_type = p.get("conflict_type", "ngo_employer_mismatch")
        resolution    = p.get("resolution", "schedule_fresh_interview")
        if not cid:
            raise ValueError("candidate_id required.")
        self._state["conflict_resolved"] = True
        task  = TASKS[self._task_id]
        r     = 0.10
        r    += 0.05 if conflict_type == task.success_criteria.get("conflict_type") else 0
        r    += 0.05 if resolution    == task.success_criteria.get("resolution")    else 0
        return {
            "message"         : (
                f"Conflict resolved for {cid}. "
                f"Type: {conflict_type}. Resolution: {resolution}."
            ),
            "candidate_id"    : cid,
            "conflict_type"   : conflict_type,
            "resolution"      : resolution,
            "conflict_details": {
                "ngo_record"    : "Placed at Infosys BPM — Job ID: job_002",
                "employer_record": "No hire record found for this candidate",
                "followup_record": "Candidate still actively seeking employment",
                "root_cause"    : "NGO marked placement prematurely before offer confirmation",
            },
        }, r, ["conflict_identified", "conflict_resolved"]

    def _publish_dashboard_update(self, p: dict) -> tuple[dict, float, list[str]]:
        ward  = p.get("ward", "All")
        utype = p.get("update_type", "placement_status")
        self._state["dashboard_updated"] = True
        return {
            "message"         : f"Dashboard updated. Ward: {ward}. Update: {utype}.",
            "ward"            : ward,
            "update_type"     : utype,
            "timestamp"       : "2026-04-01T10:30:00+05:30",
            "metrics_updated" : ["placement_rate", "active_candidates", "employer_count"],
        }, 0.08, ["ward_dashboard_updated"]

    def _finalize_task(self, p: dict) -> tuple[dict, float, list[str]]:
        score, details = GRADERS[self._task_id](self._history, self._state)
        return {
            "message"     : f"Task finalized. Final grade: {score:.2%}",
            "task_id"     : self._task_id,
            "final_score" : score,
            "grade_details": details,
        }, 0.0, []
