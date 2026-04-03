"""
samaarthya_ops_env/reward_shaping.py
=====================================
Standalone utilities for the SamaarthyaSetu reward function.

This module makes the reward design transparent, testable, and reusable.
It can be imported by graders, tests, or third-party evaluators without
instantiating the full environment.

Reward Formula
--------------
  R_total = W_PROGRESS  * progress_fraction
           + W_ACCURACY  * accuracy_signal
           + W_EFFICIENCY* step_efficiency(steps, max_steps)
           + Σ partial_signal_bonus(signal_id)   [each fires at most once]
           - Σ penalty(reason)

Partial Progress Signals
------------------------
Dense intermediate rewards fire on the FIRST step each sub-goal is
satisfied.  This gives the agent informative feedback at every step,
rather than only at episode end (sparse reward), which dramatically
accelerates policy learning.

Signal taxonomy:
  Tier 1 (information retrieval)  — small bonuses (0.03–0.04)
  Tier 2 (computation / matching) — medium bonuses (0.06–0.08)
  Tier 3 (decision quality)       — larger bonuses (0.08–0.09)
  Tier 4 (completeness)           — completeness bonuses (0.04–0.05)
"""
from __future__ import annotations
import math

# ── Weights ───────────────────────────────────────────────────────────────────
W_PROGRESS   = 0.50
W_ACCURACY   = 0.35
W_EFFICIENCY = 0.15

# ── Partial signal catalogue ──────────────────────────────────────────────────
#   key → (bonus_value, tier, description)
SIGNAL_CATALOGUE: dict[str, tuple[float, int, str]] = {
    # Tier 1 — information retrieval
    "candidate_loaded"       : (0.04, 1, "Candidate profile fetched from state"),
    "job_loaded"             : (0.03, 1, "Job listing fetched with accommodation details"),
    # Tier 2 — computation & matching
    "match_computed"         : (0.08, 2, "Weighted composite match score computed (60/25/15)"),
    "scheme_checked"         : (0.07, 2, "Scheme eligibility evaluated for candidate"),
    # Tier 3 — decision quality
    "top_match_found"        : (0.06, 3, "Optimal candidate-job pair correctly identified"),
    "high_quality_match"     : (0.04, 3, "Match score ≥ 0.75 — high-confidence placement"),
    "conflict_identified"    : (0.05, 3, "Placement conflict type correctly classified"),
    "conflict_resolved"      : (0.08, 3, "Conflict resolution strategy selected and executed"),
    "interview_confirmed"    : (0.09, 3, "Fresh interview booked with the correct employer"),
    "checklist_generated"    : (0.08, 3, "Prioritised document checklist generated"),
    # Tier 4 — completeness / quality bonuses
    "all_schemes_found"      : (0.05, 4, "All expected eligible schemes identified (recall = 1.0)"),
    "ward_dashboard_updated" : (0.05, 4, "Ward-level dashboard record published"),
}

# ── Penalty catalogue ─────────────────────────────────────────────────────────
PENALTY_CATALOGUE: dict[str, float] = {
    "invalid_action" : -0.05,
    "repeated_action": -0.02,
    "early_finalize" : -0.10,
    "wrong_target"   : -0.02,
}


def step_efficiency(steps_taken: int, max_steps: int) -> float:
    """
    Smooth cosine-decay step efficiency ∈ [0, 1].

    Rewards agents that complete tasks with fewer steps:
      steps = 0           → 1.00
      steps = max_steps/2 → 0.50
      steps = max_steps   → 0.00

    Using cosine ensures a smooth, differentiable gradient throughout
    the episode rather than a hard cliff at max_steps.

    Args:
        steps_taken: number of steps used so far
        max_steps:   maximum allowed steps for this task

    Returns:
        float in [0, 1]
    """
    if max_steps <= 0:
        return 0.0
    ratio = min(steps_taken / max_steps, 1.0)
    return round(0.5 * (1.0 + math.cos(math.pi * ratio)), 4)


def compute_partial_bonus(
    signals_fired: list[str],
    already_fired: set[str],
) -> tuple[float, list[str]]:
    """
    Compute the total partial-progress bonus for a list of newly fired signals,
    excluding any that have already fired in this episode.

    Args:
        signals_fired:  signal IDs that fired on this step
        already_fired:  set of signal IDs that fired in previous steps

    Returns:
        (total_bonus, list_of_newly_awarded_signal_ids)
    """
    total = 0.0
    newly_awarded: list[str] = []
    for sid in signals_fired:
        if sid in already_fired:
            continue
        if sid not in SIGNAL_CATALOGUE:
            continue
        bonus, _tier, _desc = SIGNAL_CATALOGUE[sid]
        total += bonus
        newly_awarded.append(sid)
    return round(total, 4), newly_awarded


def compute_episode_reward(
    progress: float,
    accuracy: float,
    steps_taken: int,
    max_steps: int,
    partial_signals_fired: list[str],
) -> dict[str, float]:
    """
    Compute the full episode reward breakdown.

    Args:
        progress:              fraction of required actions completed [0, 1]
        accuracy:              grader accuracy component [0, 1]
        steps_taken:           total steps used in episode
        max_steps:             maximum allowed steps for this task
        partial_signals_fired: list of all signal IDs fired during episode

    Returns:
        dict with keys: progress_component, accuracy_component,
        efficiency_component, partial_bonus, total (all floats)
    """
    eff         = step_efficiency(steps_taken, max_steps)
    partial_sum = sum(
        SIGNAL_CATALOGUE[s][0]
        for s in partial_signals_fired
        if s in SIGNAL_CATALOGUE
    )
    progress_c   = W_PROGRESS   * progress
    accuracy_c   = W_ACCURACY   * accuracy
    efficiency_c = W_EFFICIENCY * eff
    total        = min(1.5, progress_c + accuracy_c + efficiency_c + partial_sum)

    return {
        "progress_component"  : round(progress_c, 4),
        "accuracy_component"  : round(accuracy_c, 4),
        "efficiency_component": round(efficiency_c, 4),
        "partial_bonus"       : round(partial_sum, 4),
        "total"               : round(total, 4),
        "efficiency_raw"      : round(eff, 4),
    }


def max_achievable_partial_bonus() -> float:
    """Return the theoretical maximum partial signal bonus (all signals fired)."""
    return round(sum(v[0] for v in SIGNAL_CATALOGUE.values()), 4)


def signals_for_task(task_id: str) -> list[str]:
    """
    Return the subset of signals most relevant to a given task.
    Useful for graders and evaluation tooling.
    """
    task_signals = {
        "task_001": [
            "candidate_loaded", "job_loaded", "match_computed",
            "top_match_found", "high_quality_match",
        ],
        "task_002": [
            "candidate_loaded", "scheme_checked", "all_schemes_found",
            "checklist_generated",
        ],
        "task_003": [
            "candidate_loaded", "job_loaded", "conflict_identified",
            "conflict_resolved", "interview_confirmed", "ward_dashboard_updated",
        ],
    }
    return task_signals.get(task_id, [])
