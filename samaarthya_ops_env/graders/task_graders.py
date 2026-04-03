"""
Deterministic graders for SamaarthyaSetu tasks.
All graders return float in [0.0, 1.0]. No randomness.
"""
from __future__ import annotations
from typing import Any


def grade_task_001(state_history: list[dict], final_state: dict) -> tuple[float, dict]:
    """
    Grade Task 1: Verified Job Match.
    Target: cand_003 matched to job_002 with score >= 0.75.
    """
    details: dict[str, Any] = {}

    matched_candidate = final_state.get("selected_candidate_id")
    matched_job = final_state.get("selected_job_id")
    match_score = final_state.get("last_match_score", 0.0)
    actions_taken = final_state.get("actions_taken", [])

    target_candidate = "cand_003"
    target_job = "job_002"

    # Correctness: did they match the right pair?
    correct_match = (matched_candidate == target_candidate and matched_job == target_job)
    accuracy = 1.0 if correct_match else 0.0

    # Partial credit: correct candidate but wrong job, or vice versa
    if not correct_match:
        if matched_candidate == target_candidate or matched_job == target_job:
            accuracy = 0.4
        elif match_score >= 0.6:
            accuracy = 0.2

    details["correct_match"] = correct_match
    details["matched_candidate"] = matched_candidate
    details["matched_job"] = matched_job
    details["match_score"] = match_score

    # Progress: were required actions taken?
    required = {"list_candidates", "list_jobs", "match_candidate_to_job", "finalize_task"}
    taken_set = set(actions_taken)
    progress = len(required & taken_set) / len(required)

    # Efficiency: fewer steps = better (max_steps=15, target ≤8)
    steps = final_state.get("steps_taken", 15)
    efficiency = max(0.0, 1.0 - (steps - 4) / 11.0) if steps >= 4 else 1.0

    score = 0.50 * progress + 0.35 * accuracy + 0.15 * efficiency
    details["progress"] = progress
    details["accuracy"] = accuracy
    details["efficiency"] = efficiency

    return round(min(1.0, max(0.0, score)), 4), details


def grade_task_002(state_history: list[dict], final_state: dict) -> tuple[float, dict]:
    """
    Grade Task 2: Scheme Navigator.
    Target: Generate correct checklist for cand_005 with eligible schemes.
    """
    details: dict[str, Any] = {}

    actions_taken = final_state.get("actions_taken", [])
    generated_checklist = final_state.get("generated_checklist", {})
    candidate_id = final_state.get("selected_candidate_id")
    eligible_found = final_state.get("eligible_scheme_ids", [])

    # Required actions
    required = {"get_candidate", "check_scheme_eligibility", "generate_scheme_checklist", "finalize_task"}
    taken_set = set(actions_taken)
    progress = len(required & taken_set) / len(required)

    # Accuracy: correct candidate targeted?
    correct_candidate = candidate_id == "cand_005"
    details["correct_candidate"] = correct_candidate

    # Accuracy: correct schemes found?
    # All schemes covering "All" disability types are eligible for cand_005 (Speech & Language)
    # Engine correctly returns all 9 "All" coverage schemes — grader measures recall across them
    expected_schemes = {"sch_001", "sch_002", "sch_004", "sch_005",
                        "sch_006", "sch_007", "sch_008", "sch_009", "sch_010"}
    found_set = set(eligible_found)
    scheme_recall = len(expected_schemes & found_set) / len(expected_schemes) if expected_schemes else 0
    scheme_precision = len(expected_schemes & found_set) / len(found_set) if found_set else 0
    scheme_f1 = (2 * scheme_precision * scheme_recall / (scheme_precision + scheme_recall)
                 if (scheme_precision + scheme_recall) > 0 else 0)

    details["scheme_recall"] = scheme_recall
    details["scheme_precision"] = scheme_precision
    details["eligible_found"] = list(found_set)

    # Checklist completeness: has documents?
    has_docs = bool(generated_checklist.get("documents"))
    has_priority = bool(generated_checklist.get("priority_order"))
    checklist_score = (0.5 * int(has_docs) + 0.5 * int(has_priority))

    details["has_documents"] = has_docs
    details["has_priority_order"] = has_priority

    accuracy = (0.5 * int(correct_candidate) + 0.3 * scheme_f1 + 0.2 * checklist_score)
    steps = final_state.get("steps_taken", 20)
    efficiency = max(0.0, 1.0 - (steps - 4) / 16.0)

    score = 0.50 * progress + 0.35 * accuracy + 0.15 * efficiency
    details["progress"] = progress
    details["accuracy"] = accuracy
    details["efficiency"] = efficiency

    return round(min(1.0, max(0.0, score)), 4), details


def grade_task_003(state_history: list[dict], final_state: dict) -> tuple[float, dict]:
    """
    Grade Task 3: Placement Reconciliation (Hard).
    Target: Identify conflict, resolve, schedule interview with emp_001, publish update.
    """
    details: dict[str, Any] = {}

    actions_taken = final_state.get("actions_taken", [])
    conflict_resolved = final_state.get("conflict_resolved", False)
    interview_scheduled = final_state.get("interview_scheduled", False)
    interview_employer = final_state.get("interview_employer_id")
    dashboard_updated = final_state.get("dashboard_updated", False)
    candidate_id = final_state.get("selected_candidate_id")

    required = {
        "get_candidate", "get_job", "resolve_placement_conflict",
        "schedule_interview", "publish_dashboard_update", "finalize_task"
    }
    taken_set = set(actions_taken)
    progress = len(required & taken_set) / len(required)

    # Accuracy breakdown
    correct_candidate = candidate_id == "cand_011"
    correct_employer = interview_employer == "emp_001"

    accuracy = (
        0.25 * int(correct_candidate) +
        0.25 * int(conflict_resolved) +
        0.25 * int(interview_scheduled and correct_employer) +
        0.25 * int(dashboard_updated)
    )

    steps = final_state.get("steps_taken", 30)
    efficiency = max(0.0, 1.0 - (steps - 6) / 24.0)

    details["correct_candidate"] = correct_candidate
    details["conflict_resolved"] = conflict_resolved
    details["interview_scheduled"] = interview_scheduled
    details["correct_employer"] = correct_employer
    details["dashboard_updated"] = dashboard_updated
    details["progress"] = progress
    details["accuracy"] = accuracy
    details["efficiency"] = efficiency

    score = 0.50 * progress + 0.35 * accuracy + 0.15 * efficiency
    return round(min(1.0, max(0.0, score)), 4), details


GRADERS = {
    "task_001": grade_task_001,
    "task_002": grade_task_002,
    "task_003": grade_task_003,
}
