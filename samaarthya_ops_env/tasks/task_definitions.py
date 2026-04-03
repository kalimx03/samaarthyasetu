"""Task definitions for SamaarthyaSetu OpenEnv."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class TaskDefinition:
    task_id: str
    name: str
    description: str
    difficulty: str  # easy / medium / hard
    required_actions: list[str]
    success_criteria: dict[str, Any]
    max_steps: int


TASKS = {
    "task_001": TaskDefinition(
        task_id="task_001",
        name="Verified Job Match",
        description=(
            "An AI agent must identify the single best candidate-job match from the pool. "
            "List candidates, list jobs, call match_candidate_to_job for the optimal pair, "
            "then finalize. The correct answer is cand_003 → job_002 (highest combined score)."
        ),
        difficulty="easy",
        required_actions=["list_candidates", "list_jobs", "match_candidate_to_job", "finalize_task"],
        success_criteria={
            "target_candidate_id": "cand_003",
            "target_job_id": "job_002",
            "min_match_score": 0.75,
        },
        max_steps=15,
    ),
    "task_002": TaskDefinition(
        task_id="task_002",
        name="Scheme Navigator",
        description=(
            "Given candidate cand_005 (Speech & Language Disability, BCom, Karnataka resident), "
            "check eligibility for all schemes and generate a complete checklist with priority order. "
            "The agent must call check_scheme_eligibility and generate_scheme_checklist correctly."
        ),
        difficulty="medium",
        required_actions=["get_candidate", "check_scheme_eligibility", "generate_scheme_checklist", "finalize_task"],
        success_criteria={
            "candidate_id": "cand_005",
            "expected_eligible_schemes": ["sch_001", "sch_002", "sch_004", "sch_005", "sch_006", "sch_007", "sch_008", "sch_009", "sch_010"],
            "must_include_documents": True,
        },
        max_steps=20,
    ),
    "task_003": TaskDefinition(
        task_id="task_003",
        name="Placement Reconciliation",
        description=(
            "Candidate cand_011 (Deepak Gowda) has conflicting records: NGO says 'placed at Infosys BPM', "
            "employer says 'no such hire', and follow-up says 'still looking'. "
            "Agent must call get_candidate, get_job, resolve_placement_conflict, "
            "schedule_interview with the correct employer, and publish_dashboard_update."
        ),
        difficulty="hard",
        required_actions=[
            "get_candidate", "get_job", "resolve_placement_conflict",
            "schedule_interview", "publish_dashboard_update", "finalize_task"
        ],
        success_criteria={
            "candidate_id": "cand_011",
            "conflict_type": "ngo_employer_mismatch",
            "resolution": "schedule_fresh_interview",
            "target_employer_id": "emp_001",
        },
        max_steps=30,
    ),
}
