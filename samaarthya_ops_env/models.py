"""Core Pydantic v2 models for SamaarthyaSetu OpenEnv."""
from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field


class Action(BaseModel):
    """Typed action model for the SamaarthyaSetu environment."""
    action_type: Literal[
        "list_candidates",
        "get_candidate",
        "list_jobs",
        "get_job",
        "match_candidate_to_job",
        "check_scheme_eligibility",
        "generate_scheme_checklist",
        "schedule_interview",
        "resolve_placement_conflict",
        "publish_dashboard_update",
        "finalize_task",
    ]
    parameters: dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    """Environment observation returned after each step."""
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    available_actions: list[str] = Field(default_factory=list)
    progress: float = Field(ge=0.0, le=1.0)


class State(BaseModel):
    """Full environment state snapshot."""
    current_task_id: str
    selected_candidate_id: str | None = None
    selected_job_id: str | None = None
    steps_taken: int = 0
    progress_score: float = Field(ge=0.0, le=1.0, default=0.0)
    completed: bool = False
    actions_taken: list[str] = Field(default_factory=list)
    reward_history: list[float] = Field(default_factory=list)


class TaskResult(BaseModel):
    """Result of a completed task evaluation."""
    task_id: str
    score: float = Field(ge=0.0, le=1.0)
    steps_taken: int
    total_reward: float
    details: dict[str, Any] = Field(default_factory=dict)


class Candidate(BaseModel):
    id: str
    name: str
    age: int
    disability_type: str
    disability_certificate: str
    skills: list[str]
    language: list[str]
    accommodation_needs: list[str]
    ward: str
    ngo_trained_by: str
    training_program: str
    experience_years: int
    education: str
    location: str = "Bangalore"
    available: bool = True


class Job(BaseModel):
    id: str
    employer_id: str
    employer_name: str
    title: str
    required_skills: list[str]
    disability_friendly: bool
    accommodations_provided: list[str]
    languages_accepted: list[str]
    ward: str
    salary_range: str
    inclusivity_score: float
    open: bool = True


class Scheme(BaseModel):
    id: str
    name: str
    authority: str
    disability_types_covered: list[str]
    min_disability_percentage: int
    income_limit: int | None
    documents_required: list[str]
    benefit: str
    application_url: str


class NGO(BaseModel):
    id: str
    name: str
    ward: str
    programs: list[str]
    total_trained: int
    placed: int


class Employer(BaseModel):
    id: str
    name: str
    sector: str
    inclusivity_score: float
    disability_roles_count: int
    accommodations: list[str]
    pledge_signed: bool
