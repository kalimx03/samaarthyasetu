"""
SamaarthyaSetu FastAPI Server
Exposes OpenEnv environment via REST API for HuggingFace Spaces deployment.
"""
from __future__ import annotations
import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any

from samaarthya_ops_env import SamaarthyaSetuEnvironment, Action, Observation, State
from samaarthya_ops_env.data import CANDIDATES, JOBS, SCHEMES, NGOS, EMPLOYERS, WARDS
from samaarthya_ops_env.matching_engine import get_top_matches, check_scheme_eligibility, generate_scheme_checklist

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SamaarthyaSetu OpenEnv API",
    description="Inclusive Employment OS for Persons with Disabilities — Bangalore",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment instances per task
_envs: dict[str, SamaarthyaSetuEnvironment] = {}


def get_env(task_id: str) -> SamaarthyaSetuEnvironment:
    if task_id not in _envs:
        _envs[task_id] = SamaarthyaSetuEnvironment(task_id=task_id)
    return _envs[task_id]


# ─── Health & Info ────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
def root():
    return {
        "name": "SamaarthyaSetu OpenEnv",
        "version": "1.0.0",
        "status": "operational",
        "tasks": ["task_001", "task_002", "task_003"],
        "endpoints": ["/reset", "/step", "/state", "/candidates", "/jobs", "/schemes", "/dashboard"],
    }


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}


# ─── OpenEnv Interface ────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "task_001"


class _ActionPayload(BaseModel):
    action_type: str
    parameters: dict = {}

class StepRequest(BaseModel):
    task_id: str = "task_001"
    action: _ActionPayload


@app.post("/reset", response_model=Observation, tags=["openenv"])
def reset(req: ResetRequest):
    """Reset the environment for the given task."""
    if req.task_id not in ["task_001", "task_002", "task_003"]:
        raise HTTPException(400, f"Invalid task_id. Choose from: task_001, task_002, task_003")
    env = get_env(req.task_id)
    obs = env.reset()
    return obs


@app.post("/step", tags=["openenv"])
def step(req: StepRequest):
    """Execute one action in the environment."""
    if req.task_id not in ["task_001", "task_002", "task_003"]:
        raise HTTPException(400, "Invalid task_id.")
    env = get_env(req.task_id)
    action = Action(action_type=req.action.action_type, parameters=req.action.parameters)
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state/{task_id}", response_model=State, tags=["openenv"])
def state(task_id: str):
    """Get the current state of the environment."""
    if task_id not in ["task_001", "task_002", "task_003"]:
        raise HTTPException(400, "Invalid task_id.")
    env = get_env(task_id)
    return env.state()


# ─── Data Endpoints ────────────────────────────────────────────────────────────

@app.get("/candidates", tags=["data"])
def list_candidates(ward: str | None = None, disability_type: str | None = None):
    candidates = CANDIDATES
    if ward:
        candidates = [c for c in candidates if c.ward == ward]
    if disability_type:
        candidates = [c for c in candidates if c.disability_type == disability_type]
    return {"candidates": [c.model_dump() for c in candidates], "total": len(candidates)}


@app.get("/candidates/{candidate_id}", tags=["data"])
def get_candidate(candidate_id: str):
    c = next((c for c in CANDIDATES if c.id == candidate_id), None)
    if not c:
        raise HTTPException(404, f"Candidate {candidate_id} not found")
    return c.model_dump()


@app.get("/jobs", tags=["data"])
def list_jobs(ward: str | None = None, disability_friendly_only: bool = False):
    jobs = JOBS
    if ward:
        jobs = [j for j in jobs if j.ward == ward]
    if disability_friendly_only:
        jobs = [j for j in jobs if j.disability_friendly]
    return {"jobs": [j.model_dump() for j in jobs], "total": len(jobs)}


@app.get("/jobs/{job_id}", tags=["data"])
def get_job(job_id: str):
    j = next((j for j in JOBS if j.id == job_id), None)
    if not j:
        raise HTTPException(404, f"Job {job_id} not found")
    return j.model_dump()


@app.get("/match/{candidate_id}", tags=["matching"])
def match_candidate(candidate_id: str, top_n: int = 3):
    c = next((c for c in CANDIDATES if c.id == candidate_id), None)
    if not c:
        raise HTTPException(404, f"Candidate {candidate_id} not found")
    matches = get_top_matches(c, JOBS, top_n=top_n)
    return {"candidate_id": candidate_id, "matches": matches}


@app.get("/schemes", tags=["data"])
def list_schemes():
    return {"schemes": [s.model_dump() for s in SCHEMES], "total": len(SCHEMES)}


@app.get("/schemes/eligibility/{candidate_id}", tags=["schemes"])
def scheme_eligibility(candidate_id: str):
    c = next((c for c in CANDIDATES if c.id == candidate_id), None)
    if not c:
        raise HTTPException(404, f"Candidate {candidate_id} not found")
    results = check_scheme_eligibility(c, SCHEMES)
    checklist = generate_scheme_checklist(results)
    return {"candidate_id": candidate_id, "eligibility": results, "checklist": checklist}


@app.get("/ngos", tags=["data"])
def list_ngos():
    return {"ngos": [n.model_dump() for n in NGOS], "total": len(NGOS)}


@app.get("/employers", tags=["data"])
def list_employers():
    return {"employers": [e.model_dump() for e in EMPLOYERS], "total": len(EMPLOYERS)}


@app.get("/dashboard", tags=["dashboard"])
def dashboard():
    """Live dashboard stats — ward-level placement analytics."""
    candidates = CANDIDATES
    placed = sum(1 for c in candidates if not c.available)
    ward_stats = {}
    for ward in WARDS:
        ward_cands = [c for c in candidates if c.ward == ward]
        ward_jobs = [j for j in JOBS if j.ward == ward]
        ward_stats[ward] = {
            "total_candidates": len(ward_cands),
            "open_jobs": len(ward_jobs),
            "disability_types": list({c.disability_type for c in ward_cands}),
        }
    return {
        "total_candidates": len(candidates),
        "total_jobs": len(JOBS),
        "total_employers": len(EMPLOYERS),
        "total_ngos": len(NGOS),
        "total_schemes": len(SCHEMES),
        "placement_rate": f"{(placed / len(candidates) * 100):.1f}%",
        "wards_covered": len(WARDS),
        "ward_stats": ward_stats,
        "top_employers_by_inclusivity": sorted(
            [{"name": e.name, "score": e.inclusivity_score} for e in EMPLOYERS],
            key=lambda x: x["score"], reverse=True
        )[:3],
    }


@app.get("/tasks", tags=["openenv"])
def list_tasks():
    from samaarthya_ops_env.tasks import TASKS
    return {
        "tasks": [
            {
                "task_id": t.task_id,
                "name": t.name,
                "difficulty": t.difficulty,
                "description": t.description,
                "max_steps": t.max_steps,
            }
            for t in TASKS.values()
        ]
    }
