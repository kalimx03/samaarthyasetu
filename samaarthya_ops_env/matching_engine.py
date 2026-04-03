"""
Matching engine for SamaarthyaSetu.
Implements weighted scoring: Skill 60%, Accommodation 25%, Language/Extra 15%.
"""
from __future__ import annotations
from samaarthya_ops_env.models import Candidate, Job


def compute_skill_score(candidate: Candidate, job: Job) -> float:
    """Jaccard similarity between candidate skills and job required skills."""
    if not job.required_skills:
        return 0.5
    cand_skills = {s.lower() for s in candidate.skills}
    job_skills = {s.lower() for s in job.required_skills}
    intersection = cand_skills & job_skills
    union = cand_skills | job_skills
    return len(intersection) / len(union) if union else 0.0


def compute_accommodation_score(candidate: Candidate, job: Job) -> float:
    """How well the job's accommodations cover candidate's needs."""
    if not candidate.accommodation_needs:
        return 1.0
    cand_needs = {n.lower() for n in candidate.accommodation_needs}
    job_acc = {a.lower() for a in job.accommodations_provided}
    # Check keyword overlap
    covered = 0
    for need in cand_needs:
        need_words = set(need.split())
        for acc in job_acc:
            acc_words = set(acc.split())
            if need_words & acc_words:
                covered += 1
                break
    return covered / len(cand_needs)


def compute_language_score(candidate: Candidate, job: Job) -> float:
    """Language compatibility."""
    cand_langs = {l.lower() for l in candidate.language}
    job_langs = {l.lower() for l in job.languages_accepted}
    common = cand_langs & job_langs
    return min(1.0, len(common) / max(1, len(cand_langs)))


def compute_match_score(candidate: Candidate, job: Job) -> float:
    """
    Weighted composite match score.
    Skill: 60%, Accommodation: 25%, Language: 15%
    """
    if not job.open or not candidate.available:
        return 0.0
    skill = compute_skill_score(candidate, job)
    accommodation = compute_accommodation_score(candidate, job)
    language = compute_language_score(candidate, job)
    return round(0.60 * skill + 0.25 * accommodation + 0.15 * language, 4)


def get_top_matches(candidate: Candidate, jobs: list[Job], top_n: int = 3) -> list[dict]:
    """Return top N jobs for a candidate with detailed scores."""
    results = []
    for job in jobs:
        score = compute_match_score(candidate, job)
        results.append({
            "job_id": job.id,
            "job_title": job.title,
            "employer": job.employer_name,
            "overall_score": score,
            "skill_score": compute_skill_score(candidate, job),
            "accommodation_score": compute_accommodation_score(candidate, job),
            "language_score": compute_language_score(candidate, job),
            "salary_range": job.salary_range,
            "ward": job.ward,
        })
    results.sort(key=lambda x: x["overall_score"], reverse=True)
    return results[:top_n]


def check_scheme_eligibility(candidate: Candidate, schemes: list) -> list[dict]:
    """
    Check which schemes a candidate is eligible for.
    Returns list of {scheme_id, eligible, reason}.
    """
    results = []
    for scheme in schemes:
        eligible = True
        reasons = []

        # Check disability type
        if "All" not in scheme.disability_types_covered:
            if candidate.disability_type not in scheme.disability_types_covered:
                eligible = False
                reasons.append(f"Disability type '{candidate.disability_type}' not covered")

        # Income check (simplified — assume candidate is below limit if not specified)
        # In real system this would come from income data

        if eligible:
            reasons.append("Meets all eligibility criteria")

        results.append({
            "scheme_id": scheme.id,
            "scheme_name": scheme.name,
            "authority": scheme.authority,
            "eligible": eligible,
            "benefit": scheme.benefit,
            "documents_required": scheme.documents_required,
            "reason": "; ".join(reasons),
            "application_url": scheme.application_url,
        })

    return results


def generate_scheme_checklist(eligible_schemes: list[dict]) -> dict:
    """Generate a prioritized checklist for eligible schemes."""
    # Priority order: employment first, then financial, then assistive devices
    priority_keywords = ["employment", "job", "kaushal", "training", "exchange"]
    
    def priority_key(s: dict) -> int:
        name_lower = s["scheme_name"].lower()
        for i, kw in enumerate(priority_keywords):
            if kw in name_lower:
                return i
        return 99

    eligible = [s for s in eligible_schemes if s["eligible"]]
    eligible.sort(key=priority_key)

    all_docs: set[str] = set()
    for s in eligible:
        all_docs.update(s.get("documents_required", []))

    return {
        "priority_order": [s["scheme_id"] for s in eligible],
        "schemes": eligible,
        "documents": sorted(all_docs),
        "first_step": "Obtain UDID Card from nearest District Disability Rehabilitation Centre (DDRC)",
        "helpline": "1800-11-4515 (Divyangjan Helpline - Toll Free)",
        "bangalore_ddrc": "DDRC Bangalore: No. 1, K.G. Road, Bangalore - 560001",
    }
