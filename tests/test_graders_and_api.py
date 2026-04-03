"""tests/test_graders.py — Deterministic grader unit tests."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from samaarthya_ops_env.graders import GRADERS, grade_task_001, grade_task_002, grade_task_003


class TestGraders:
    def _perfect_state_001(self):
        return {
            "current_task_id": "task_001",
            "selected_candidate_id": "cand_003",
            "selected_job_id": "job_002",
            "steps_taken": 5,
            "progress_score": 1.0,
            "completed": True,
            "actions_taken": ["list_candidates", "list_jobs", "match_candidate_to_job", "finalize_task"],
            "last_match_score": 0.85,
            "eligible_scheme_ids": [],
            "generated_checklist": {},
            "conflict_resolved": False,
            "interview_scheduled": False,
            "interview_employer_id": None,
            "dashboard_updated": False,
            "reward_history": [],
        }

    def test_grade_task_001_perfect_score(self):
        state = self._perfect_state_001()
        score, details = grade_task_001([], state)
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # optimal path should score > 0.5

    def test_grade_task_001_wrong_candidate(self):
        state = self._perfect_state_001()
        state["selected_candidate_id"] = "cand_001"  # wrong
        score, details = grade_task_001([], state)
        assert score < 0.9  # should penalize wrong match

    def test_grade_task_001_empty_actions(self):
        state = self._perfect_state_001()
        state["actions_taken"] = []
        score, details = grade_task_001([], state)
        # progress=0 (no required actions taken), but accuracy still counts
        # from pre-filled correct candidate/job in state → score ~0.48
        assert score < 0.6  # definitely below a passing agent
        assert details["progress"] == 0.0  # no actions means zero progress

    def test_grade_task_002_full_eligible(self):
        state = {
            "current_task_id": "task_002",
            "selected_candidate_id": "cand_005",
            "steps_taken": 4,
            "actions_taken": ["get_candidate", "check_scheme_eligibility",
                              "generate_scheme_checklist", "finalize_task"],
            "eligible_scheme_ids": ["sch_001", "sch_005", "sch_006", "sch_008"],
            "generated_checklist": {
                "documents": ["UDID Card", "Aadhaar"],
                "priority_order": ["sch_001", "sch_005"],
            },
            "conflict_resolved": False,
            "interview_scheduled": False,
            "interview_employer_id": None,
            "dashboard_updated": False,
            "reward_history": [],
        }
        score, details = grade_task_002([], state)
        assert 0.0 <= score <= 1.0
        assert score > 0.6

    def test_grade_task_003_full_completion(self):
        state = {
            "current_task_id": "task_003",
            "selected_candidate_id": "cand_011",
            "selected_job_id": "job_002",
            "steps_taken": 6,
            "actions_taken": [
                "get_candidate", "get_job", "resolve_placement_conflict",
                "schedule_interview", "publish_dashboard_update", "finalize_task"
            ],
            "conflict_resolved": True,
            "interview_scheduled": True,
            "interview_employer_id": "emp_001",
            "dashboard_updated": True,
            "eligible_scheme_ids": [],
            "generated_checklist": {},
            "reward_history": [],
        }
        score, details = grade_task_003([], state)
        assert 0.0 <= score <= 1.0
        assert score > 0.7

    def test_all_scores_in_range(self):
        for grader_fn in GRADERS.values():
            empty_state = {
                "selected_candidate_id": None, "selected_job_id": None,
                "steps_taken": 0, "actions_taken": [],
                "last_match_score": 0.0, "eligible_scheme_ids": [],
                "generated_checklist": {}, "conflict_resolved": False,
                "interview_scheduled": False, "interview_employer_id": None,
                "dashboard_updated": False, "reward_history": [],
            }
            score, _ = grader_fn([], empty_state)
            assert 0.0 <= score <= 1.0

    def test_grader_is_deterministic(self):
        """Same state → same score always."""
        state = {
            "selected_candidate_id": "cand_003", "selected_job_id": "job_002",
            "steps_taken": 5, "actions_taken": ["list_candidates", "list_jobs",
                                                  "match_candidate_to_job", "finalize_task"],
            "last_match_score": 0.85, "eligible_scheme_ids": [],
            "generated_checklist": {}, "conflict_resolved": False,
            "interview_scheduled": False, "interview_employer_id": None,
            "dashboard_updated": False, "reward_history": [],
        }
        score1, _ = grade_task_001([], state)
        score2, _ = grade_task_001([], state)
        assert score1 == score2


# ──────────────────────────────────────────────────────────────────────────────

"""tests/test_api_smoke.py — API smoke tests using TestClient."""
from fastapi.testclient import TestClient

def get_client():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from server.app import app
    return TestClient(app)


class TestAPISmoke:
    def test_root_returns_200(self):
        client = get_client()
        r = client.get("/")
        assert r.status_code == 200

    def test_health_endpoint(self):
        client = get_client()
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_reset_task_001(self):
        client = get_client()
        r = client.post("/reset", json={"task_id": "task_001"})
        assert r.status_code == 200
        data = r.json()
        assert "message" in data
        assert "available_actions" in data
        assert data["progress"] == 0.0

    def test_reset_all_tasks(self):
        client = get_client()
        for task_id in ["task_001", "task_002", "task_003"]:
            r = client.post("/reset", json={"task_id": task_id})
            assert r.status_code == 200

    def test_step_list_candidates(self):
        client = get_client()
        client.post("/reset", json={"task_id": "task_001"})
        r = client.post("/step", json={
            "task_id": "task_001",
            "action": {"action_type": "list_candidates", "parameters": {}}
        })
        assert r.status_code == 200
        data = r.json()
        assert "observation" in data
        assert "reward" in data
        assert "done" in data

    def test_state_endpoint(self):
        client = get_client()
        client.post("/reset", json={"task_id": "task_001"})
        r = client.get("/state/task_001")
        assert r.status_code == 200
        data = r.json()
        assert data["current_task_id"] == "task_001"
        assert data["steps_taken"] == 0

    def test_candidates_endpoint(self):
        client = get_client()
        r = client.get("/candidates")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 15

    def test_jobs_endpoint(self):
        client = get_client()
        r = client.get("/jobs")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 10

    def test_match_endpoint(self):
        client = get_client()
        r = client.get("/match/cand_003")
        assert r.status_code == 200
        data = r.json()
        assert "matches" in data
        assert len(data["matches"]) == 3

    def test_schemes_endpoint(self):
        client = get_client()
        r = client.get("/schemes")
        assert r.status_code == 200

    def test_dashboard_endpoint(self):
        client = get_client()
        r = client.get("/dashboard")
        assert r.status_code == 200
        data = r.json()
        assert "total_candidates" in data
        assert "ward_stats" in data

    def test_invalid_task_reset(self):
        client = get_client()
        r = client.post("/reset", json={"task_id": "task_999"})
        assert r.status_code == 400

    def test_tasks_list(self):
        client = get_client()
        r = client.get("/tasks")
        assert r.status_code == 200
        data = r.json()
        assert len(data["tasks"]) == 3
