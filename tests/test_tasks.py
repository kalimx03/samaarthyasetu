"""tests/test_tasks.py — Task completion integration tests."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from samaarthya_ops_env import SamaarthyaSetuEnvironment, Action
from samaarthya_ops_env.tasks import TASKS


class TestTaskDefinitions:
    def test_all_tasks_exist(self):
        for tid in ["task_001", "task_002", "task_003"]:
            assert tid in TASKS

    def test_difficulty_levels(self):
        assert TASKS["task_001"].difficulty == "easy"
        assert TASKS["task_002"].difficulty == "medium"
        assert TASKS["task_003"].difficulty == "hard"

    def test_max_steps_increasing(self):
        assert TASKS["task_001"].max_steps <= TASKS["task_002"].max_steps
        assert TASKS["task_002"].max_steps <= TASKS["task_003"].max_steps

    def test_required_actions_nonempty(self):
        for task in TASKS.values():
            assert len(task.required_actions) > 0


class TestTask001Integration:
    def test_optimal_path_scores_high(self):
        env = SamaarthyaSetuEnvironment("task_001")
        env.reset()
        script = [
            Action(action_type="list_candidates", parameters={}),
            Action(action_type="list_jobs", parameters={}),
            Action(action_type="get_candidate", parameters={"candidate_id": "cand_003"}),
            Action(action_type="get_job", parameters={"job_id": "job_002"}),
            Action(action_type="match_candidate_to_job",
                   parameters={"candidate_id": "cand_003", "job_id": "job_002"}),
            Action(action_type="finalize_task", parameters={}),
        ]
        total_reward = 0
        for action in script:
            _, reward, done, info = env.step(action)
            total_reward += reward
        assert done is True
        assert total_reward > 0
        s = env.state()
        assert s.completed is True

    def test_progress_increases_with_actions(self):
        env = SamaarthyaSetuEnvironment("task_001")
        env.reset()
        prev = 0.0
        for at in ["list_candidates", "list_jobs", "match_candidate_to_job"]:
            if at == "match_candidate_to_job":
                env.step(Action(action_type="get_candidate",
                               parameters={"candidate_id": "cand_003"}))
                env.step(Action(action_type=at,
                               parameters={"candidate_id": "cand_003", "job_id": "job_002"}))
            else:
                env.step(Action(action_type=at, parameters={}))
            curr = env.state().progress_score
            assert curr >= prev
            prev = curr


class TestTask002Integration:
    def test_scheme_navigator_optimal_path(self):
        env = SamaarthyaSetuEnvironment("task_002")
        env.reset()
        script = [
            Action(action_type="get_candidate", parameters={"candidate_id": "cand_005"}),
            Action(action_type="check_scheme_eligibility",
                   parameters={"candidate_id": "cand_005"}),
            Action(action_type="generate_scheme_checklist",
                   parameters={"candidate_id": "cand_005"}),
            Action(action_type="finalize_task", parameters={}),
        ]
        for action in script:
            obs, reward, done, info = env.step(action)
        assert done is True
        assert env.state().completed is True

    def test_scheme_eligibility_populates_state(self):
        env = SamaarthyaSetuEnvironment("task_002")
        env.reset()
        env.step(Action(action_type="get_candidate",
                        parameters={"candidate_id": "cand_005"}))
        env.step(Action(action_type="check_scheme_eligibility",
                        parameters={"candidate_id": "cand_005"}))
        s = env.state()
        assert s.selected_candidate_id == "cand_005"
        # eligible_scheme_ids should be populated (accessed via internal state)


class TestTask003Integration:
    def test_placement_reconciliation_optimal_path(self):
        env = SamaarthyaSetuEnvironment("task_003")
        env.reset()
        script = [
            Action(action_type="get_candidate",
                   parameters={"candidate_id": "cand_011"}),
            Action(action_type="get_job", parameters={"job_id": "job_002"}),
            Action(action_type="resolve_placement_conflict", parameters={
                "candidate_id": "cand_011",
                "conflict_type": "ngo_employer_mismatch",
                "resolution": "schedule_fresh_interview",
            }),
            Action(action_type="schedule_interview", parameters={
                "candidate_id": "cand_011",
                "employer_id": "emp_001",
                "date": "2026-04-15",
            }),
            Action(action_type="publish_dashboard_update", parameters={
                "ward": "Malleshwaram",
                "update_type": "placement_status",
            }),
            Action(action_type="finalize_task", parameters={}),
        ]
        total_reward = 0
        for action in script:
            _, reward, done, info = env.step(action)
            total_reward += reward
        assert done is True
        assert total_reward > 0
