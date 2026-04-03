"""
tests/test_environment.py
Tests for the SamaarthyaSetu OpenEnv environment interface.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from samaarthya_ops_env import SamaarthyaSetuEnvironment, Action, Observation, State


class TestEnvironmentReset:
    def test_reset_returns_observation(self):
        env = SamaarthyaSetuEnvironment("task_001")
        obs = env.reset()
        assert isinstance(obs, Observation)

    def test_reset_progress_is_zero(self):
        env = SamaarthyaSetuEnvironment("task_001")
        obs = env.reset()
        assert obs.progress == 0.0

    def test_reset_has_available_actions(self):
        env = SamaarthyaSetuEnvironment("task_001")
        obs = env.reset()
        assert len(obs.available_actions) > 0
        assert "list_candidates" in obs.available_actions

    def test_reset_all_tasks(self):
        for task_id in ["task_001", "task_002", "task_003"]:
            env = SamaarthyaSetuEnvironment(task_id)
            obs = env.reset()
            assert obs.progress == 0.0
            assert task_id in obs.data["task_id"]


class TestEnvironmentState:
    def test_state_returns_state_model(self):
        env = SamaarthyaSetuEnvironment("task_001")
        env.reset()
        s = env.state()
        assert isinstance(s, State)

    def test_initial_state_fields(self):
        env = SamaarthyaSetuEnvironment("task_001")
        env.reset()
        s = env.state()
        assert s.current_task_id == "task_001"
        assert s.steps_taken == 0
        assert s.completed is False
        assert s.progress_score == 0.0

    def test_state_updates_after_step(self):
        env = SamaarthyaSetuEnvironment("task_001")
        env.reset()
        action = Action(action_type="list_candidates", parameters={})
        env.step(action)
        s = env.state()
        assert s.steps_taken == 1


class TestEnvironmentStep:
    def test_step_returns_tuple(self):
        env = SamaarthyaSetuEnvironment("task_001")
        env.reset()
        action = Action(action_type="list_candidates", parameters={})
        result = env.step(action)
        assert len(result) == 4
        obs, reward, done, info = result
        assert isinstance(obs, Observation)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_reward_is_finite(self):
        env = SamaarthyaSetuEnvironment("task_001")
        env.reset()
        for at in ["list_candidates", "list_jobs"]:
            _, reward, _, _ = env.step(Action(action_type=at, parameters={}))
            assert -1.0 <= reward <= 2.0

    def test_repeated_action_penalty(self):
        env = SamaarthyaSetuEnvironment("task_001")
        env.reset()
        env.step(Action(action_type="list_candidates", parameters={}))
        _, reward2, _, info2 = env.step(Action(action_type="list_candidates", parameters={}))
        assert reward2 < 0 or "penalties" in info2 or "penalty" in info2

    def test_done_on_finalize(self):
        env = SamaarthyaSetuEnvironment("task_001")
        env.reset()
        for at in ["list_candidates", "list_jobs"]:
            env.step(Action(action_type=at, parameters={}))
        env.step(Action(action_type="get_candidate", parameters={"candidate_id": "cand_003"}))
        env.step(Action(action_type="get_job", parameters={"job_id": "job_002"}))
        env.step(Action(action_type="match_candidate_to_job",
                        parameters={"candidate_id": "cand_003", "job_id": "job_002"}))
        _, _, done, _ = env.step(Action(action_type="finalize_task", parameters={}))
        assert done is True

    def test_no_step_after_done(self):
        env = SamaarthyaSetuEnvironment("task_001")
        env.reset()
        env.step(Action(action_type="finalize_task", parameters={}))
        obs, reward, done, _ = env.step(Action(action_type="list_candidates", parameters={}))
        assert done is True

    def test_invalid_candidate_raises(self):
        env = SamaarthyaSetuEnvironment("task_001")
        env.reset()
        _, reward, _, info = env.step(Action(action_type="get_candidate",
                                             parameters={"candidate_id": "cand_FAKE"}))
        assert reward < 0 or "penalty" in info


class TestActionValidation:
    def test_all_valid_action_types(self):
        valid_actions = [
            "list_candidates", "get_candidate", "list_jobs", "get_job",
            "match_candidate_to_job", "check_scheme_eligibility",
            "generate_scheme_checklist", "schedule_interview",
            "resolve_placement_conflict", "publish_dashboard_update", "finalize_task",
        ]
        for at in valid_actions:
            action = Action(action_type=at, parameters={})
            assert action.action_type == at

    def test_invalid_action_type_raises(self):
        with pytest.raises(Exception):
            Action(action_type="fly_to_moon", parameters={})
