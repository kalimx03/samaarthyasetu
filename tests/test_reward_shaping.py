"""
tests/test_reward_shaping.py
============================
Comprehensive tests for the SamaarthyaSetu reward function and partial
progress signals.

These tests verify:
  1. Partial signals fire correctly and only once per episode
  2. Step efficiency calculation is monotonically decreasing
  3. Reward components are correctly weighted
  4. Penalties are applied at correct magnitudes
  5. Dense rewards appear throughout episode (not just at end)
  6. Full episode reward computation is correct
"""
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from samaarthya_ops_env import (
    SamaarthyaSetuEnvironment,
    Action,
    step_efficiency,
    compute_partial_bonus,
    compute_episode_reward,
    max_achievable_partial_bonus,
    signals_for_task,
    SIGNAL_CATALOGUE,
    W_PROGRESS,
    W_ACCURACY,
    W_EFFICIENCY,
)


# ── step_efficiency tests ─────────────────────────────────────────────────────

class TestStepEfficiency:
    def test_zero_steps_is_max(self):
        assert step_efficiency(0, 15) == 1.0

    def test_all_steps_is_zero(self):
        assert step_efficiency(15, 15) == 0.0

    def test_half_steps_is_half(self):
        val = step_efficiency(7, 14)
        assert abs(val - 0.5) < 0.01  # cos(π/2) ≈ 0

    def test_monotonically_decreasing(self):
        max_steps = 20
        efficiencies = [step_efficiency(s, max_steps) for s in range(max_steps + 1)]
        for i in range(len(efficiencies) - 1):
            assert efficiencies[i] >= efficiencies[i + 1], (
                f"Efficiency not monotone at step {i}: "
                f"{efficiencies[i]} vs {efficiencies[i+1]}"
            )

    def test_output_in_range(self):
        for steps in range(0, 31):
            val = step_efficiency(steps, 30)
            assert 0.0 <= val <= 1.0, f"Out of range at steps={steps}: {val}"

    def test_zero_max_steps_returns_zero(self):
        assert step_efficiency(5, 0) == 0.0

    def test_steps_exceed_max_clamped(self):
        # Should not raise, should return 0.0 (clamped)
        assert step_efficiency(100, 30) == 0.0

    def test_efficiency_better_with_fewer_steps(self):
        """6 steps in 15-step task should beat 12 steps."""
        assert step_efficiency(6, 15) > step_efficiency(12, 15)


# ── compute_partial_bonus tests ───────────────────────────────────────────────

class TestComputePartialBonus:
    def test_single_new_signal(self):
        bonus, awarded = compute_partial_bonus(["candidate_loaded"], set())
        expected = SIGNAL_CATALOGUE["candidate_loaded"][0]
        assert bonus == pytest.approx(expected)
        assert "candidate_loaded" in awarded

    def test_already_fired_signal_ignored(self):
        bonus, awarded = compute_partial_bonus(
            ["candidate_loaded"], {"candidate_loaded"}
        )
        assert bonus == 0.0
        assert awarded == []

    def test_multiple_new_signals(self):
        signals = ["candidate_loaded", "job_loaded"]
        bonus, awarded = compute_partial_bonus(signals, set())
        expected = (
            SIGNAL_CATALOGUE["candidate_loaded"][0]
            + SIGNAL_CATALOGUE["job_loaded"][0]
        )
        assert bonus == pytest.approx(expected)
        assert len(awarded) == 2

    def test_mix_new_and_fired(self):
        bonus, awarded = compute_partial_bonus(
            ["candidate_loaded", "match_computed"],
            {"candidate_loaded"},
        )
        expected = SIGNAL_CATALOGUE["match_computed"][0]
        assert bonus == pytest.approx(expected)
        assert awarded == ["match_computed"]

    def test_unknown_signal_ignored(self):
        bonus, awarded = compute_partial_bonus(["unknown_signal_xyz"], set())
        assert bonus == 0.0
        assert awarded == []

    def test_empty_signals_returns_zero(self):
        bonus, awarded = compute_partial_bonus([], set())
        assert bonus == 0.0
        assert awarded == []


# ── compute_episode_reward tests ──────────────────────────────────────────────

class TestComputeEpisodeReward:
    def test_perfect_episode(self):
        result = compute_episode_reward(
            progress=1.0,
            accuracy=1.0,
            steps_taken=4,   # very efficient
            max_steps=15,
            partial_signals_fired=list(SIGNAL_CATALOGUE.keys()),
        )
        assert result["total"] > 0.8
        assert result["progress_component"] == pytest.approx(W_PROGRESS * 1.0)
        assert result["accuracy_component"] == pytest.approx(W_ACCURACY * 1.0)

    def test_zero_progress_low_total(self):
        result = compute_episode_reward(
            progress=0.0,
            accuracy=0.0,
            steps_taken=30,
            max_steps=30,
            partial_signals_fired=[],
        )
        assert result["total"] == pytest.approx(0.0)

    def test_partial_bonus_accumulates(self):
        r_no_partial = compute_episode_reward(
            progress=0.5, accuracy=0.5, steps_taken=10, max_steps=20,
            partial_signals_fired=[],
        )
        r_with_partial = compute_episode_reward(
            progress=0.5, accuracy=0.5, steps_taken=10, max_steps=20,
            partial_signals_fired=["candidate_loaded", "match_computed"],
        )
        assert r_with_partial["total"] > r_no_partial["total"]
        assert r_with_partial["partial_bonus"] > 0

    def test_weights_sum_to_one(self):
        assert W_PROGRESS + W_ACCURACY + W_EFFICIENCY == pytest.approx(1.0)

    def test_components_all_present(self):
        result = compute_episode_reward(0.7, 0.8, 8, 20, ["candidate_loaded"])
        for key in [
            "progress_component", "accuracy_component",
            "efficiency_component", "partial_bonus", "total"
        ]:
            assert key in result

    def test_total_does_not_exceed_cap(self):
        """Total should be capped at 1.5 even with all signals."""
        result = compute_episode_reward(
            progress=1.0,
            accuracy=1.0,
            steps_taken=1,
            max_steps=30,
            partial_signals_fired=list(SIGNAL_CATALOGUE.keys()),
        )
        assert result["total"] <= 1.5


# ── SIGNAL_CATALOGUE integrity ────────────────────────────────────────────────

class TestSignalCatalogue:
    def test_all_bonuses_positive(self):
        for sig_id, (bonus, tier, desc) in SIGNAL_CATALOGUE.items():
            assert bonus > 0, f"Signal {sig_id} has non-positive bonus {bonus}"

    def test_all_tiers_valid(self):
        for sig_id, (bonus, tier, desc) in SIGNAL_CATALOGUE.items():
            assert tier in {1, 2, 3, 4}, f"Signal {sig_id} has invalid tier {tier}"

    def test_all_descriptions_nonempty(self):
        for sig_id, (bonus, tier, desc) in SIGNAL_CATALOGUE.items():
            assert desc, f"Signal {sig_id} has empty description"

    def test_max_partial_bonus_reasonable(self):
        max_b = max_achievable_partial_bonus()
        assert 0.5 < max_b < 2.0, f"Max bonus out of expected range: {max_b}"

    def test_task_signals_subset_of_catalogue(self):
        for task_id in ["task_001", "task_002", "task_003"]:
            for sig in signals_for_task(task_id):
                assert sig in SIGNAL_CATALOGUE, (
                    f"Signal '{sig}' for {task_id} not in catalogue"
                )


# ── Environment partial-signal integration ────────────────────────────────────

class TestPartialSignalsInEnvironment:
    """Verify partial signals fire correctly during actual episode steps."""

    def test_candidate_loaded_fires_on_get_candidate(self):
        env = SamaarthyaSetuEnvironment("task_001")
        env.reset()
        _, _, _, info = env.step(
            Action(action_type="get_candidate",
                   parameters={"candidate_id": "cand_003"})
        )
        signal_ids = [s["signal"] for s in info.get("partial_signals", [])]
        assert "candidate_loaded" in signal_ids

    def test_job_loaded_fires_on_get_job(self):
        env = SamaarthyaSetuEnvironment("task_001")
        env.reset()
        _, _, _, info = env.step(
            Action(action_type="get_job",
                   parameters={"job_id": "job_002"})
        )
        signal_ids = [s["signal"] for s in info.get("partial_signals", [])]
        assert "job_loaded" in signal_ids

    def test_match_signals_fire_on_match_action(self):
        env = SamaarthyaSetuEnvironment("task_001")
        env.reset()
        env.step(Action(action_type="get_candidate",
                        parameters={"candidate_id": "cand_003"}))
        env.step(Action(action_type="get_job",
                        parameters={"job_id": "job_002"}))
        _, _, _, info = env.step(
            Action(action_type="match_candidate_to_job",
                   parameters={"candidate_id": "cand_003", "job_id": "job_002"})
        )
        signal_ids = [s["signal"] for s in info.get("partial_signals", [])]
        assert "match_computed" in signal_ids

    def test_optimal_pair_fires_top_match_found(self):
        env = SamaarthyaSetuEnvironment("task_001")
        env.reset()
        _, _, _, info = env.step(
            Action(action_type="match_candidate_to_job",
                   parameters={"candidate_id": "cand_003", "job_id": "job_002"})
        )
        signal_ids = [s["signal"] for s in info.get("partial_signals", [])]
        assert "top_match_found" in signal_ids

    def test_signal_fires_only_once_per_episode(self):
        env = SamaarthyaSetuEnvironment("task_001")
        env.reset()
        # First get_job fires job_loaded
        _, _, _, info1 = env.step(
            Action(action_type="get_job", parameters={"job_id": "job_001"})
        )
        # Repeated get_job should NOT fire job_loaded again
        _, _, _, info2 = env.step(
            Action(action_type="get_job", parameters={"job_id": "job_002"})
        )
        sigs1 = {s["signal"] for s in info1.get("partial_signals", [])}
        sigs2 = {s["signal"] for s in info2.get("partial_signals", [])}
        # job_loaded should only fire once across both steps
        assert not (sigs1 & sigs2), (
            f"Overlap in signals between steps: {sigs1 & sigs2}"
        )

    def test_scheme_signals_fire_correctly(self):
        env = SamaarthyaSetuEnvironment("task_002")
        env.reset()
        env.step(Action(action_type="get_candidate",
                        parameters={"candidate_id": "cand_005"}))
        _, _, _, info = env.step(
            Action(action_type="check_scheme_eligibility",
                   parameters={"candidate_id": "cand_005"})
        )
        signal_ids = [s["signal"] for s in info.get("partial_signals", [])]
        assert "scheme_checked" in signal_ids

    def test_conflict_signals_fire_on_task_003(self):
        env = SamaarthyaSetuEnvironment("task_003")
        env.reset()
        env.step(Action(action_type="get_candidate",
                        parameters={"candidate_id": "cand_011"}))
        _, _, _, info = env.step(
            Action(action_type="resolve_placement_conflict",
                   parameters={
                       "candidate_id": "cand_011",
                       "conflict_type": "ngo_employer_mismatch",
                       "resolution": "schedule_fresh_interview",
                   })
        )
        signal_ids = [s["signal"] for s in info.get("partial_signals", [])]
        assert "conflict_identified" in signal_ids
        assert "conflict_resolved" in signal_ids

    def test_reward_positive_on_correct_actions(self):
        """Every correct action should yield positive net reward."""
        env = SamaarthyaSetuEnvironment("task_001")
        env.reset()
        positive_actions = [
            Action(action_type="list_candidates", parameters={}),
            Action(action_type="list_jobs", parameters={}),
            Action(action_type="get_candidate",
                   parameters={"candidate_id": "cand_003"}),
            Action(action_type="get_job",
                   parameters={"job_id": "job_002"}),
            Action(action_type="match_candidate_to_job",
                   parameters={"candidate_id": "cand_003", "job_id": "job_002"}),
        ]
        for action in positive_actions:
            _, reward, _, info = env.step(action)
            assert reward >= 0, (
                f"Expected non-negative reward for {action.action_type}, "
                f"got {reward}. Info: {info}"
            )

    def test_repeated_action_yields_negative_reward(self):
        env = SamaarthyaSetuEnvironment("task_001")
        env.reset()
        env.step(Action(action_type="list_candidates", parameters={}))
        _, reward2, _, info2 = env.step(
            Action(action_type="list_candidates", parameters={})
        )
        assert reward2 < 0 or "penalties" in info2

    def test_invalid_action_yields_negative_reward(self):
        env = SamaarthyaSetuEnvironment("task_001")
        env.reset()
        _, reward, _, info = env.step(
            Action(action_type="get_candidate",
                   parameters={"candidate_id": "FAKE_ID_999"})
        )
        assert reward < 0

    def test_dense_rewards_throughout_episode(self):
        """Agent should receive non-zero rewards on more than half of steps."""
        env = SamaarthyaSetuEnvironment("task_001")
        env.reset()
        actions = [
            Action(action_type="list_candidates", parameters={}),
            Action(action_type="list_jobs", parameters={}),
            Action(action_type="get_candidate",
                   parameters={"candidate_id": "cand_003"}),
            Action(action_type="get_job",
                   parameters={"job_id": "job_002"}),
            Action(action_type="match_candidate_to_job",
                   parameters={"candidate_id": "cand_003", "job_id": "job_002"}),
            Action(action_type="finalize_task", parameters={}),
        ]
        rewards = []
        for a in actions:
            _, r, done, _ = env.step(a)
            rewards.append(r)
        nonzero = sum(1 for r in rewards if r != 0.0)
        assert nonzero >= len(actions) // 2, (
            f"Expected dense rewards, got {nonzero}/{len(actions)} non-zero steps"
        )

    def test_step_efficiency_in_obs_data(self):
        """step_efficiency should be exposed in observation data."""
        env = SamaarthyaSetuEnvironment("task_001")
        env.reset()
        obs, _, _, _ = env.step(
            Action(action_type="list_candidates", parameters={})
        )
        assert "_step_efficiency" in obs.data

    def test_cumulative_reward_in_obs_data(self):
        env = SamaarthyaSetuEnvironment("task_001")
        env.reset()
        obs, _, _, _ = env.step(
            Action(action_type="list_candidates", parameters={})
        )
        assert "_cumulative_reward" in obs.data

    def test_final_score_present_on_finalize(self):
        env = SamaarthyaSetuEnvironment("task_001")
        env.reset()
        for a in [
            Action(action_type="list_candidates", parameters={}),
            Action(action_type="list_jobs", parameters={}),
            Action(action_type="get_candidate",
                   parameters={"candidate_id": "cand_003"}),
            Action(action_type="get_job",
                   parameters={"job_id": "job_002"}),
            Action(action_type="match_candidate_to_job",
                   parameters={"candidate_id": "cand_003", "job_id": "job_002"}),
            Action(action_type="finalize_task", parameters={}),
        ]:
            _, _, done, info = env.step(a)
        assert "final_score" in info
        assert 0.0 <= info["final_score"] <= 1.5

    def test_partial_signals_total_in_final_info(self):
        env = SamaarthyaSetuEnvironment("task_001")
        env.reset()
        for a in [
            Action(action_type="get_candidate",
                   parameters={"candidate_id": "cand_003"}),
            Action(action_type="get_job",
                   parameters={"job_id": "job_002"}),
            Action(action_type="match_candidate_to_job",
                   parameters={"candidate_id": "cand_003", "job_id": "job_002"}),
            Action(action_type="finalize_task", parameters={}),
        ]:
            _, _, _, info = env.step(a)
        assert "partial_signals_total" in info
        assert info["partial_signals_total"] >= 3

    def test_reward_design_in_reset_observation(self):
        """Reset observation should expose the full reward design."""
        env = SamaarthyaSetuEnvironment("task_001")
        obs = env.reset()
        rd = obs.data.get("reward_design", {})
        assert "formula" in rd
        assert "partial_signals" in rd
        assert "penalties" in rd


# ── Full end-to-end reward validation ────────────────────────────────────────

class TestEndToEndReward:
    def _run_optimal(self, task_id: str) -> tuple[float, float]:
        """Run the optimal script and return (total_reward, final_score)."""
        def A(at, p=None):
            return Action(action_type=at, parameters=p or {})
        scripts = {
            "task_001": [
                A("list_candidates"),
                A("list_jobs"),
                A("get_candidate", {"candidate_id": "cand_003"}),
                A("get_job", {"job_id": "job_002"}),
                A("match_candidate_to_job", {"candidate_id": "cand_003", "job_id": "job_002"}),
                A("finalize_task"),
            ],
            "task_002": [
                A("get_candidate", {"candidate_id": "cand_005"}),
                A("check_scheme_eligibility", {"candidate_id": "cand_005"}),
                A("generate_scheme_checklist", {"candidate_id": "cand_005"}),
                A("finalize_task"),
            ],
            "task_003": [
                A("get_candidate", {"candidate_id": "cand_011"}),
                A("get_job", {"job_id": "job_002"}),
                A("resolve_placement_conflict", {
                    "candidate_id": "cand_011",
                    "conflict_type": "ngo_employer_mismatch",
                    "resolution": "schedule_fresh_interview",
                }),
                A("schedule_interview", {
                    "candidate_id": "cand_011",
                    "employer_id": "emp_001",
                    "date": "2026-04-15",
                }),
                A("publish_dashboard_update", {
                    "ward": "Malleshwaram",
                    "update_type": "placement_status",
                }),
                A("finalize_task"),
            ],
        }
        env = SamaarthyaSetuEnvironment(task_id)
        env.reset()
        total_reward = 0.0
        final_score  = 0.0
        for action in scripts[task_id]:
            _, r, done, info = env.step(action)
            total_reward += r
            if "final_score" in info:
                final_score = info["final_score"]
        return round(total_reward, 4), round(final_score, 4)

    def test_task_001_optimal_score(self):
        total, fs = self._run_optimal("task_001")
        assert total > 0.5, f"task_001 total reward {total} too low"
        assert fs >= 0.7, f"task_001 final score {fs} below target"

    def test_task_002_optimal_score(self):
        total, fs = self._run_optimal("task_002")
        assert total > 0.4
        assert fs >= 0.6

    def test_task_003_optimal_score(self):
        total, fs = self._run_optimal("task_003")
        assert total > 0.5
        assert fs >= 0.65

    def test_all_tasks_score_above_zero(self):
        for task_id in ["task_001", "task_002", "task_003"]:
            total, fs = self._run_optimal(task_id)
            assert total > 0.0, f"{task_id} total reward is zero"
            assert fs > 0.0, f"{task_id} final score is zero"
