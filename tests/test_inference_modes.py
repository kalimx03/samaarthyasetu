"""
tests/test_inference_modes.py
==============================
Tests for the inference.py CLI modes.
No API key required — tests the rules mode end-to-end and
verifies the argument parser and mode routing logic.
"""
import sys
import os
import json
import subprocess
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


INFERENCE_PY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "inference.py"
)


def _run_inference(*args, timeout=60) -> tuple[int, str, str]:
    """Run inference.py as a subprocess, return (returncode, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, INFERENCE_PY] + list(args),
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "HF_TOKEN": ""},  # ensure no accidental LLM calls
    )
    return result.returncode, result.stdout, result.stderr


class TestRulesModeAllTasks:
    """Rules mode must complete all tasks and produce valid scores."""

    def test_rules_mode_exits_zero(self):
        rc, stdout, stderr = _run_inference("--mode", "rules")
        assert rc == 0, f"Non-zero exit.\nstdout: {stdout}\nstderr: {stderr}"

    def test_rules_mode_produces_json_output(self):
        _run_inference("--mode", "rules")
        assert os.path.exists("inference_results.json"), \
            "inference_results.json not created"
        with open("inference_results.json") as f:
            data = json.load(f)
        assert "average_score" in data
        assert "task_results" in data
        assert len(data["task_results"]) == 3
        os.remove("inference_results.json")

    def test_rules_mode_all_tasks_complete(self):
        _run_inference("--mode", "rules")
        with open("inference_results.json") as f:
            data = json.load(f)
        for r in data["task_results"]:
            assert r["completed"] is True, \
                f"{r['task_id']} did not complete in rules mode"
        os.remove("inference_results.json")

    def test_rules_mode_average_score_above_threshold(self):
        _run_inference("--mode", "rules")
        with open("inference_results.json") as f:
            data = json.load(f)
        avg = data["average_score"]
        assert avg >= 0.75, f"Average score {avg} below 0.75 threshold"
        os.remove("inference_results.json")

    def test_rules_mode_no_llm_steps(self):
        _run_inference("--mode", "rules")
        with open("inference_results.json") as f:
            data = json.load(f)
        for r in data["task_results"]:
            assert r["llm_steps"] == 0, \
                f"{r['task_id']}: rules mode made LLM calls (should be 0)"
        os.remove("inference_results.json")


class TestRulesModePerTask:
    """Single-task runs produce consistent results."""

    @pytest.mark.parametrize("task_id", ["task_001", "task_002", "task_003"])
    def test_single_task_completes(self, task_id):
        rc, stdout, _ = _run_inference("--mode", "rules", "--task", task_id)
        assert rc == 0
        with open("inference_results.json") as f:
            data = json.load(f)
        assert data["tasks_run"] == [task_id]
        assert data["task_results"][0]["completed"] is True
        os.remove("inference_results.json")

    def test_task_001_score_above_threshold(self):
        _run_inference("--mode", "rules", "--task", "task_001")
        with open("inference_results.json") as f:
            data = json.load(f)
        score = data["task_results"][0]["final_score"]
        assert score >= 0.90, f"task_001 score {score} < 0.90"
        os.remove("inference_results.json")

    def test_task_002_score_above_threshold(self):
        _run_inference("--mode", "rules", "--task", "task_002")
        with open("inference_results.json") as f:
            data = json.load(f)
        score = data["task_results"][0]["final_score"]
        assert score >= 0.85, f"task_002 score {score} < 0.85"
        os.remove("inference_results.json")

    def test_task_003_score_above_threshold(self):
        _run_inference("--mode", "rules", "--task", "task_003")
        with open("inference_results.json") as f:
            data = json.load(f)
        score = data["task_results"][0]["final_score"]
        assert score >= 0.85, f"task_003 score {score} < 0.85"
        os.remove("inference_results.json")


class TestOutputStructure:
    """inference_results.json has the correct schema."""

    def test_output_has_mode_field(self):
        _run_inference("--mode", "rules", "--task", "task_001")
        with open("inference_results.json") as f:
            data = json.load(f)
        assert data["mode"] == "rules"
        os.remove("inference_results.json")

    def test_output_has_llm_rule_step_counts(self):
        _run_inference("--mode", "rules", "--task", "task_001")
        with open("inference_results.json") as f:
            data = json.load(f)
        r = data["task_results"][0]
        assert "llm_steps" in r
        assert "rule_steps" in r
        assert r["llm_steps"] + r["rule_steps"] == r["steps_taken"]
        os.remove("inference_results.json")

    def test_steps_have_source_field(self):
        _run_inference("--mode", "rules", "--task", "task_001")
        with open("inference_results.json") as f:
            data = json.load(f)
        for step in data["task_results"][0]["steps"]:
            assert "source" in step, f"Step missing 'source': {step}"
        os.remove("inference_results.json")


class TestCLIArguments:
    """Argument parser handles edge cases correctly."""

    def test_llm_mode_without_token_exits_nonzero(self):
        rc, stdout, stderr = _run_inference("--mode", "llm", "--task", "task_001")
        # Should exit with error code since HF_TOKEN is empty
        assert rc != 0, "LLM mode without token should fail"

    def test_hybrid_mode_without_token_exits_nonzero(self):
        rc, stdout, stderr = _run_inference("--mode", "hybrid", "--task", "task_001")
        assert rc != 0, "Hybrid mode without token should fail"

    def test_invalid_mode_exits_nonzero(self):
        rc, _, _ = _run_inference("--mode", "turbo_boost")
        assert rc != 0

    def test_invalid_task_exits_nonzero(self):
        rc, _, _ = _run_inference("--mode", "rules", "--task", "task_999")
        assert rc != 0

    def test_demo_flag_runs_single_task(self):
        rc, stdout, _ = _run_inference(
            "--mode", "rules", "--task", "task_001", "--demo"
        )
        assert rc == 0
        with open("inference_results.json") as f:
            data = json.load(f)
        assert data["tasks_run"] == ["task_001"]
        os.remove("inference_results.json")
