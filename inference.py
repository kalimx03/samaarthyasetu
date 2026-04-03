#!/usr/bin/env python3
"""
SamaarthyaSetu — Inference Script  (v3)
========================================
Runs all 3 tasks using three agent modes:

  --mode rules   : Rule-based baseline only (no LLM, always reproducible)
  --mode llm     : LLM-only — model drives every step from reset -> finalize
  --mode hybrid  : LLM primary, rule-based only on JSON parse failures

Usage
-----
  # Rule-based (always works, no API key needed)
  python inference.py --mode rules

  # LLM-only (requires API key — shows the agent reasoning loop)
  export API_BASE_URL="https://api.openai.com/v1"
  export MODEL_NAME="gpt-4o-mini"
  export HF_TOKEN="sk-..."
  python inference.py --mode llm

  # LLM-only, single task, verbose tracing
  python inference.py --mode llm --task task_001 --verbose

  # Quick demo of the LLM solving task_001 independently
  python inference.py --mode llm --task task_001 --demo
"""
import argparse
import json
import logging
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from samaarthya_ops_env import SamaarthyaSetuEnvironment, Action

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("samaarthyasetu.inference")

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")

MAX_STEPS = {"task_001": 15, "task_002": 20, "task_003": 30}
ALL_TASKS  = ["task_001", "task_002", "task_003"]

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an AI agent solving tasks inside SamaarthyaSetu, an employment platform
for persons with disabilities in Bangalore, India.

RESPONSE CONTRACT
-----------------
Respond with ONLY a single JSON object — no markdown, no explanation, no extra text:
  {"action_type": "<action>", "parameters": {<key-value pairs>}}

AVAILABLE ACTIONS
-----------------
list_candidates      {"ward"?: str, "disability_type"?: str}
get_candidate        {"candidate_id": str}
list_jobs            {"ward"?: str, "disability_friendly_only"?: bool}
get_job              {"job_id": str}
match_candidate_to_job  {"candidate_id": str, "job_id"?: str}
check_scheme_eligibility   {"candidate_id": str}
generate_scheme_checklist  {"candidate_id": str}
schedule_interview   {"candidate_id": str, "employer_id": str, "date"?: str}
resolve_placement_conflict {"candidate_id": str, "conflict_type": str, "resolution": str}
publish_dashboard_update   {"ward"?: str, "update_type"?: str}
finalize_task        {}

KNOWN ENTITY IDs
----------------
Candidates: cand_001 through cand_015
Jobs:       job_001 through job_010
Employers:  emp_001 (Infosys BPM), emp_002 (BigBasket), emp_003 (Namdhari's),
            emp_004 (Myntra), emp_005 (Apollo), emp_006 (HDFC Bank),
            emp_007 (Decathlon), emp_008 (Wipro GreenTech)

STRATEGY
--------
1. Read the task message carefully — it names the exact target candidate and job.
2. Complete all required actions in a logical order before calling finalize_task.
3. Do NOT repeat the same action_type — you will be penalised each time.
4. Call finalize_task only when your required actions are done.
5. Fewer steps = higher efficiency score.
"""

# ── Rule-based scripts ────────────────────────────────────────────────────────
RULE_SCRIPTS: dict[str, list[dict]] = {
    "task_001": [
        {"action_type": "list_candidates", "parameters": {}},
        {"action_type": "list_jobs",       "parameters": {"disability_friendly_only": True}},
        {"action_type": "get_candidate",   "parameters": {"candidate_id": "cand_003"}},
        {"action_type": "get_job",         "parameters": {"job_id": "job_002"}},
        {"action_type": "match_candidate_to_job",
         "parameters": {"candidate_id": "cand_003", "job_id": "job_002"}},
        {"action_type": "finalize_task",   "parameters": {}},
    ],
    "task_002": [
        {"action_type": "get_candidate",
         "parameters": {"candidate_id": "cand_005"}},
        {"action_type": "check_scheme_eligibility",
         "parameters": {"candidate_id": "cand_005"}},
        {"action_type": "generate_scheme_checklist",
         "parameters": {"candidate_id": "cand_005"}},
        {"action_type": "finalize_task",   "parameters": {}},
    ],
    "task_003": [
        {"action_type": "get_candidate",
         "parameters": {"candidate_id": "cand_011"}},
        {"action_type": "get_job",
         "parameters": {"job_id": "job_002"}},
        {"action_type": "resolve_placement_conflict",
         "parameters": {
             "candidate_id":  "cand_011",
             "conflict_type": "ngo_employer_mismatch",
             "resolution":    "schedule_fresh_interview",
         }},
        {"action_type": "schedule_interview",
         "parameters": {
             "candidate_id": "cand_011",
             "employer_id":  "emp_001",
             "date":         "2026-04-20",
         }},
        {"action_type": "publish_dashboard_update",
         "parameters": {"ward": "Malleshwaram", "update_type": "placement_status"}},
        {"action_type": "finalize_task",   "parameters": {}},
    ],
}


# ── LLM call ─────────────────────────────────────────────────────────────────

def _call_llm(client: Any, messages: list[dict], verbose: bool = False) -> dict | None:
    """Query LLM. Returns parsed action dict, or None on any failure."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=300,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()

        if verbose:
            logger.info(f"    [LLM RAW] {raw[:300]}")

        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) >= 2 else raw
            if raw.startswith("json"):
                raw = raw[4:]

        parsed = json.loads(raw.strip())

        if "action_type" not in parsed:
            logger.warning("    [LLM] Missing 'action_type' key")
            return None
        if "parameters" not in parsed:
            parsed["parameters"] = {}

        return parsed

    except json.JSONDecodeError as exc:
        logger.warning(f"    [LLM] JSON parse error: {exc}")
        return None
    except Exception as exc:
        logger.warning(f"    [LLM] API error: {exc}")
        return None


# ── Episode runner ────────────────────────────────────────────────────────────

def run_task(
    task_id: str,
    mode: str,
    client: Any,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Run one task episode.

    mode = 'rules'  → rule-based script, no LLM calls
    mode = 'llm'    → LLM drives every step; finalize_task safety fallback only
                       after 2 consecutive parse failures
    mode = 'hybrid' → LLM primary, rule script step on LLM failure only
    """
    logger.info("\n" + "=" * 64)
    logger.info(f"  TASK: {task_id.upper()}   MODE: {mode.upper()}")
    logger.info("=" * 64)

    env = SamaarthyaSetuEnvironment(task_id=task_id)
    obs = env.reset()
    max_steps = MAX_STEPS.get(task_id, 30)

    logger.info(f"  {obs.message[:140]}")
    logger.info(f"  Required actions: {obs.data.get('required_actions', [])}")
    logger.info(f"  Step budget: {max_steps}\n")

    history: list[dict] = [
        {
            "role": "user",
            "content": (
                f"TASK: {obs.message}\n\n"
                f"Required actions: {obs.data.get('required_actions', [])}\n"
                f"Step budget: {max_steps}\n\n"
                "Solve the task now."
            ),
        }
    ]

    rule_script = RULE_SCRIPTS.get(task_id, [])
    rule_idx    = 0
    step_count  = 0
    total_reward = 0.0
    final_score  = 0.0
    done         = False
    llm_steps    = 0
    rule_steps   = 0
    steps_log: list[dict] = []

    while not done and step_count < max_steps:
        step_count += 1
        action_dict: dict | None = None
        source = "unknown"

        if mode == "rules":
            if rule_idx < len(rule_script):
                action_dict = rule_script[rule_idx]
                rule_idx += 1
            else:
                action_dict = {"action_type": "finalize_task", "parameters": {}}
            source = "rules"
            rule_steps += 1

        elif mode == "llm":
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + history
            for attempt in range(2):
                action_dict = _call_llm(client, msgs, verbose=verbose)
                if action_dict is not None:
                    break
                logger.warning(f"    [LLM] Attempt {attempt + 1}/2 failed, retrying...")
            if action_dict is not None:
                source = "llm"
                llm_steps += 1
            else:
                logger.warning("    [LLM] All retries exhausted — safety finalize")
                action_dict = {"action_type": "finalize_task", "parameters": {}}
                source = "llm-safety-fallback"

        elif mode == "hybrid":
            if client is not None:
                msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + history
                action_dict = _call_llm(client, msgs, verbose=verbose)
                if action_dict is not None:
                    source = "llm"
                    llm_steps += 1
            if action_dict is None:
                if rule_idx < len(rule_script):
                    action_dict = rule_script[rule_idx]
                    rule_idx += 1
                else:
                    action_dict = {"action_type": "finalize_task", "parameters": {}}
                source = "rules-fallback"
                rule_steps += 1

        logger.info(
            f"  Step {step_count:02d} [{source:<18}] "
            f"{action_dict['action_type']}  "
            f"{action_dict['parameters'] or ''}"
        )

        try:
            action = Action(**action_dict)
            obs, reward, done, info = env.step(action)
            total_reward += reward

            logger.info(
                f"           reward={reward:+.4f}  "
                f"progress={obs.progress:.0%}  done={done}"
            )
            if obs.message:
                logger.info(f"           env: {obs.message[:100]}")
            if info.get("partial_signals_awarded"):
                logger.info(f"           signals: {info['partial_signals_awarded']}")
            if info.get("penalty"):
                logger.warning(f"           PENALTY: {info['penalty']}")
            if "final_score" in info:
                final_score = info["final_score"]
                logger.info(f"           FINAL SCORE: {final_score:.4f}")
                if verbose and info.get("grade_details"):
                    logger.info(f"           grade: {info['grade_details']}")

            steps_log.append({
                "step":       step_count,
                "source":     source,
                "action":     action_dict["action_type"],
                "parameters": action_dict["parameters"],
                "reward":     round(reward, 4),
                "progress":   round(obs.progress, 4),
                "done":       done,
            })

            # Update conversation history
            data_preview = json.dumps(obs.data, default=str)[:500]
            history.append({"role": "assistant", "content": json.dumps(action_dict)})
            history.append({
                "role": "user",
                "content": (
                    f"Step {step_count} result:\n"
                    f"  message:  {obs.message}\n"
                    f"  progress: {obs.progress:.0%}\n"
                    f"  reward:   {reward:+.4f}\n"
                    f"  done:     {done}\n"
                    f"  data:     {data_preview}"
                ),
            })

        except Exception as exc:
            logger.error(f"  Step {step_count} ERROR: {exc}")
            steps_log.append({
                "step":   step_count,
                "source": source,
                "action": action_dict.get("action_type"),
                "error":  str(exc),
            })

    final_state = env.state()
    logger.info(f"\n  {'─' * 60}")
    logger.info(f"  Done | steps={step_count}/{max_steps} | "
                f"llm={llm_steps} rules={rule_steps} | "
                f"score={final_score:.4f} | complete={final_state.completed}")

    return {
        "task_id":      task_id,
        "mode":         mode,
        "steps_taken":  step_count,
        "llm_steps":    llm_steps,
        "rule_steps":   rule_steps,
        "total_reward": round(total_reward, 4),
        "final_score":  round(final_score, 4),
        "progress":     round(final_state.progress_score, 4),
        "completed":    final_state.completed,
        "steps":        steps_log,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description="SamaarthyaSetu inference — rules / llm / hybrid",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--mode",    choices=["rules", "llm", "hybrid"], default="rules")
    p.add_argument("--task",    choices=ALL_TASKS + ["all"], default="all")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--demo",    action="store_true",
                   help="Demo mode: run task_001 with verbose output")
    args = p.parse_args()

    tasks_to_run = ALL_TASKS if args.task == "all" else [args.task]
    if args.demo:
        args.verbose = True
        if args.task == "all":
            tasks_to_run = ["task_001"]

    start = time.time()
    logger.info("SamaarthyaSetu v3 — Inference Runner")
    logger.info(f"Mode:  {args.mode.upper()}")
    logger.info(f"Tasks: {tasks_to_run}")
    logger.info(f"Model: {MODEL_NAME}  |  API: {API_BASE_URL}")
    logger.info(f"Token: {'SET' if HF_TOKEN else 'NOT SET'}\n")

    client = None
    if args.mode in ("llm", "hybrid"):
        if not HF_TOKEN:
            logger.error("LLM/hybrid mode requires HF_TOKEN env var. Use --mode rules otherwise.")
            return 1
        try:
            from openai import OpenAI
            client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
            logger.info("✅ LLM client initialised\n")
        except Exception as exc:
            logger.error(f"Cannot init LLM client: {exc}")
            return 1

    results: list[dict] = []
    for tid in tasks_to_run:
        results.append(run_task(tid, args.mode, client, verbose=args.verbose))

    elapsed = time.time() - start
    avg = sum(r["final_score"] for r in results) / len(results)

    print("\n" + "=" * 72)
    print("  SAMAARTHYASETU — RESULTS")
    print("=" * 72)
    print(f"  Mode: {args.mode.upper():<10}  Model: {MODEL_NAME}")
    print(f"  {'Task':<12} {'Steps':>5} {'LLM':>4} {'Rule':>5} {'Score':>8}  {'Done':>5}")
    print("  " + "-" * 60)
    for r in results:
        print(
            f"  {r['task_id']:<12} {r['steps_taken']:>5} "
            f"{r['llm_steps']:>4} {r['rule_steps']:>5} "
            f"{r['final_score']:>8.4f}  {str(r['completed']):>5}"
        )
    print("  " + "-" * 60)
    print(f"  {'AVERAGE':<12} {'':>5} {'':>4} {'':>5} {avg:>8.4f}")
    print(f"\n  Runtime: {elapsed:.1f}s")
    print("=" * 72)

    out = {
        "mode":            args.mode,
        "model":           MODEL_NAME,
        "api_base":        API_BASE_URL,
        "tasks_run":       tasks_to_run,
        "runtime_seconds": round(elapsed, 2),
        "average_score":   round(avg, 4),
        "task_results":    results,
    }
    with open("inference_results.json", "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Results saved → inference_results.json")

    return 0 if avg >= 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())
