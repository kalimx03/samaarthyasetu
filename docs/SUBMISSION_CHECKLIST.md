# SamaarthyaSetu — Submission Checklist

Run through this before every submission attempt. Every ✅ must be confirmed.

---

## 1. Environment Interface

```bash
python -c "
from samaarthya_ops_env import SamaarthyaSetuEnvironment, Action
for tid in ['task_001', 'task_002', 'task_003']:
    env = SamaarthyaSetuEnvironment(tid)
    obs = env.reset()
    assert obs.progress == 0.0
    _, r, _, _ = env.step(Action(action_type='list_candidates', parameters={}))
    s = env.state()
    assert s.steps_taken == 1
    print(f'✅ {tid}: reset/step/state OK  reward={r:.4f}')
"
```

- [ ] `reset()` returns Observation with `progress=0.0`
- [ ] `step()` returns (Observation, float, bool, dict)
- [ ] `state()` returns typed State snapshot
- [ ] All three tasks initialise without error

---

## 2. Baseline Scores (rule-based, no API key)

```bash
python inference.py --mode rules
```

Expected output:
```
task_001   6 steps   score ≥ 0.90   Done=True
task_002   4 steps   score ≥ 0.90   Done=True
task_003   6 steps   score ≥ 0.90   Done=True
AVERAGE              score ≥ 0.90
```

- [ ] task_001 final_score ≥ 0.90
- [ ] task_002 final_score ≥ 0.90
- [ ] task_003 final_score ≥ 0.90
- [ ] Average ≥ 0.75 (competition disqualification threshold)
- [ ] All three tasks `completed = True`
- [ ] Runtime < 60 seconds

---

## 3. LLM-Only Mode (requires API key)

```bash
export HF_TOKEN="your-key"
python inference.py --mode llm --task task_001 --demo
```

- [ ] Every step shows `[llm]` source label (not `[rules]`)
- [ ] No `[rules-fallback]` label appears (LLM is solving cleanly)
- [ ] task_001 final_score ≥ 0.70
- [ ] Task completes (`done=True`)

---

## 4. Test Suite

```bash
python -m pytest tests/ -v --tb=short
```

- [ ] All tests pass
- [ ] No import errors
- [ ] `test_reward_shaping.py` — all step efficiency + partial signal tests pass
- [ ] `test_environment.py` — all reset/step/state/action tests pass
- [ ] `test_tasks.py` — all task definition + integration tests pass
- [ ] `test_graders_and_api.py` — grader unit tests + API smoke tests pass

---

## 5. FastAPI Server

```bash
# Terminal 1
make run-api

# Terminal 2
curl http://localhost:7860/health
curl http://localhost:7860/tasks
curl http://localhost:7860/candidates
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id": "task_001"}'
```

- [ ] `/health` returns `{"status": "ok"}`
- [ ] `/tasks` returns all 3 task definitions
- [ ] `/candidates` returns 15 candidates
- [ ] `/reset` returns Observation with `progress=0.0`
- [ ] `/docs` interactive Swagger UI loads

---

## 6. Docker Build

```bash
docker build -t samaarthyasetu:latest .
docker run -d --name ss_test -p 7860:7860 samaarthyasetu:latest
sleep 8
curl -f http://localhost:7860/health
docker stop ss_test && docker rm ss_test
```

- [ ] Docker image builds without error
- [ ] Container starts and exposes port 7860
- [ ] Health check passes from host
- [ ] Container stops cleanly

---

## 7. File Structure

Required files for submission:

- [ ] `inference.py` — with `--mode rules/llm/hybrid` flags
- [ ] `openenv.yaml` — version 2.0.0
- [ ] `Dockerfile` — port 7860
- [ ] `README.md` — no false claims
- [ ] `requirements.txt` — pinned versions
- [ ] `samaarthya_ops_env/environment.py` — reset/step/state
- [ ] `samaarthya_ops_env/models.py` — Pydantic v2 Action/Observation/State
- [ ] `samaarthya_ops_env/graders/task_graders.py` — deterministic
- [ ] `samaarthya_ops_env/data/seed_data.py` — job_002 is React/JS role
- [ ] `server/app.py` — FastAPI
- [ ] `ml/model.py` — PyTorch Siamese network
- [ ] `tests/` — all 4 test modules
- [ ] `docs/REWARD_DESIGN.md`
- [ ] `docs/JUDGE_QA.md`
- [ ] `docs/DEMO_SCRIPT.md`
- [ ] `.github/workflows/ci.yml`

---

## 8. Claims Verification

Read through README.md and confirm every claim is backed by code:

- [ ] "15 candidates" → count rows in `seed_data.py` CANDIDATES list
- [ ] "10 jobs" → count rows in JOBS list
- [ ] "44 tests passing" → `pytest tests/ -v | grep PASSED | wc -l`
- [ ] "< 1 second runtime" → timed output from `python inference.py --mode rules`
- [ ] "LLM can solve task independently" → `python inference.py --mode llm --task task_001`
- [ ] No mention of `apps/web`, Next.js frontend, or features not in the repo

---

## 9. HuggingFace Spaces Readiness

- [ ] Dockerfile exposes port 7860
- [ ] `CMD` starts uvicorn on `0.0.0.0:7860`
- [ ] Secrets documented: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
- [ ] `openenv.yaml` has `hf_space_compatible: true`
- [ ] `requirements.txt` has no local-path dependencies

---

## 10. Final Zip

```bash
make clean && make zip
unzip -l samaarthyasetu_submission.zip | grep -c "\.py"
```

- [ ] Zip contains all required files
- [ ] No `__pycache__`, `.pyc`, `.pt`, `inference_results.json` in zip
- [ ] Zip is under 50 MB
- [ ] Submitter name and repo name match registration

---

## Sign-off

| Check | Person | Date |
|-------|--------|------|
| Environment interface | | |
| Baseline scores ≥ 0.75 | | |
| LLM-only mode runs | | |
| All tests pass | | |
| Docker builds | | |
| README claims verified | | |
