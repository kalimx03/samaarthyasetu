.PHONY: install test test-env test-tasks test-graders test-api test-reward \
        run-api run-inference run-llm run-demo validate \
        docker-build docker-run clean zip

install:
	pip install -r requirements.txt

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	python -m pytest tests/ -v --tb=short

test-env:
	python -m pytest tests/test_environment.py -v

test-tasks:
	python -m pytest tests/test_tasks.py -v

test-graders:
	python -m pytest tests/test_graders_and_api.py::TestGraders -v

test-api:
	python -m pytest tests/test_graders_and_api.py::TestAPISmoke -v

test-reward:
	python -m pytest tests/test_reward_shaping.py -v

# ── Inference modes ────────────────────────────────────────────────────────────
run-inference:
	python inference.py --mode rules

run-llm:
	python inference.py --mode llm

run-hybrid:
	python inference.py --mode hybrid

run-demo:
	python inference.py --mode llm --task task_001 --demo

run-demo-rules:
	python inference.py --mode rules --task task_001 --verbose

# ── API server ─────────────────────────────────────────────────────────────────
run-api:
	uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# ── Model training ─────────────────────────────────────────────────────────────
train-model:
	python ml/model.py

# ── Pre-submission validation ──────────────────────────────────────────────────
validate:
	@echo "=== SamaarthyaSetu Pre-Submission Validation ==="
	@python -c "\
from samaarthya_ops_env import SamaarthyaSetuEnvironment, Action; \
env = SamaarthyaSetuEnvironment('task_001'); obs = env.reset(); \
print('✅ reset() OK  progress=0.0'); \
_, r, _, _ = env.step(Action(action_type='list_candidates', parameters={})); \
print(f'✅ step()  OK  reward={r}'); \
s = env.state(); print(f'✅ state() OK  steps={s.steps_taken}'); \
"
	@python inference.py --mode rules --task task_001
	@echo "=== ✅ Validation passed ==="

# ── Docker ─────────────────────────────────────────────────────────────────────
docker-build:
	docker build -t samaarthyasetu:latest .

docker-run:
	docker run -p 7860:7860 \
		-e API_BASE_URL=$(API_BASE_URL) \
		-e MODEL_NAME=$(MODEL_NAME) \
		-e HF_TOKEN=$(HF_TOKEN) \
		samaarthyasetu:latest

# ── Packaging ──────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -f inference_results.json
	rm -rf .pytest_cache

zip:
	zip -r samaarthyasetu_submission.zip . \
		--exclude "*.pyc" --exclude "*__pycache__*" \
		--exclude ".git/*" --exclude "*.pt" \
		--exclude "inference_results.json" --exclude "*.zip"
	@echo "✅ samaarthyasetu_submission.zip created"
