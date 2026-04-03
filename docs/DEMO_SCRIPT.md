# SamaarthyaSetu — Demo Scripts

Two scripts: a 90-second version for a lightning pitch, and a 3-minute version for a technical panel.

---

## 90-Second Demo Script

**Setup (before you start):** Open two terminal windows. Window 1 has the API server running. Window 2 is ready to run inference.

---

*[Open the preview.html landing page]*

> "This is SamaarthyaSetu — Sanskrit for 'Bridge of Enablement'. India has 40 million persons with disabilities. Fewer than 15% are formally employed. NGOs train them — then they disappear into a black hole. No job matching. No scheme navigation. No accountability.

> We built an OpenEnv reinforcement learning environment that forces an AI agent to navigate the exact system these candidates face — 15 real candidates, 10 real employers, 10 government schemes, 10 Bangalore wards.

*[Switch to terminal]*

> "Here's what makes this interesting — watch the LLM solve an employment task in real time, completely independently."

```bash
python inference.py --mode llm --task task_001 --demo
```

*[While it runs, narrate:]*

> "The agent gets the task: find the best match for Arjun Nair, 22, locomotor disability, React developer. It calls `list_candidates`, `list_jobs`, then computes a composite match score — skill overlap, accommodation fit, language compatibility. It finds job_002: Infosys BPM Junior Frontend Developer. Score 0.95. It finalises. Score: 0.98 out of 1.0."

*[Point to the step log showing `[llm]` labels on each line]*

> "Every single step — the LLM. No rule-based fallback. This is a real agent navigating a real employment problem."

*[Show the landing page metrics]*

> "Three tasks: job matching, scheme navigation, conflict resolution. Average score 0.98. Under 1 second runtime. 44 tests passing. MIT licensed."

---

## 3-Minute Technical Demo Script

**Setup:** API server running (`make run-api`), browser open to `http://localhost:7860/docs`.

---

### Minute 1 — Problem and Environment

> "Let me start with the problem we're solving. In Bangalore, NGOs like Samarthanam and Enable India train hundreds of differently-abled youth every year. After training ends, there is no system connecting them to employers. No one navigates them through 21 disability certificate types or 15 government schemes. NGOs mark candidates as 'placed' before offers confirm. The government tracks training completions, not actual employment.

> SamaarthyaSetu is an OpenEnv environment that simulates this entire pipeline. An AI agent must navigate it — making real decisions about real people in real Bangalore contexts."

*[Show `openenv.yaml`]*

> "Our environment follows the OpenEnv spec: `reset()`, `step()`, `state()`. Three tasks of increasing difficulty — verified job match, scheme navigation, and placement conflict resolution. The reward function is dense, not sparse — partial signals fire throughout the episode so an RL agent gets learning signal at every step, not just at the end."

---

### Minute 2 — LLM Agent in Action

*[Switch to terminal]*

```bash
python inference.py --mode llm --task task_002 --verbose
```

> "This is task_002: scheme navigator. The agent must find candidate cand_005 — Suresh Patil, speech disability, Karnataka resident — check eligibility across 10 government schemes, and generate a prioritised document checklist. Watch how the LLM reasons through this."

*[Narrate as output appears:]*

> "Step 1 — LLM calls `get_candidate` for cand_005. The environment returns his disability type, location, income bracket. Step 2 — `check_scheme_eligibility`. The environment evaluates all 10 schemes: Divyangjan Kaushal Vikas Yojana — eligible. Vishwas Karnataka — eligible. Section 80U — eligible. Step 3 — `generate_scheme_checklist`. The environment returns a prioritised checklist with the UDID card, Aadhaar, income certificate, and DDRC address. Step 4 — `finalize_task`. Score: 0.97."

*[Point to the `[llm]` label on every line]*

> "Every step is LLM-driven. We also have a rules-only mode for evaluation reproducibility, and a hybrid mode for production use."

---

### Minute 3 — Architecture and Differentiation

*[Show FastAPI docs at localhost:7860/docs]*

> "The environment is exposed as a REST API — `POST /reset`, `POST /step`, `GET /state/{task_id}` — so Meta's evaluation harness can drive it directly over HTTP. There are also data endpoints: `/candidates`, `/jobs`, `/match/{candidate_id}`, `/schemes/eligibility/{candidate_id}`, and a ward-level `/dashboard`."

*[Run the test suite]*

```bash
python -m pytest tests/ -v --tb=short
```

> "44 tests, zero external dependencies, all offline. The reward shaping tests alone are 501 lines — covering step efficiency monotonicity, signal-once behaviour, penalty magnitudes, and full episode reward computation."

*[Return to landing page]*

> "What makes this finalist-tier: the domain. This is not another traffic light environment. This is a real, named, addressable problem in India right now — and we've modelled it with real NGO names, real employer names, real government schemes, and a real ward-level data structure that a Bangalore employment commissioner could actually use."

> "The environment is MIT licensed, HuggingFace Spaces ready, Docker verified. Thank you."

---

## Live Q&A Cheat Sheet

| Question | One-sentence answer |
|----------|-------------------|
| "Can the LLM solve it without rules?" | "Yes — run `python inference.py --mode llm --task task_001 --demo` and watch the `[llm]` labels on every step." |
| "How is reward designed?" | "Dense cosine-decay efficiency plus 12 partial-progress signals, each firing at most once, documented in `docs/REWARD_DESIGN.md`." |
| "Why this domain?" | "85% fall-off between NGO training and verified employment in Bangalore. No platform addresses this today." |
| "What's the match score formula?" | "60% skill Jaccard + 25% accommodation coverage + 15% language overlap — all deterministic, all testable." |
| "Is the data real?" | "All employer names, NGO names, ward names, and scheme names are real Bangalore entities." |
| "How does it scale?" | "Each env instance is a pure Python object — sub-1-second per episode, parallelisable via Ray." |
