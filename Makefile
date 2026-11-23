PYTHON := python
PIP := pip

.PHONY: install setup test test-p0 test-p1 test-p2 test-p3 test-p4 test-p5 eval demo voice phase3 clean

install:
	$(PIP) install -r requirements.txt

setup:
	$(PYTHON) scripts/setup_models.py

# Run all tests
test:
	$(PYTHON) -m pytest tests/ -v

# Phase-specific test targets
test-p0:
	$(PYTHON) -m pytest tests/test_phase0_scaffold.py -v

test-p1:
	$(PYTHON) -m pytest tests/test_phase1_text_core.py -v

test-p2:
	$(PYTHON) -m pytest tests/test_phase2_voice.py -v

test-p3:
	$(PYTHON) -m pytest tests/test_phase3_turn_latency.py -v

test-p4:
	$(PYTHON) -m pytest tests/test_phase4_safety.py -v

test-p5:
	$(PYTHON) -m pytest tests/test_phase5_evals.py -v

test-p6:
	$(PYTHON) -m pytest tests/test_phase6_observability.py -v

# Phase 5: Run full evaluation suite and generate reports
eval:
	$(PYTHON) -m evals.runner

# Phase 3: run the benchmark + latency profile and print tables
phase3:
	$(PYTHON) -c "from app.audio.phase3_tuning import run_detector_benchmark, print_comparison_table, TTFAProfiler, PromptCachingAdvisor; import json; st, sm = run_detector_benchmark(); print_comparison_table(st, sm); p = TTFAProfiler(); [p.profile_turn(q, dead_air_ms=da, turn_detector_name=d) for q, da, d in [('What is the Python course fee?', 600.0, 'silence_timer'), ('Can you tell me about Data Science?', 350.0, 'semantic'), ('What is the address?', 600.0, 'silence_timer'), ('I want to book a demo session.', 280.0, 'semantic')]]; p.print_report(); advisor = PromptCachingAdvisor(); print(json.dumps(advisor.report(), indent=2))"

# Text-only demo (offline mode — no network, no Groq quota used)
demo:
	$(PYTHON) run.py "What are the courses offered by AI Adventures and what is the fee for Python?"

# Browser voice demo (needs Groq API key in .env)
voice:
	$(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Semantic turn detector enabled voice mode
voice-semantic:
	USE_SEMANTIC_TURN_DETECTOR=true $(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

clean:
	rmdir /s /q __pycache__ .pytest_cache 2>nul || true
