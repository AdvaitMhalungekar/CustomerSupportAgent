PYTHON := .\venv\Scripts\python.exe
PIP := .\venv\Scripts\pip.exe

.PHONY: install setup test demo voice clean

install:
	$(PIP) install -r requirements.txt

setup:
	$(PYTHON) scripts/setup_models.py

test:
	$(PYTHON) -m pytest tests/ -v

demo:
	$(PYTHON) run.py "What are the courses offered by AI Adventures and what is the fee for Python?"

voice:
	$(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

clean:
	rmdir /s /q __pycache__ .pytest_cache
