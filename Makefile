.PHONY: setup lint test join generate features train eval app clean

PYTHON := .venv/Scripts/python.exe
PIP := .venv/Scripts/pip.exe

setup:
	python -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e .[dev]
	$(PYTHON) -m spacy download en_core_web_sm

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m mypy src

test:
	$(PYTHON) -m pytest -q

join:
	$(PYTHON) -m detector.ingest.join_ell

generate:
	$(PYTHON) -m detector.generate.run_polish
	$(PYTHON) -m detector.generate.run_modern

features:
	$(PYTHON) -m detector.features.build_all

train:
	$(PYTHON) -m detector.model.train

eval:
	$(PYTHON) -m detector.model.evaluate

app:
	$(PYTHON) -m detector.app.main

clean:
	rm -rf dataset/interim/* dataset/generated/* dataset/features/*
