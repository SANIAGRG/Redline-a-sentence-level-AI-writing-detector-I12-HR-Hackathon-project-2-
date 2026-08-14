.PHONY: setup lint test join sample generate manifests stylometric stylometric-all likelihood features train eval analysis app clean

PYTHON := .venv/Scripts/python.exe
PIP := .venv/Scripts/pip.exe

setup:
	python -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e .[dev]
	$(PYTHON) -m spacy download en_core_web_sm
	$(PYTHON) -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m mypy src

test:
	$(PYTHON) -m pytest -q

join:
	$(PYTHON) -m detector.ingest.join_ell

sample:
	$(PYTHON) -m detector.ingest.build_corpus

generate:
	$(PYTHON) -m detector.generate.run_polish
	$(PYTHON) -m detector.generate.run_modern
	$(PYTHON) -m detector.generate.run_adversarial

manifests:
	$(PYTHON) -m detector.generate.build_manifests

stylometric:
	$(PYTHON) -m detector.features.build_baseline

stylometric-all:
	$(PYTHON) -m detector.features.build_stylometric_all

likelihood:
	$(PYTHON) -m detector.features.build_likelihood

# Deadline-scoped Module 4 pipeline (see ADR 0008): stylometric-all and
# likelihood must both run before train/eval -- they build the same
# deadline-scoped document set (build_likelihood.py's POOL_SIZES).
features: stylometric-all likelihood

train:
	$(PYTHON) -m detector.model.train

eval:
	$(PYTHON) -m detector.model.evaluate

analysis:
	$(PYTHON) -m detector.model.analysis

app:
	$(PYTHON) -m detector.app.main

clean:
	rm -rf dataset/interim/* dataset/generated/* dataset/features/*
