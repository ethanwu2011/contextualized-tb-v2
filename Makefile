.PHONY: all setup data endpoint-build build train figures public-smoke public-release-check test lint format type clean help
UV := uv run --locked

help:
	@echo "Targets:"
	@echo "  setup    Create venv and install (editable, dev extras)"
	@echo "  all      Build the endpoint-explicit gated-data artifacts"
	@echo "  endpoint-build  Verify raw data, run ETL, and build the PATCH matrix"
	@echo "  train    Run the manuscript model-ladder analysis after endpoint-build"
	@echo "  figures  Regenerate the main contextualized figures after endpoint-build"
	@echo "  public-smoke  Run the synthetic-only endpoint and model smoke test"
	@echo "  public-release-check  Stage and scan a source-only public tree"
	@echo "  test     Run pytest with coverage"
	@echo "  lint     ruff check + format --check"
	@echo "  type     mypy --strict on src"
	@echo "  clean    Remove generated outputs (keeps raw data)"

setup:
	uv sync --locked --python 3.11 --extra dev

data:
	$(UV) python scripts/00_download_data.py

endpoint-build:
	bash scripts/run_all.sh

build: endpoint-build

train: endpoint-build
	$(UV) python scripts/15_model_ladder_unfavorable.py

figures: endpoint-build
	$(UV) python scripts/06_contextualized_figures.py

public-smoke:
	$(UV) python scripts/reproduce_synthetic.py --n 300 --epochs 50

public-release-check:
	public_dir=$$(mktemp -d); trap 'rm -rf "$$public_dir"' EXIT; \
	  bash scripts/sync_to_public.sh "$$public_dir" && \
	  bash scripts/scan_public.sh "$$public_dir"

all: endpoint-build

test:
	$(UV) pytest --cov=ctb --cov-report=term-missing -q

lint:
	$(UV) ruff check src tests
	$(UV) ruff format --check src tests
	$(UV) python -m compileall -q src scripts figures

format:
	$(UV) ruff check --fix src tests
	$(UV) ruff format src tests

type:
	$(UV) mypy

clean:
	rm -rf results/figures/* results/metrics/* results/tables/* \
	       data/interim/* data/processed/* logs/*
