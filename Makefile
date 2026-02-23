VENV ?= .venv
PYTHON_BIN ?= $(shell command -v python3.12 >/dev/null 2>&1 && echo python3.12 || echo python3)
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
RUFF := $(VENV)/bin/ruff
BLACK := $(VENV)/bin/black
MYPY := $(VENV)/bin/mypy
PYTEST := $(VENV)/bin/pytest
PIP_AUDIT := $(VENV)/bin/pip-audit
BANDIT := $(VENV)/bin/bandit

.PHONY: venv install lint format-check format typecheck test security ci

venv:
	@test -x $(PY) || $(PYTHON_BIN) -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

lint: install
	$(RUFF) check src tests
	$(BLACK) --check src tests

format-check: install
	$(BLACK) --check src tests

format: install
	$(BLACK) src tests
	$(RUFF) check --fix src tests

typecheck: install
	$(MYPY) src/monomarket

test: install
	$(PYTEST)

security: install
	$(PIP_AUDIT)
	$(BANDIT) -q -r src/monomarket

ci: lint typecheck test security
