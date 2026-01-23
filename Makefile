.PHONY: setup lint format test

setup:
	uv venv
	. .venv/bin/activate && uv pip install -e ".[dev]"

lint:
	. .venv/bin/activate && ruff check .

format:
	. .venv/bin/activate && ruff format .

test:
	. .venv/bin/activate && pytest