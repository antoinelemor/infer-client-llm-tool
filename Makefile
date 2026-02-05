.PHONY: install install-dev install-pandas run test clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-pandas:
	pip install -e ".[pandas]"

run:
	infer

test:
	pytest tests/

clean:
	rm -rf build/ dist/ *.egg-info infer_client.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
