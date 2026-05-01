.PHONY: install test lint format download train train-disabled sweep docker-build docker-train clean

PYTHON ?= python

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

download:
	$(PYTHON) -m scripts.download_data

train:
	$(PYTHON) -m src.training.train

train-disabled:
	$(PYTHON) -m src.training.train wandb.mode=disabled

sweep:
	$(PYTHON) -m scripts.run_sweep --count 20

docker-build:
	docker compose -f docker/docker-compose.yml build

docker-train:
	docker compose -f docker/docker-compose.yml up train

clean:
	rm -rf outputs/ multirun/ wandb/ .hydra/ .pytest_cache/ .ruff_cache/
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
