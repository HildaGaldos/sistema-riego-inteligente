.PHONY: install download-data eda train-fast train-full reports test lint up down bootstrap

install:
	python -m pip install -r backend/requirements.txt
	python -m pip install -r ml/requirements.txt
	cd frontend && npm install

download-data:
	python scripts/download_data.py

eda:
	python scripts/train.py --eda-only

train-fast:
	python scripts/train.py --fast

train-full:
	python scripts/train.py --full

reports:
	python scripts/generate_reports.py

test:
	pytest -q

lint:
	ruff check backend ml scripts
	black --check backend ml scripts

up:
	docker compose up --build

down:
	docker compose down

bootstrap:
	python scripts/bootstrap.py
