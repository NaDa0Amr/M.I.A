.PHONY: help setup setup-dev prepare-data extract-features train evaluate test lint format app api docker-build docker-run clean

help:
	@echo "Available commands:"
	@echo "  setup            - Install dependencies and download NLTK data"
	@echo "  setup-dev        - Install development dependencies"
	@echo "  prepare-data     - Download and prepare dataset"
	@echo "  extract-features - Precompute image feature embeddings"
	@echo "  train            - Train the image captioning model"
	@echo "  evaluate         - Evaluate the model on test set"
	@echo "  test             - Run unit and integration tests with coverage"
	@echo "  lint             - Check formatting and linting"
	@echo "  format           - Format code with black and isort"
	@echo "  app              - Launch Streamlit web application"
	@echo "  api              - Launch FastAPI REST API server"
	@echo "  docker-build     - Build Docker image"
	@echo "  docker-run       - Run all services via docker-compose"
	@echo "  clean            - Remove cached files, coverage reports, and bytecode"

setup:
	pip install -r requirements.txt && python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt_tab')"

setup-dev:
	pip install -r requirements-dev.txt

prepare-data:
	python setup_data.py

extract-features:
	python scripts/extract_features.py

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	black --check src/ tests/ app/ api/ scripts/ && isort --check-only src/ tests/ app/ api/ scripts/ && flake8 src/ tests/ app/ api/ scripts/

format:
	black src/ tests/ app/ api/ scripts/ && isort src/ tests/ app/ api/ scripts/

app:
	streamlit run app/app.py

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t image-captioning .

docker-run:
	docker-compose up

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').glob('.pytest_cache')]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').glob('htmlcov')]; [p.unlink() for p in pathlib.Path('.').glob('.coverage*') if p.is_file()]"
