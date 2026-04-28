## AnimeRecommendationSystem
Anime Recommendation System is an AI-powered recommendation platform that combines collaborative filtering, content-based signals, hybrid ranking, Redis caching, FAISS/vector search, FastAPI, Streamlit, Docker, Kubernetes, and Prometheus-based observability. The goal is to provide fast, explainable anime recommendations with a production-style MLOps workflow.

## Architecture Summary
The system follows an end-to-end flow: data ingestion and preprocessing, model training, artifact generation, API serving, Redis caching, UI rendering, and observability through Prometheus and logs. The backend exposes authentication, recommendation, health, metrics, and admin cache endpoints, while the Streamlit frontend handles user interaction and display.

## Key Features
- Hybrid recommendation logic with collaborative and content-based ranking.
- Redis caching for fast repeat requests and cache invalidation.
- FAISS/vector-search-style similarity workflow and ranking support.
- JWT authentication for protected endpoints.
- Prometheus metrics for request latency, cache behavior, pipeline performance, and readiness.
- Docker and Kubernetes deployment manifests for backend, UI, Redis, and observability stack.

## Technology Stack

* Python 3.8+
* FastAPI and Uvicorn
* Streamlit
* Redis
* FAISS CPU
* TensorFlow / Keras
* Pandas, NumPy, scikit-learn, Joblib
* Prometheus client and prometheus-fastapi-instrumentator
* Docker and Kubernetes

Folder Structure
api/ backend service, auth, cache, metrics, schemas, and server.

config/ app settings and path configuration.

core/ exceptions and shared core helpers.

pipeline/ hybrid recommendation inference pipeline.

src/ model/data processing and logging utilities.

ui/ Streamlit application pages and UI helpers.

utils/ shared helper functions.

artifacts/ processed data, weights, and model files.

*.yaml Kubernetes manifests for backend, UI, Redis, Prometheus, and Grafana datasource.

## Prerequisites
- Python 3.8 or later
- Redis available locally or in Kubernetes
- Access to the processed training artifacts and model files
- Docker and Kubernetes tooling for container deployment
- A valid internet connection for Jikan poster lookups if you use the UI.

## Environment Setup
- Clone the repository
- Create and activate a Python virtual environment
- Install dependencies from requirements.txt
- Ensure the artifacts/ directory contains the trained model and processed data

## Python virtual environment setup
**Linux/Unix:**
python -m venv .venv
source .venv/bin/activate
**On Windows:**
python -m venv .venv
.venv\Scripts\activate

**Required environment variables**
APIHOST — backend host name or service name.
APIPORT — backend port, usually 8000.
REDISHOST — Redis host, usually redis in Kubernetes.
REDISPORT — Redis port, usually 6379.
STREAMLITHOST — Streamlit bind address, usually 0.0.0.0.
STREAMLITPORT — Streamlit port, usually 8501.
LOGINUSER and LOGINPASSWORD — demo login values for the UI.
ENVIRONMENT — development, staging, or production.

## Dependency installation
pip install --upgrade pip
pip install -r requirements.txt

## Data preparation steps
Place raw files such as animelist.csv, anime.csv, and animewithsynopsis.csv in the expected raw data path.
Run the data processing pipeline to generate cleaned datasets and mappings.
Verify that processed outputs such as ratingdf.parquet, animedf.parquet, synopsisdf.csv, and encoding artifacts are created.

## Model training steps
python -m src.basemodel

This step should build or retrain the recommender network, generate training artifacts, and save model-related outputs into the configured artifact directories.

**Model inference steps**
The recommendation pipeline reads trained artifacts, encoded mappings, and processed datasets from artifacts/ before generating ranked suggestions. The API service calls the hybrid pipeline and caches results in Redis for repeated requests.

**Running the API locally**
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

## Testing authentication endpoint
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=demo&password=demo"


