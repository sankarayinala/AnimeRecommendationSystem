# AnimeRecommendationSystem
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


## Testing recommendation endpoint

curl -X GET "http://localhost:8000/recommend/11880?userweight=0.6&contentweight=0.4&topk=10" \
  -H "Authorization: Bearer <ACCESS_TOKEN>"

## Docker image build steps
docker build -t anime-recommender-backend:latest .

## Docker container run steps
docker run -p 8000:8000 \
  -e REDISHOST=redis \
  -e REDISPORT=6379 \
  anime-recommender-backend:latest

## Pushing Docker image

docker tag anime-recommender-backend:latest <registry>/anime-recommender-backend:latest
docker push <registry>/anime-recommender-backend:latest  

## Kubernetes deployment steps
kubectl apply -f redis.yaml
kubectl apply -f backend-deployment.yaml
kubectl apply -f ui-deployment.yaml
kubectl apply -f prometheus-observability.yaml
kubectl apply -f grafana-datasource.yaml

## Redis deployment steps
kubectl apply -f redis.yaml
kubectl get pods
kubectl get svc

## Prometheus/Grafana observability setup
Apply the observability manifests, confirm Prometheus is running in the observability namespace, and add the Grafana datasource config map if Grafana is already installed. The backend is annotated for Prometheus scraping on /metrics, so the metrics endpoint should be visible once the service is reachable.

## Health check and metrics endpoint testing
curl http://localhost:8000/healthz
curl http://localhost:8000/metrics

## Troubleshooting common errors

- Missing artifact files usually mean the training or preprocessing step did not finish successfully.
- Redis connection errors usually point to a host or service-name mismatch.
- Authentication failures usually indicate bad demo credentials or a missing bearer token.
- Empty recommendations often mean the user has no history in the rating dataset or the input weights are too restrictive.

## Git commit checklist
Verify code formatting and imports.
Confirm all docstrings and section comments are in place.
Check that README.md matches the actual run commands.
Validate Docker build and Kubernetes manifests.
Ensure secrets are not committed to the repository.

## Future Enhancements
  - Replace demo login with proper user management.
  - Externalize all secrets to Kubernetes secrets or environment injection.
  - Add automated tests for API, cache, and pipeline behavior.
  - Add CI checks for formatting, linting, and smoke tests.
  - Expand explainability with richer recommendation reasons.

## Deployment and testing commands

### Virtual environment
python -m venv .venv
source .venv/bin/activate

### Install dependencies
pip install -r requirements.txt

### Train / process data
python -m src.basemodel

### Run backend
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

# Build Docker image
docker build -t anime-recommender-backend:latest .

# Push image
docker tag anime-recommender-backend:latest <registry>/anime-recommender-backend:latest
docker push <registry>/anime-recommender-backend:latest

# Apply Kubernetes manifests
kubectl apply -f redis.yaml
kubectl apply -f backend-deployment.yaml
kubectl apply -f ui-deployment.yaml
kubectl apply -f prometheus-observability.yaml
kubectl apply -f grafana-datasource.yaml

# Check pods and services
kubectl get pods
kubectl get svc

# View logs
kubectl logs deployment/backend
kubectl logs deployment/ui

# Port forwarding
kubectl port-forward svc/backend-service 8000:8000
kubectl port-forward svc/ui-service 8501:8501

# Test auth
curl -X POST "http://localhost:8000/auth/login" -H "Content-Type: application/x-www-form-urlencoded" -d "username=demo&password=demo"

# Test recommendations
curl -X GET "http://localhost:8000/recommend/11880?userweight=0.6&contentweight=0.4&topk=10" -H "Authorization: Bearer <ACCESS_TOKEN>"

# Test metrics
curl http://localhost:8000/metrics
