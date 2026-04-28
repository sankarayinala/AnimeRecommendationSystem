# Anime Recommendation System

## Project Overview
Anime Recommendation System is a hybrid recommendation platform that combines collaborative filtering, content-based ranking, Redis caching, FAISS/vector search, FastAPI, Streamlit, Docker, Kubernetes, and Prometheus observability.

## Architecture Summary
The system follows an end-to-end flow:
1. Raw anime and rating data is processed into training artifacts.
2. The model is trained and embeddings are saved.
3. The FastAPI backend serves recommendations and authentication.
4. Redis caches recommendation responses.
5. Streamlit provides the user interface.
6. Prometheus scrapes backend metrics for observability.

## Key Features
- Hybrid anime recommendations using collaborative and content-based signals.
- JWT authentication for protected endpoints.
- Redis-backed caching with user-level invalidation.
- FAISS-based similarity search for fast nearest-neighbor lookup.
- Prometheus metrics for latency, cache, and pipeline monitoring.
- Docker and Kubernetes deployment support.

## Technology Stack
- Python 3.8+
- FastAPI
- Streamlit
- Redis
- FAISS CPU
- TensorFlow / Keras
- Pandas, NumPy, scikit-learn, Joblib
- Prometheus client libraries
- Docker
- Kubernetes

## Folder Structure
- `api/` FastAPI app, auth, cache, metrics, and schemas.
- `config/` environment and path configuration.
- `src/` preprocessing, model construction, and logging.
- `pipeline/` recommendation inference pipeline.
- `ui/` Streamlit UI pages and helpers.
- `utils/` shared helper functions.
- `artifacts/` raw, processed, model, and weight files.
- `*.yaml` Kubernetes deployment manifests.

## Prerequisites
- Python 3.8 or later
- Docker
- Kubernetes cluster or local Kubernetes environment
- Redis
- Access to the processed model artifacts
- Internet access for poster lookups in the UI

## Environment Setup
1. Clone the repository.
2. Create a virtual environment.
3. Install dependencies.
4. Generate or copy the processed artifacts into `artifacts/`.
5. Start Redis.
6. Start the backend API.
7. Start the Streamlit UI.

## Python virtual environment setup
```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

## Required environment variables
- `API_HOST`
- `API_PORT`
- `REDIS_HOST`
- `REDIS_PORT`
- `STREAMLIT_HOST`
- `STREAMLIT_PORT`
- `AUTH_TIMEOUT_SECONDS`
- `READ_TIMEOUT_SECONDS`
- `LOG_LEVEL`
- `LOG_DIR`

## Dependency installation
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Data preparation steps
1. Place raw files in the expected raw artifact directory.
2. Run the data processing pipeline.
3. Verify the following files are generated:
   - processed rating dataframe
   - anime metadata dataframe
   - synopsis export
   - user and anime encoding mappings
   - train/test arrays

## Model training steps
```bash
python -m src.data_processing
python -m src.training
```

If your training entry point differs, update the command to match the final trainer script in your repo.

## Model inference steps
The API reads the saved model, embeddings, and mapping artifacts from the `artifacts/` folder. The recommendation endpoint uses the trained artifacts and may cache results in Redis for repeated requests.

## Running the API locally
```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

## Testing authentication endpoint
```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=demo&password=demo"
```

## Testing recommendation endpoint
```bash
curl -X GET "http://localhost:8000/recommend/11880?user_weight=0.6&content_weight=0.4&top_k=10" \
  -H "Authorization: Bearer <ACCESS_TOKEN>"
```

## Docker image build steps
```bash
docker build -t anime-recommender-backend:latest .
```

## Docker container run steps
```bash
docker run -p 8000:8000 \
  -e REDIS_HOST=redis \
  -e REDIS_PORT=6379 \
  anime-recommender-backend:latest
```

## Pushing Docker image
```bash
docker tag anime-recommender-backend:latest <registry>/anime-recommender-backend:latest
docker push <registry>/anime-recommender-backend:latest
```

## Kubernetes deployment steps
```bash
kubectl apply -f redis.yaml
kubectl apply -f backend-deployment.yaml
kubectl apply -f ui-deployment.yaml
kubectl apply -f prometheus-observability.yaml
kubectl apply -f grafana-datasource.yaml
```

## Redis deployment steps
```bash
kubectl apply -f redis.yaml
kubectl get pods
kubectl get svc
```

## Prometheus/Grafana observability setup
Apply the observability manifests to the `observability` namespace and confirm Prometheus is reachable on the configured NodePort. Import the Grafana datasource config if Grafana is already deployed.

## Health check and metrics endpoint testing
```bash
curl http://localhost:8000/healthz
curl http://localhost:8000/metrics
```

## Troubleshooting common errors
- Missing artifact files usually mean preprocessing or training did not complete.
- Redis connection errors usually mean the service name or port is incorrect.
- Authentication failures usually indicate invalid demo credentials or a missing bearer token.
- Empty recommendations usually mean the user has no history in the rating dataset.

## Git commit checklist
- Verify no secrets are committed.
- Confirm docstrings and comments are added.
- Check that commands match the actual file names.
- Run a local smoke test for API and UI.
- Validate Docker and Kubernetes manifests.

## Future enhancements
- Replace demo authentication with real user management.
- Move all secrets to environment variables or secret management.
- Add automated tests for API, cache, and pipeline behavior.
- Add CI linting and formatting checks.
- Expand recommendation explanations.

## Deployment and testing commands
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

docker build -t anime-recommender-backend:latest .
docker tag anime-recommender-backend:latest <registry>/anime-recommender-backend:latest
docker push <registry>/anime-recommender-backend:latest

kubectl apply -f redis.yaml
kubectl apply -f backend-deployment.yaml
kubectl apply -f ui-deployment.yaml
kubectl apply -f prometheus-observability.yaml
kubectl apply -f grafana-datasource.yaml

kubectl get pods
kubectl get svc
kubectl logs deployment/backend
kubectl logs deployment/ui

kubectl port-forward svc/backend-service 8000:8000
kubectl port-forward svc/ui-service 8501:8501

curl -X POST "http://localhost:8000/auth/login" -H "Content-Type: application/x-www-form-urlencoded" -d "username=demo&password=demo"
curl -X GET "http://localhost:8000/recommend/11880?user_weight=0.6&content_weight=0.4&top_k=10" -H "Authorization: Bearer <ACCESS_TOKEN>"
curl http://localhost:8000/metrics
```

### Test metrics
curl http://localhost:8000/metrics


# Deep dive into coding :

### File: src/data_processing.py
**Purpose:** Cleans rating and anime metadata, encodes IDs, creates train/test splits, and saves all processed artifacts for training and inference.

### File: src/base_model.py
**Purpose:** Builds the neural collaborative filtering model used for anime recommendation training and inference.

### File: api/auth.py
**Purpose:** Creates and validates JWT access tokens for protected API endpoints.

### File: api/cache.py
**Purpose:** Wraps Redis operations for storing recommendations, listing keys, reading TTLs, and invalidating user-specific cache entries.

### File: api/models.py
**Purpose:** Defines response schemas used by the API

### File: api/server.py
**Purpose:** Hosts the FastAPI application, health and metrics endpoints, login, recommendation generation, and admin cache operations.

### File: api/metrics.py
**Purpose:** Defines all Prometheus counters, gauges, and histograms used by the API and recommendation pipeline

### File: ui/app.py
**Purpose:** Provides the Streamlit user interface for login, search, recommendation requests, and recommendation display

