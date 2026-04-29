"""Prometheus metrics for the anime recommendation service."""

from prometheus_client import Counter, Gauge, Histogram

# -------------------------------------------------
# API-level metrics
# -------------------------------------------------
REQUEST_COUNT = Counter(
    "anime_api_requests_total",
    "Total API requests",
    ["endpoint", "method", "status"],
)

REQUEST_ERRORS = Counter(
    "anime_api_request_errors_total",
    "Total API request errors",
    ["endpoint", "error_type"],
)

REQUEST_LATENCY = Histogram(
    "anime_api_request_latency_seconds",
    "API request latency in seconds",
    ["endpoint", "method"],
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 30),
)

INFLIGHT_REQUESTS = Gauge(
    "anime_api_inflight_requests",
    "Current in-flight API requests",
    ["endpoint"],
)

CACHE_HITS = Counter(
    "anime_cache_hits_total",
    "Total cache hits",
    ["endpoint"],
)

CACHE_MISSES = Counter(
    "anime_cache_misses_total",
    "Total cache misses",
    ["endpoint"],
)

EMPTY_RESULTS = Counter(
    "anime_empty_results_total",
    "Total empty recommendation responses",
    ["endpoint"],
)

RECOMMENDATION_COUNT = Histogram(
    "anime_recommendation_count",
    "Number of recommendations returned per request",
    ["endpoint"],
    buckets=(0, 1, 3, 5, 10, 15, 20, 30, 50),
)

# -------------------------------------------------
# App state / readiness
# -------------------------------------------------
APP_WARMUP_STATE = Gauge(
    "anime_app_warmup_state",
    "Application warmup state: 0=starting, 1=ready, -1=failed",
)

MODEL_READY_STATE = Gauge(
    "anime_model_ready_state",
    "Model/index readiness state: 0=not ready, 1=ready",
)

DATAFRAME_READY_STATE = Gauge(
    "anime_dataframe_ready_state",
    "Dataframe cache readiness state: 0=not ready, 1=ready",
)

EMBEDDINGS_READY_STATE = Gauge(
    "anime_embeddings_ready_state",
    "Embedding readiness state: 0=not ready, 1=ready",
)

FAISS_INDEX_READY_STATE = Gauge(
    "anime_faiss_index_ready_state",
    "FAISS index readiness state: 0=not ready, 1=ready",
)

# -------------------------------------------------
# Pipeline metrics
# -------------------------------------------------
PIPELINE_REQUESTS = Counter(
    "anime_pipeline_requests_total",
    "Total recommendation pipeline executions",
)

PIPELINE_FAILURES = Counter(
    "anime_pipeline_failures_total",
    "Total recommendation pipeline failures",
)

PIPELINE_EMPTY_HISTORY = Counter(
    "anime_pipeline_empty_history_total",
    "Total requests where user had no rating history",
)

PIPELINE_STAGE_LATENCY = Histogram(
    "anime_pipeline_stage_latency_seconds",
    "Latency of pipeline stages in seconds",
    ["stage"],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 30),
)

PIPELINE_CANDIDATE_COUNT = Histogram(
    "anime_pipeline_candidate_count",
    "Number of candidates produced at each pipeline stage",
    ["source"],
    buckets=(0, 1, 5, 10, 20, 50, 100, 250, 500, 1000, 5000),
)

# -------------------------------------------------
# FAISS metrics
# -------------------------------------------------
FAISS_USER_LATENCY = Histogram(
    "anime_faiss_user_search_latency_seconds",
    "Latency of FAISS user similarity search",
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5),
)

FAISS_ANIME_LATENCY = Histogram(
    "anime_faiss_anime_search_latency_seconds",
    "Latency of FAISS anime similarity search",
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5),
)