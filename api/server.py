"""FastAPI application for the anime recommendation system.

This module exposes auth, recommendation, health, metrics, and admin cache
endpoints while tracking request and pipeline telemetry.
"""

from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.security import OAuth2PasswordRequestForm
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.concurrency import run_in_threadpool

from api.auth import authenticate, create_access_token
from api.cache import (
    delete_key,
    get_cached,
    get_ttl,
    invalidate_user_cache,
    list_keys,
    set_cached,
)
from api.metrics import (
    APP_WARMUP_STATE,
    CACHE_HITS,
    CACHE_MISSES,
    EMPTY_RESULTS,
    INFLIGHT_REQUESTS,
    MODEL_READY_STATE,
    RECOMMENDATION_COUNT,
    REQUEST_COUNT,
    REQUEST_ERRORS,
    REQUEST_LATENCY,
)
from pipeline.prediction_pipeline import hybrid_recommendation
from src.logger import get_logger

logger = get_logger(__name__)
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up dataframes and set readiness gauges before serving traffic."""
    APP_WARMUP_STATE.set(0)
    try:
        from pipeline.prediction_pipeline import _load_dataframes

        _load_dataframes()
        APP_WARMUP_STATE.set(1)
        MODEL_READY_STATE.set(1)
        logger.info("Application warmup completed successfully")
    except Exception:
        APP_WARMUP_STATE.set(-1)
        MODEL_READY_STATE.set(0)
        logger.exception("Warmup failed")
        raise
    yield


app = FastAPI(
    title="Anime Recommendation API",
    description="Hybrid Anime Recommendation API with explanations, caching, and admin tools",
    version="3.7.0",
    lifespan=lifespan,
)

app.state.limiter = limiter


@app.get("/healthz", tags=["System"])
def healthz():
    """Return basic service health and warmup readiness."""
    return {
        "status": "ok",
        "warmup_state": "ready" if APP_WARMUP_STATE._value.get() == 1 else "not_ready",
        "model_ready": True if MODEL_READY_STATE._value.get() == 1 else False,
    }


@app.get("/metrics", tags=["System"])
def metrics():
    """Expose Prometheus metrics for scraping."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/", tags=["System"])
def root():
    """Return a simple service status message."""
    return {"message": "Anime Recommendation API is running!"}


@app.get("/auth/login", tags=["Auth"])
def login_info():
    """Explain how to obtain a JWT token."""
    return {"message": "POST /auth/login with username & password"}


@app.post("/auth/login", tags=["Auth"])
def login(form: OAuth2PasswordRequestForm = Depends()):
    """Validate demo credentials and return a bearer token."""
    endpoint = "/auth/login"
    method = "POST"

    start = perf_counter()
    INFLIGHT_REQUESTS.labels(endpoint=endpoint).inc()

    try:
        username = form.username
        password = form.password

        if username != "demo" or password != "demo":
            REQUEST_COUNT.labels(endpoint=endpoint, method=method, status="401").inc()
            REQUEST_ERRORS.labels(endpoint=endpoint, error_type="invalid_credentials").inc()
            raise HTTPException(status_code=401, detail="Invalid credentials")

        token = create_access_token({"username": username})
        REQUEST_COUNT.labels(endpoint=endpoint, method=method, status="200").inc()
        return {"access_token": token, "token_type": "bearer"}

    except HTTPException:
        raise
    except Exception as exc:
        REQUEST_COUNT.labels(endpoint=endpoint, method=method, status="500").inc()
        REQUEST_ERRORS.labels(endpoint=endpoint, error_type=type(exc).__name__).inc()
        logger.exception("Login request failed")
        raise HTTPException(status_code=500, detail="Login failed") from exc
    finally:
        REQUEST_LATENCY.labels(endpoint=endpoint, method=method).observe(perf_counter() - start)
        INFLIGHT_REQUESTS.labels(endpoint=endpoint).dec()


@app.get("/recommend/{user_id}", tags=["Recommend"])
@limiter.limit("20/minute")
async def recommend(
    request: Request,
    user_id: int,
    user_weight: float = 0.6,
    content_weight: float = 0.4,
    top_k: int = 10,
    user=Depends(authenticate),
):
    """Return cached or freshly generated recommendations for one user."""
    endpoint = "/recommend/{user_id}"
    method = "GET"

    INFLIGHT_REQUESTS.labels(endpoint=endpoint).inc()
    start = perf_counter()

    try:
        cache_key = f"rec:{user_id}:{user_weight:.2f}:{content_weight:.2f}:{top_k}"

        cached = get_cached(cache_key)
        if cached:
            CACHE_HITS.labels(endpoint=endpoint).inc()

            recs = cached.get("recommendations", [])
            RECOMMENDATION_COUNT.labels(endpoint=endpoint).observe(len(recs))

            if len(recs) == 0:
                EMPTY_RESULTS.labels(endpoint=endpoint).inc()

            REQUEST_COUNT.labels(endpoint=endpoint, method=method, status="200").inc()

            logger.info(
                f"Cache hit for user_id={user_id}, top_k={top_k}, "
                f"elapsed={(perf_counter() - start):.2f}s"
            )
            return cached

        CACHE_MISSES.labels(endpoint=endpoint).inc()

        result = await run_in_threadpool(
            hybrid_recommendation,
            user_id,
            user_weight,
            content_weight,
            top_k,
        )

        recs = result.get("recommendations", [])
        RECOMMENDATION_COUNT.labels(endpoint=endpoint).observe(len(recs))

        if len(recs) == 0:
            EMPTY_RESULTS.labels(endpoint=endpoint).inc()

        set_cached(cache_key, result, ttl=300)

        REQUEST_COUNT.labels(endpoint=endpoint, method=method, status="200").inc()

        logger.info(
            f"Generated recommendations for user_id={user_id}, count={len(recs)}, "
            f"elapsed={(perf_counter() - start):.2f}s"
        )
        return result

    except HTTPException as exc:
        REQUEST_COUNT.labels(endpoint=endpoint, method=method, status=str(exc.status_code)).inc()
        REQUEST_ERRORS.labels(endpoint=endpoint, error_type="HTTPException").inc()
        raise
    except Exception as exc:
        REQUEST_COUNT.labels(endpoint=endpoint, method=method, status="500").inc()
        REQUEST_ERRORS.labels(endpoint=endpoint, error_type=type(exc).__name__).inc()
        logger.exception("Recommendation request failed")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {exc}") from exc
    finally:
        REQUEST_LATENCY.labels(endpoint=endpoint, method=method).observe(perf_counter() - start)
        INFLIGHT_REQUESTS.labels(endpoint=endpoint).dec()


@app.get("/admin/cache/keys", tags=["Admin"])
def admin_list_keys(user=Depends(authenticate)):
    """List all cache keys and their TTLs."""
    endpoint = "/admin/cache/keys"
    method = "GET"
    start = perf_counter()
    INFLIGHT_REQUESTS.labels(endpoint=endpoint).inc()

    try:
        keys = list_keys("*")
        REQUEST_COUNT.labels(endpoint=endpoint, method=method, status="200").inc()
        return [{"key": k, "ttl": get_ttl(k)} for k in keys]
    except Exception as exc:
        REQUEST_COUNT.labels(endpoint=endpoint, method=method, status="500").inc()
        REQUEST_ERRORS.labels(endpoint=endpoint, error_type=type(exc).__name__).inc()
        logger.exception("Failed to list cache keys")
        raise HTTPException(status_code=500, detail="Failed to list cache keys") from exc
    finally:
        REQUEST_LATENCY.labels(endpoint=endpoint, method=method).observe(perf_counter() - start)
        INFLIGHT_REQUESTS.labels(endpoint=endpoint).dec()


@app.delete("/admin/cache/key/{key}", tags=["Admin"])
def admin_delete_key(key: str, user=Depends(authenticate)):
    """Delete one cache key."""
    endpoint = "/admin/cache/key/{key}"
    method = "DELETE"
    start = perf_counter()
    INFLIGHT_REQUESTS.labels(endpoint=endpoint).inc()

    try:
        deleted = delete_key(key)
        REQUEST_COUNT.labels(endpoint=endpoint, method=method, status="200").inc()
        return {"deleted": deleted, "key": key}
    except Exception as exc:
        REQUEST_COUNT.labels(endpoint=endpoint, method=method, status="500").inc()
        REQUEST_ERRORS.labels(endpoint=endpoint, error_type=type(exc).__name__).inc()
        logger.exception("Failed to delete cache key")
        raise HTTPException(status_code=500, detail="Failed to delete cache key") from exc
    finally:
        REQUEST_LATENCY.labels(endpoint=endpoint, method=method).observe(perf_counter() - start)
        INFLIGHT_REQUESTS.labels(endpoint=endpoint).dec()


@app.delete("/admin/cache/user/{user_id}", tags=["Admin"])
def admin_delete_user_cache(user_id: int, user=Depends(authenticate)):
    """Delete every cached recommendation entry for one user."""
    endpoint = "/admin/cache/user/{user_id}"
    method = "DELETE"
    start = perf_counter()
    INFLIGHT_REQUESTS.labels(endpoint=endpoint).inc()

    try:
        removed = invalidate_user_cache(user_id)
        REQUEST_COUNT.labels(endpoint=endpoint, method=method, status="200").inc()
        return {"user_id": user_id, "removed_keys": removed}
    except Exception as exc:
        REQUEST_COUNT.labels(endpoint=endpoint, method=method, status="500").inc()
        REQUEST_ERRORS.labels(endpoint=endpoint, error_type=type(exc).__name__).inc()
        logger.exception("Failed to invalidate user cache")
        raise HTTPException(status_code=500, detail="Failed to invalidate user cache") from exc
    finally:
        REQUEST_LATENCY.labels(endpoint=endpoint, method=method).observe(perf_counter() - start)
        INFLIGHT_REQUESTS.labels(endpoint=endpoint).dec()