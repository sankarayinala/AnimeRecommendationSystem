"""Redis cache helpers for the anime recommendation API.

This module stores serialized responses and exposes small helper functions
for key listing, TTL inspection, and user-level invalidation.
"""

import json
import os
import redis


def _redis_port() -> int:
    """Resolve Redis port from environment values, including tcp:// style inputs."""
    raw = os.getenv("REDIS_PORT", "6379")
    if raw.startswith("tcp://"):
        return int(raw.rsplit(":", 1)[1])
    return int(raw)


cache = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis"),
    port=_redis_port(),
    db=0,
    decode_responses=True,
)


def get_cached(key: str):
    """Load and deserialize a cached JSON response."""
    v = cache.get(key)
    return json.loads(v) if v else None


def set_cached(key: str, value, ttl: int = 300):
    """Serialize and store a response in Redis with a TTL."""
    cache.set(key, json.dumps(value), ex=ttl)


def list_keys(pattern="*"):
    """Return Redis keys matching the provided pattern."""
    return cache.keys(pattern)


def get_ttl(key: str):
    """Return the remaining TTL for a key in seconds."""
    return cache.ttl(key)


def delete_key(key: str):
    """Delete a key from Redis."""
    return cache.delete(key)


def invalidate_user_cache(user_id: int):
    """Delete all recommendation cache entries for one user."""
    keys = list_keys(f"rec:{user_id}:*")
    for k in keys:
        delete_key(k)
    return len(keys)