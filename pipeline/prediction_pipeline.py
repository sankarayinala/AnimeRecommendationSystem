"""Prediction pipeline for hybrid anime recommendations.

This module loads model artifacts, runs collaborative and content-based
scoring, applies diversification, and returns ranked recommendations with
explanations.
"""

from time import perf_counter
from typing import Dict, List, Tuple

import joblib
import numpy as np

from config.paths_config import (
    ANIME2ANIME_DECODED_PATH,
    ANIME2ANIME_ENCODED_PATH,
    ANIME_WEIGHTS_PATH,
    DF,
    RATING_DF,
    USER2USER_DECODED_PATH,
    USER2USER_ENCODED_PATH,
    USER_WEIGHTS_PATH,
)
from api.metrics import (
    PIPELINE_CANDIDATE_COUNT,
    PIPELINE_EMPTY_HISTORY,
    PIPELINE_FAILURES,
    PIPELINE_REQUESTS,
    PIPELINE_STAGE_LATENCY,
)
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

_anime_df = None
_rating_df = None
_user2user_encoded = None
_user2user_decoded = None
_anime2anime_encoded = None
_anime2anime_decoded = None
user_weights = None
anime_weights = None
anime_id_to_name = None


def _load_dataframes():
    """Load processed dataframes and weight artifacts once per process."""
    global _anime_df, _rating_df, _user2user_encoded, _user2user_decoded
    global _anime2anime_encoded, _anime2anime_decoded, user_weights, anime_weights, anime_id_to_name

    if _anime_df is not None:
        return

    try:
        _anime_df = load_data(DF)
        _rating_df = load_data(RATING_DF)
    except Exception as e:
        raise CustomException(f"Failed loading processed parquet files: {e}") from e

    _user2user_encoded = joblib.load(USER2USER_ENCODED_PATH)
    _user2user_decoded = joblib.load(USER2USER_DECODED_PATH)
    _anime2anime_encoded = joblib.load(ANIME2ANIME_ENCODED_PATH)
    _anime2anime_decoded = joblib.load(ANIME2ANIME_DECODED_PATH)
    user_weights = joblib.load(USER_WEIGHTS_PATH)
    anime_weights = joblib.load(ANIME_WEIGHTS_PATH)

    anime_id_to_name = dict(zip(_anime_df["anime_id"], _anime_df["eng_version"]))


def _get_user_slice(user_id: int):
    """Return anime IDs and ratings for one user."""
    rows = _rating_df[_rating_df["user_id"] == user_id]
    if rows.empty:
        return np.array([]), np.array([])
    return rows["anime_id"].values, rows["rating"].values


def get_similar_users_faiss(user_id: int, top_k: int = 5) -> List[int]:
    """Return user IDs similar to the requested user."""
    if user_id not in _user2user_encoded:
        return []

    encoded = _user2user_encoded[user_id]
    vec = user_weights[encoded].reshape(1, -1)

    sims = np.dot(user_weights, vec.T).reshape(-1)
    best = np.argsort(sims)[::-1]

    results = []
    for idx in best:
        uid = _user2user_decoded.get(idx)
        if uid is None or uid == user_id:
            continue
        results.append(uid)
        if len(results) >= top_k:
            break
    return results


def popularity_score(anime_id: int) -> float:
    """Compute a simple popularity score for one anime."""
    row = _anime_df[_anime_df["anime_id"] == anime_id]
    if row.empty or "Members" not in row.columns:
        return 0.0
    members = row["Members"].fillna(0).iloc[0]
    return float(np.log1p(members))


def genre_overlap(anime_a: int, anime_b: int) -> float:
    """Return a basic genre overlap score between two anime."""
    try:
        row_a = _anime_df[_anime_df["anime_id"] == anime_a].iloc[0]
        row_b = _anime_df[_anime_df["anime_id"] == anime_b].iloc[0]
        g1 = set(str(row_a.get("Genres", "")).split(","))
        g2 = set(str(row_b.get("Genres", "")).split(","))
        if not g1 or not g2:
            return 0.0
        return len(g1 & g2) / max(len(g1 | g2), 1)
    except Exception:
        return 0.0


def _aggregate_user_scores(similar_users: List[int], watched: set) -> Dict[int, float]:
    """Aggregate candidate anime scores from similar users."""
    scores: Dict[int, float] = {}
    for uid in similar_users:
        rows = _rating_df[_rating_df["user_id"] == uid]
        for _, row in rows.iterrows():
            aid = int(row["anime_id"])
            if aid in watched:
                continue
            scores[aid] = scores.get(aid, 0.0) + float(row["rating"])
    return scores


def _aggregate_content_scores(top_watched: List[int], watched: set) -> Dict[int, float]:
    """Aggregate content-based candidate scores from watched anime."""
    scores: Dict[int, float] = {}
    for aid in top_watched:
        if aid not in _anime_df["anime_id"].values:
            continue
        row = _anime_df[_anime_df["anime_id"] == aid].iloc[0]
        genres = set(str(row.get("Genres", "")).split(","))
        if not genres:
            continue
        for _, cand in _anime_df.iterrows():
            cid = int(cand["anime_id"])
            if cid in watched or cid == aid:
                continue
            cand_genres = set(str(cand.get("Genres", "")).split(","))
            if not cand_genres:
                continue
            overlap = len(genres & cand_genres)
            if overlap > 0:
                scores[cid] = scores.get(cid, 0.0) + float(overlap)
    return scores


def mmr(ranked: List[Tuple[int, float]], emb_matrix, lambda_mmr: float = 0.7, top_k: int = 10):
    """Apply a simple MMR reranking step."""
    selected = []
    candidates = ranked.copy()

    while candidates and len(selected) < top_k:
        if not selected:
            selected.append(candidates.pop(0))
            continue

        best_item = None
        best_score = -1e9

        for item in candidates:
            aid, score = item
            diversity = 0.0
            for sid, _ in selected:
                diversity = max(diversity, genre_overlap(aid, sid))
            mmr_score = lambda_mmr * score - (1 - lambda_mmr) * diversity
            if mmr_score > best_score:
                best_score = mmr_score
                best_item = item

        candidates.remove(best_item)
        selected.append(best_item)

    return selected


def hybrid_recommendation(
    user_id: int,
    user_weight: float = 0.6,
    content_weight: float = 0.4,
    top_k: int = 10,
    mmr_lambda: float = 0.7,
):
    """Generate hybrid recommendations for one user."""
    PIPELINE_REQUESTS.inc()
    total_start = perf_counter()

    logger.info(
        f"Generating recommendations for user {user_id} "
        f"(user_w={user_weight:.2f}, content_w={content_weight:.2f}, top_k={top_k})"
    )

    try:
        stage_start = perf_counter()
        _load_dataframes()
        PIPELINE_STAGE_LATENCY.labels(stage="load_dataframes").observe(perf_counter() - stage_start)

        stage_start = perf_counter()
        similar_users = get_similar_users_faiss(user_id, top_k=5)
        PIPELINE_STAGE_LATENCY.labels(stage="similar_users").observe(perf_counter() - stage_start)

        stage_start = perf_counter()
        user_anime_ids, user_ratings = _get_user_slice(user_id)
        PIPELINE_STAGE_LATENCY.labels(stage="load_user_history").observe(perf_counter() - stage_start)

        if user_anime_ids.size == 0:
            PIPELINE_EMPTY_HISTORY.inc()
            return {"recommendations": [], "explanations": {}}

        watched = set(user_anime_ids.tolist())

        stage_start = perf_counter()
        user_scores = _aggregate_user_scores(similar_users=similar_users, watched=watched)
        PIPELINE_STAGE_LATENCY.labels(stage="user_scoring").observe(perf_counter() - stage_start)
        PIPELINE_CANDIDATE_COUNT.labels(source="user_scores").observe(len(user_scores))

        stage_start = perf_counter()
        top_watched = user_anime_ids[np.argsort(user_ratings)[-8:][::-1]].astype(np.int32).tolist() if user_ratings.size > 0 else []
        content_scores = _aggregate_content_scores(top_watched=top_watched, watched=watched)
        PIPELINE_STAGE_LATENCY.labels(stage="content_scoring").observe(perf_counter() - stage_start)
        PIPELINE_CANDIDATE_COUNT.labels(source="content_scores").observe(len(content_scores))

        stage_start = perf_counter()
        combined: Dict[int, float] = {}
        explanations: Dict[int, dict] = {}

        max_user = max(user_scores.values()) if user_scores else 1.0
        max_content = max(content_scores.values()) if content_scores else 1.0
        candidate_ids = set(user_scores.keys()) | set(content_scores.keys())
        PIPELINE_CANDIDATE_COUNT.labels(source="combined_candidates").observe(len(candidate_ids))

        seed_subset = top_watched[:3] if top_watched else []

        for aid in candidate_ids:
            u = user_scores.get(aid, 0.0) / max_user
            c = content_scores.get(aid, 0.0) / max_content
            pop = popularity_score(aid)
            genre_boost = max((genre_overlap(aid, sid) for sid in seed_subset), default=0.0)

            score = (user_weight * u) + (content_weight * c) + (0.05 * pop) + (0.10 * genre_boost)
            combined[aid] = float(score)
            explanations[aid] = {
                "raw_user_score": float(u),
                "raw_content_score": float(c),
                "popularity": float(pop),
                "genre_overlap": float(genre_boost),
            }

        PIPELINE_STAGE_LATENCY.labels(stage="hybrid_scoring").observe(perf_counter() - stage_start)

        stage_start = perf_counter()
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[: max(top_k * 3, top_k)]
        reranked = mmr(ranked, anime_weights, lambda_mmr=mmr_lambda, top_k=top_k)
        PIPELINE_STAGE_LATENCY.labels(stage="ranking_mmr").observe(perf_counter() - stage_start)

        stage_start = perf_counter()
        recommendations: List[str] = []
        final_explanations: Dict[str, dict] = {}

        for aid, _score in reranked:
            anime_name = anime_id_to_name.get(aid)
            if not anime_name:
                continue
            recommendations.append(str(anime_name))
            final_explanations[str(aid)] = explanations.get(aid, {})

        PIPELINE_STAGE_LATENCY.labels(stage="format_response").observe(perf_counter() - stage_start)
        PIPELINE_CANDIDATE_COUNT.labels(source="final_recommendations").observe(len(recommendations))
        PIPELINE_STAGE_LATENCY.labels(stage="total_pipeline").observe(perf_counter() - total_start)

        return {
            "recommendations": recommendations,
            "explanations": final_explanations,
        }

    except Exception:
        PIPELINE_FAILURES.inc()
        logger.exception("Pipeline execution failed")
        raise


def warmup_dataframes():
    """Warm the global caches at startup."""
    _load_dataframes()